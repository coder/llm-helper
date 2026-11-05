package detect

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/coder/llm-helper/internal/config"
)

// Server is one detected vLLM server and everything we learned about it.
// Cmdline is deliberately kept out of JSON: it can contain --api-key values.
type Server struct {
	PID     int      `json:"pid"`
	Cmdline []string `json:"-"`

	// Parsed from the command line:
	Model          string `json:"model"`
	Port           string `json:"port"`
	MaxModelLen    string `json:"max_model_len,omitempty"`
	MemUtilization string `json:"mem_utilization,omitempty"` // --gpu-memory-utilization
	ExtraArgs      string `json:"extra_args,omitempty"`      // every other serving flag, reassembled
	APIKeySet      bool   `json:"api_key_set"`

	// How it's being run:
	VLLMBin       string `json:"vllm_bin,omitempty"`       // resolved vllm executable ("" if unknown)
	SystemdUnit   string `json:"systemd_unit,omitempty"`   // managing unit: systemd unit or launchd label ("" if none)
	UserScope     bool   `json:"user_scope"`               // true if under the user manager (user@.service slice, or a launchd user agent)
	InContainer   bool   `json:"in_container"`             // cgroup says it's inside a container
	ContainerHint string `json:"container_hint,omitempty"` // raw cgroup fragment for container correlation
}

// vLLM's OpenAI server shows up either as "... vllm serve <model> ..." or
// "python -m vllm.entrypoints.openai.api_server --model <model> ...".
func looksLikeVLLM(args []string) bool {
	joined := strings.Join(args, " ")
	return strings.Contains(joined, "vllm serve") ||
		strings.Contains(joined, "vllm.entrypoints.openai.api_server")
}

// rawProc is one process as the platform reported it.
type rawProc struct {
	PID  int
	Args []string
}

// ScanProcesses finds running vLLM servers. Worker subprocesses (spawned
// engine processes share the same cmdline root) are deduped by keeping only
// the lowest PID per port.
func ScanProcesses() []Server {
	byPort := map[string]Server{}
	for _, p := range listProcesses() {
		if !looksLikeVLLM(p.Args) {
			continue
		}
		s := parseServer(p.PID, p.Args)
		enrich(&s)
		key := s.Port
		if prev, ok := byPort[key]; !ok || s.PID < prev.PID {
			byPort[key] = s
		}
	}
	out := make([]Server, 0, len(byPort))
	for _, s := range byPort {
		out = append(out, s)
	}
	return out
}

// valuelessFlags never consume the following argument. Any other flag is
// assumed to take a value when the next argument doesn't look like a flag,
// which is how unrecognized flags get carried over intact.
var valuelessFlags = map[string]bool{
	"--enable-auto-tool-choice": true,
	"--enable-prefix-caching":   true,
	"--trust-remote-code":       true,
}

// LooksSecret reports whether a flag carries a credential. Those are left
// out of ExtraArgs entirely: it lands in a profile file and in
// 'doctor --json', which is the same reason Cmdline is withheld from JSON.
// Matching on the trailing word keeps real flags like --tokenizer-mode and
// --ssl-keyfile from being mistaken for secrets.
func LooksSecret(flag string) bool {
	for _, suffix := range []string{"-key", "-token", "-secret", "-password"} {
		if strings.HasSuffix(strings.ToLower(flag), suffix) {
			return true
		}
	}
	return false
}

// looksLikeValue distinguishes a flag's value from the next flag. Negative
// numbers are values despite the leading dash (e.g. --seed -1).
func looksLikeValue(s string) bool {
	if s == "" {
		return false
	}
	if !strings.HasPrefix(s, "-") {
		return true
	}
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

func parseServer(pid int, args []string) Server {
	s := Server{PID: pid, Cmdline: args, Port: "8000"} // vllm's default port
	var extra []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		next := func() string {
			if i+1 < len(args) {
				i++
				return args[i]
			}
			return ""
		}
		switch {
		case a == "serve":
			// Positional model follows "serve" unless it's another flag.
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
				i++
				s.Model = args[i]
			}
		case a == "--model":
			s.Model = next()
		case a == "--port":
			s.Port = next()
		case a == "--max-model-len":
			s.MaxModelLen = next()
		case a == "--gpu-memory-utilization":
			s.MemUtilization = next()
		case a == "--api-key":
			next()
			s.APIKeySet = true
		case strings.HasPrefix(a, "--") && strings.Contains(a, "="):
			// --flag=value form
			flag, val, _ := strings.Cut(a, "=")
			switch {
			case flag == "--model":
				s.Model = val
			case flag == "--port":
				s.Port = val
			case flag == "--max-model-len":
				s.MaxModelLen = val
			case flag == "--gpu-memory-utilization":
				s.MemUtilization = val
			case flag == "--api-key":
				s.APIKeySet = true
			case valuelessFlags[flag]: // e.g. --trust-remote-code=true
				extra = append(extra, flag)
			case LooksSecret(flag):
				// Deliberately dropped, like --api-key.
			default:
				extra = append(extra, flag+"="+config.Quote(val))
			}
		case valuelessFlags[a]:
			extra = append(extra, a)
		case strings.HasPrefix(a, "--"):
			// Anything else the server was started with: carry it over so
			// the profile can reproduce this configuration.
			hasValue := i+1 < len(args) && looksLikeValue(args[i+1])
			if LooksSecret(a) {
				if hasValue {
					next() // consume it so it can't be read as a flag
				}
				break
			}
			if hasValue {
				extra = append(extra, a+" "+config.Quote(next()))
			} else {
				extra = append(extra, a)
			}
		}
	}
	s.ExtraArgs = strings.Join(extra, " ")
	return s
}

// enrich adds context that isn't on the command line: the vllm binary, the
// service unit (if any), and whether the process lives in a container.
func enrich(s *Server) {
	// Find the vllm console script in argv. The process's own executable is
	// useless here: venv pythons symlink to the system interpreter.
	for _, a := range s.Cmdline {
		if strings.HasSuffix(a, "/vllm") && isExecutable(a) {
			s.VLLMBin = a
			break
		}
	}
	if s.VLLMBin == "" && len(s.Cmdline) > 0 {
		if cand := filepath.Join(filepath.Dir(s.Cmdline[0]), "vllm"); isExecutable(cand) {
			s.VLLMBin = cand
		}
	}
	enrichPlatform(s)
}

func isExecutable(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir() && info.Mode()&0o111 != 0
}
