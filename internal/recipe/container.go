package recipe

import (
	"fmt"
	"sort"
	"strings"

	"github.com/coder/llm-helper/internal/engine"
)

// ContainerSpec is everything needed to start one node of a recipe. RankSpec
// builds it from the recipe and its engine.
type ContainerSpec struct {
	Name  string
	Image string
	// Entrypoint replaces the image's own; empty keeps it.
	Entrypoint string
	// Flags are the engine's docker run flags.
	Flags []string
	// Mounts are host directories, as absolute paths, bind-mounted rather
	// than baked in so a re-launch does not re-download weights or re-JIT
	// kernels, which is most of the startup cost.
	Mounts []engine.Mount
	Env    map[string]string
	Args   []string
	// APIKey is kept out of Env so it can be redacted when the command is
	// printed, and passed as KeyEnv rather than a flag. Docker is told only
	// the variable's name (-e VLLM_API_KEY) and reads the value from its own
	// environment, so the key is on no process command line, docker's
	// included.
	APIKey string
	KeyEnv string
}

// RunArgs renders the argv for "docker run". Env is emitted in sorted order so
// the rendered command is stable and two launches can be diffed against
// each other. With an APIKey set, the caller must put KeyEnv in docker's
// environment; ShellCommand does that for a command run through a shell.
func (s ContainerSpec) RunArgs() ([]string, error) { return s.runArgs() }

func (s ContainerSpec) runArgs() ([]string, error) {
	if s.Name == "" || s.Image == "" {
		return nil, fmt.Errorf("container spec needs a name and image")
	}
	if !strings.Contains(s.Image, "@sha256:") {
		return nil, fmt.Errorf("image %q is not pinned by digest", s.Image)
	}
	if s.APIKey != "" && s.KeyEnv == "" {
		return nil, fmt.Errorf("container spec has an API key but no variable to pass it in")
	}
	// Deliberately not --rm: an auto-removed container takes its logs with it,
	// and those logs are the only post-mortem available when a node dies. The
	// launcher clears a stale container by name before starting, so keeping the
	// exited one costs nothing.
	out := []string{"docker", "run", "-d", "--name", s.Name}
	out = append(out, s.Flags...)
	for _, m := range s.Mounts {
		out = append(out, "-v", m.Host+":"+m.Container)
	}
	keys := make([]string, 0, len(s.Env))
	for k := range s.Env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		out = append(out, "-e", k+"="+s.Env[k])
	}
	if s.APIKey != "" {
		out = append(out, "-e", s.KeyEnv)
	}
	if s.Entrypoint != "" {
		out = append(out, "--entrypoint", s.Entrypoint)
	}
	out = append(out, s.Image)
	return append(out, s.Args...), nil
}

// ShellCommand renders RunArgs as a single command line safe to pipe to a
// remote shell. The key rides as an environment assignment in front of
// docker, which the shell consumes; it must only be fed to a shell on stdin,
// never passed as an argument, or it lands on a command line after all.
func (s ContainerSpec) ShellCommand() (string, error) { return s.shellCommand(false) }

// DisplayCommand is ShellCommand with the API key redacted, for printing.
func (s ContainerSpec) DisplayCommand() (string, error) { return s.shellCommand(true) }

func (s ContainerSpec) shellCommand(redact bool) (string, error) {
	args, err := s.runArgs()
	if err != nil {
		return "", err
	}
	quoted := make([]string, 0, len(args)+1)
	if s.APIKey != "" {
		key := ShellQuote(s.APIKey)
		if redact {
			key = "'<redacted>'"
		}
		quoted = append(quoted, s.KeyEnv+"="+key)
	}
	for _, a := range args {
		quoted = append(quoted, ShellQuote(a))
	}
	return strings.Join(quoted, " "), nil
}

// ShellQuote wraps a value so a POSIX shell passes it through untouched. The
// compilation-config argument is JSON with braces and quotes, so unquoted
// interpolation would corrupt it.
func ShellQuote(s string) string {
	if s == "" {
		return "''"
	}
	safe := true
	for _, r := range s {
		if !(r == '-' || r == '_' || r == '.' || r == '/' || r == ':' || r == '=' || r == ',' || r == '@' ||
			(r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')) {
			safe = false
			break
		}
	}
	if safe {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'"'"'`) + "'"
}
