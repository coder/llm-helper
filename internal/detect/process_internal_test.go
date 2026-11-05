package detect

import (
	"reflect"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
)

// Real-world argv observed on a GB10 box (venv console-script form).
var realArgv = strings.Fields(
	"/home/dell/venvs/vllm-native/bin/python3 /home/dell/venvs/vllm-native/bin/vllm serve poolside/Laguna-S-2.1-NVFP4 " +
		"--port 8001 --max-model-len 131072 --gpu-memory-utilization 0.85 --api-key sk-secret " +
		"--enable-auto-tool-choice --tool-call-parser poolside_v1 --reasoning-parser poolside_v1")

func TestParseServerConsoleScript(t *testing.T) {
	s := parseServer(1, realArgv)
	if s.Model != "poolside/Laguna-S-2.1-NVFP4" {
		t.Errorf("Model = %q", s.Model)
	}
	if s.Port != "8001" {
		t.Errorf("Port = %q", s.Port)
	}
	if s.MaxModelLen != "131072" {
		t.Errorf("MaxModelLen = %q", s.MaxModelLen)
	}
	if !s.APIKeySet {
		t.Error("APIKeySet should be true")
	}
	want := "--enable-auto-tool-choice --tool-call-parser poolside_v1 --reasoning-parser poolside_v1"
	if s.ExtraArgs != want {
		t.Errorf("ExtraArgs = %q, want %q", s.ExtraArgs, want)
	}
	if strings.Contains(s.ExtraArgs, "api-key") {
		t.Error("api-key must never leak into ExtraArgs")
	}
}

func TestParseServerAPIServerForm(t *testing.T) {
	args := strings.Fields(
		"python -m vllm.entrypoints.openai.api_server --model=openai/gpt-oss-120b --port=8000 --max-model-len=131072 --trust-remote-code")
	s := parseServer(2, args)
	if s.Model != "openai/gpt-oss-120b" {
		t.Errorf("Model = %q", s.Model)
	}
	if s.Port != "8000" {
		t.Errorf("Port = %q", s.Port)
	}
	if s.ExtraArgs != "--trust-remote-code" {
		t.Errorf("ExtraArgs = %q", s.ExtraArgs)
	}
}

// Adoption has to reproduce the server it took over, so every serving flag
// on the command line must survive into the profile -- not just the handful
// the parser knows by name.
func TestParseServerCarriesUnrecognizedFlags(t *testing.T) {
	s := parseServer(4, strings.Fields(
		"/usr/bin/vllm serve poolside/Laguna-S-2.1-NVFP4 --tensor-parallel-size 2 "+
			"--dtype bfloat16 --enforce-eager --max-num-seqs 64"))
	want := "--tensor-parallel-size 2 --dtype bfloat16 --enforce-eager --max-num-seqs 64"
	if s.ExtraArgs != want {
		t.Errorf("ExtraArgs = %q, want %q", s.ExtraArgs, want)
	}
}

// --gpu-memory-utilization belongs to the profile, not EXTRA_ARGS: 'start'
// passes it explicitly, so adopting it into both would duplicate the flag.
func TestParseServerCapturesMemUtilization(t *testing.T) {
	for _, argv := range []string{
		"/usr/bin/vllm serve m/x --gpu-memory-utilization 0.92",
		"python -m vllm.entrypoints.openai.api_server --model=m/x --gpu-memory-utilization=0.92",
	} {
		s := parseServer(5, strings.Fields(argv))
		if s.MemUtilization != "0.92" {
			t.Errorf("%s: MemUtilization = %q, want 0.92", argv, s.MemUtilization)
		}
		if strings.Contains(s.ExtraArgs, "gpu-memory-utilization") {
			t.Errorf("%s: memory fraction leaked into ExtraArgs %q", argv, s.ExtraArgs)
		}
	}
}

// A carried-over value containing spaces has to survive the round trip
// through the profile, so it is quoted on the way out.
func TestParseServerQuotesValuesWithSpaces(t *testing.T) {
	s := parseServer(6, []string{
		"/usr/bin/vllm", "serve", "m/x",
		"--override-generation-config", `{"temperature": 0.7}`,
	})
	got, err := config.SplitArgs(s.ExtraArgs)
	if err != nil {
		t.Fatalf("SplitArgs(%q): %v", s.ExtraArgs, err)
	}
	want := []string{"--override-generation-config", `{"temperature": 0.7}`}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("round trip = %#v, want %#v", got, want)
	}
}

// The API key must never reach a profile, in either flag form.
func TestParseServerNeverLeaksAPIKey(t *testing.T) {
	for _, argv := range []string{
		"/usr/bin/vllm serve m/x --api-key sk-secret --dtype bfloat16",
		"/usr/bin/vllm serve m/x --api-key=sk-secret --dtype bfloat16",
	} {
		s := parseServer(7, strings.Fields(argv))
		if !s.APIKeySet {
			t.Errorf("%s: APIKeySet should be true", argv)
		}
		if strings.Contains(s.ExtraArgs, "sk-secret") || strings.Contains(s.ExtraArgs, "api-key") {
			t.Errorf("%s: api key leaked into ExtraArgs %q", argv, s.ExtraArgs)
		}
	}
}

// ExtraArgs reaches both a profile file and 'doctor --json', so credentials
// must not ride along -- without catching real flags that merely look like
// they might.
func TestParseServerDropsSecretsButKeepsLookalikes(t *testing.T) {
	s := parseServer(8, strings.Fields(
		"/usr/bin/vllm serve m/x --hf-token hf_secret --tokenizer-mode auto --ssl-keyfile /etc/k.pem"))
	if strings.Contains(s.ExtraArgs, "hf_secret") || strings.Contains(s.ExtraArgs, "--hf-token") {
		t.Errorf("secret leaked into ExtraArgs %q", s.ExtraArgs)
	}
	for _, want := range []string{"--tokenizer-mode auto", "--ssl-keyfile /etc/k.pem"} {
		if !strings.Contains(s.ExtraArgs, want) {
			t.Errorf("ExtraArgs %q lost %q", s.ExtraArgs, want)
		}
	}
}

func TestParseServerDefaultsPort(t *testing.T) {
	s := parseServer(3, strings.Fields("/usr/bin/vllm serve meta-llama/Llama-3-8B"))
	if s.Port != "8000" {
		t.Errorf("default port = %q, want vllm's default 8000", s.Port)
	}
	if !looksLikeVLLM(s.Cmdline) {
		t.Error("looksLikeVLLM failed on a vllm serve command")
	}
}

// 'start' passes model, port, context, memory fraction and api key from the
// profile's own fields, so leaving any of them in EXTRA_ARGS would hand vLLM
// the same flag twice.
func TestParseServerKeepsRepassedFlagsOutOfExtraArgs(t *testing.T) {
	t.Parallel()

	for _, argv := range []string{
		"/usr/bin/vllm serve m/x --port 8001 --max-model-len 131072 " +
			"--gpu-memory-utilization 0.85 --api-key sk-secret --dtype bfloat16",
		"python -m vllm.entrypoints.openai.api_server --model=m/x --port=8001 " +
			"--max-model-len=131072 --gpu-memory-utilization=0.85 --api-key=sk-secret --dtype=bfloat16",
	} {
		s := parseServer(9, strings.Fields(argv))
		for _, flag := range []string{"--model", "--port", "--max-model-len", "--gpu-memory-utilization", "--api-key"} {
			if strings.Contains(s.ExtraArgs, flag) {
				t.Errorf("%s: %s would be passed twice, ExtraArgs = %q", argv, flag, s.ExtraArgs)
			}
		}
		if !strings.Contains(s.ExtraArgs, "--dtype") {
			t.Errorf("%s: an unrecognized flag must still survive, ExtraArgs = %q", argv, s.ExtraArgs)
		}
	}
}

// Values that begin with a dash are the parser's ambiguous case: a negative
// number belongs to the flag before it, a flag does not.
func TestParseServerBoundaryBetweenFlagsAndValues(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		argv string
		want string
	}{{
		name: "negative number is a value",
		argv: "/usr/bin/vllm serve m/x --seed -1 --enforce-eager",
		want: "--seed -1 --enforce-eager",
	}, {
		name: "a flag is not the previous flag's value",
		argv: "/usr/bin/vllm serve m/x --enforce-eager --dtype bfloat16",
		want: "--enforce-eager --dtype bfloat16",
	}, {
		name: "trailing flag with no value",
		argv: "/usr/bin/vllm serve m/x --enforce-eager",
		want: "--enforce-eager",
	}, {
		name: "known boolean never swallows the next argument",
		argv: "/usr/bin/vllm serve m/x --trust-remote-code --dtype bfloat16",
		want: "--trust-remote-code --dtype bfloat16",
	}, {
		name: "boolean written as --flag=value keeps just the flag",
		argv: "/usr/bin/vllm serve m/x --trust-remote-code=true",
		want: "--trust-remote-code",
	}, {
		name: "single-dash flags are ignored",
		argv: "/usr/bin/vllm serve m/x -q --dtype bfloat16",
		want: "--dtype bfloat16",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := parseServer(10, strings.Fields(tt.argv)).ExtraArgs; got != tt.want {
				t.Errorf("ExtraArgs = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestParseServerModelPositionAfterServe(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		argv []string
		want string
	}{
		{"positional after serve", []string{"/usr/bin/vllm", "serve", "org/model"}, "org/model"},
		{"a flag is not the model", []string{"/usr/bin/vllm", "serve", "--model", "org/model"}, "org/model"},
		{"serve with nothing after it", []string{"/usr/bin/vllm", "serve"}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := parseServer(11, tt.argv).Model; got != tt.want {
				t.Errorf("Model = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLooksLikeVLLM(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		argv string
		want bool
	}{
		{"console script", "/opt/venv/bin/python3 /opt/venv/bin/vllm serve org/model", true},
		{"api server module", "python -m vllm.entrypoints.openai.api_server --model org/model", true},
		{"vllm without serve", "/usr/bin/vllm --help", false},
		{"a different server", "python -m sglang.launch_server --model org/model", false},
		{"our own name in an argument", "grep vllm /var/log/syslog", false},
		{"empty", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := looksLikeVLLM(strings.Fields(tt.argv)); got != tt.want {
				t.Errorf("looksLikeVLLM(%q) = %v, want %v", tt.argv, got, tt.want)
			}
		})
	}
}

func TestLooksSecret(t *testing.T) {
	t.Parallel()

	tests := []struct {
		flag string
		want bool
	}{
		{"--api-key", true},
		{"--hf-token", true},
		{"--HF-TOKEN", true},
		{"--some-vendor-secret", true},
		{"--redis-password", true},
		{"--tokenizer-mode", false},
		{"--ssl-keyfile", false},
		{"--served-model-name", false},
		{"--dtype", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.flag, func(t *testing.T) {
			t.Parallel()
			if got := LooksSecret(tt.flag); got != tt.want {
				t.Errorf("LooksSecret(%q) = %v, want %v", tt.flag, got, tt.want)
			}
		})
	}
}

func TestLooksLikeValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		in   string
		want bool
	}{
		{"bfloat16", true},
		{"0.85", true},
		{"-1", true},
		{"-0.5", true},
		{"--dtype", false},
		{"-q", false},
		{"", false},
		{`{"temperature": 0.7}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			t.Parallel()
			if got := looksLikeValue(tt.in); got != tt.want {
				t.Errorf("looksLikeValue(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}
