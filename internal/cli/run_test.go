package cli_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/coder/vllm-helper/internal/cli"
)

const sampleProfile = `# Status: TESTED AND WORKING
MODEL=Qwen/Qwen3-8B-FP8
MAX_MODEL_LEN=40960
GPU_MEMORY_UTILIZATION=0.85
EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes"
`

// runCLI drives a command the way the shell does, capturing what the user
// would see. fmt.Print* resolves os.Stdout at call time, so swapping the
// streams needs no production change -- but it is process-wide, which is
// why nothing here runs in parallel.
func runCLI(t *testing.T, args ...string) (code int, stdout, stderr string) {
	t.Helper()

	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	errR, errW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}

	var outBuf, errBuf bytes.Buffer
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); _, _ = io.Copy(&outBuf, outR) }()
	go func() { defer wg.Done(); _, _ = io.Copy(&errBuf, errR) }()

	origOut, origErr := os.Stdout, os.Stderr
	defer func() {
		os.Stdout, os.Stderr = origOut, origErr
		outW.Close()
		errW.Close()
		wg.Wait()
		outR.Close()
		errR.Close()
	}()
	os.Stdout, os.Stderr = outW, errW

	code = cli.Run(args)
	os.Stdout, os.Stderr = origOut, origErr
	outW.Close()
	errW.Close()
	wg.Wait()
	return code, outBuf.String(), errBuf.String()
}

func portOf(t *testing.T, rawURL string) string {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatal(err)
	}
	return u.Port()
}

func closedPort(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	port := portOf(t, srv.URL)
	srv.Close()
	return port
}

// newTree gives the CLI a config tree of its own and one saved profile. The
// network is closed off for the whole package in TestMain.
func newTree(t *testing.T) (configDir, profilePath string) {
	t.Helper()
	root := t.TempDir()
	configDir = filepath.Join(root, "config")
	t.Setenv("VLLM_HELPER_CONFIG_DIR", configDir)
	t.Setenv("VLLM_HELPER_STATE_DIR", filepath.Join(root, "state"))
	t.Setenv("VLLM_HELPER_DATA_DIR", filepath.Join(root, "data"))

	profiles := filepath.Join(configDir, "models")
	if err := os.MkdirAll(profiles, 0o755); err != nil {
		t.Fatal(err)
	}
	config := "PORT=" + closedPort(t) + "\nSERVICE_NAME=vllm-test\nSERVICE_SCOPE=user\n"
	if err := os.WriteFile(filepath.Join(configDir, "config.env"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	profilePath = filepath.Join(profiles, "qwen.env")
	if err := os.WriteFile(profilePath, []byte(sampleProfile), 0o644); err != nil {
		t.Fatal(err)
	}
	return configDir, profilePath
}

func TestRunHelp(t *testing.T) {
	newTree(t)
	for _, args := range [][]string{{"-h"}, {"--help"}, {"help"}} {
		code, stdout, _ := runCLI(t, args...)
		if code != 0 {
			t.Errorf("%v exited %d, want 0", args, code)
		}
		if !strings.Contains(stdout, "Usage:") {
			t.Errorf("%v printed no usage:\n%s", args, stdout)
		}
		// The groups are what keep the command list readable; an ungrouped
		// command silently lands under "Additional Commands".
		for _, group := range []string{"Model profiles:", "Running server:", "This machine:", "More than one machine:"} {
			if !strings.Contains(stdout, group) {
				t.Errorf("%v is missing the %q group:\n%s", args, group, stdout)
			}
		}
	}
}

// A bare invocation is a usage error, not a successful run.
func TestRunWithNoArguments(t *testing.T) {
	newTree(t)
	code, stdout, _ := runCLI(t)
	if code == 0 {
		t.Error("exited 0, want non-zero")
	}
	if !strings.Contains(stdout, "Usage:") {
		t.Errorf("printed no help:\n%s", stdout)
	}
}

func TestRunUnknownCommand(t *testing.T) {
	newTree(t)
	code, _, stderr := runCLI(t, "frobnicate")
	if code == 0 {
		t.Error("exited 0, want non-zero")
	}
	if !strings.Contains(stderr, `unknown command "frobnicate"`) {
		t.Errorf("stderr = %q, want it to name the command", stderr)
	}
}

func TestRunVersion(t *testing.T) {
	newTree(t)
	code, stdout, _ := runCLI(t, "version")
	if code != 0 {
		t.Errorf("exited %d, want 0", code)
	}
	if !strings.Contains(stdout, "llm-helper") {
		t.Errorf("stdout = %q, want the version line", stdout)
	}
}

// Flags are accepted wherever the user puts them: before the positional,
// after it, and in --flag=value form. All three have to reach the same
// place, which here is the profile lookup failing on a name that doesn't
// exist -- never a flag-parsing error.
func TestRunFlagsAreAcceptedAnywhere(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"flag after the positional", []string{"switch", "nosuchprofile", "--wait"}},
		{"flag before the positional", []string{"switch", "--wait", "nosuchprofile"}},
		{"equals form", []string{"switch", "nosuchprofile", "--timeout=30s"}},
		{"equals form before the positional", []string{"switch", "--timeout=30s", "nosuchprofile"}},
		{"valued flag with a separate value", []string{"switch", "nosuchprofile", "--timeout", "30s"}},
		{"several flags around the positional", []string{"switch", "--no-follow", "nosuchprofile", "--timeout=1m"}},
		{"terminator before the positional", []string{"switch", "--", "nosuchprofile"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			newTree(t)
			code, _, stderr := runCLI(t, tt.args...)
			if code == 0 {
				t.Errorf("%v exited 0, want non-zero for a profile that doesn't exist", tt.args)
			}
			if !strings.Contains(stderr, "nosuchprofile") {
				t.Errorf("stderr = %q, want the unknown profile reported", stderr)
			}
			if strings.Contains(stderr, "unknown flag") || strings.Contains(stderr, "invalid argument") {
				t.Errorf("stderr = %q, want the flags accepted rather than rejected", stderr)
			}
		})
	}
}

// The value in --flag=value really reaches the flag, so a bad one is caught
// rather than silently ignored.
func TestRunRejectsABadFlagValue(t *testing.T) {
	newTree(t)
	code, _, stderr := runCLI(t, "switch", "nosuchprofile", "--timeout=nonsense")
	if code == 0 {
		t.Error("exited 0, want non-zero")
	}
	if !strings.Contains(stderr, "invalid argument") {
		t.Errorf("stderr = %q, want the unparseable duration reported", stderr)
	}
}

func TestRunRejectsUnknownFlags(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"after a positional", []string{"switch", "nosuchprofile", "--bogus"}},
		{"before a positional", []string{"switch", "--bogus", "nosuchprofile"}},
		{"on a command with no positionals", []string{"status", "--bogus"}},
		{"equals form", []string{"list", "--bogus=1"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			newTree(t)
			code, _, stderr := runCLI(t, tt.args...)
			if code == 0 {
				t.Errorf("%v exited 0, want non-zero", tt.args)
			}
			if !strings.Contains(stderr, "unknown flag") {
				t.Errorf("stderr = %q, want the unknown flag reported", stderr)
			}
		})
	}
}

// -h on a subcommand prints that command's usage and is not an error.
func TestRunSubcommandHelp(t *testing.T) {
	for _, args := range [][]string{{"status", "-h"}, {"list", "--help"}, {"doctor", "-h"}} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			newTree(t)
			code, stdout, _ := runCLI(t, args...)
			if code != 0 {
				t.Errorf("exited %d, want 0", code)
			}
			if !strings.Contains(stdout, "llm-helper "+args[0]) {
				t.Errorf("stdout = %q, want the command's usage", stdout)
			}
		})
	}
}

func TestRunList(t *testing.T) {
	newTree(t)

	code, stdout, _ := runCLI(t, "list")
	if code != 0 {
		t.Fatalf("exited %d, want 0", code)
	}
	if !strings.Contains(stdout, "qwen") || !strings.Contains(stdout, "TESTED AND WORKING") {
		t.Errorf("stdout = %q, want the profile and its status", stdout)
	}

	code, stdout, _ = runCLI(t, "list", "--json")
	if code != 0 {
		t.Fatalf("list --json exited %d, want 0", code)
	}
	var got []struct {
		Name        string `json:"name"`
		Model       string `json:"model"`
		MaxModelLen string `json:"max_model_len"`
		GPUMemUtil  string `json:"gpu_memory_utilization"`
		ExtraArgs   string `json:"extra_args"`
		Status      string `json:"status"`
		Active      bool   `json:"active"`
	}
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatalf("list --json isn't JSON: %v\n%s", err, stdout)
	}
	if len(got) != 1 {
		t.Fatalf("got %d profiles, want 1", len(got))
	}
	if got[0].Name != "qwen" || got[0].Model != "Qwen/Qwen3-8B-FP8" {
		t.Errorf("profile = %+v", got[0])
	}
	if got[0].ExtraArgs != "--enable-auto-tool-choice --tool-call-parser hermes" {
		t.Errorf("ExtraArgs = %q, want the quotes stripped", got[0].ExtraArgs)
	}
	if got[0].Active {
		t.Error("Active should be false with nothing switched to")
	}
}

func TestRunStatusJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	t.Cleanup(srv.Close)

	configDir, _ := newTree(t)
	port := portOf(t, srv.URL)
	config := "PORT=" + port + "\nSERVICE_NAME=vllm-test\nSERVICE_SCOPE=user\n"
	if err := os.WriteFile(filepath.Join(configDir, "config.env"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := runCLI(t, "status", "--json")
	if code != 0 {
		t.Fatalf("exited %d (%s), want 0", code, stderr)
	}
	var got struct {
		Service    string `json:"service"`
		Profile    string `json:"profile"`
		Model      string `json:"model"`
		Port       string `json:"port"`
		Healthy    bool   `json:"endpoint_healthy"`
		HTTPStatus int    `json:"http_status"`
	}
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatalf("status --json isn't JSON: %v\n%s", err, stdout)
	}
	if got.Port != port {
		t.Errorf("port = %q, want %q", got.Port, port)
	}
	if !got.Healthy || got.HTTPStatus != http.StatusOK {
		t.Errorf("healthy = %v, http_status = %d, want true and 200", got.Healthy, got.HTTPStatus)
	}
	if got.Service == "" {
		t.Error("service should always report a state")
	}
	if got.Profile != "unknown" || got.Model != "unknown" {
		t.Errorf("profile/model = %q/%q, want unknown before the first switch", got.Profile, got.Model)
	}
}

func TestRunSet(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{{
		name: "plain assignment",
		args: []string{"set", "qwen", "MAX_MODEL_LEN=8192"},
		want: "MAX_MODEL_LEN=8192",
	}, {
		name: "everything after -- is an operand",
		args: []string{"set", "qwen", "--", "MAX_MODEL_LEN=4096"},
		want: "MAX_MODEL_LEN=4096",
	}, {
		name: "a value that begins with a dash needs no terminator",
		args: []string{"set", "qwen", "EXTRA_ARGS=--enable-prefix-caching"},
		want: "EXTRA_ARGS=--enable-prefix-caching",
	}, {
		name: "several pairs at once",
		args: []string{"set", "qwen", "MAX_MODEL_LEN=16384", "GPU_MEMORY_UTILIZATION=0.9"},
		want: "GPU_MEMORY_UTILIZATION=0.9",
	}, {
		name: "a value with spaces is quoted back",
		args: []string{"set", "qwen", "EXTRA_ARGS=--tool-call-parser hermes"},
		want: `EXTRA_ARGS="--tool-call-parser hermes"`,
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, profilePath := newTree(t)
			code, stdout, stderr := runCLI(t, tt.args...)
			if code != 0 {
				t.Fatalf("exited %d, want 0\nstdout: %s\nstderr: %s", code, stdout, stderr)
			}
			raw, err := os.ReadFile(profilePath)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(raw), tt.want) {
				t.Errorf("profile is missing %q:\n%s", tt.want, raw)
			}
			if !strings.Contains(string(raw), "# Status: TESTED AND WORKING") {
				t.Errorf("set dropped the profile's comments:\n%s", raw)
			}
		})
	}
}

// A rejected `set` must not leave a half-applied profile behind, so the
// file has to come out byte for byte the way it went in.
func TestRunSetRejectionsWriteNothing(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{{
		name:    "not a KEY=VALUE pair",
		args:    []string{"set", "qwen", "MAX_MODEL_LEN"},
		wantErr: "not KEY=VALUE",
	}, {
		name:    "empty key",
		args:    []string{"set", "qwen", "=8192"},
		wantErr: "not KEY=VALUE",
	}, {
		name:    "a good pair followed by a bad one",
		args:    []string{"set", "qwen", "MAX_MODEL_LEN=8192", "BROKEN"},
		wantErr: "nothing was written",
	}, {
		name:    "EXTRA_ARGS that won't parse",
		args:    []string{"set", "qwen", `EXTRA_ARGS=--served-model-name 'unterminated`},
		wantErr: "won't parse",
	}, {
		name:    "emptying MODEL",
		args:    []string{"set", "qwen", "MODEL="},
		wantErr: "no MODEL",
	}, {
		name:    "a provider this build doesn't know",
		args:    []string{"set", "qwen", "PROVIDER=nope"},
		wantErr: "unknown PROVIDER",
	}, {
		name:    "unknown profile",
		args:    []string{"set", "nosuchprofile", "MODEL=org/x"},
		wantErr: "nosuchprofile",
	}, {
		name:    "no pairs at all",
		args:    []string{"set", "qwen"},
		wantErr: "usage",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, profilePath := newTree(t)
			before, err := os.ReadFile(profilePath)
			if err != nil {
				t.Fatal(err)
			}
			code, _, stderr := runCLI(t, tt.args...)
			if code == 0 {
				t.Errorf("exited 0, want non-zero")
			}
			if !strings.Contains(stderr, tt.wantErr) {
				t.Errorf("stderr = %q, want it to mention %q", stderr, tt.wantErr)
			}
			after, err := os.ReadFile(profilePath)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before, after) {
				t.Errorf("the profile was modified by a rejected set:\nbefore:\n%s\nafter:\n%s", before, after)
			}
		})
	}
}

func TestRunWithoutConfig(t *testing.T) {
	configDir, _ := newTree(t)
	if err := os.Remove(filepath.Join(configDir, "config.env")); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"list"}, {"status"}, {"set", "qwen", "MODEL=org/x"}} {
		t.Run(args[0], func(t *testing.T) {
			code, _, stderr := runCLI(t, args...)
			if code == 0 {
				t.Error("exited 0, want non-zero before setup")
			}
			if !strings.Contains(stderr, "setup") {
				t.Errorf("stderr = %q, want it to point at 'vllm-helper setup'", stderr)
			}
		})
	}
}

// The Hub is unreachable here, so these exercise the failure path scripts
// actually hit -- an error on stderr and a non-zero exit, not a panic.
func TestRunHubCommandsWithoutNetwork(t *testing.T) {
	for _, args := range [][]string{
		{"search", "qwen3", "--json"},
		{"search", "--json", "qwen3"},
		{"inspect", "Qwen/Qwen3-8B-FP8", "--json"},
	} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			newTree(t)
			code, _, stderr := runCLI(t, args...)
			if code == 0 {
				t.Error("exited 0, want non-zero with no route to the Hub")
			}
			if stderr == "" {
				t.Error("nothing on stderr to tell the user what failed")
			}
		})
	}
}

func TestRunSearchAndInspectNeedAnArgument(t *testing.T) {
	for _, args := range [][]string{{"search"}, {"inspect"}, {"edit"}, {"remove"}, {"coder-sync"}} {
		t.Run(args[0], func(t *testing.T) {
			newTree(t)
			code, _, stderr := runCLI(t, args...)
			if code == 0 {
				t.Error("exited 0, want non-zero")
			}
			if !strings.Contains(stderr, "usage") {
				t.Errorf("stderr = %q, want a usage line", stderr)
			}
		})
	}
}

// start is what the service manager runs. Before the first switch there is no
// active model, and exiting non-zero here would send systemd's
// Restart=on-failure -- or launchd's KeepAlive -- into a crash loop serving
// nothing.
func TestRunStartWithoutAnActiveModel(t *testing.T) {
	newTree(t)
	code, stdout, stderr := runCLI(t, "start")
	if code != 0 {
		t.Fatalf("exited %d (%s), want 0 so the service manager doesn't restart it", code, stderr)
	}
	if !strings.Contains(stdout, "switch") {
		t.Errorf("stdout = %q, want it to point at 'vllm-helper switch'", stdout)
	}
}

// Only a missing active.env means "not set up yet". A file that exists and
// can't be read is a fault, and swallowing it would hide the breakage behind
// the same quiet exit.
func TestRunStartReportsAnUnreadableActiveModel(t *testing.T) {
	newTree(t)
	state := os.Getenv("VLLM_HELPER_STATE_DIR")
	// A directory in its place fails to read for every user, including root.
	if err := os.MkdirAll(filepath.Join(state, "active.env"), 0o755); err != nil {
		t.Fatal(err)
	}
	code, _, stderr := runCLI(t, "start")
	if code == 0 {
		t.Fatal("exited 0, want non-zero when the active model config can't be read")
	}
	if !strings.Contains(stderr, "active.env") {
		t.Errorf("stderr = %q, want it to name the file it couldn't read", stderr)
	}
}

// A provider this build doesn't know is a fault to report, never a reason to
// fall back to vLLM and serve the profile with the wrong server.
func TestRunStartRefusesAnUnknownProvider(t *testing.T) {
	newTree(t)
	state := os.Getenv("VLLM_HELPER_STATE_DIR")
	if err := os.MkdirAll(state, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(state, "active.env"), []byte("PROVIDER=nope\nMODEL=m\nPORT=1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	code, _, stderr := runCLI(t, "start")
	if code == 0 || !strings.Contains(stderr, "nope") {
		t.Fatalf("exited %d, stderr %q; want a failure naming the provider", code, stderr)
	}
}

// --quant picks one GGUF file out of a repo, which mlx does not load. Under
// mlx it is refused before anything is fetched or written.
func TestRunNewRefusesAQuantForAnotherProvider(t *testing.T) {
	configDir, _ := newTree(t)
	f, err := os.OpenFile(filepath.Join(configDir, "config.env"), os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("PROVIDER=mlx\n"); err != nil {
		t.Fatal(err)
	}
	f.Close()
	code, _, stderr := runCLI(t, "new", "org/model-GGUF", "--quant", "Q8_0")
	if code == 0 || !strings.Contains(stderr, "mlx") {
		t.Fatalf("exited %d, stderr %q; want --quant refused naming the provider", code, stderr)
	}
	if _, err := os.Stat(filepath.Join(configDir, "models", "model-gguf-q8_0.env")); err == nil {
		t.Error("a profile was written for a refused pull")
	}
}

// Every command has to stay reachable through Run. A rejected flag is enough
// to prove the arm is wired without running what's behind it, which keeps
// commands that install units or drive a service out of the test.
func TestRunDispatchesEveryCommand(t *testing.T) {
	for _, cmd := range []string{
		"status", "list", "switch", "new", "remove", "search", "inspect",
		"restart", "logs", "coder-sync", "doctor",
	} {
		t.Run(cmd, func(t *testing.T) {
			newTree(t)
			code, _, stderr := runCLI(t, cmd, "--not-a-real-flag")
			if code == 0 {
				t.Errorf("exited 0, want a rejected flag to fail")
			}
			if strings.Contains(stderr, "unknown command") {
				t.Errorf("%q is not wired into Run: %s", cmd, stderr)
			}
		})
	}
}

// An operand too many used to be dropped on the floor, so "switch qwen typo"
// switched to qwen and said nothing about the typo.
func TestRunRejectsExtraOperands(t *testing.T) {
	tests := [][]string{
		{"switch", "qwen", "typo"},
		{"new", "org/model", "typo"},
		{"remove", "qwen", "typo"},
		{"edit", "qwen", "typo"},
		{"inspect", "org/model", "typo"},
		{"coder-sync", "register", "qwen", "typo"},
		{"status", "typo"},
		{"list", "typo"},
		{"stop", "typo"},
		{"logs", "typo"},
		{"version", "typo"},
	}
	for _, args := range tests {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			newTree(t)
			code, _, stderr := runCLI(t, args...)
			if code == 0 {
				t.Errorf("exited 0, want the stray operand rejected")
			}
			// Commands with an operand of their own report a count;
			// commands with none read the extra word as a subcommand.
			if !strings.Contains(stderr, "accepts at most") &&
				!strings.Contains(stderr, `unknown command "typo"`) {
				t.Errorf("stderr = %q, want the operand rejected by name or count", stderr)
			}
		})
	}
}

// The operand bound must not swallow the arguments a command legitimately
// takes more than one of.
func TestRunKeepsVariadicOperands(t *testing.T) {
	newTree(t)
	// search joins its operands into one query.
	_, _, stderr := runCLI(t, "search", "qwen", "7b", "instruct")
	if strings.Contains(stderr, "accepts at most") {
		t.Errorf("search rejected a multi-word query: %s", stderr)
	}
	// set takes any number of assignments.
	code, _, stderr := runCLI(t, "set", "qwen", "MAX_MODEL_LEN=2048", "GPU_MEMORY_UTILIZATION=0.5")
	if code != 0 {
		t.Errorf("set rejected two assignments (exit %d): %s", code, stderr)
	}
}

// stdlib flag treated -json and --json alike; pflag reads the first as a
// cluster of shorthands. Anyone upgrading with the old spelling in a script
// gets told what to write instead of a complaint about a letter they never
// typed.
func TestRunRejectsSingleDashLongFlags(t *testing.T) {
	newTree(t)
	code, _, stderr := runCLI(t, "list", "-json")
	if code == 0 {
		t.Error("exited 0, want non-zero")
	}
	if !strings.Contains(stderr, "takes --json, not -json") {
		t.Errorf("stderr = %q, want the double-dash spelling suggested", stderr)
	}

	// A flag that isn't ours keeps pflag's own message: we only claim to
	// know what someone meant when the letters spell a real flag.
	_, _, stderr = runCLI(t, "list", "-xyz")
	if !strings.Contains(stderr, "unknown shorthand flag") {
		t.Errorf("stderr = %q, want pflag's message left alone", stderr)
	}
}

// stop is safe to run against a service that was never installed, and says
// so rather than failing.
func TestRunStopWhenNothingIsRunning(t *testing.T) {
	newTree(t)
	code, stdout, stderr := runCLI(t, "stop")
	if code != 0 {
		t.Fatalf("exited %d (%s), want 0", code, stderr)
	}
	if !strings.Contains(stdout, "nothing to stop") {
		t.Errorf("stdout = %q, want it to say there was nothing to stop", stdout)
	}
}

func TestRunRestartWithoutAnActiveProfile(t *testing.T) {
	newTree(t)
	code, _, stderr := runCLI(t, "restart")
	if code == 0 {
		t.Error("exited 0, want non-zero")
	}
	if !strings.Contains(stderr, "no active profile") {
		t.Errorf("stderr = %q, want it to explain there's nothing to restart", stderr)
	}
}

func TestRunDoctor(t *testing.T) {
	newTree(t)
	code, stdout, stderr := runCLI(t, "doctor")
	if code != 0 {
		t.Fatalf("exited %d (%s), want 0", code, stderr)
	}
	for _, want := range []string{"HARDWARE", "CONFIG", "RUNNING SERVERS", "ENDPOINTS", "INTEGRATIONS", "BUILD TOOLCHAIN"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("doctor printed no %s section:\n%s", want, stdout)
		}
	}
}

func TestRunDoctorJSON(t *testing.T) {
	newTree(t)
	code, stdout, stderr := runCLI(t, "doctor", "--json")
	if code != 0 {
		t.Fatalf("exited %d (%s), want 0", code, stderr)
	}
	var got struct {
		Hardware struct {
			TotalMemGB   int    `json:"total_mem_gb"`
			BudgetReason string `json:"budget_reason"`
		} `json:"hardware"`
		Config *struct {
			Path string `json:"path"`
			Port string `json:"port"`
		} `json:"config"`
		Servers   []map[string]any `json:"servers"`
		Endpoints []map[string]any `json:"endpoints"`
		Toolchain []struct {
			Name string `json:"name"`
		} `json:"toolchain"`
	}
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatalf("doctor --json isn't JSON: %v\n%s", err, stdout)
	}
	if got.Config == nil {
		t.Fatal("config section is null with a config present")
	}
	if got.Hardware.BudgetReason == "" {
		t.Error("hardware.budget_reason should explain the budget")
	}
	// Cmdline is withheld from JSON because it can carry --api-key values.
	for _, s := range got.Servers {
		if _, ok := s["cmdline"]; ok {
			t.Errorf("a server's raw command line reached JSON: %v", s)
		}
	}
}

// Coder sync writes one entry per profile and remembers its id, so the next
// switch updates rather than duplicating it.
func TestRunCoderSync(t *testing.T) {
	var created int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			created++
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"mc-1"}`)
			return
		}
		_, _ = io.WriteString(w, `[]`)
	}))
	t.Cleanup(srv.Close)

	configDir, _ := newTree(t)
	config := "PORT=" + closedPort(t) + "\nSERVICE_NAME=vllm-test\nSERVICE_SCOPE=user\n" +
		"CODER_URL=" + srv.URL + "\nCODER_API_TOKEN=tok\nCODER_AI_PROVIDER_ID=prov\n"
	if err := os.WriteFile(filepath.Join(configDir, "config.env"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	stateDir := filepath.Join(filepath.Dir(configDir), "state")

	code, stdout, stderr := runCLI(t, "coder-sync", "register", "qwen")
	if code != 0 {
		t.Fatalf("exited %d (%s), want 0", code, stderr)
	}
	if !strings.Contains(stdout, "registering") {
		t.Errorf("stdout = %q, want it to report the registration", stdout)
	}
	ids, err := os.ReadFile(filepath.Join(stateDir, "coder-ids.env"))
	if err != nil {
		t.Fatalf("no id state written: %v", err)
	}
	if !strings.Contains(string(ids), "qwen=mc-1") {
		t.Errorf("id state = %q, want the profile mapped to its Coder entry", ids)
	}

	code, stdout, _ = runCLI(t, "coder-sync", "register", "qwen")
	if code != 0 {
		t.Fatalf("second register exited %d, want 0", code)
	}
	if !strings.Contains(stdout, "already has") {
		t.Errorf("stdout = %q, want the existing entry left alone", stdout)
	}
	if created != 1 {
		t.Errorf("the deployment saw %d creates, want 1", created)
	}
}

func TestRunCoderSyncWithoutCoderConfigured(t *testing.T) {
	newTree(t)
	code, stdout, stderr := runCLI(t, "coder-sync", "register-all")
	if code == 0 {
		t.Errorf("exited 0 (%s / %s), want non-zero", stdout, stderr)
	}
	if !strings.Contains(stderr, "not configured") {
		t.Errorf("stderr = %q, want it to say Coder sync isn't configured", stderr)
	}
}

func TestRunCoderSyncRejectsAnUnknownOperation(t *testing.T) {
	newTree(t)
	code, _, stderr := runCLI(t, "coder-sync", "frobnicate", "qwen")
	if code == 0 {
		t.Error("exited 0, want non-zero")
	}
	if !strings.Contains(stderr, "usage") {
		t.Errorf("stderr = %q, want a usage line", stderr)
	}
}

// main hands Run's return value to os.Exit, so the shell sees the same code
// the tests above assert on.
func TestBinaryExitCodes(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain to build the binary with")
	}
	bin := filepath.Join(t.TempDir(), "vllm-helper")
	build := exec.Command("go", "build", "-o", bin, "github.com/coder/vllm-helper")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	tests := []struct {
		name string
		args []string
		want int
	}{
		{"version", []string{"version"}, 0},
		{"help", []string{"--help"}, 0},
		{"unknown command", []string{"frobnicate"}, 1},
		{"no arguments", nil, 1},
		{"unknown profile", []string{"switch", "nosuchprofile"}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configDir, _ := newTree(t)
			cmd := exec.Command(bin, tt.args...)
			cmd.Env = append(os.Environ(),
				"VLLM_HELPER_CONFIG_DIR="+configDir,
				"VLLM_HELPER_STATE_DIR="+filepath.Join(filepath.Dir(configDir), "state"),
			)
			got := 0
			if err := cmd.Run(); err != nil {
				var exit *exec.ExitError
				if !errors.As(err, &exit) {
					t.Fatalf("running the binary: %v", err)
				}
				got = exit.ExitCode()
			}
			if got != tt.want {
				t.Errorf("%v exited %d, want %d", tt.args, got, tt.want)
			}
		})
	}
}
