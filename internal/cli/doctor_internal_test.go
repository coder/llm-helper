package cli

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/profile"
)

// Nothing in this package's tests runs in parallel: they swap os.Stdout and
// the process environment, both of which are shared.

// sleeperEnv marks a re-exec of this test binary that stands in for a vLLM
// server: it ignores its arguments and stays alive until the test kills it.
const sleeperEnv = "VLLM_HELPER_TEST_SLEEPER"

func TestMain(m *testing.M) {
	if os.Getenv(sleeperEnv) != "" {
		time.Sleep(2 * time.Minute)
		os.Exit(0)
	}
	goOffline()
	os.Exit(m.Run())
}

// goOffline routes outbound HTTP at a port nothing listens on, so no test
// depends on huggingface.co being reachable. net/http resolves the proxy
// environment once per process and caches it, so setting this per-test would
// only take effect for whichever test happened to run first -- and a later
// test that made a request without setting it would put the whole package back
// on the real network. Port 1 needs root to bind and is outside the ephemeral
// range, so it cannot collide with a test server. Loopback is exempt from
// proxying, which is what lets the httptest servers here still work.
func goOffline() {
	for _, k := range []string{"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"} {
		os.Unsetenv(k)
	}
	os.Setenv("HTTP_PROXY", "http://127.0.0.1:1")
	os.Setenv("HTTPS_PROXY", "http://127.0.0.1:1")

	req, err := http.NewRequest("GET", "https://huggingface.co", nil)
	if err != nil {
		panic(err)
	}
	if _, err := http.ProxyFromEnvironment(req); err != nil {
		panic(err)
	}
}

// newTree points the CLI at a temp config/state/data tree and keeps it off
// the network, so no test depends on huggingface.co being reachable.
func newTree(t *testing.T) config.Dirs {
	t.Helper()
	root := t.TempDir()
	t.Setenv("VLLM_HELPER_CONFIG_DIR", filepath.Join(root, "config"))
	t.Setenv("VLLM_HELPER_STATE_DIR", filepath.Join(root, "state"))
	t.Setenv("VLLM_HELPER_DATA_DIR", filepath.Join(root, "data"))

	dirs := config.ResolveDirs()
	if err := os.MkdirAll(dirs.Profiles, 0o755); err != nil {
		t.Fatal(err)
	}
	return dirs
}

// TestTheHubIsUnreachable guards the isolation itself. Without it, a change
// that resolved the proxy environment before TestMain set it would put every
// Hub call in this package on the real network, and the tests that assert a
// lookup fails would start failing for a reason nothing points at.
func TestTheHubIsUnreachable(t *testing.T) {
	hub, err := http.NewRequest("GET", "https://huggingface.co/api/models/x", nil)
	if err != nil {
		t.Fatal(err)
	}
	proxy, err := http.ProxyFromEnvironment(hub)
	if err != nil {
		t.Fatal(err)
	}
	if proxy == nil {
		t.Error("hub requests are going to the real network, not the dead proxy")
	}

	local, err := http.NewRequest("GET", "http://127.0.0.1:8080/v1/models", nil)
	if err != nil {
		t.Fatal(err)
	}
	if proxy, err := http.ProxyFromEnvironment(local); err != nil {
		t.Fatal(err)
	} else if proxy != nil {
		t.Errorf("loopback is being proxied through %s, so the test servers are unreachable", proxy)
	}
}

func closedPort(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	port := portOf(t, srv.URL)
	srv.Close()
	return port
}

func portOf(t *testing.T, rawURL string) string {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatal(err)
	}
	return u.Port()
}

func writeConfig(t *testing.T, dirs config.Dirs, content string) {
	t.Helper()
	if err := os.MkdirAll(dirs.Config, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dirs.ConfigFile(), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestGatherDoctorReportsTheConfig(t *testing.T) {
	dirs := newTree(t)
	bin := filepath.Join(t.TempDir(), "vllm")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeConfig(t, dirs, "PORT=9101\nVLLM_BIN="+bin+"\nSERVICE_NAME=vllm-test\nSERVICE_SCOPE=user\n")

	r := gatherDoctor()
	if r.Config == nil {
		t.Fatalf("Config is nil, ConfigError = %q", r.ConfigError)
	}
	tests := []struct {
		name string
		got  string
		want string
	}{
		{"path", r.Config.Path, dirs.ConfigFile()},
		{"port", r.Config.Port, "9101"},
		{"vllm bin", r.Config.VLLMBin, bin},
		{"service", r.Config.Service, "vllm-test"},
		{"scope", r.Config.Scope, "user"},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %q, want %q", tt.name, tt.got, tt.want)
		}
	}
	if !r.Config.VLLMBinOK {
		t.Error("VLLMBinOK should be true for an executable that exists")
	}
	if r.Config.CoderConfigured {
		t.Error("CoderConfigured should be false without Coder settings")
	}
	if r.Config.ServiceState == "" {
		t.Error("ServiceState should always say something")
	}
}

// A VLLM_BIN that has gone missing is the difference between "configured"
// and "will actually start", so doctor reports them separately.
func TestGatherDoctorFlagsAnUnusableVLLMBin(t *testing.T) {
	tests := []struct {
		name    string
		bin     string
		wantBin bool
	}{
		{"unset", "", false},
		{"missing", "/nonexistent/vllm", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dirs := newTree(t)
			writeConfig(t, dirs, "VLLM_BIN="+tt.bin+"\n")
			r := gatherDoctor()
			if r.Config == nil {
				t.Fatalf("Config is nil, ConfigError = %q", r.ConfigError)
			}
			if r.Config.VLLMBinOK != tt.wantBin {
				t.Errorf("VLLMBinOK = %v, want %v", r.Config.VLLMBinOK, tt.wantBin)
			}
		})
	}
}

// With new profiles going to another provider, the binary that has to exist
// is that provider's, and doctor has to say which one it checked.
func TestGatherDoctorReportsTheProvidersBin(t *testing.T) {
	dirs := newTree(t)
	bin := filepath.Join(t.TempDir(), "mlx_lm.server")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeConfig(t, dirs, "PROVIDER=mlx\nMLX_LM_BIN="+bin+"\n")
	r := gatherDoctor()
	if r.Config == nil {
		t.Fatalf("Config is nil, ConfigError = %q", r.ConfigError)
	}
	if r.Config.Provider != "mlx" || r.Config.ProviderBin != bin || !r.Config.ProviderBinOK {
		t.Errorf("provider = %q, bin = %q (ok %v), want mlx at %s", r.Config.Provider, r.Config.ProviderBin, r.Config.ProviderBinOK, bin)
	}

	// An install that never chose one reports vLLM and nothing more.
	writeConfig(t, dirs, "VLLM_BIN=/nonexistent/vllm\n")
	r = gatherDoctor()
	if r.Config.Provider != "vllm" || r.Config.ProviderBin != "" {
		t.Errorf("provider = %q, bin = %q, want vllm and no provider bin", r.Config.Provider, r.Config.ProviderBin)
	}
}

func TestGatherDoctorWithoutConfig(t *testing.T) {
	newTree(t)

	r := gatherDoctor()
	if r.Config != nil {
		t.Errorf("Config = %+v, want nil before setup", r.Config)
	}
	if !strings.Contains(r.ConfigError, "setup") {
		t.Errorf("ConfigError = %q, want it to point at 'vllm-helper setup'", r.ConfigError)
	}
	if r.Hardware.TotalMemGB == 0 && (runtime.GOOS == "linux" || runtime.GOOS == "darwin") {
		t.Error("hardware should still be probed without a config")
	}
}

// The drift check is the part a human wouldn't spot: config claims one
// model, the port is actually serving another.
func TestGatherDoctorWarnsWhenThePortServesAnotherModel(t *testing.T) {
	dirs := newTree(t)
	const port = "18201"
	writeConfig(t, dirs, "PORT="+port+"\n")
	if err := os.MkdirAll(dirs.State, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dirs.ActiveEnvFile(), []byte("MODEL=org/configured-model\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	startFakeServer(t, port, "org/running-model")

	var warning string
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		for _, w := range gatherDoctor().Warnings {
			if strings.Contains(w, "org/running-model") {
				warning = w
			}
		}
		if warning != "" {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}
	if warning == "" {
		t.Fatal("no warning about the port serving a different model than the config names")
	}
	for _, want := range []string{"org/configured-model", port, "adopt"} {
		if !strings.Contains(warning, want) {
			t.Errorf("warning %q should mention %q", warning, want)
		}
	}
}

// startFakeServer launches a process whose argv looks like a vLLM server's,
// so the scan underneath doctor has something real to find.
func startFakeServer(t *testing.T, port, model string) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Skipf("can't locate the test binary to re-exec: %v", err)
	}
	bin := filepath.Join(t.TempDir(), "vllm")
	if err := os.Symlink(self, bin); err != nil {
		t.Skipf("can't stage a vllm-named executable: %v", err)
	}
	cmd := exec.Command(bin, "serve", model, "--port", port)
	cmd.Env = append(os.Environ(), sleeperEnv+"=1")
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting the stand-in server: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})
}

func TestGatherCoderStatus(t *testing.T) {
	t.Run("not configured", func(t *testing.T) {
		dirs := newTree(t)
		writeConfig(t, dirs, "PORT=8001\n")
		cfg, err := config.Load()
		if err != nil {
			t.Fatal(err)
		}
		st := gatherCoderStatus(cfg)
		if st.Configured || st.Reachable || st.Error != "" {
			t.Errorf("status = %+v, want everything empty", st)
		}
	})

	t.Run("nil config", func(t *testing.T) {
		if st := gatherCoderStatus(nil); st.Configured {
			t.Errorf("status = %+v, want not configured", st)
		}
	})

	t.Run("reachable", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`[]`))
		}))
		t.Cleanup(srv.Close)
		st := coderStatusFor(t, srv.URL)
		if !st.Configured || !st.Reachable {
			t.Errorf("status = %+v, want configured and reachable", st)
		}
		if st.Error != "" {
			t.Errorf("Error = %q, want none", st.Error)
		}
	})

	// An expired token has to show up in doctor rather than on the next
	// switch, which is why this calls the API instead of checking that the
	// variables are merely set.
	t.Run("credentials rejected", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		}))
		t.Cleanup(srv.Close)
		st := coderStatusFor(t, srv.URL)
		if !st.Configured {
			t.Error("Configured should be true when all three settings are present")
		}
		if st.Reachable {
			t.Error("Reachable should be false when the token is rejected")
		}
		if !strings.Contains(st.Error, "401") {
			t.Errorf("Error = %q, want it to report the rejection", st.Error)
		}
	})
}

func coderStatusFor(t *testing.T, coderURL string) coderStatus {
	t.Helper()
	dirs := newTree(t)
	writeConfig(t, dirs, "CODER_URL="+coderURL+"\nCODER_API_TOKEN=tok\nCODER_AI_PROVIDER_ID=prov\n")
	cfg, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	return gatherCoderStatus(cfg)
}

func TestGatherHFToken(t *testing.T) {
	t.Run("absent", func(t *testing.T) {
		dirs := newTree(t)
		for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"} {
			t.Setenv(k, "")
		}
		writeConfig(t, dirs, "PORT=8001\n")
		cfg, err := config.Load()
		if err != nil {
			t.Fatal(err)
		}
		st := gatherHFToken(cfg)
		if st.Present || st.Source != "" {
			t.Errorf("status = %+v, want no token", st)
		}
	})

	t.Run("from config", func(t *testing.T) {
		dirs := newTree(t)
		writeConfig(t, dirs, "HF_TOKEN=hf_from_config\n")
		cfg, err := config.Load()
		if err != nil {
			t.Fatal(err)
		}
		st := gatherHFToken(cfg)
		if !st.Present || st.Source != "config" {
			t.Errorf("status = %+v, want a token sourced from config", st)
		}
		if st.Error == "" {
			t.Error("Error should report that the Hub couldn't be asked")
		}
	})

	t.Run("from the environment", func(t *testing.T) {
		newTree(t)
		for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"} {
			t.Setenv(k, "")
		}
		t.Setenv("HUGGING_FACE_HUB_TOKEN", "hf_from_env")
		st := gatherHFToken(nil)
		if !st.Present {
			t.Fatalf("status = %+v, want the environment token found", st)
		}
		if !strings.Contains(st.Source, "HUGGING_FACE_HUB_TOKEN") {
			t.Errorf("Source = %q, want it to name the variable it came from", st.Source)
		}
	})
}

func TestGatherToolchain(t *testing.T) {
	checks := gatherToolchain()
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no toolchain checks on %s", runtime.GOOS)
	}
	if len(checks) == 0 {
		t.Fatal("expected toolchain checks on this platform")
	}
	for _, c := range checks {
		if c.Name == "" {
			t.Errorf("check %+v has no name to show the user", c)
		}
		if !c.OK && c.Package == "" && c.Hint == "" {
			t.Errorf("check %q is missing but offers no way to fix it", c.Name)
		}
	}
}

func TestGatherStatus(t *testing.T) {
	t.Run("healthy endpoint", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
		t.Cleanup(srv.Close)
		port := portOf(t, srv.URL)

		dirs := newTree(t)
		writeConfig(t, dirs, "PORT="+port+"\nSERVICE_NAME=vllm-test\nSERVICE_SCOPE=user\n")
		if err := os.MkdirAll(dirs.State, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dirs.ActiveEnvFile(), []byte("MODEL=org/model\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dirs.ActiveProfileFile(), []byte("qwen\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		cfg, err := config.Load()
		if err != nil {
			t.Fatal(err)
		}

		r := gatherStatus(cfg)
		if !r.Healthy || r.HTTPStatus != http.StatusOK {
			t.Errorf("healthy = %v, status = %d, want true and 200", r.Healthy, r.HTTPStatus)
		}
		if r.Model != "org/model" || r.Profile != "qwen" || r.Port != port {
			t.Errorf("report = %+v, want the active model, profile and port", r)
		}
	})

	t.Run("nothing listening", func(t *testing.T) {
		dirs := newTree(t)
		writeConfig(t, dirs, "PORT="+closedPort(t)+"\nSERVICE_NAME=vllm-test\nSERVICE_SCOPE=user\n")
		cfg, err := config.Load()
		if err != nil {
			t.Fatal(err)
		}
		r := gatherStatus(cfg)
		if r.Healthy || r.HTTPStatus != 0 {
			t.Errorf("healthy = %v, status = %d, want false and 0", r.Healthy, r.HTTPStatus)
		}
		if r.Profile != "unknown" || r.Model != "unknown" {
			t.Errorf("report = %+v, want unknown where nothing is set yet", r)
		}
	})
}

func TestOrUnknown(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"org/model", "org/model"},
		{"", "unknown"},
	}
	for _, tt := range tests {
		if got := orUnknown(tt.in); got != tt.want {
			t.Errorf("orUnknown(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestValidateProfile(t *testing.T) {
	dirs := newTree(t)
	tests := []struct {
		name    string
		content string
		wantErr string
	}{{
		name:    "servable",
		content: "MODEL=org/model\nEXTRA_ARGS=\"--tool-call-parser hermes\"\n",
	}, {
		name:    "no model",
		content: "MAX_MODEL_LEN=4096\n",
		wantErr: "no MODEL",
	}, {
		name:    "extra args that won't split",
		content: "MODEL=org/model\nEXTRA_ARGS=\"--served-model-name 'unterminated\"\n",
		wantErr: "won't parse",
	}, {
		name:    "empty extra args is fine",
		content: "MODEL=org/model\nEXTRA_ARGS=\n",
	}, {
		// A context length vLLM can't use reaches its command line
		// verbatim and kills the server on startup, so it has to be
		// refused at the point someone writes it.
		name:    "negative context length",
		content: "MODEL=org/model\nMAX_MODEL_LEN=-5\n",
		wantErr: "MAX_MODEL_LEN",
	}, {
		name:    "zero context length",
		content: "MODEL=org/model\nMAX_MODEL_LEN=0\n",
		wantErr: "MAX_MODEL_LEN",
	}, {
		name:    "non-numeric context length",
		content: "MODEL=org/model\nMAX_MODEL_LEN=lots\n",
		wantErr: "MAX_MODEL_LEN",
	}, {
		name:    "a real context length is fine",
		content: "MODEL=org/model\nMAX_MODEL_LEN=262144\n",
	}, {
		// Unset is how a profile asks for the default, so it must stay
		// valid -- profiles in the field predate this check.
		name:    "unset context length is fine",
		content: "MODEL=org/model\n",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(dirs.ProfileFile("p"), []byte(tt.content), 0o644); err != nil {
				t.Fatal(err)
			}
			p, err := profile.Load(dirs, "p")
			if err != nil {
				t.Fatal(err)
			}
			err = validateProfile(p, "p")
			switch {
			case tt.wantErr == "" && err != nil:
				t.Errorf("unexpected error: %v", err)
			case tt.wantErr != "" && err == nil:
				t.Errorf("expected an error mentioning %q", tt.wantErr)
			case tt.wantErr != "" && !strings.Contains(err.Error(), tt.wantErr):
				t.Errorf("error = %q, want it to mention %q", err, tt.wantErr)
			}
		})
	}
}

func TestFitMemUtilizationLeavesUnusableValuesAlone(t *testing.T) {
	for _, in := range []string{"", "not-a-number", "0", "-0.5"} {
		for _, pinned := range []bool{false, true} {
			if got := fitMemUtilization(in, pinned); got != in {
				t.Errorf("fitMemUtilization(%q, %v) = %q, want it unchanged", in, pinned, got)
			}
		}
	}
}

// vLLM rejects anything outside (0, 1], so whatever the clamp returns for a
// usable value still has to be one.
func TestFitMemUtilizationStaysInRange(t *testing.T) {
	for _, in := range []string{"0.3", "0.85", "0.99"} {
		got := fitMemUtilization(in, false)
		f, err := strconv.ParseFloat(got, 64)
		if err != nil {
			t.Errorf("fitMemUtilization(%q) = %q, which vLLM can't parse", in, got)
			continue
		}
		if f <= 0 || f > 1 {
			t.Errorf("fitMemUtilization(%q) = %q, outside vLLM's (0, 1]", in, got)
		}
		if want, _ := strconv.ParseFloat(in, 64); f > want {
			t.Errorf("fitMemUtilization(%q) = %q, higher than the profile asked for", in, got)
		}
	}
}
