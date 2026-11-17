package control

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/resolve"
)

func configWith(t *testing.T, body string) *config.Config {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.env")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	f, err := config.ParseEnvFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return &config.Config{File: f}
}

func TestHubCacheIsWhatTheServerSees(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("HF_HOME", "/shell/only")
	t.Setenv("HF_HUB_CACHE", "")

	cases := []struct {
		name, env, want string
	}{
		{"the shell's HF_HOME is not the server's", "", filepath.Join(home, ".cache", "huggingface", "hub")},
		{"HF_HOME in config.env", "HF_HOME=/mnt/models\n", "/mnt/models/hub"},
		{"HF_HUB_CACHE beats HF_HOME", "HF_HOME=/a\nHF_HUB_CACHE=/b\n", "/b"},
		{"llama.cpp reads LLAMA_CACHE first", "PROVIDER=llamacpp\nHF_HUB_CACHE=/b\nLLAMA_CACHE=/c\n", "/c"},
		{"LLAMA_CACHE is ignored by other engines", "PROVIDER=vllm\nHF_HUB_CACHE=/b\nLLAMA_CACHE=/c\n", "/b"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := HubCache(configWith(t, c.env)); got != c.want {
				t.Fatalf("HubCache = %q, want %q", got, c.want)
			}
		})
	}
}

func TestHubOfflineReadsTheServerAndTheProcess(t *testing.T) {
	t.Setenv("HF_HUB_OFFLINE", "")
	t.Setenv("TRANSFORMERS_OFFLINE", "")
	if HubOffline(configWith(t, "PORT=8001\n")) {
		t.Fatal("offline with nothing set")
	}
	if !HubOffline(configWith(t, "HF_HUB_OFFLINE=1\n")) {
		t.Fatal("config.env HF_HUB_OFFLINE=1 not seen")
	}
	t.Setenv("HF_HUB_OFFLINE", "1")
	if !HubOffline(configWith(t, "PORT=8001\n")) {
		t.Fatal("process HF_HUB_OFFLINE=1 not seen")
	}
}

// fakeServer is a stand-in for an installed server whose --help says help.
func fakeServer(t *testing.T, help string) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "server")
	script := "#!/bin/sh\ncat <<'EOF'\n" + help + "\nEOF\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin
}

func TestKeepOfflineAsksTheInstalledServer(t *testing.T) {
	hub := t.TempDir()
	hf.UseCache(hub, true)
	t.Cleanup(func() { hf.UseCache("", false) })
	prov, err := engine.Lookup("llamacpp")
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		name, help, want string
	}{
		{"lists the flag", "  --offline  no network", "--offline"},
		{"does not list it", "  --port N", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := configWith(t, prov.BinKey()+"="+fakeServer(t, c.help)+"\n")
			var plan resolve.Plan
			KeepOffline(context.Background(), cfg, prov, &plan)
			if plan.Opts.ExtraArgs != c.want {
				t.Errorf("ExtraArgs = %q, want %q", plan.Opts.ExtraArgs, c.want)
			}
		})
	}
}

func TestKeepOfflineDoesNothingOnline(t *testing.T) {
	hf.UseCache("", false)
	prov, _ := engine.Lookup("vllm")
	var plan resolve.Plan
	KeepOffline(context.Background(), configWith(t, "PORT=1\n"), prov, &plan)
	if len(plan.Opts.Env) != 0 || plan.Opts.ExtraArgs != "" || len(plan.Notes) != 0 {
		t.Errorf("plan changed: %+v", plan)
	}
}

func TestKeepOfflineWritesTheProfileEveryEngineNeeds(t *testing.T) {
	hf.UseCache(t.TempDir(), true)
	t.Cleanup(func() { hf.UseCache("", false) })
	for _, id := range []string{"vllm", "mlx", "llamacpp"} {
		prov, err := engine.Lookup(id)
		if err != nil {
			t.Fatal(err)
		}
		cfg := configWith(t, prov.BinKey()+"="+fakeServer(t, "--offline")+"\n")
		plan := resolve.Plan{Opts: profile.CreateOpts{Repo: "org/model", Provider: id, Source: "test"}}
		KeepOffline(context.Background(), cfg, prov, &plan)

		dirs := config.Dirs{Config: t.TempDir(), Profiles: t.TempDir()}
		if _, err := profile.Create(dirs, plan.Opts); err != nil {
			t.Fatal(err)
		}
		body, err := os.ReadFile(dirs.ProfileFile("model"))
		if err != nil {
			t.Fatal(err)
		}
		off := prov.OfflineSetting()
		for _, line := range off.EnvLines() {
			if !strings.Contains(string(body), line) {
				t.Errorf("%s: profile lacks %q:\n%s", id, line, body)
			}
		}
		for _, a := range off.Args {
			if !strings.Contains(string(body), a) {
				t.Errorf("%s: profile lacks %q:\n%s", id, a, body)
			}
		}
	}
}
