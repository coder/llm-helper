package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/engine"
)

// fakeExecutable writes an executable file called name in dir.
func fakeExecutable(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

// A distribution's llama-server on PATH can be years older than the build the
// user means to serve with; setup found /usr/bin/llama-server (0.2.0) and
// ignored LLAMA_SERVER_BIN. The engine's bin key in the environment wins, for
// every engine, and one that names nothing runnable is an error rather than a
// quiet fall back to the server the user was avoiding.
func TestSetupPrefersTheBinKeyFromTheEnvironment(t *testing.T) {
	prov, err := engine.Lookup("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	onPath, mine := t.TempDir(), t.TempDir()
	old := fakeExecutable(t, onPath, "llama-server")
	built := fakeExecutable(t, mine, "llama-server")
	t.Setenv("PATH", onPath)

	t.Setenv(prov.BinKey(), "")
	if got := lookBin(prov); got != old {
		t.Errorf("unset: lookBin = %q, want the one on PATH, %q", got, old)
	}

	t.Setenv(prov.BinKey(), built)
	if got := lookBin(prov); got != built {
		t.Errorf("%s=%s: lookBin = %q", prov.BinKey(), built, got)
	}

	// A bare name is looked up on PATH and recorded as a full path.
	t.Setenv(prov.BinKey(), "llama-server")
	if got, err := envBin(prov); err != nil || got != old {
		t.Errorf("a bare name: envBin = %q, %v; want %q", got, err, old)
	}

	notExec := filepath.Join(mine, "llama-server.txt")
	if err := os.WriteFile(notExec, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{notExec, filepath.Join(mine, "missing")} {
		t.Setenv(prov.BinKey(), bad)
		if _, err := envBin(prov); err == nil || !strings.Contains(err.Error(), prov.BinKey()+"="+bad) {
			t.Errorf("%s: envBin error = %v, want one naming the variable", bad, err)
		}
	}

	// vLLM's key is honored the same way, ahead of its own search.
	vllm, err := engine.Lookup(engine.VLLM)
	if err != nil {
		t.Fatal(err)
	}
	venv := fakeExecutable(t, mine, "vllm")
	t.Setenv(vllm.BinKey(), venv)
	if got := findVLLMOrEnv(); got != venv {
		t.Errorf("VLLM_BIN=%s: findVLLMOrEnv = %q", venv, got)
	}
}

// A system-scope unit needs sudo, which fails without a terminal on a box
// where it asks for a password, so setup can write a user-scope config. A
// scope that is neither is refused before anything is written.
func TestSetupWritesTheServiceScopeAskedFor(t *testing.T) {
	prov, err := engine.Lookup("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "config.env")
	if err := os.WriteFile(path, []byte(newConfigEnv(prov, "/opt/llama/llama-server", "user", detect.Hardware{})), 0o600); err != nil {
		t.Fatal(err)
	}
	env, err := config.ParseEnvFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := env.Get("SERVICE_SCOPE"); got != "user" {
		t.Errorf("SERVICE_SCOPE = %q, want user", got)
	}
	if got := env.Get("LLAMA_SERVER_BIN"); got != "/opt/llama/llama-server" {
		t.Errorf("LLAMA_SERVER_BIN = %q", got)
	}

	dir := t.TempDir()
	t.Setenv("VLLM_HELPER_CONFIG_DIR", dir)
	if err := runSetup("llamacpp", "users"); err == nil || !strings.Contains(err.Error(), "system or user") {
		t.Errorf("--service-scope users: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.env")); err == nil {
		t.Error("config.env was written for a refused scope")
	}
}

// Setup keeps an existing config.env, so a bin key set in the environment on
// a second run did nothing and said nothing. It now says when the key differs
// from the kept config or names nothing runnable, and stays quiet otherwise.
func TestSetupSaysWhenAKeptConfigIgnoresTheBinKey(t *testing.T) {
	prov, err := engine.Lookup("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	key := prov.BinKey()
	for _, name := range engine.ProcessNames() {
		if other, err := engine.Lookup(name); err == nil && other.BinKey() != "" {
			t.Setenv(other.BinKey(), "")
		}
	}
	dir := t.TempDir()
	recorded := fakeExecutable(t, t.TempDir(), "llama-server")
	built := fakeExecutable(t, dir, "llama-server")
	cfg := filepath.Join(dir, "config.env")
	if err := os.WriteFile(cfg, []byte(key+"="+recorded+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	kept, err := config.ParseEnvFile(cfg)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name, env string
		want      []string
	}{
		{"unset", "", nil},
		{"the same", recorded, nil},
		{"another", built, []string{key + " in the environment (" + built + ") differs from config.env (" + recorded + ")", "set " + key + "=" + built}},
		{"not an executable", filepath.Join(dir, "missing"), []string{key + "=" + filepath.Join(dir, "missing") + " in the environment is not an executable", "keeps config.env"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(key, tc.env)
			notes := envBinNotes(kept)
			if tc.want == nil {
				if len(notes) != 0 {
					t.Errorf("notes = %q, want none", notes)
				}
				return
			}
			if len(notes) != 1 {
				t.Fatalf("notes = %q, want one", notes)
			}
			for _, w := range tc.want {
				if !strings.Contains(notes[0], w) {
					t.Errorf("note %q lacks %q", notes[0], w)
				}
			}
		})
	}
}
