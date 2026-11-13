package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

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
