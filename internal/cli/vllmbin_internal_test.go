package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
)

// /opt/homebrew/bin/vllm resolves into a versioned Cellar directory that the
// next 'brew upgrade' deletes. The opt link tracks the installed version.
func TestStableVLLMPathKeepsHomebrewsOptLink(t *testing.T) {
	t.Parallel()

	prefix := t.TempDir()
	keg := filepath.Join(prefix, "Cellar", "vllm-metal", "0.32.0")
	envBin := filepath.Join(keg, "libexec", "bin")
	if err := os.MkdirAll(envBin, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(envBin, "vllm"), nil, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, d := range []string{filepath.Join(prefix, "bin"), filepath.Join(prefix, "opt")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(filepath.Join("..", "Cellar", "vllm-metal", "0.32.0"), filepath.Join(prefix, "opt", "vllm-metal")); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(prefix, "bin", "vllm")
	if err := os.Symlink(filepath.Join("..", "Cellar", "vllm-metal", "0.32.0", "libexec", "bin", "vllm"), link); err != nil {
		t.Fatal(err)
	}

	got := stableVLLMPath(link)
	realPrefix, err := filepath.EvalSymlinks(prefix)
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(realPrefix, "opt", "vllm-metal", "libexec", "bin", "vllm")
	if got != want {
		t.Errorf("stableVLLMPath(%s) = %s, want %s", link, got, want)
	}
}

// Anything outside a Homebrew keg resolves to its real path, as setup always
// did: a venv's bin/vllm is usually reached through a symlink on PATH.
func TestStableVLLMPathResolvesOtherSymlinks(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	target := filepath.Join(dir, "venv", "bin", "vllm")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "vllm")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	want, err := filepath.EvalSymlinks(target)
	if err != nil {
		t.Fatal(err)
	}
	if got := stableVLLMPath(link); got != want {
		t.Errorf("stableVLLMPath(%s) = %s, want %s", link, got, want)
	}
}

// The first setup writes an empty VLLM_BIN when it finds no vllm and says to
// run setup again after installing one, so the rerun has to record it.
func TestFillMissingVLLMBin(t *testing.T) {
	t.Parallel()

	found := "/opt/homebrew/opt/vllm-metal/libexec/bin/vllm"
	cases := []struct {
		name, config, find string
		wantFilled         bool
		wantBin            string
	}{
		{"empty is filled", "PORT=8001\nVLLM_BIN=\nSERVICE_SCOPE=user\n", found, true, found},
		{"missing is filled", "PORT=8001\n", found, true, found},
		{"set is kept", "VLLM_BIN=/venv/bin/vllm\n", found, false, "/venv/bin/vllm"},
		{"still nothing to find", "VLLM_BIN=\n", "", false, ""},
		{"vllm provider is filled", "PROVIDER=vllm\nVLLM_BIN=\n", found, true, found},
		{"other provider is left alone", "PROVIDER=mlx\nVLLM_BIN=\n", found, false, ""},
		{"unknown provider is left alone", "PROVIDER=nope\n", found, false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			path := filepath.Join(t.TempDir(), "config.env")
			if err := os.WriteFile(path, []byte(tc.config), 0o644); err != nil {
				t.Fatal(err)
			}
			bin, filled, err := fillMissingVLLMBin(path, func() string { return tc.find })
			if err != nil {
				t.Fatal(err)
			}
			if filled != tc.wantFilled || (filled && bin != tc.find) {
				t.Errorf("fillMissingVLLMBin() = %q, %v; want filled %v", bin, filled, tc.wantFilled)
			}
			f, err := config.ParseEnvFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if got := f.Get("VLLM_BIN"); got != tc.wantBin {
				t.Errorf("VLLM_BIN = %q, want %q", got, tc.wantBin)
			}
			if got := f.Get("PORT"); strings.Contains(tc.config, "PORT=") && got != "8001" {
				t.Errorf("PORT = %q, want the rest of config.env kept", got)
			}
		})
	}
}
