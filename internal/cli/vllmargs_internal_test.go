package cli

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/nodes"
	"github.com/coder/vllm-helper/internal/profile"
	"github.com/coder/vllm-helper/internal/provider"
)

func TestRedactServeArgs(t *testing.T) {
	got := shellJoin(redactServeArgs([]string{"vllm", "serve", "m", "--api-key", "sk-secret", "--port", "8001"}))
	if containsSecret(got, "sk-secret") {
		t.Fatalf("secret was not redacted: %s", got)
	}
	if !containsSecret(got, "<redacted>") {
		t.Fatalf("redacted placeholder missing: %s", got)
	}
}

// EXTRA_ARGS is free text, so a key can arrive inline, as the last token, or
// under another credential-shaped flag; none of them may survive a dry run.
func TestRedactServeArgsCoversEverySpelling(t *testing.T) {
	for _, args := range [][]string{
		{"vllm", "serve", "m", "--api-key=sk-secret", "--port", "8001"},
		{"vllm", "serve", "m", "--port", "8001", "--api-key=sk-secret"},
		{"vllm", "serve", "m", "--hf-token", "sk-secret"},
	} {
		got := shellJoin(redactServeArgs(args))
		if containsSecret(got, "sk-secret") {
			t.Errorf("secret was not redacted: %s", got)
		}
		if !containsSecret(got, "<redacted>") {
			t.Errorf("redacted placeholder missing: %s", got)
		}
	}
	// A real flag that merely contains "key" is left alone, and so is the
	// value that follows it.
	got := shellJoin(redactServeArgs([]string{"vllm", "--ssl-keyfile", "/etc/k.pem", "--port", "8001"}))
	if !containsSecret(got, "/etc/k.pem") {
		t.Errorf("non-secret flag was redacted: %s", got)
	}
}

func containsSecret(s, needle string) bool {
	for i := 0; i+len(needle) <= len(s); i++ {
		if s[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

// An active.env with no PROVIDER is every install from before providers
// existed, and must produce the vllm serve command it always did.
func TestServeSpecFromActivePicksTheProvider(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string, mode os.FileMode) string {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), mode); err != nil {
			t.Fatal(err)
		}
		return path
	}
	vllmBin := write("vllm", "", 0o755)
	mlxBin := write("mlx_lm.server", "", 0o755)

	tests := []struct {
		name   string
		active string
		want   []string
	}{{
		name: "no provider is vllm",
		active: "VLLM_BIN=" + vllmBin + "\nMLX_LM_BIN=" + mlxBin + "\nPORT=8001\nVLLM_API_KEY=k\n" +
			"MODEL=org/m\nMAX_MODEL_LEN=4096\nGPU_MEMORY_UTILIZATION=0.5\nKV_CACHE_MEMORY_BYTES=1000\nEXTRA_ARGS=--x y\n",
		want: []string{vllmBin, "serve", "org/m", "--port", "8001", "--max-model-len", "4096",
			"--gpu-memory-utilization", "0.5", "--api-key", "k", "--kv-cache-memory-bytes", "1000", "--x", "y"},
	}, {
		name: "mlx",
		active: "VLLM_BIN=" + vllmBin + "\nMLX_LM_BIN=" + mlxBin + "\nPORT=8001\nVLLM_API_KEY=k\n" +
			"PROVIDER=mlx\nMODEL=mlx-community/m-4bit\nEXTRA_ARGS=--max-tokens 4096\n",
		want: []string{mlxBin, "--model", "mlx-community/m-4bit", "--port", "8001", "--max-tokens", "4096"},
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			active, err := config.ParseEnvFile(write("active.env", tt.active, 0o600))
			if err != nil {
				t.Fatal(err)
			}
			spec, err := serveSpecFromActive(active)
			if err != nil {
				t.Fatal(err)
			}
			got, err := spec.args()
			if err != nil {
				t.Fatal(err)
			}
			// The memory fit may lower vLLM's fraction on a busy machine;
			// that is memfit's business, so compare around it.
			for i := range got {
				if i > 0 && got[i-1] == "--gpu-memory-utilization" {
					got[i] = "0.5"
				}
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("argv\n got %q\nwant %q", got, tt.want)
			}
		})
	}
}

func TestServeSpecFromActiveRejectsAnUnknownProvider(t *testing.T) {
	path := filepath.Join(t.TempDir(), "active.env")
	if err := os.WriteFile(path, []byte("PROVIDER=ollama\nMODEL=m\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	active, err := config.ParseEnvFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := serveSpecFromActive(active); err == nil || !strings.Contains(err.Error(), "ollama") {
		t.Fatalf("err = %v, want it to name the unknown provider", err)
	}
}

// A shard drives vLLM's own executors, so an mlx profile has to be refused
// before anything is started on the nodes.
func TestRequireVLLM(t *testing.T) {
	dir := t.TempDir()
	dirs := config.Dirs{Config: dir, Profiles: dir}
	for name, body := range map[string]string{"old": "MODEL=m\n", "mlx": "PROVIDER=mlx\nMODEL=m\n"} {
		if err := os.WriteFile(dirs.ProfileFile(name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	old, _ := profile.Load(dirs, "old")
	if err := requireVLLM(old, "shard"); err != nil {
		t.Errorf("a profile naming no provider was refused: %v", err)
	}
	mlx, _ := profile.Load(dirs, "mlx")
	if err := requireVLLM(mlx, "shard"); err == nil || !strings.Contains(err.Error(), "mlx") {
		t.Errorf("err = %v, want an mlx profile refused by name", err)
	}
}

// The key reaches llama-server through its environment, replacing any copy it
// inherited, and a vLLM spec's environment is left as it was.
func TestWithProviderEnv(t *testing.T) {
	t.Parallel()
	llama, err := provider.Lookup(provider.LlamaCPP)
	if err != nil {
		t.Fatal(err)
	}
	env := []string{"PATH=/usr/bin", "LLAMA_API_KEY=stale", "VLLM_API_KEY=sk-x"}
	got := withProviderEnv(env, serveSpec{Provider: llama, Spec: provider.Spec{APIKey: "sk-x"}})
	want := []string{"PATH=/usr/bin", "VLLM_API_KEY=sk-x", "LLAMA_API_KEY=sk-x"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("llamacpp env = %q, want %q", got, want)
	}
	if got := withProviderEnv(env, serveSpec{Spec: provider.Spec{APIKey: "sk-x"}}); !reflect.DeepEqual(got, env) {
		t.Errorf("vllm env = %q, want it unchanged", got)
	}
}

// useLocalArch makes this machine report arch for the length of the test.
func useLocalArch(t *testing.T, arch string) {
	t.Helper()
	old := localArch
	localArch = func() string { return arch }
	t.Cleanup(func() { localArch = old })
}

// The forbidden flags were only checked for tested recipes, so the same
// SM100-only kernel in a plain profile's EXTRA_ARGS still reached vllm serve
// on a GB10 and aborted it at startup.
func TestServeArgsRefuseAFlagThisArchitectureAbortsOn(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "vllm")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	spec := serveSpec{Spec: provider.Spec{
		Bin: bin, Model: "m", Port: "8001",
		ExtraArgs: "--moe-backend deep_gemm_mega_moe",
	}}
	useLocalArch(t, "sm121")
	if _, err := spec.args(); err == nil || !strings.Contains(err.Error(), "sm121") {
		t.Fatalf("args() = %v, want a refusal naming sm121", err)
	}
	// The flag is only fatal on the architecture that lists it.
	useLocalArch(t, "sm90")
	if _, err := spec.args(); err != nil {
		t.Fatalf("args() on sm90 = %v, want it allowed", err)
	}
	// A shard's own args are checked too, not only the profile's.
	useLocalArch(t, "sm121")
	spec.ExtraArgs = ""
	if _, err := spec.args("--moe-backend", "deep_gemm_mega_moe"); err == nil {
		t.Fatal("args() accepted the flag when it came from the caller")
	}
}

// A shard's workers run the head's command line on their own hardware, so a
// worker the registry knows is a GB10 is checked before any node is touched.
func TestCheckNodesRefusesAWorkerThatWouldAbort(t *testing.T) {
	spec := serveSpec{Spec: provider.Spec{
		Bin: "/opt/vllm/bin/vllm", Model: "m", Port: "8001",
		ExtraArgs: "--moe-backend deep_gemm_mega_moe",
	}}
	head := nodes.Node{Name: "head", Caps: nodes.Capabilities{Arch: "sm90"}}
	unprobed := nodes.Node{Name: "old"}
	if err := spec.checkNodes([]nodes.Node{head, unprobed}); err != nil {
		t.Fatalf("checkNodes = %v, want nodes with no rule for the flag allowed", err)
	}
	worker := nodes.Node{Name: "spark-2", Caps: nodes.Capabilities{Arch: "sm121"}}
	err := spec.checkNodes([]nodes.Node{head, worker})
	if err == nil || !strings.Contains(err.Error(), "spark-2") {
		t.Fatalf("checkNodes = %v, want a refusal naming spark-2", err)
	}
}
