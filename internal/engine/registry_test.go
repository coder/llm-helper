package engine_test

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/engine"
)

// Every install from before providers existed has no PROVIDER line, and must
// keep serving with vLLM.
func TestLookupDefaultsToVLLM(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"", "  ", "vllm", "VLLM"} {
		p, err := engine.Lookup(name)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", name, err)
		}
		if p.Name() != engine.VLLM {
			t.Errorf("Lookup(%q) = %s, want vllm", name, p.Name())
		}
	}
}

func TestLookupRejectsUnknownNamingTheChoices(t *testing.T) {
	t.Parallel()
	_, err := engine.Lookup("ollama")
	if err == nil {
		t.Fatal("an unknown provider must be an error, not a silent fallback to vLLM")
	}
	for _, want := range []string{"PROVIDER", "ollama", "llamacpp", "mlx", "vllm"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// The argv vLLM gets is pinned, so moving it behind the seam changes nothing.
func TestVLLMCommand(t *testing.T) {
	t.Parallel()
	p, _ := engine.Lookup(engine.VLLM)
	got, err := p.Command(engine.Spec{
		Bin: "/opt/vllm/bin/vllm", Model: "Qwen/Qwen3-32B", Port: "8001",
		MaxModelLen: "32768", MemUtilization: "0.85", APIKey: "sk-x",
		KVCacheBytes: "1000", ExtraArgs: `--tool-call-parser hermes --chat-template "/a b.jinja"`,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"/opt/vllm/bin/vllm", "serve", "Qwen/Qwen3-32B",
		"--port", "8001", "--max-model-len", "32768", "--gpu-memory-utilization", "0.85",
		"--api-key", "sk-x", "--kv-cache-memory-bytes", "1000",
		"--tool-call-parser", "hermes", "--chat-template", "/a b.jinja"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("argv\n got %q\nwant %q", got, want)
	}
}

// mlx_lm.server has no API key or vLLM sizing flags, and passing any of them
// is an argparse error that crash-loops the service.
func TestMLXCommandCarriesNoVLLMFlags(t *testing.T) {
	t.Parallel()
	p, _ := engine.Lookup(engine.MLX)
	got, err := p.Command(engine.Spec{
		Bin: "/v/bin/mlx_lm.server", Model: "mlx-community/Qwen3-4B-4bit", Port: "8001",
		MaxModelLen: "32768", MemUtilization: "0.85", APIKey: "sk-x", KVCacheBytes: "1000",
		ExtraArgs: "--host 0.0.0.0 --max-tokens 8192",
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"/v/bin/mlx_lm.server", "--model", "mlx-community/Qwen3-4B-4bit",
		"--port", "8001", "--host", "0.0.0.0", "--max-tokens", "8192"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("argv\n got %q\nwant %q", got, want)
	}
	if caps := p.Capabilities(); caps.APIKey || caps.VLLMArgs || caps.GGUFQuant {
		t.Errorf("mlx capabilities = %+v, want none", caps)
	}
}

// llama-server gets none of vLLM's sizing flags, which it does not have, and
// its key through the environment rather than argv, which ps shows to every
// user. --alias keeps /v1/models reporting MODEL, which is the name Coder sync
// and status look for.
func TestLlamaCPPCommand(t *testing.T) {
	t.Parallel()
	p, _ := engine.Lookup(engine.LlamaCPP)
	bin := "/opt/homebrew/bin/llama-server"
	for _, tc := range []struct {
		name, model, apiKey, source string
		wantKey                     bool
	}{
		{"hf repo with quant", "bartowski/Qwen_Qwen3-32B-GGUF:Q8_0", "sk-x", "-hf", true},
		{"hf repo", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "sk-x", "-hf", true},
		{"local file", "/models/qwen3-32b-q8_0.gguf", "sk-x", "-m", true},
		{"relative file", "qwen3-32b-q8_0.GGUF", "sk-x", "-m", true},
		{"no key", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "", "-hf", false},
	} {
		got, err := p.Command(engine.Spec{
			Bin: bin, Model: tc.model, Port: "8001", APIKey: tc.apiKey,
			MaxModelLen: "32768", MemUtilization: "0.85", KVCacheBytes: "1000",
			ExtraArgs: `-c 65536 --chat-template-file "/a b.jinja"`,
		})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		// Without a key there is nothing to stop other machines, so it stays
		// on loopback.
		host := "127.0.0.1"
		if tc.wantKey {
			host = "0.0.0.0"
		}
		want := []string{bin, tc.source, tc.model, "--alias", tc.model, "--host", host, "--port", "8001",
			"-c", "65536", "--chat-template-file", "/a b.jinja"}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: argv\n got %q\nwant %q", tc.name, got, want)
		}
		envp, ok := p.(engine.EnvProvider)
		if !ok {
			t.Fatalf("llamacpp does not pass its key through the environment")
		}
		var wantEnv []string
		if tc.wantKey {
			wantEnv = []string{"LLAMA_API_KEY=" + tc.apiKey}
		}
		if gotEnv := envp.Env(engine.Spec{APIKey: tc.apiKey}); !reflect.DeepEqual(gotEnv, wantEnv) {
			t.Errorf("%s: env = %q, want %q", tc.name, gotEnv, wantEnv)
		}
	}
	if caps := p.Capabilities(); !caps.APIKey || caps.VLLMArgs || !caps.GGUFQuant {
		t.Errorf("llamacpp capabilities = %+v, want APIKey and GGUFQuant only", caps)
	}
}

func TestCommandRejectsUnparseableExtraArgs(t *testing.T) {
	t.Parallel()
	for _, name := range engine.Names() {
		p, _ := engine.Lookup(name)
		if _, err := p.Command(engine.Spec{Bin: "/x", Model: "m", Port: "1", ExtraArgs: `--a "unterminated`}); err == nil {
			t.Errorf("%s: unbalanced quote in EXTRA_ARGS was accepted", name)
		}
	}
}

// The complaint names the provider's own setting, so an MLX install is not
// told to fix VLLM_BIN.
func TestValidateNamesTheProvidersBinKey(t *testing.T) {
	t.Parallel()
	p, _ := engine.Lookup(engine.MLX)
	err := engine.Validate(p, engine.Spec{Model: "m", Port: "1"})
	if err == nil || !strings.Contains(err.Error(), "MLX_LM_BIN") {
		t.Fatalf("Validate with no bin = %v, want it to name MLX_LM_BIN", err)
	}

	notExec := filepath.Join(t.TempDir(), "mlx_lm.server")
	if err := os.WriteFile(notExec, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := engine.Validate(p, engine.Spec{Bin: notExec, Model: "m", Port: "1"}); err == nil {
		t.Fatal("a non-executable bin passed validation")
	}

	bin := filepath.Join(t.TempDir(), "mlx_lm.server")
	if err := os.WriteFile(bin, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := engine.Validate(p, engine.Spec{Bin: bin, Port: "1"}); err == nil || !strings.Contains(err.Error(), "MODEL") {
		t.Fatalf("Validate with no model = %v", err)
	}
	if err := engine.Validate(p, engine.Spec{Bin: bin, Model: "m", Port: "1"}); err != nil {
		t.Fatalf("a complete spec failed validation: %v", err)
	}
}
