package hf

import (
	"strings"
	"testing"
)

// The real NVIDIA Nemotron-3-Nano card. It exercises every rule at once: a
// plugin-provided parser, memory-shaping flags worth keeping, topology flags
// that describe someone else's box, and a pasted credential.
const nemotronSnippet = `vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --served-model-name model \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --api-key sk-should-never-be-copied`

func TestFlagsFromSnippet(t *testing.T) {
	t.Parallel()
	args, maxLen := flagsFromSnippet(nemotronSnippet)

	if maxLen != "262144" {
		t.Errorf("maxModelLen = %q, want 262144", maxLen)
	}
	for _, want := range []string{
		"--trust-remote-code",
		"--max-num-seqs 8", // memory-shaping: the allowlist dropped this
		"--enable-auto-tool-choice",
		"--tool-call-parser qwen3_coder",
		"--reasoning-parser nano_v3",
		// Carried so the validator can explain why nano_v3 is unknown.
		"--reasoning-parser-plugin nano_v3_reasoning_parser.py",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("missing %q in %q", want, args)
		}
	}
	for _, unwanted := range []string{
		"--max-model-len",        // owned: becomes MAX_MODEL_LEN
		"--tensor-parallel-size", // someone else's GPU count
		"--served-model-name",    // vllm-helper names the model
		"--api-key",              // never into a profile file
		"sk-should-never",
	} {
		if strings.Contains(args, unwanted) {
			t.Errorf("%q should not be adopted, got %q", unwanted, args)
		}
	}
}

func TestFlagsFromSnippetKeepsThinkingDefault(t *testing.T) {
	t.Parallel()
	args, _ := flagsFromSnippet(`vllm serve org/m --reasoning-parser deepseek_v3`)
	if !strings.Contains(args, `--default-chat-template-kwargs {"thinking":true}`) {
		t.Errorf("deepseek_v3 should still get the thinking default, got %q", args)
	}
	other, _ := flagsFromSnippet(`vllm serve org/m --reasoning-parser qwen3`)
	if strings.Contains(other, "thinking") {
		t.Errorf("only the deepseek-style parsers need it, got %q", other)
	}
}

func TestFlagsFromSnippetEdgeCases(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, snippet, want string
	}{
		{"equals form", `vllm serve org/m --tool-call-parser=hermes`, "--tool-call-parser hermes"},
		{"negative value is not a flag", `vllm serve org/m --seed -1`, "--seed -1"},
		{"bare flag at the end", `vllm serve org/m --trust-remote-code`, "--trust-remote-code"},
		{"quoted value stays one argument", `vllm serve org/m --chat-template "a b.jinja"`, `--chat-template "a b.jinja"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, _ := flagsFromSnippet(tt.snippet)
			if !strings.Contains(got, tt.want) {
				t.Errorf("flagsFromSnippet(%q) = %q, want it to contain %q", tt.snippet, got, tt.want)
			}
		})
	}
}

// A snippet that isn't valid shell yields nothing rather than half of it.
func TestFlagsFromSnippetUnparseable(t *testing.T) {
	t.Parallel()
	if got, _ := flagsFromSnippet(`vllm serve org/m --chat-template "unterminated`); got != "" {
		t.Errorf("got %q, want nothing from an unparseable snippet", got)
	}
}

func TestParseSnippetReplacesTheParser(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, snippet, parser, want string
	}{
		{"into deepseek_v3", `vllm serve org/m --reasoning-parser deepseek_r1`, "deepseek_v3", `--reasoning-parser deepseek_v3 --default-chat-template-kwargs {"thinking":true}`},
		{"away from deepseek_v3", `vllm serve org/m --reasoning-parser deepseek_v3`, "qwen3", "--reasoning-parser qwen3"},
		{"equals form", `vllm serve org/m --reasoning-parser=deepseek_r1 --enforce-eager`, "qwen3", "--reasoning-parser qwen3 --enforce-eager"},
		{"no parser in the snippet", `vllm serve org/m --enforce-eager`, "qwen3", "--enforce-eager"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, _, _ := parseSnippet(tt.snippet, tt.parser)
			if got != tt.want {
				t.Errorf("parseSnippet() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSGLangReasoningParser(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, card, want string
	}{
		{"qwen3 hybrid card", "- SGLang:\n    ```shell\n    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3\n    ```\n- vLLM:\n    ```shell\n    vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1\n    ```\n", "qwen3"},
		{"continued lines", "python3 -m sglang.launch_server \\\n    --model-path ./m \\\n    --reasoning-parser=deepseek-r1\n", "deepseek-r1"},
		{"commands disagree", "sglang.launch_server --reasoning-parser qwen3\nsglang.launch_server --reasoning-parser deepseek-r1\n", ""},
		{"no parser", "python -m sglang.launch_server --model-path m\n", ""},
		{"no sglang command", "vllm serve m --reasoning-parser deepseek_r1\n", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := sglangReasoningParser(tt.card); got != tt.want {
				t.Errorf("sglangReasoningParser() = %q, want %q", got, tt.want)
			}
		})
	}
}
