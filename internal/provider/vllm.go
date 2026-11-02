package provider

import (
	"fmt"

	"github.com/coder/vllm-helper/internal/config"
)

// vllm is vLLM's OpenAI-compatible server, 'vllm serve'. It is the default,
// and the only provider whose profiles carry tuned flags.
type vllm struct{}

func (vllm) Name() string    { return VLLM }
func (vllm) BinKey() string  { return "VLLM_BIN" }
func (vllm) BinName() string { return "vllm" }

func (vllm) Capabilities() Capabilities {
	return Capabilities{APIKey: true, VLLMArgs: true, GGUFQuant: true}
}

// Command expects MemUtilization already fitted to this machine: what fits
// depends on live memory, which the caller measures and this package does not.
func (vllm) Command(s Spec) ([]string, error) {
	args := []string{s.Bin, "serve", s.Model,
		"--port", s.Port,
		"--max-model-len", s.MaxModelLen,
		"--gpu-memory-utilization", s.MemUtilization,
		"--api-key", s.APIKey,
	}
	if s.KVCacheBytes != "" {
		args = append(args, "--kv-cache-memory-bytes", s.KVCacheBytes)
	}
	return appendExtraArgs(args, s.ExtraArgs)
}

func appendExtraArgs(args []string, extra string) ([]string, error) {
	if extra == "" {
		return args, nil
	}
	words, err := config.SplitArgs(extra)
	if err != nil {
		return nil, fmt.Errorf("EXTRA_ARGS: %w", err)
	}
	return append(args, words...), nil
}
