package cli

import (
	"github.com/coder/vllm-helper/internal/detect"
	"github.com/coder/vllm-helper/internal/vllm"
)

// useMetalProbe tells detect how to find out whether bin serves on the Apple
// Silicon GPU. The answer depends on which vLLM is installed, not on the Mac:
// vllm-metal runs on the GPU, a plain build runs the CPU backend. The backend
// is a module lookup; the working set imports MLX, so only callers that size
// memory pay for it.
func useMetalProbe(bin string) {
	if bin == "" {
		detect.UseMetalProbe(nil, nil)
		return
	}
	detect.UseMetalProbe(
		func() bool { return vllm.MetalInstalled(bin) },
		func() detect.MetalMemory {
			m := vllm.ProbeMetal(bin)
			return detect.MetalMemory{WorkingSetBytes: m.WorkingSetBytes, FixedKVCache: m.FixedKVCache}
		},
	)
}
