//go:build darwin

package cli

import (
	"testing"

	"github.com/coder/vllm-helper/internal/detect"
)

// vllm-metal sizes its cache against the Metal working set and checks for
// memory pressure itself, so the CPU backend's free-RAM clamp must not cut a
// GPU profile down to whatever happens to be free.
func TestFitMemUtilizationLeavesVLLMMetalAlone(t *testing.T) {
	detect.UseMetalProbe(func() bool { return true }, func() detect.MetalMemory { return detect.MetalMemory{WorkingSetBytes: 239143780352} })
	t.Cleanup(func() { detect.UseMetalProbe(nil, nil) })

	for _, want := range []string{"0.9", "0.95", "0.5"} {
		if got := fitMemUtilization(want, false); got != want {
			t.Errorf("fitMemUtilization(%s) = %s under vllm-metal, want it unchanged", want, got)
		}
	}
}
