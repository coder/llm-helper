package cli

import (
	"fmt"

	"github.com/coder/vllm-helper/internal/detect"
)

// fitMemUtilization keeps --gpu-memory-utilization within what this Mac can
// give vLLM. What the flag means depends on which vLLM is installed:
//
//   - vllm-metal (the Apple Silicon GPU): a share of the Metal working set,
//     which macOS already holds below installed RAM. vllm-metal allocates
//     the cache lazily and checks for memory pressure at startup, shrinking
//     the cache rather than failing, so the request stands.
//   - vLLM's CPU backend: a share of system RAM that must be free right now,
//     or startup aborts. A number saved in a profile goes stale the moment
//     someone opens a browser, so it is clamped here, immediately before
//     exec, the only place the real figure is known.
func fitMemUtilization(want string, _ bool) string {
	if detect.MetalRuntime() {
		return want
	}
	value, msg := clampMemUtilization(want, detect.Probe().TotalMemMB(), detect.AvailableMemMB())
	if msg != "" {
		fmt.Print(msg)
	}
	return value
}
