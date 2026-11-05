package cli

import (
	"fmt"

	"github.com/coder/llm-helper/internal/detect"
)

// fitMemUtilization keeps --gpu-memory-utilization within what this machine can
// actually give vLLM. What that means depends on whose memory the flag spends:
//
//   - Discrete VRAM: nobody else can take it, so the request stands.
//   - Unified memory (DGX Spark / GB10, Grace): CPU and GPU share one pool, so
//     the flag is a slice of the whole machine and the remainder is all the OS
//     and page cache get. Overshooting is worse here than elsewhere -- on GB10
//     the driver's allocation failure lands below the CUDA runtime, so rather
//     than a clean OOM the host can wedge hard enough to need a power cycle
//     (vllm#46307). We cap instead of measuring; see capMemUtilization for why.
//   - No NVIDIA GPU: vLLM falls back to the CPU backend, where the flag
//     reserves system RAM and the 0.85 default asks for 85% of the box. That
//     aborts startup on a machine that is also doing something else, so here
//     the live reading is the right signal. Seen on a 15 GiB WSL2 box; #13.
func fitMemUtilization(want string, kvPinned bool) string {
	value, msg := pickMemFit(detect.Probe(), detect.AvailableMemMB(), want, kvPinned)
	if msg != "" {
		fmt.Print(msg)
	}
	return value
}
