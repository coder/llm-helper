package cli

import (
	"fmt"
	"math"
	"strconv"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/control"
	"github.com/coder/vllm-helper/internal/detect"
)

// useServerGPUs tells detect which GPUs a server started with these env
// files will be allowed to use. See control.CUDAVisible for the layering.
func useServerGPUs(files ...*config.EnvFile) {
	detect.UseCUDAVisibleDevices(control.CUDAVisible(files...))
}

// memHeadroom leaves a margin below what's free: vLLM samples available memory
// at startup, and the figure keeps drifting while the engine boots.
const memHeadroom = 0.85

// maxUnifiedMemUtilization is the most of a shared CPU/GPU pool to ask for when
// the KV cache is left to vLLM's profiling, where this number alone decides how
// much of the machine gets eaten. Both vLLM's own GB10 guidance and sparkrun's
// platform default land on 0.85.
const maxUnifiedMemUtilization = 0.85

// capMemUtilization limits --gpu-memory-utilization to ceiling, returning the
// value to use plus a line for the user ("" when the request was already fine).
//
// This deliberately ignores how much memory is free right now. On a unified
// machine the reading is dominated by whichever model is loaded, and the next
// model only starts once that one has exited -- so sampling it during a restart
// measures the outgoing server and would size the incoming one to nothing.
// Measured on a DGX Spark serving a 68 GB model: 0.85 became 0.06.
func capMemUtilization(want string, ceiling float64) (value, msg string) {
	f, err := strconv.ParseFloat(want, 64)
	if err != nil || f <= 0 || f <= ceiling {
		return want, ""
	}
	return strconv.FormatFloat(ceiling, 'f', 2, 64), fmt.Sprintf(
		"Lowering --gpu-memory-utilization %s -> %.2f: this GPU shares one memory pool with the OS, so the rest of the machine has to fit in what's left.\n",
		want, ceiling)
}

// pickMemFit chooses how to treat --gpu-memory-utilization for this machine.
// It is the pure half of fitMemUtilization: everything that decides *which*
// rule applies, with the probing left to the caller so it can be tested.
//
// The order matters. Unified is checked first because such a machine also
// reports a GPU name, and reaching the discrete case would leave the request
// unbounded on exactly the hardware that most needs a bound.
// kvPinned changes what the flag means: with the cache pinned it is only a
// ceiling, so the measured one beats a fixed 0.85. Unpinned, the flag *is* the
// cache size and 0.85 is the most of a shared pool worth risking.
func pickMemFit(hw detect.Hardware, availMB int, want string, kvPinned bool) (value, msg string) {
	switch {
	case hw.Unified:
		ceiling := maxUnifiedMemUtilization
		if kvPinned {
			if m := hw.PlanFor(0, 0, 0, 0).MemUtilization; m > ceiling {
				ceiling = m
			}
		}
		return capMemUtilization(want, ceiling)
	case hw.GPUName != "" || hw.GPUMemGB > 0:
		return want, "" // discrete VRAM: nothing else competes for it
	default:
		return clampMemUtilization(want, hw.TotalMemMB(), availMB)
	}
}

// clampMemUtilization is the pure (no I/O) half of fitMemUtilization. It caps
// --gpu-memory-utilization at what this machine can actually satisfy right now
// and returns the value to use plus a line to show the user ("" when there is
// nothing worth saying). Non-positive totals mean we learned nothing about the
// machine, so the request passes through untouched.
func clampMemUtilization(want string, totalMB, availMB int) (value, msg string) {
	f, err := strconv.ParseFloat(want, 64)
	if err != nil || f <= 0 {
		return want, ""
	}
	if totalMB <= 0 || availMB <= 0 {
		return want, ""
	}

	limit := memHeadroom * float64(availMB) / float64(totalMB)
	if f <= limit {
		return want, ""
	}
	lowered := math.Floor(limit*100) / 100
	if lowered < 0.05 {
		return want, fmt.Sprintf(
			"warning: only %d MiB of %d MiB is free -- vLLM will likely refuse to start. Close some apps and switch again.\n",
			availMB, totalMB)
	}
	return strconv.FormatFloat(lowered, 'f', 2, 64), fmt.Sprintf(
		"Lowering --gpu-memory-utilization %s -> %.2f: only %d MiB of %d MiB is free, and on the CPU backend that flag reserves system RAM.\n",
		want, lowered, availMB, totalMB)
}
