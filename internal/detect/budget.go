package detect

import (
	"math"
	"strconv"
	"strings"
)

// Budget sizing, from vLLM's memory profiler across 14 starts spanning a 125M
// to a 35B model:
//
//   - weights in memory == weights on disk, within 0.2%
//   - non-torch runtime and CUDA graphs are constants, not fractions of the
//     model (0.05-4.00 GiB and at most 0.98 GiB, at every size)
//   - only activation scales, at ~5% of weights
//
// So the budget solves for the largest model whose weights, overhead and a
// usable KV cache all fit -- not a percentage of RAM, which was wrong in both
// directions at once.
const (
	gibibyte = 1 << 30
	gigabyte = 1e9

	runtimeReserve     = 4.5 * gibibyte // Python + CUDA context; max seen 4.00
	graphReserve       = 1.5 * gibibyte // torch.compile + CUDA graphs; max seen 0.98
	activationFraction = 0.06           // peak activation / weights; max seen 4.7%

	// Without CUDA there is no context to hold and no graphs to capture, so
	// most of the above is not spent. Charging for them refuses small models
	// on machines that can serve them: a 7 GB box came out at a budget of 0.
	cpuRuntimeReserve = 2 * gibibyte

	// vllm-metal holds MLX and a torch MPS context, and captures no graphs.
	metalRuntimeReserve = 2 * gibibyte

	// A model you cannot serve does not fit, so hold back a usable cache --
	// capped by fraction so a small card is not budgeted to zero.
	kvFloor     = 6 * gibibyte
	kvFloorFrac = 0.25

	// A discrete card holds only what CUDA puts there.
	discreteBaseline = 0.5 * gibibyte
	discreteRuntime  = 1.5 * gibibyte
)

// weightsBudget solves w + baseline + runtime + graphs + activationFraction*w
// + kv <= pool. Closed form: every term is constant or linear in w. Returns 0
// when the fixed costs already exceed the pool, which is a real answer.
func weightsBudget(pool, baseline, runtime, graphs float64) float64 {
	kv := math.Min(kvFloor, kvFloorFrac*pool)
	free := pool - baseline - runtime - graphs - kv
	if free <= 0 {
		return 0
	}
	return free / (1 + activationFraction)
}

// defaultBaseline estimates what the OS holds when we could not measure it.
// Proportional in the middle, bounded at both ends: 6% of an 8 GB box is far
// less than any Linux install actually uses.
func defaultBaseline(pool float64) float64 {
	const (
		frac = 0.06
		lo   = 2 * gibibyte
		hi   = 8 * gibibyte
	)
	switch b := frac * pool; {
	case b < lo:
		return lo
	case b > hi:
		return hi
	default:
		return b
	}
}

// baselineCeiling is the most of the pool a reading may claim before we stop
// believing it describes an idle machine. Sampling while our own model is
// loaded reads 111 of 121 GiB, which as a baseline would budget zero. Treating
// an implausible reading as no reading avoids coupling this to service state.
const baselineCeiling = 0.15

// Budget reports the largest model weights this machine should load, in bytes,
// with a line explaining the figure. usedBytes is memory in use now, or 0 when
// unknown. A discrete GPU is measured against its own VRAM.
func (h Hardware) Budget(usedBytes int64) (int64, string) {
	poolBytes, poolKind := h.Pool()
	pool := float64(poolBytes)
	runtime, graphs := float64(runtimeReserve), float64(graphReserve)
	baseline := 0.0
	reason := "unified/system memory, minus the OS and vLLM's own overhead"

	// vLLM falls back to its CPU backend without CUDA, where there is no
	// context and no captured graphs to pay for.
	if !h.CUDA {
		runtime, graphs = cpuRuntimeReserve, 0
		reason = "system memory, minus the OS and vLLM's CPU-backend overhead"
	}

	switch {
	case poolKind == "metal":
		// vllm-metal spends --gpu-memory-utilization of the working set on
		// weights and cache together, and macOS keeps the rest of RAM
		// outside the working set, so there is no OS baseline to subtract.
		pool *= MetalMemUtilization
		runtime, graphs = metalRuntimeReserve, 0
		reason = "the Metal GPU working set, minus vllm-metal's overhead and room for a KV cache"
	case h.Metal:
		// The working set could not be read, so estimate it from RAM.
		pool *= metalWorkingSetGuess * MetalMemUtilization
		runtime, graphs = metalRuntimeReserve, 0
		reason = "an estimated Metal GPU working set (MLX could not be asked), minus vllm-metal's overhead"
	case poolKind == "vram":
		runtime, graphs = discreteRuntime, float64(graphReserve)
		baseline = discreteBaseline
		reason = "discrete VRAM, minus CUDA's overhead and room for a KV cache"
	case usedBytes > 0 && float64(usedBytes) < baselineCeiling*pool:
		baseline = float64(usedBytes)
		reason += ", against a measured OS baseline"
	default:
		baseline = defaultBaseline(pool)
	}

	if pool <= 0 {
		return 0, "unknown: this machine's memory could not be measured"
	}
	b := int64(weightsBudget(pool, baseline, runtime, graphs))
	if b <= 0 {
		// A real answer, not a failure to compute one: vLLM's own runtime
		// and a usable cache already exceed what this machine has.
		return 0, "none: this machine has too little memory to serve a model"
	}
	return b, reason
}

// BudgetGB reports the weights budget in decimal gigabytes, honouring an
// override from config.env. An unparseable or non-positive override is treated
// as absent: a zero budget would silently reject every model on the Hub.
func BudgetGB(override string) float64 {
	if v, ok := overrideGB(override); ok {
		return v
	}
	return float64(Probe().BudgetBytes) / gigabyte
}

// overrideGB reads a configured budget, reporting false when there is none to
// use. Kept apart from the probe so the parsing can be checked on its own: the
// probe reads memory that is in use right now, so two calls a moment apart
// legitimately disagree and cannot be compared to each other.
func overrideGB(override string) (float64, bool) {
	v, err := strconv.ParseFloat(strings.TrimSpace(override), 64)
	return v, err == nil && v > 0
}
