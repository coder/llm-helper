package detect

// DefaultContextTarget is how many full-length requests a KV cache is sized to
// hold. Four rather than "whatever is left", which is what this replaced: a
// 1.5 GB model took 101.77 GiB of cache, 23 full contexts, on a 121 GiB box.
// A default rather than a rule -- the right number is a property of the
// workload, so KV_CONTEXT_TARGET overrides it.
const DefaultContextTarget = 4

// Plan is how a specific model should be launched on this machine.
type Plan struct {
	// KVCacheBytes is --kv-cache-memory-bytes, or 0 to leave vLLM to profile.
	KVCacheBytes int64
	// MemUtilization is the ceiling for --gpu-memory-utilization.
	MemUtilization float64
	// Tokens is how much context KVCacheBytes buys; 0 when unknown.
	Tokens int64
	// Note explains the sizing, or is empty when there is nothing to say.
	Note string
}

// PlanFor sizes the KV cache and the memory ceiling for one model. A zero
// kvPerToken means the config did not say enough, and the plan then sets no KV
// size at all: vLLM's own profiling beats a guess.
func (h Hardware) PlanFor(weightBytes, kvPerToken int64, maxLen, contextTarget int) Plan {
	if contextTarget <= 0 {
		contextTarget = DefaultContextTarget
	}
	pool := float64(h.TotalMemBytes)
	if !h.Unified && h.GPUMemBytes > 0 {
		pool = float64(h.GPUMemBytes)
	}
	if pool <= 0 {
		return Plan{}
	}

	util := h.memCeiling(pool)
	reserve := float64(runtimeReserve + graphReserve)
	if h.Metal {
		// vllm-metal reads the flag as a share of the working set, which
		// macOS already holds below RAM, and captures no graphs.
		pool = h.metalPool()
		util = MetalMemUtilization
		reserve = metalRuntimeReserve
	}
	p := Plan{MemUtilization: util}

	if kvPerToken <= 0 || maxLen <= 0 {
		return p
	}

	// Everything that is not KV cache. Weights are known exactly here, so
	// this is tighter than the budget's estimate.
	overhead := float64(weightBytes)*(1+activationFraction) + reserve
	spare := pool*util - overhead
	if spare < float64(kvFloor) {
		// Leave vLLM to profile it, which fails loudly rather than
		// quietly serving one truncated request.
		p.Note = "this model leaves little room for a KV cache -- vLLM will size it, and may not start"
		return p
	}
	if h.Metal && !h.MetalFixedKVCache {
		// A size this vllm-metal ignores would be reported as the cache
		// while the real one is whatever the ceiling leaves.
		p.Note = "this vllm-metal ignores KV_CACHE_MEMORY_BYTES, so it sizes the KV cache itself from what GPU_MEMORY_UTILIZATION leaves after the weights, not from KV_CONTEXT_TARGET"
		return p
	}

	want := float64(contextTarget) * float64(maxLen) * float64(kvPerToken)
	if want > spare {
		want = spare
	}
	p.KVCacheBytes = int64(want)
	p.Tokens = p.KVCacheBytes / kvPerToken

	switch n := float64(p.Tokens) / float64(maxLen); {
	case n < 1:
		p.Note = "only enough KV cache for a partial context -- lower MAX_MODEL_LEN if requests get preempted"
	default:
		p.Note = ""
	}
	return p
}

// memCeiling is everything vLLM may touch: the machine minus the OS, minus a
// margin for drift between our reading and vLLM's at startup. A safety limit,
// which is what this flag should always have been.
func (h Hardware) memCeiling(pool float64) float64 {
	baseline := defaultBaseline(pool)
	if h.BaselineGB > 0 && h.BaselineGB*gigabyte < baselineCeiling*pool {
		baseline = h.BaselineGB * gigabyte
	}
	const drift = 2 * gibibyte
	util := (pool - baseline - drift) / pool
	switch {
	case util > 0.95:
		return 0.95
	case util < 0.30:
		return 0.30
	}
	return util
}

// metalPool is the memory vllm-metal sizes against: the working set MLX
// reports, or an estimate from RAM when it could not be read.
func (h Hardware) metalPool() float64 {
	if h.MetalWorkingSetBytes > 0 {
		return float64(h.MetalWorkingSetBytes)
	}
	return metalWorkingSetGuess * float64(h.TotalMemBytes)
}
