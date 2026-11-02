package detect

import (
	"strings"
	"testing"
)

// The M5 Ultra this was measured on: 256 GiB of RAM, of which MLX reports a
// 239.1 GB Metal working set.
const (
	macTotal      = 274877906944
	macWorkingSet = 239143780352
)

// vllm-metal reads --gpu-memory-utilization as a share of the Metal working
// set, so that, not installed RAM, is what weights and cache have to fit in.
func TestAMetalBudgetIsMeasuredAgainstTheWorkingSet(t *testing.T) {
	t.Parallel()

	hw := Hardware{TotalMemBytes: macTotal, Unified: true, Metal: true, MetalWorkingSetBytes: macWorkingSet}

	if pool, kind := hw.Pool(); pool != macWorkingSet || kind != "metal" {
		t.Fatalf("Pool() = %d %q, want the working set", pool, kind)
	}
	got, reason := hw.Budget(0)
	if limit := MetalMemUtilization * macWorkingSet; got <= 0 || float64(got) >= limit {
		t.Errorf("budget %.1f GB, want above 0 and below the %.1f GB vllm-metal may use", float64(got)/1e9, limit/1e9)
	}
	if !strings.Contains(reason, "Metal") {
		t.Errorf("reason = %q, want it to name the Metal working set", reason)
	}

	// The CPU backend on the same Mac is budgeted against all of RAM.
	cpu, _ := Hardware{TotalMemBytes: macTotal, Unified: true}.Budget(0)
	if cpu <= got {
		t.Errorf("CPU-backend budget %.1f GB should exceed the Metal one %.1f GB on this Mac", float64(cpu)/1e9, float64(got)/1e9)
	}
}

// When MLX could not be asked, the working set is estimated rather than taken
// to be all of RAM, which macOS never gives the GPU.
func TestAMetalBudgetWithoutAWorkingSetIsAnEstimate(t *testing.T) {
	t.Parallel()

	hw := Hardware{TotalMemBytes: macTotal, Unified: true, Metal: true}
	got, reason := hw.Budget(0)
	if got <= 0 || float64(got) >= metalWorkingSetGuess*macTotal {
		t.Errorf("budget %.1f GB, want above 0 and below the estimated working set", float64(got)/1e9)
	}
	if !strings.Contains(reason, "estimated") {
		t.Errorf("reason = %q, want it to say the figure is an estimate", reason)
	}
}

func TestAMetalPlanFitsInsideWhatVLLMMetalMayUse(t *testing.T) {
	t.Parallel()

	hw := Hardware{TotalMemBytes: macTotal, Unified: true, Metal: true, MetalWorkingSetBytes: macWorkingSet, MetalFixedKVCache: true}
	for _, weights := range []float64{1.5e9, 30e9, 150e9} {
		p := hw.PlanFor(int64(weights), smallModelKVPerToken, 262144, 0)
		if p.MemUtilization != MetalMemUtilization {
			t.Errorf("%.0f GB model: MemUtilization %.2f, want %.2f", weights/1e9, p.MemUtilization, MetalMemUtilization)
		}
		if p.KVCacheBytes <= 0 {
			t.Errorf("%.0f GB model: no KV cache size, want one under a vllm-metal that honors it", weights/1e9)
		}
		ceiling := MetalMemUtilization * macWorkingSet
		committed := weights*(1+activationFraction) + metalRuntimeReserve + float64(p.KVCacheBytes)
		if committed > ceiling {
			t.Errorf("%.0f GB model: commits %.1f GiB against a %.1f GiB ceiling",
				weights/1e9, committed/gibibyte, ceiling/gibibyte)
		}
	}
}

// vllm-metal 0.32.0 accepts --kv-cache-memory-bytes and ignores it, so a plan
// must not claim a cache size it cannot hold the server to.
func TestAMetalPlanLeavesTheCacheToAVLLMMetalThatIgnoresTheSize(t *testing.T) {
	t.Parallel()

	hw := Hardware{TotalMemBytes: macTotal, Unified: true, Metal: true, MetalWorkingSetBytes: macWorkingSet}
	p := hw.PlanFor(30e9, smallModelKVPerToken, 262144, 0)
	if p.KVCacheBytes != 0 || p.Tokens != 0 {
		t.Errorf("plan sized the cache at %d bytes (%d tokens), want it left to vllm-metal", p.KVCacheBytes, p.Tokens)
	}
	if p.MemUtilization != MetalMemUtilization {
		t.Errorf("MemUtilization %.2f, want %.2f", p.MemUtilization, MetalMemUtilization)
	}
	if !strings.Contains(p.Note, "KV_CACHE_MEMORY_BYTES") {
		t.Errorf("Note = %q, want it to say the size is not honored", p.Note)
	}
}

// The tests below are not parallel: the registration is process-wide.

// MetalRuntime is what 'start' asks, so it must not pay for an MLX import.
func TestMetalRuntimeAsksEachProbeOnceAndOnlyWhenNeeded(t *testing.T) {
	t.Cleanup(func() { UseMetalProbe(nil, nil) })

	if MetalRuntime() {
		t.Fatal("MetalRuntime() is true with no probe registered")
	}
	installedCalls, memCalls := 0, 0
	UseMetalProbe(
		func() bool { installedCalls++; return true },
		func() MetalMemory { memCalls++; return MetalMemory{WorkingSetBytes: macWorkingSet, FixedKVCache: true} },
	)
	for range 3 {
		if !MetalRuntime() {
			t.Fatal("MetalRuntime() = false, want the probe's answer")
		}
	}
	if memCalls != 0 {
		t.Errorf("MetalRuntime read the working set %d times; it imports MLX and is not needed there", memCalls)
	}
	m := currentMetal()
	for range 3 {
		if mem := m.memoryInfo(); mem.WorkingSetBytes != macWorkingSet || !mem.FixedKVCache {
			t.Errorf("memoryInfo() = %+v, want the probe's answer", mem)
		}
	}
	if installedCalls != 1 || memCalls != 1 {
		t.Errorf("probes ran %d and %d times, want once each: each starts a Python interpreter", installedCalls, memCalls)
	}
}

// Without vllm-metal there is no MLX to ask.
func TestTheWorkingSetIsNotProbedWithoutVLLMMetal(t *testing.T) {
	t.Cleanup(func() { UseMetalProbe(nil, nil) })

	memCalls := 0
	UseMetalProbe(func() bool { return false }, func() MetalMemory { memCalls++; return MetalMemory{WorkingSetBytes: macWorkingSet} })
	if mem := currentMetal().memoryInfo(); mem != (MetalMemory{}) || memCalls != 0 {
		t.Errorf("memoryInfo() = %+v after %d probes, want nothing without probing", mem, memCalls)
	}
}

// A probe that finishes after a newer registration must not change what the
// newer one reports.
func TestAStaleProbeCannotOverwriteANewerRegistration(t *testing.T) {
	t.Cleanup(func() { UseMetalProbe(nil, nil) })

	release, started := make(chan struct{}), make(chan struct{})
	UseMetalProbe(func() bool {
		close(started)
		<-release
		return true
	}, nil)
	stale := currentMetal()
	done := make(chan bool)
	go func() { done <- stale.runtime() }()
	<-started

	UseMetalProbe(func() bool { return false }, nil)
	close(release)
	if !<-done {
		t.Error("the old registration lost its own answer")
	}
	if MetalRuntime() {
		t.Error("MetalRuntime() = true: the stale probe's answer leaked into the newer registration")
	}
}
