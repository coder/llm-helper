package detect

import "testing"

// A small model with a long context: 1.5 GB of weights that vLLM gave 101.77
// GiB of KV cache, 23 full contexts, on a 121 GiB machine. That is the case
// this planning exists to prevent.
const (
	smallModelKVPerToken = 114688
	smallModelWeights    = 1.5e9
	smallModelMaxLen     = 40960
)

func TestATinyModelNoLongerTakesTheWholeMachine(t *testing.T) {
	hw := Hardware{TotalMemBytes: unifiedTotal, Unified: true}
	p := hw.PlanFor(smallModelWeights, smallModelKVPerToken, smallModelMaxLen, 0)

	if p.KVCacheBytes == 0 {
		t.Fatal("no KV size planned for a model whose shape is fully known")
	}
	// The measured disaster was 101.77 GiB. Anything near it is the bug.
	if got := float64(p.KVCacheBytes) / gibibyte; got > 25 {
		t.Errorf("planned %.1f GiB of KV cache for a 1.5 GB model", got)
	}
	// And it must still hold the context it advertises, several times over.
	if p.Tokens < smallModelMaxLen {
		t.Errorf("planned %d tokens, less than the %d it will be started with", p.Tokens, smallModelMaxLen)
	}
	if want := int64(DefaultContextTarget) * smallModelMaxLen; p.Tokens != want {
		t.Errorf("planned %d tokens, want %d (%dx one full context)", p.Tokens, want, DefaultContextTarget)
	}
}

// A machine whose GPUs CUDA_VISIBLE_DEVICES hides cannot start vLLM, so it
// gets no context estimate, which inspect would otherwise print as if the
// GPUs were there.
func TestNoPlanWhenEveryGPUIsHidden(t *testing.T) {
	hw := Hardware{TotalMemBytes: unifiedTotal, Unified: true, GPUsHidden: 1}
	if !hw.AllGPUsHidden() {
		t.Fatal("the fixture does not hide its GPU")
	}
	p := hw.PlanFor(smallModelWeights, smallModelKVPerToken, smallModelMaxLen, 0)
	if p.KVCacheBytes != 0 || p.Tokens != 0 || p.MemUtilization != 0 {
		t.Errorf("plan = %+v, want none", p)
	}
	if p.Note == "" {
		t.Error("the plan does not say why it is empty")
	}
}

func TestThePlanNeverExceedsWhatVLLMIsAllowedToTouch(t *testing.T) {
	hw := Hardware{TotalMemBytes: unifiedTotal, Unified: true}
	for _, weights := range []float64{1.5e9, 20e9, 63e9, 95e9} {
		p := hw.PlanFor(int64(weights), smallModelKVPerToken, 262144, 0)
		ceiling := p.MemUtilization * float64(unifiedTotal)
		committed := weights*(1+activationFraction) + runtimeReserve + graphReserve + float64(p.KVCacheBytes)
		if committed > ceiling {
			t.Errorf("%.0f GB model: commits %.1f GiB against a %.1f GiB ceiling",
				weights/1e9, committed/gibibyte, ceiling/gibibyte)
		}
	}
}

func TestAModelWithNoRoomLeftIsToldSoRatherThanGivenAScrap(t *testing.T) {
	hw := Hardware{TotalMemBytes: unifiedTotal, Unified: true}
	p := hw.PlanFor(115*gibibyte, smallModelKVPerToken, 262144, 0)
	if p.KVCacheBytes != 0 {
		t.Errorf("planned %.1f GiB of cache for a model that barely fits", float64(p.KVCacheBytes)/gibibyte)
	}
	if p.Note == "" {
		t.Error("no warning for a model with no room for a cache")
	}
}

func TestAnUnreadableConfigLeavesVLLMToProfile(t *testing.T) {
	// Worse than our answer, but far better than a wrong number: a KV cache
	// sized from a guess is an OOM partway through serving.
	hw := Hardware{TotalMemBytes: unifiedTotal, Unified: true}
	p := hw.PlanFor(20e9, 0, 40960, 0)
	if p.KVCacheBytes != 0 {
		t.Errorf("planned %d bytes of cache without knowing the model's shape", p.KVCacheBytes)
	}
	if p.MemUtilization <= 0 {
		t.Error("no utilization ceiling either, which leaves nothing to bound vLLM")
	}
}

func TestTheUtilizationCeilingLeavesTheOSItsMemory(t *testing.T) {
	hw := Hardware{TotalMemBytes: unifiedTotal, Unified: true, BaselineGB: 5.7}
	p := hw.PlanFor(0, 0, 0, 0)
	spare := (1 - p.MemUtilization) * float64(unifiedTotal)
	if spare < 5.7*gigabyte {
		t.Errorf("ceiling %.2f leaves %.1f GiB, less than the %.1f GiB measured in use",
			p.MemUtilization, spare/gibibyte, 5.7*gigabyte/gibibyte)
	}
	// It should still beat the flat 0.85 it replaces, or there was no point.
	if p.MemUtilization <= 0.85 {
		t.Errorf("ceiling %.2f is no better than the fixed 0.85", p.MemUtilization)
	}
}

func TestADiscreteCardPlansAgainstItsOwnMemory(t *testing.T) {
	hw := Hardware{TotalMemBytes: 128 * gibibyte, GPUMemBytes: 24 * gibibyte}
	p := hw.PlanFor(14e9, smallModelKVPerToken, 32768, 0)
	committed := 14e9*(1+activationFraction) + runtimeReserve + graphReserve + float64(p.KVCacheBytes)
	if committed > 24*gibibyte {
		t.Errorf("commits %.1f GiB to a 24 GiB card", committed/gibibyte)
	}
}
