package resolve

import (
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hardware"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/vllm"
)

// station is a DGX Station GB300 as detect sizes it: about 244 GB of weights
// on the GPU and 478 GB more of CPU memory for offloaded experts.
var station = detect.Hardware{
	Memory: string(hardware.MemoryCoherent), CUDA: true,
	GPUMemBytes: 269172604928, TotalMemBytes: 496e9,
	BudgetBytes: 244e9, BudgetGB: 244, OffloadBytes: 478e9, OffloadGB: 478,
}

var deepseek = hf.Arch{ModelType: "deepseek_v3", NRoutedExperts: 256}

func TestTheBestCopyFitsOnTheGPUBeforeAnyIsOffloaded(t *testing.T) {
	t.Parallel()
	fp8 := hf.Variant{Name: "fp8", Label: "FP8", SizeGB: 641}
	nvfp4 := hf.Variant{Name: "nvfp4", Label: "NVFP4", SizeGB: 395}
	q2 := hf.Variant{Name: "q2", Label: "Q2", SizeGB: 149}
	tests := []struct {
		name        string
		all         []hf.Variant
		size        float64
		moe         bool
		offloadGB   float64
		wantBest    string
		wantOffload *bool
	}{
		{"one copy fits on the GPU", []hf.Variant{fp8, nvfp4, q2}, 149, true, 478, "Q2", nil},
		{"none fits on the GPU", []hf.Variant{fp8, nvfp4}, 395, true, 478, "FP8", new(true)},
		{"a dense model is not offloaded", []hf.Variant{fp8, nvfp4}, 395, false, 478, "", nil},
		{"no CPU memory to offload into", []hf.Variant{fp8, nvfp4}, 395, true, 0, "", nil},
		{"one copy, offloaded", []hf.Variant{nvfp4}, 395, true, 478, "", new(true)},
		{"one copy, too big even offloaded", []hf.Variant{{Name: "pro", SizeGB: 805}}, 805, true, 478, "", new(false)},
		{"only an unloadable copy fits offloaded", []hf.Variant{{Name: "bf16", SizeGB: 900}, {Name: "q4", SizeGB: 395, Blocked: "split"}}, 395, true, 478, "", new(false)},
		{"one copy that cannot load", []hf.Variant{{Name: "q4", SizeGB: 395, Blocked: "split"}}, 395, true, 478, "", new(false)},
		{"the copy that fits the GPU cannot load", []hf.Variant{nvfp4, {Name: "q2", Label: "Q2", SizeGB: 149, Blocked: "split"}}, 149, true, 478, "NVFP4", new(true)},
	}
	for _, tt := range tests {
		d := Detail{All: tt.all, SizeGB: tt.size, BudgetGB: 244, BudgetKnown: true, MoE: tt.moe, OffloadGB: tt.offloadGB}
		d.rankVariants(true)
		best := ""
		if d.Best != nil {
			best = d.Best.Label
		}
		if best != tt.wantBest {
			t.Errorf("%s: best %q, want %q", tt.name, best, tt.wantBest)
		}
		if (d.FitsWithOffload == nil) != (tt.wantOffload == nil) || (d.FitsWithOffload != nil && *d.FitsWithOffload != *tt.wantOffload) {
			t.Errorf("%s: fits with offload %v, want %v", tt.name, d.FitsWithOffload, tt.wantOffload)
		}
		if d.FitsWithOffload != nil && *d.FitsWithOffload && d.OffloadNeededGB <= 0 {
			t.Errorf("%s: fits with offload but needs %v GB of it", tt.name, d.OffloadNeededGB)
		}
	}
}

// An engine with no way to offload ranks only what fits on the GPU.
func TestAnEngineThatCannotOffloadIsNotOfferedIt(t *testing.T) {
	t.Parallel()
	mlx, err := engine.Lookup(engine.MLX)
	if err != nil {
		t.Fatal(err)
	}
	d := Detail{
		All:    []hf.Variant{{Name: "a", Label: "A", SizeGB: 641}, {Name: "b", Label: "B", SizeGB: 395}},
		SizeGB: 395, BudgetGB: 244, BudgetKnown: true, MoE: true, OffloadGB: 478,
	}
	d.rankVariants(true)
	got := ForProvider(d, mlx)
	if got.OffloadGB != 0 || got.FitsWithOffload != nil || got.Best != nil {
		t.Errorf("mlx-lm was offered offload: %+v", got)
	}
}

func TestOffloadIsPlannedForAMoEModelTooBigForTheGPU(t *testing.T) {
	t.Parallel()
	vl, err := engine.Lookup(engine.VLLM)
	if err != nil {
		t.Fatal(err)
	}
	off := planOffload(station, deepseek, 395e9, vl, vllm.Caps{})
	if off.GiB <= 0 || !slices.Contains(off.Args, "--cpu-offload-gb") || !slices.Contains(off.Args, "experts") {
		t.Fatalf("no offload planned for a 395 GB MoE model: %+v", off)
	}
	if i := slices.Index(off.Args, "--cpu-offload-gb"); off.Args[i+1] != strconv.Itoa(off.GiB) {
		t.Errorf("--cpu-offload-gb %s, want %d", off.Args[i+1], off.GiB)
	}
	if !slices.Contains(off.Env, "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1") {
		t.Errorf("env %v does not turn off pinning", off.Env)
	}
	if off.Note == "" {
		t.Error("no note explaining the offload")
	}

	for name, o := range map[string]offloadPlan{
		"fits on the GPU": planOffload(station, deepseek, 149e9, vl, vllm.Caps{}),
		"dense":           planOffload(station, hf.Arch{ModelType: "llama"}, 395e9, vl, vllm.Caps{}),
		"discrete GPU":    planOffload(detect.Hardware{CUDA: true, GPUMemBytes: 85e9, BudgetBytes: 70e9}, deepseek, 395e9, vl, vllm.Caps{}),
	} {
		if len(o.Args) > 0 || o.Note != "" {
			t.Errorf("%s: planned %+v", name, o)
		}
	}

	old := vllm.Caps{ReasoningParsers: []string{"deepseek_r1"}, Flags: []string{"--cpu-offload-gb"}}
	if o := planOffload(station, deepseek, 395e9, vl, old); len(o.Args) > 0 || !strings.Contains(o.Note, "cannot") {
		t.Errorf("a vLLM without --cpu-offload-params got %+v", o)
	}
}

// llama-server places experts itself, so the profile gets a note and no flags.
func TestAnAutomaticEngineGetsOnlyANote(t *testing.T) {
	t.Parallel()
	llama, err := engine.Lookup(engine.LlamaCPP)
	if err != nil {
		t.Fatal(err)
	}
	off := planOffload(station, deepseek, 395e9, llama, vllm.Caps{})
	if len(off.Args) > 0 || len(off.Env) > 0 || off.Note == "" {
		t.Errorf("llama.cpp offload plan %+v, want only a note", off)
	}
}
