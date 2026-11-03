package detect

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// Four rows from a multi-GPU box, with the spacing nvidia-smi really emits.
const fourH100 = `0, GPU-11111111-aaaa-bbbb-cccc-000000000000, NVIDIA H100 80GB HBM3, 81559
1, GPU-22222222-aaaa-bbbb-cccc-000000000000, NVIDIA H100 80GB HBM3, 81559
2, GPU-33333333-aaaa-bbbb-cccc-000000000000, NVIDIA H100 80GB HBM3, 81559
3, GPU-34444444-aaaa-bbbb-cccc-000000000000, NVIDIA H100 80GB HBM3, 81559
`

// Only the first row used to be read, so every GPU after it was invisible and
// a mixed box was described by whichever card enumerated first.
func TestParseNVIDIASMI(t *testing.T) {
	t.Parallel()

	gpus := parseNVIDIASMI(fourH100)
	if len(gpus) != 4 {
		t.Fatalf("got %d GPUs, want 4: %+v", len(gpus), gpus)
	}
	if g := gpus[3]; g.Index != 3 || g.Name != "NVIDIA H100 80GB HBM3" || g.MemBytes != 81559<<20 || !strings.HasPrefix(g.UUID, "GPU-3444") {
		t.Errorf("gpus[3] = %+v", g)
	}

	// A GB10 reports no memory of its own; that is 0, not a parse failure.
	gb10 := parseNVIDIASMI("0, GPU-5f0e, NVIDIA GB10, [N/A]\n")
	if len(gb10) != 1 || gb10[0].Name != "NVIDIA GB10" || gb10[0].MemBytes != 0 {
		t.Errorf("GB10 = %+v", gb10)
	}

	// One unreadable row must not hide the rest.
	mixed := parseNVIDIASMI("garbage\n1, GPU-x, NVIDIA RTX A6000, 49140\n")
	if len(mixed) != 1 || mixed[0].Index != 1 {
		t.Errorf("mixed = %+v", mixed)
	}
	if got := parseNVIDIASMI(""); len(got) != 0 {
		t.Errorf("empty output = %+v", got)
	}
}

// CUDA_VISIBLE_DEVICES decides which GPU a server loads onto, so describing
// any other one sizes models for the wrong card.
func TestVisibleGPUs(t *testing.T) {
	t.Parallel()
	all := parseNVIDIASMI(fourH100)

	tests := []struct {
		name  string
		value string
		set   bool
		want  []int
	}{
		{name: "unset shows every GPU", set: false, want: []int{0, 1, 2, 3}},
		{name: "empty hides every GPU", value: "", set: true, want: nil},
		{name: "-1 hides every GPU", value: "-1", set: true, want: nil},
		{name: "order is the server's", value: "2,0", set: true, want: []int{2, 0}},
		{name: "spaces are ignored", value: " 1 , 3 ", set: true, want: []int{1, 3}},
		{name: "an unknown index ends the list", value: "1,7,2", set: true, want: []int{1}},
		{name: "a repeat ends the list", value: "1,1,2", set: true, want: []int{1}},
		{name: "full UUID", value: "GPU-22222222-aaaa-bbbb-cccc-000000000000", set: true, want: []int{1}},
		{name: "unique UUID prefix", value: "GPU-1111,GPU-2", set: true, want: []int{0, 1}},
		{name: "ambiguous UUID prefix names nothing", value: "GPU-3", set: true, want: nil},
		{name: "MIG instance is not described", value: "0,MIG-abcdef", set: true, want: []int{0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := visibleGPUs(all, tt.value, tt.set)
			var idx []int
			for _, g := range got {
				idx = append(idx, g.Index)
			}
			if !slices.Equal(idx, tt.want) {
				t.Errorf("visibleGPUs(%q, %v) = %v, want %v", tt.value, tt.set, idx, tt.want)
			}
		})
	}
}

// A box with several cards is sized for one, because that is what a server
// without --tensor-parallel-size uses -- but it has to say so, or 80 GB on an
// 8-GPU machine reads as the machine's limit.
func TestFactsHardwareMultiGPU(t *testing.T) {
	t.Parallel()

	f := facts{totalMem: 1 << 40, availMem: 1<<40 - 8<<30, smiOut: fourH100, smiOK: true, nvidiaDevice: true}
	hw := f.hardware(nil, nil)
	if len(hw.GPUs) != 4 || hw.GPUsHidden != 0 {
		t.Fatalf("GPUs = %d, hidden = %d", len(hw.GPUs), hw.GPUsHidden)
	}
	if hw.GPUName != "NVIDIA H100 80GB HBM3" || hw.GPUMemBytes != 81559<<20 || hw.Unified || !hw.CUDA {
		t.Errorf("hw = %+v", hw)
	}
	if got := hw.GPUSummary(); got != "4x NVIDIA H100 80GB HBM3" {
		t.Errorf("GPUSummary = %q", got)
	}
	if !strings.Contains(hw.BudgetReason, "one of 4 GPUs") {
		t.Errorf("BudgetReason = %q, want it to say the budget is one GPU's", hw.BudgetReason)
	}

	f.visible, f.visibleSet = "3", true
	one := f.hardware(nil, nil)
	if len(one.GPUs) != 1 || one.GPUs[0].Index != 3 || one.GPUsHidden != 3 {
		t.Errorf("CUDA_VISIBLE_DEVICES=3: GPUs = %+v, hidden = %d", one.GPUs, one.GPUsHidden)
	}
	if strings.Contains(one.BudgetReason, "GPUs") {
		t.Errorf("one visible GPU: BudgetReason = %q", one.BudgetReason)
	}

	f.visible = ""
	none := f.hardware(nil, nil)
	if none.GPUName != "" || none.GPUsHidden != 4 || !none.CUDA {
		t.Errorf("CUDA_VISIBLE_DEVICES='': %+v", none)
	}
}

// vLLM chooses CUDA from NVML's device count, which ignores
// CUDA_VISIBLE_DEVICES, and then cannot start with no device left to it. A
// Spark told to hide its GPU still came out at a 103 GB budget.
func TestHidingEveryGPUBudgetsNothing(t *testing.T) {
	t.Parallel()
	for _, value := range []string{"", "-1", "7"} {
		f := facts{totalMem: unifiedTotal, availMem: unifiedTotal - 8<<30, smiOut: "0, GPU-5f0e, NVIDIA GB10, [N/A]\n", smiOK: true, nvidiaDevice: true, visible: value, visibleSet: true}
		hw := f.hardware(nil, nil)
		if !hw.CUDA || !hw.AllGPUsHidden() {
			t.Errorf("CUDA_VISIBLE_DEVICES=%q: CUDA = %v, AllGPUsHidden = %v", value, hw.CUDA, hw.AllGPUsHidden())
		}
		if hw.BudgetBytes != 0 || !strings.Contains(hw.BudgetReason, "CUDA_VISIBLE_DEVICES") {
			t.Errorf("CUDA_VISIBLE_DEVICES=%q: budget %d (%s), want 0 naming CUDA_VISIBLE_DEVICES", value, hw.BudgetBytes, hw.BudgetReason)
		}
	}

	// nvidia-smi answering with rows nobody could read hides nothing: that
	// is a GPU we could not describe, sized as before.
	f := facts{totalMem: unifiedTotal, availMem: unifiedTotal - 8<<30, smiOut: "garbage\n", smiOK: true, nvidiaDevice: true}
	if hw := f.hardware(nil, nil); hw.AllGPUsHidden() || hw.BudgetBytes <= 0 {
		t.Errorf("unparseable nvidia-smi: AllGPUsHidden = %v, budget %d", hw.AllGPUsHidden(), hw.BudgetBytes)
	}
}

// Memory a loaded model holds is no baseline, so the budget ignores it; but
// it is still reported, because the next vLLM sizes its cache from what is
// free.
func TestMemoryInUseIsReportedEvenWhenTheBudgetIgnoresIt(t *testing.T) {
	t.Parallel()
	idle := facts{totalMem: unifiedTotal, availMem: unifiedTotal - 8<<30, smiOut: "0, GPU-5f0e, NVIDIA GB10, [N/A]\n", smiOK: true, nvidiaDevice: true}
	busy := idle
	busy.availMem = 7 << 30
	a, b := idle.hardware(nil, nil), busy.hardware(nil, nil)
	if a.Busy() || !b.Busy() {
		t.Errorf("Busy: idle = %v, loaded = %v", a.Busy(), b.Busy())
	}
	if b.InUseBytes != unifiedTotal-7<<30 || b.BaselineGB != 0 {
		t.Errorf("loaded: InUseBytes = %d, BaselineGB = %v", b.InUseBytes, b.BaselineGB)
	}
	if want, _ := b.Budget(0); b.BudgetBytes != want || want <= 0 {
		t.Errorf("loaded: budget %d, want the unmeasured-baseline figure %d", b.BudgetBytes, want)
	}
}

// A wedged driver hangs nvidia-smi instead of failing it, and every command
// that sizes memory used to hang with it.
func TestAHungNVIDIASMIIsAbandoned(t *testing.T) {
	script := filepath.Join(t.TempDir(), "nvidia-smi")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nsleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	oldCmd, oldTimeout := smiCommand, smiTimeout
	smiCommand, smiTimeout = script, 100*time.Millisecond
	t.Cleanup(func() { smiCommand, smiTimeout = oldCmd, oldTimeout })

	start := time.Now()
	if _, ok := runNVIDIASMI(nvidiaSMIArgs...); ok {
		t.Error("a hung nvidia-smi was reported as answering")
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("runNVIDIASMI took %v, want it abandoned after the timeout", d)
	}
}

func TestGPUSummaryMixed(t *testing.T) {
	t.Parallel()
	hw := Hardware{GPUName: "NVIDIA GeForce RTX 4090", GPUs: []GPU{{Name: "NVIDIA GeForce RTX 4090"}, {Index: 1, Name: "NVIDIA GeForce RTX 3090"}}}
	if got := hw.GPUSummary(); got != "NVIDIA GeForce RTX 4090 + NVIDIA GeForce RTX 3090" {
		t.Errorf("GPUSummary = %q", got)
	}
	if got := (Hardware{GPUName: "Apple M4 Max"}).GPUSummary(); got != "Apple M4 Max" {
		t.Errorf("no NVIDIA list: GPUSummary = %q", got)
	}
}

// A node is read over SSH with RemoteScript and must come out as it would
// have locally, so the multi-node view and that node's own doctor agree.
func TestParseRemote(t *testing.T) {
	t.Parallel()

	spark := "memtotal 127535012\nmemavail 120000000\nnvidiactl 1\nsmi ok\ngpu 0, GPU-5f0e, NVIDIA GB10, [N/A]\n"
	hw := ParseRemote(spark)
	if hw.TotalMemBytes != 127535012*1024 || hw.GPUName != "NVIDIA GB10" || !hw.Unified || !hw.CUDA || len(hw.GPUs) != 1 {
		t.Errorf("Spark = %+v", hw)
	}
	if pool, kind := hw.Pool(); kind != "ram" || pool != hw.TotalMemBytes {
		t.Errorf("Spark pool = %d %s", pool, kind)
	}
	if hw.BudgetBytes <= 0 {
		t.Errorf("Spark budget = %d", hw.BudgetBytes)
	}

	cpu := ParseRemote("memtotal 16000000\nmemavail 12000000\n")
	if cpu.CUDA || cpu.GPUName != "" || cpu.TotalMemBytes != 16000000*1024 {
		t.Errorf("CPU box = %+v", cpu)
	}

	// nvidia-smi failing on a box with the driver's device node is still a
	// CUDA machine, as the local probe treats it.
	broken := ParseRemote("memtotal 16000000\nnvidiactl 1\n")
	if !broken.CUDA || broken.GPUName != "" {
		t.Errorf("driver mismatch = %+v", broken)
	}

	if got := ParseRemote(""); got.TotalMemBytes != 0 {
		t.Errorf("no output = %+v", got)
	}
}

// The script and the parser are one contract; a key renamed on one side
// would silently drop that reading.
func TestRemoteScriptEmitsWhatParseRemoteReads(t *testing.T) {
	t.Parallel()
	s := RemoteScript()
	for _, key := range []string{`"memtotal `, `"memavail `, `"nvidiactl `, `"smi ok"`, `s/^/gpu /`, strings.Join(nvidiaSMIArgs, " ")} {
		if !strings.Contains(s, key) {
			t.Errorf("RemoteScript does not contain %q", key)
		}
	}
}
