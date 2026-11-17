package detect

import (
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/hardware"
)

// stationReadings is a DGX Station GB300 as the synthetic report records it.
func stationReadings(t *testing.T) Readings {
	t.Helper()
	for _, f := range loadFixtures(t) {
		if f.name == "dgx-station-gb300.json" {
			return f.Readings
		}
	}
	t.Fatal("no dgx-station-gb300.json fixture")
	return Readings{}
}

// A GB300 is sized by its own HBM, with the Grace CPU's memory offered only
// for offloaded experts.
func TestAGraceGPUIsSizedByItsOwnMemory(t *testing.T) {
	hw := FromReadings(stationReadings(t))
	if hw.Unified {
		t.Fatal("a GB300 was read as unified memory")
	}
	if pool, kind := hw.Pool(); kind != "vram" || pool != hw.GPUMemBytes {
		t.Errorf("Pool() = %d bytes of %s, want the GPU's %d bytes of vram", pool, kind, hw.GPUMemBytes)
	}
	if hw.BudgetBytes <= 200e9 || hw.BudgetBytes >= hw.GPUMemBytes {
		t.Errorf("budget %d GB for a %d GB GPU", hw.BudgetGB, hw.GPUMemGB)
	}
	if !strings.Contains(hw.BudgetReason, "first of 2 GPUs") || strings.Contains(hw.BudgetReason, "--tensor-parallel-size") {
		t.Errorf("reason %q should name the first of two different GPUs, not tensor parallelism", hw.BudgetReason)
	}
	want := hw.TotalMemBytes - int64(defaultBaseline(float64(hw.TotalMemBytes))) - offloadHostReserve
	if hw.OffloadBytes != want {
		t.Errorf("offload pool %d bytes, want system memory less the OS and the server: %d", hw.OffloadBytes, want)
	}
	if plan := hw.PlanFor(200e9, 100_000, 131072, 0); plan.KVCacheBytes <= 0 || float64(plan.KVCacheBytes)+200e9 > float64(hw.GPUMemBytes) {
		t.Errorf("planned %d bytes of KV cache beside 200 GB of weights on a %d GB GPU", plan.KVCacheBytes, hw.GPUMemGB)
	}
}

// Offloading is for a GPU that reads CPU memory coherently. A PCIe card or a
// unified-memory machine has nowhere faster to put the weights.
func TestOnlyACoherentGPUOffloads(t *testing.T) {
	for _, f := range loadFixtures(t) {
		hw := FromReadings(f.Readings)
		coherent := hw.Memory == string(hardware.MemoryCoherent)
		if got := hw.OffloadBytes > 0; got != coherent {
			t.Errorf("%s: offload %d bytes with %q memory", f.name, hw.OffloadBytes, hw.Memory)
		}
	}
	hidden := Hardware{Memory: string(hardware.MemoryCoherent), CUDA: true, GPUMemBytes: bi(252), TotalMemBytes: bi(480), GPUsHidden: 1}
	if got := hidden.offloadPool(0); got != 0 {
		t.Errorf("offload pool %d bytes with every GPU hidden", got)
	}
}

func TestOffloadForCoversTheShortfall(t *testing.T) {
	t.Parallel()
	hw := Hardware{BudgetBytes: 240e9, OffloadBytes: 450e9}
	tests := []struct {
		name    string
		weights int64
		ok      bool
	}{
		{"fits on the GPU", 200e9, false},
		{"fits with offload", 395e9, true},
		{"too big even with offload", 700e9, false},
	}
	for _, tt := range tests {
		gib, ok := hw.OffloadFor(tt.weights)
		if ok != tt.ok {
			t.Errorf("%s: OffloadFor(%d) ok = %v, want %v", tt.name, tt.weights, ok, tt.ok)
			continue
		}
		if !ok {
			continue
		}
		short := tt.weights - hw.BudgetBytes
		if moved := int64(gib) * gibibyte; moved < short || moved > short+offloadMargin+gibibyte {
			t.Errorf("%s: %d GiB offloaded for a %d byte shortfall", tt.name, gib, short)
		}
	}
	if gib, ok := (Hardware{BudgetBytes: 240e9, OffloadBytes: 160e9}).OffloadFor(398e9); !ok || int64(gib)*gibibyte > 160e9 {
		t.Errorf("OffloadFor beyond the pool's margin = %d GiB, %v; want capped at the pool", gib, ok)
	}
}

func TestTheGPUPinNamesTheBiggestModelByUUID(t *testing.T) {
	t.Parallel()
	gb300 := GPU{Name: "NVIDIA GB300", MemBytes: 269e9, UUID: "GPU-aaaa"}
	rtx := GPU{Name: "NVIDIA RTX PRO 2000 Blackwell", MemBytes: 17e9, UUID: "GPU-bbbb"}
	h100 := GPU{Name: "NVIDIA H100", MemBytes: 85e9, UUID: "GPU-cccc"}
	h100b := GPU{Name: "NVIDIA H100", MemBytes: 85e9, UUID: "GPU-dddd"}
	tests := []struct {
		name string
		gpus []GPU
		want string
	}{
		{"one GPU", []GPU{gb300}, ""},
		{"all one model", []GPU{h100, h100b}, ""},
		{"display GPU listed first", []GPU{rtx, gb300}, "GPU-aaaa"},
		{"every GPU of the biggest model", []GPU{rtx, h100, h100b}, "GPU-cccc,GPU-dddd"},
		{"no UUID to name it by", []GPU{rtx, {Name: "NVIDIA GB300", MemBytes: 269e9}}, ""},
	}
	for _, tt := range tests {
		if got := gpuPin(tt.gpus); got != tt.want {
			t.Errorf("%s: gpuPin = %q, want %q", tt.name, got, tt.want)
		}
	}
}
