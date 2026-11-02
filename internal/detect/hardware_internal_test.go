package detect

import (
	"testing"

	"github.com/coder/vllm-helper/internal/hardware"
)

// Which machines share one memory pool with the CPU. This decides whether
// --gpu-memory-utilization gets capped, so a wrong answer either leaves a
// Spark uncapped (it can wedge the host) or caps a discrete card for no reason.
func TestIsUnified(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		gpuName      string
		gpuMemGB     int
		totalMemGB   int
		nvidiaDevice bool
		want         bool
	}{
		// nvidia-smi reports "[N/A]" for memory.total on a GB10, so the name
		// is the only thing that identifies it.
		{name: "DGX Spark", gpuName: "NVIDIA GB10", totalMemGB: 121, want: true},
		{name: "GH200", gpuName: "NVIDIA GH200 480GB", totalMemGB: 480, want: true},
		{name: "Grace-Blackwell", gpuName: "NVIDIA GB200", totalMemGB: 384, want: true},
		{name: "Jetson", gpuName: "Orin (nvgpu)", totalMemGB: 62, want: true},
		{name: "Thor", gpuName: "NVIDIA Thor", totalMemGB: 128, want: true},
		{
			name: "a pool the size of system RAM",
			// Anything reporting nearly all of system memory as its own is
			// sharing it, whatever it calls itself.
			gpuName: "Unknown Integrated", gpuMemGB: 120, totalMemGB: 128, want: true,
		},
		{
			name:    "discrete card stays discrete",
			gpuName: "NVIDIA RTX A6000", gpuMemGB: 48, totalMemGB: 256, want: false,
		},
		{
			// A share of the host's memory cannot be bigger than the host's
			// memory, so this is a discrete card in a small box. Reading it
			// as unified would cap a request that has no reason to be capped.
			name:    "VRAM larger than system RAM is discrete",
			gpuName: "NVIDIA A100 80GB", gpuMemGB: 80, totalMemGB: 64, want: false,
		},
		{
			name:    "consumer Blackwell is discrete",
			gpuName: "NVIDIA GeForce RTX 5090", gpuMemGB: 32, totalMemGB: 128, want: false,
		},
		{name: "no GPU at all", totalMemGB: 15, want: false},
		{
			name: "no GPU and no device node", gpuName: "", totalMemGB: 15,
			nvidiaDevice: false, want: false,
		},
		{
			// A device node says a GPU is here, not that its memory is
			// shared. Claiming unified would swap the CPU backend's live
			// clamp for a static cap, which is the issue #13 regression.
			name:    "undescribed GPU is not evidence of unified memory",
			gpuName: "", totalMemGB: 121, nvidiaDevice: true, want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := isUnified(tt.gpuName, tt.gpuMemGB, tt.totalMemGB, tt.nvidiaDevice)
			if got != tt.want {
				t.Errorf("isUnified(%q, %d, %d, %v) = %v, want %v",
					tt.gpuName, tt.gpuMemGB, tt.totalMemGB, tt.nvidiaDevice, got, tt.want)
			}
		})
	}
}

// A descriptor that names a card as discrete outranks the size heuristic,
// which would otherwise read 96 GB of VRAM in a 96 GB host as a shared pool.
func TestUnifiedDescriptorOutranksTheSizeHeuristic(t *testing.T) {
	t.Parallel()
	reg, err := hardware.Parse([]byte(`{"architectures":[],"descriptors":[
		{"id":"big_card","display_name":"Big card","match":{"gpu_name":"Big Card"},"unified":false,"why":"discrete"}
	]}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if unifiedWith(reg, "ACME Big Card", 95, 96) {
		t.Error("a card its descriptor calls discrete was read as unified")
	}
	if !unifiedWith(reg, "ACME Other Card", 95, 96) {
		t.Error("an undescribed card the size of system RAM should still read as unified")
	}
}
