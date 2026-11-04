package detect

import (
	"testing"
	"testing/fstest"

	"github.com/coder/vllm-helper/internal/hardware"
)

// unifiedFor classifies one NVIDIA GPU the way the probe does.
func unifiedFor(name string, gpuMemGB, totalMemGB int, addressing string) (bool, GPU) {
	g := GPU{Name: name, MemBytes: int64(gpuMemGB) * 1e9, AddressingMode: addressing}
	classifyGPU(hardware.Default(), &g, int64(totalMemGB)*1e9, false)
	return g.Memory == string(hardware.MemoryUnified) || g.Memory == string(hardware.MemoryCoherent), g
}

// Which machines share one memory pool with the CPU. This decides whether
// --gpu-memory-utilization gets capped, so a wrong answer either leaves a
// Spark uncapped (it can wedge the host) or caps a discrete card for no reason.
func TestIsUnified(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		gpuName    string
		gpuMemGB   int
		totalMemGB int
		addressing string
		want       bool
		source     string
	}{
		// nvidia-smi reports "[N/A]" for memory.total on a GB10, so the name
		// is the only thing that identifies it.
		{name: "DGX Spark", gpuName: "NVIDIA GB10", totalMemGB: 121, want: true, source: MemoryFromDefinition},
		{name: "GH200", gpuName: "NVIDIA GH200 480GB", totalMemGB: 480, want: true, source: MemoryFromDefinition},
		{name: "Grace-Blackwell", gpuName: "NVIDIA GB200", totalMemGB: 384, want: true, source: MemoryFromDefinition},
		{name: "Jetson", gpuName: "Orin (nvgpu)", totalMemGB: 62, want: true, source: MemoryFromDefinition},
		{name: "Thor", gpuName: "NVIDIA Thor", totalMemGB: 128, want: true, source: MemoryFromDefinition},
		{
			name: "a pool the size of system RAM",
			// Anything reporting nearly all of system memory as its own is
			// sharing it, whatever it calls itself, unless it says otherwise.
			gpuName: "Unknown Integrated", gpuMemGB: 120, totalMemGB: 128, want: true, source: MemoryFromSize,
		},
		{
			// A GPU with no memory of its own that reaches system memory
			// through ATS is a GB10 by another name.
			name:    "no memory of its own, ATS",
			gpuName: "NVIDIA GB99", totalMemGB: 128, addressing: "ATS", want: true, source: MemoryFromAddressing,
		},
		{
			// A 96 GB card in a 96 GB host passes the size test, and its
			// addressing mode is what says it has memory of its own.
			name:    "a discrete card the size of the host, by its addressing mode",
			gpuName: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", gpuMemGB: 95, totalMemGB: 100, addressing: "HMM", want: false,
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
		{
			// ATS on a card with memory of its own is a discrete card that
			// can also reach the host's, not a shared pool.
			name:    "ATS with VRAM is not evidence of unified memory",
			gpuName: "NVIDIA H100 80GB HBM3", gpuMemGB: 80, totalMemGB: 512, addressing: "ATS", want: false,
		},
		{
			// A GPU nvidia-smi could not size, with no addressing mode to
			// go on, says nothing about its memory. Claiming unified would
			// swap the CPU backend's live clamp for a static cap, which is
			// the issue #13 regression.
			name:    "an undescribed GPU is not evidence of unified memory",
			gpuName: "NVIDIA Mystery", totalMemGB: 121, want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, g := unifiedFor(tt.gpuName, tt.gpuMemGB, tt.totalMemGB, tt.addressing)
			if got != tt.want || (tt.want && g.MemorySource != tt.source) {
				t.Errorf("%q (%d GB in %d GB, %q): unified = %v from %q, want %v from %q",
					tt.gpuName, tt.gpuMemGB, tt.totalMemGB, tt.addressing, got, g.MemorySource, tt.want, tt.source)
			}
		})
	}
}

// A definition that names a card as discrete outranks the size heuristic,
// which would otherwise read 96 GB of VRAM in a 96 GB host as a shared pool.
func TestADefinitionOutranksTheSizeHeuristic(t *testing.T) {
	t.Parallel()
	reg, err := hardware.Load(fstest.MapFS{"machines/big_card.json": {Data: []byte(
		`{"schema_version":1,"id":"big_card","display_name":"Big card","match":{"gpu_name":"Big Card"},"memory":"discrete","why":"discrete"}`,
	)}})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	big := GPU{Name: "ACME Big Card", MemBytes: 95e9}
	classifyGPU(reg, &big, 96e9, false)
	if big.Memory != string(hardware.MemoryDiscrete) || big.HardwareID != "big_card" {
		t.Errorf("a card its definition calls discrete: %+v", big)
	}
	other := GPU{Name: "ACME Other Card", MemBytes: 95e9}
	classifyGPU(reg, &other, 96e9, false)
	if other.Memory != string(hardware.MemoryUnified) || other.MemorySource != MemoryFromSize {
		t.Errorf("an undescribed card the size of system RAM: %+v", other)
	}
}

// The architecture is what forbidden flags are checked against, so a GPU no
// definition names still needs one, and a measured compute capability beats
// a definition's claim.
func TestTheArchitectureComesFromTheComputeCapability(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		name, cc, want string
	}{
		{"NVIDIA GB10", "12.1", "sm121"},
		{"NVIDIA GB10", "", "sm121"},
		{"NVIDIA Some Future GB10 Box", "12.1", "sm121"},
		{"NVIDIA GeForce RTX 5090", "12.0", "sm120"},
		{"NVIDIA H100 80GB HBM3", "9.0", "sm90"},
		{"NVIDIA Unknown", "", ""},
	} {
		g := GPU{Name: tt.name, ComputeCap: tt.cc}
		classifyGPU(hardware.Default(), &g, 128e9, false)
		if g.Arch != tt.want {
			t.Errorf("%q with compute_cap %q: arch %q, want %q", tt.name, tt.cc, g.Arch, tt.want)
		}
	}
}
