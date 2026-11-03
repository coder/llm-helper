package detect

import (
	"fmt"
	"strings"

	"github.com/coder/vllm-helper/internal/hardware"
)

// What decided whether a GPU's memory is shared with the CPU, strongest
// first. Recorded so doctor, and anyone reading a report, can tell a fact
// from an inference.
const (
	// MemoryFromDefinition is a hardware definition naming the GPU.
	MemoryFromDefinition = "definition"
	// MemoryFromTegra is the Jetson platform's own marker files, which
	// vLLM reads the same way.
	MemoryFromTegra = "tegra"
	// MemoryFromAddressing is a GPU with no memory of its own that reaches
	// system memory through ATS, which is what a GB10 reports.
	MemoryFromAddressing = "addressing"
	// MemoryFromSize is a GPU reporting nearly all of system memory as its
	// own, and no addressing mode that says otherwise.
	MemoryFromSize = "size"
)

// describeGPUs names the GPU a server would load onto and decides how its
// memory relates to the CPU's. NVIDIA GPUs come from nvidia-smi; a Jetson
// whose nvidia-smi lists nothing is named from its device tree; a Mac's GPU
// is its SoC.
func (f facts) describeGPUs(hw *Hardware) {
	reg := hardware.Default()
	if f.smiOK {
		all := make([]GPU, len(f.gpus))
		copy(all, f.gpus)
		for i := range all {
			classifyGPU(reg, &all[i], f.totalMem, f.tegra)
		}
		hw.GPUs = visibleGPUs(all, f.visible, f.visibleSet)
		hw.GPUsHidden = len(all) - len(hw.GPUs)
	}

	switch {
	case len(hw.GPUs) > 0:
		g := hw.GPUs[0]
		hw.GPUName, hw.GPUVendor = g.Name, "nvidia"
		hw.GPUMemBytes = g.MemBytes
		hw.HardwareID, hw.Memory, hw.MemorySource, hw.Arch = g.HardwareID, g.Memory, g.MemorySource, g.Arch
	case f.tegra && hw.GPUsHidden == 0:
		name := f.tegraModel
		if name == "" {
			name = "NVIDIA Tegra"
		}
		g := GPU{Name: name}
		classifyGPU(reg, &g, f.totalMem, true)
		hw.GPUName, hw.GPUVendor = name, "nvidia"
		hw.HardwareID, hw.Memory, hw.MemorySource, hw.Arch = g.HardwareID, g.Memory, g.MemorySource, g.Arch
	case f.cpuBrand != "":
		// The platform probe only records an Apple SoC's brand string, so
		// an Intel Mac's CPU is never mistaken for a GPU here.
		hw.GPUName, hw.GPUVendor = f.cpuBrand, "apple"
		if d, ok := reg.Match(hardware.Observed{Vendor: "apple", GPUName: f.cpuBrand}); ok {
			hw.HardwareID, hw.Memory, hw.MemorySource, hw.Arch = d.ID, string(d.Memory), MemoryFromDefinition, d.Arch
		}
	}
	hw.GPUMemGB = decimalGB(hw.GPUMemBytes)
	hw.Unified = hw.Memory == string(hardware.MemoryUnified) || hw.Memory == string(hardware.MemoryCoherent)
	if d, ok := reg.ByID(hw.HardwareID); ok {
		hw.HardwareName = d.DisplayName
	}
}

// classifyGPU decides one NVIDIA GPU's memory kind and architecture.
//
// A definition that recognises the GPU decides its memory: naming machines
// beats inferring, and a definition can say discrete as well as unified.
// Otherwise only positive evidence counts. A GPU with no memory of its own
// that addresses system memory through ATS shares it; a GPU reporting nearly
// all of system memory shares it too, unless its addressing mode says it is
// a discrete card. Anything else is left discrete or unknown: guessing
// unified would hand the 0.85 cap to a machine whose real need is the CPU
// backend's live clamp, which is the regression issue #13 describes.
//
// The architecture comes from the compute capability the GPU reports when
// one is defined for it, since that is measured, and from the definition
// otherwise.
func classifyGPU(reg *hardware.Registry, g *GPU, totalMem int64, tegra bool) {
	gpuGB, totalGB := decimalGB(g.MemBytes), decimalGB(totalMem)
	if d, ok := reg.Match(hardware.Observed{Vendor: "nvidia", GPUName: g.Name}); ok {
		g.HardwareID, g.Memory, g.MemorySource, g.Arch = d.ID, string(d.Memory), MemoryFromDefinition, d.Arch
	} else {
		switch {
		case tegra:
			g.Memory, g.MemorySource = string(hardware.MemoryUnified), MemoryFromTegra
		case g.MemBytes == 0 && g.AddressingMode == "ATS":
			g.Memory, g.MemorySource = string(hardware.MemoryUnified), MemoryFromAddressing
		// A pool the size of system memory is a shared pool -- but only
		// when it does not exceed it, since an 80 GB card in a 64 GB host
		// is plainly its own memory, and only when the GPU does not say it
		// is discrete: a 96 GB card in a 96 GB host is not a share of it.
		case gpuGB > 0 && totalGB > 0 && gpuGB <= totalGB && float64(gpuGB) > 0.9*float64(totalGB) &&
			(g.AddressingMode == "" || g.AddressingMode == "ATS"):
			g.Memory, g.MemorySource = string(hardware.MemoryUnified), MemoryFromSize
		case g.MemBytes > 0:
			g.Memory = string(hardware.MemoryDiscrete)
		}
	}
	if a, ok := reg.ArchitectureByComputeCap(g.ComputeCap); ok {
		g.Arch = a.Arch
	}
}

// PCIDevice is one device from /sys/bus/pci/devices, as raw as sysfs gives
// it. Bus addresses are left out: they identify a slot, not a machine kind.
type PCIDevice struct {
	Vendor string `json:"vendor"` // e.g. "0x10de"
	Device string `json:"device"` // e.g. "0x2e12"
	Class  string `json:"class"`  // e.g. "0x030200"
	Driver string `json:"driver,omitempty"`
	// VRAMBytes is the memory amdgpu reports for the card, 0 elsewhere.
	VRAMBytes int64 `json:"vram_bytes,omitempty"`
}

// Accelerator is a GPU or compute accelerator found on the PCI bus.
type Accelerator struct {
	Vendor     string `json:"vendor,omitempty"` // vendor id, "" when no definition names the maker
	VendorName string `json:"vendor_name"`
	PCIID      string `json:"pci_id"` // vendor:device, e.g. "0x1002:0x744c"
	// Kind is "gpu" for a display controller and "accelerator" for a
	// processing accelerator, such as a datacenter card with no display.
	Kind      string `json:"kind"`
	Driver    string `json:"driver,omitempty"`
	VRAMBytes int64  `json:"vram_bytes,omitempty"`
}

// Describe names the accelerator in one line, for doctor.
func (a Accelerator) Describe() string {
	parts := []string{a.PCIID}
	if a.Driver != "" {
		parts = append(parts, "driver "+a.Driver)
	} else {
		parts = append(parts, "no driver")
	}
	if a.VRAMBytes > 0 {
		parts = append(parts, fmt.Sprintf("%d GB", decimalGB(a.VRAMBytes)))
	}
	return fmt.Sprintf("%s %s (%s)", a.VendorName, a.Kind, strings.Join(parts, ", "))
}

// accelerators picks the GPUs and accelerators out of a PCI listing. A
// display controller counts only from a vendor the definitions name, since
// a server's BMC shows up as one too; a processing accelerator counts from
// any vendor, since that is all the class means.
func accelerators(pci []PCIDevice) []Accelerator {
	reg := hardware.Default()
	var out []Accelerator
	for _, d := range pci {
		class := strings.ToLower(d.Class)
		v, known := reg.VendorByPCI(d.Vendor)
		var kind string
		switch {
		case strings.HasPrefix(class, "0x03") && known:
			kind = "gpu"
		case strings.HasPrefix(class, "0x12"):
			kind = "accelerator"
		default:
			continue
		}
		a := Accelerator{Vendor: v.ID, VendorName: v.Name, PCIID: d.Vendor + ":" + d.Device, Kind: kind, Driver: d.Driver, VRAMBytes: d.VRAMBytes}
		if !known {
			a.VendorName = "unknown vendor " + d.Vendor
		}
		out = append(out, a)
	}
	return out
}
