package detect

import (
	"github.com/coder/llm-helper/internal/hardware"
)

// Report is what 'hardware report' prints and what a hardware fixture holds:
// a machine's raw readings, and what llm-helper concluded from them. Saving
// one beside a new definition is how that definition is tested against the
// machine it describes.
//
// Nothing in it identifies a particular host: no hostname, serial number,
// GPU UUID or bus address, and no memory in use, which changes by the second.
type Report struct {
	// Provenance says where the readings came from: the llm-helper that took
	// them, or, for a fixture nobody has measured yet, how it was made.
	Provenance string   `json:"provenance"`
	Readings   Readings `json:"readings"`
	Detected   Detected `json:"detected"`
}

// Readings are a machine's raw hardware readings.
type Readings struct {
	SchemaVersion int    `json:"schema_version"`
	OS            string `json:"os"`
	Arch          string `json:"arch"`
	// System is the DMI vendor and product, or a Mac's model identifier,
	// for the person reading the report. Detection does not use it.
	System        string `json:"system,omitempty"`
	MemTotalBytes int64  `json:"mem_total_bytes"`
	MemLimitBytes int64  `json:"mem_limit_bytes,omitempty"`
	// CPUBrand is an Apple Silicon SoC's brand string, which names its GPU.
	CPUBrand     string `json:"cpu_brand,omitempty"`
	NVIDIADevice bool   `json:"nvidia_device,omitempty"`
	// NVIDIASMI is whether nvidia-smi answered, and GPUs what it listed.
	NVIDIASMI  bool        `json:"nvidia_smi"`
	GPUs       []ReportGPU `json:"gpus,omitempty"`
	PCI        []PCIDevice `json:"pci,omitempty"`
	Tegra      bool        `json:"tegra,omitempty"`
	TegraModel string      `json:"tegra_model,omitempty"`
}

// ReportGPU is one GPU as nvidia-smi listed it, without its UUID.
type ReportGPU struct {
	Index          int    `json:"index"`
	Name           string `json:"name"`
	MemBytes       int64  `json:"mem_bytes"`
	ComputeCap     string `json:"compute_cap,omitempty"`
	AddressingMode string `json:"addressing_mode,omitempty"`
}

// Detected is what llm-helper concluded from a report's readings.
type Detected struct {
	HardwareID   string `json:"hardware_id,omitempty"`
	GPUName      string `json:"gpu_name,omitempty"`
	Memory       string `json:"memory,omitempty"`
	MemorySource string `json:"memory_source,omitempty"`
	Arch         string `json:"arch,omitempty"`
	CUDA         bool   `json:"cuda"`
	// Accelerators names every GPU or accelerator found on the PCI bus.
	Accelerators []string `json:"accelerators,omitempty"`
}

// ProbeReport takes this machine's readings as a report. Every GPU counts
// as visible: a report describes the machine, not one server's share of it.
func ProbeReport(provenance string) Report {
	return newReport(provenance, gatherFacts().readings())
}

func newReport(provenance string, r Readings) Report {
	return Report{Provenance: provenance, Readings: r, Detected: DetectedFrom(FromReadings(r))}
}

// FromReadings derives the hardware a set of readings describes, as Probe
// would on that machine with every GPU visible and no vllm-metal.
func FromReadings(r Readings) Hardware {
	return factsFrom(r).hardware(nil)
}

// DetectedFrom summarises a probe the way a report records it.
func DetectedFrom(hw Hardware) Detected {
	d := Detected{HardwareID: hw.HardwareID, GPUName: hw.GPUName, Memory: hw.Memory, MemorySource: hw.MemorySource, Arch: hw.Arch, CUDA: hw.CUDA}
	for _, a := range hw.Accelerators {
		d.Accelerators = append(d.Accelerators, a.Describe())
	}
	return d
}

func (f facts) readings() Readings {
	r := Readings{
		SchemaVersion: hardware.SchemaVersion,
		OS:            f.os, Arch: f.arch, System: f.system,
		MemTotalBytes: f.totalMem, MemLimitBytes: f.memLimit,
		CPUBrand:     f.cpuBrand,
		NVIDIADevice: f.nvidiaDevice, NVIDIASMI: f.smiOK,
		PCI:   f.pci,
		Tegra: f.tegra, TegraModel: f.tegraModel,
	}
	for _, g := range f.gpus {
		r.GPUs = append(r.GPUs, ReportGPU{Index: g.Index, Name: g.Name, MemBytes: g.MemBytes, ComputeCap: g.ComputeCap, AddressingMode: g.AddressingMode})
	}
	return r
}

func factsFrom(r Readings) facts {
	f := facts{
		os: r.OS, arch: r.Arch, system: r.System,
		totalMem: r.MemTotalBytes, memLimit: r.MemLimitBytes,
		cpuBrand:     r.CPUBrand,
		nvidiaDevice: r.NVIDIADevice, smiOK: r.NVIDIASMI,
		pci:   r.PCI,
		tegra: r.Tegra, tegraModel: r.TegraModel,
	}
	for _, g := range r.GPUs {
		f.gpus = append(f.gpus, GPU{Index: g.Index, Name: g.Name, MemBytes: g.MemBytes, ComputeCap: g.ComputeCap, AddressingMode: g.AddressingMode})
	}
	return f
}
