package detect

import (
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// hasNVIDIADevice reports whether this machine has a GPU we could actually
// serve from, for the case where nvidia-smi told us nothing.
//
// The control device is the signal rather than /proc/driver/nvidia/version,
// because the version file outlives a usable GPU: it is still there after a
// driver upgrade without a reboot (nvidia-smi then fails on a library version
// mismatch), and it is visible inside a container started without --gpus. Both
// are ordinary CPU-backend situations, and treating them as GPU boxes would
// skip the clamp that issue #13 needs.
func hasNVIDIADevice() bool {
	_, err := os.Stat("/dev/nvidiactl")
	return err == nil
}

// Hardware describes what we can serve models with on this machine.
//
// Gigabytes here are decimal, matching how the Hub reports model sizes and how
// a machine's memory is advertised: a "128 GB" DGX Spark reads as 130 GB total
// because 2 GB of its 128 GiB is carved off before Linux sees it. Comparing a
// decimal model size against a binary budget is a silent 7.4% error in the
// pessimistic direction, so the two must agree.
type Hardware struct {
	TotalMemBytes int64  `json:"total_mem_bytes"` // system RAM, exact
	TotalMemGB    int    `json:"total_mem_gb"`    // system RAM, for display
	GPUName       string `json:"gpu_name"`        // e.g. "NVIDIA GB10", "" if no GPU we can serve on is visible
	GPUMemBytes   int64  `json:"gpu_mem_bytes"`   // VRAM reported by nvidia-smi; 0 if unknown/none
	GPUMemGB      int    `json:"gpu_mem_gb"`      // VRAM, for display
	// GPUVendor is the vendor id of the GPU GPUName names, such as "nvidia"
	// or "apple".
	GPUVendor string `json:"gpu_vendor,omitempty"`
	// GPUs is every NVIDIA GPU the server can see, in CUDA order. GPUName
	// and GPUMemBytes describe the first, because that is the one a server
	// without --tensor-parallel-size loads onto.
	GPUs []GPU `json:"gpus,omitempty"`
	// GPUsHidden counts GPUs nvidia-smi listed that CUDA_VISIBLE_DEVICES
	// keeps from the server.
	GPUsHidden int `json:"gpus_hidden,omitempty"`
	// HardwareID is the hardware definition that recognised the GPU, such
	// as "dgx_spark_gb10", and HardwareName its display name. Both are ""
	// when no definition did.
	HardwareID   string `json:"hardware_id,omitempty"`
	HardwareName string `json:"hardware_name,omitempty"`
	// Memory is how the GPU's memory relates to the CPU's: unified,
	// discrete or coherent, or "" when there is no GPU or nothing says.
	Memory string `json:"memory,omitempty"`
	// MemorySource is what decided Memory; see the MemoryFrom constants.
	MemorySource string `json:"memory_source,omitempty"`
	// Arch is the GPU architecture, such as "sm121", from its compute
	// capability or its definition. Flags known to abort there are refused.
	Arch    string `json:"arch,omitempty"`
	Unified bool   `json:"unified"` // GPU shares system memory (DGX Spark / GB10 / Grace)
	// CUDA reports an NVIDIA GPU vLLM could actually use. Without one it
	// runs its CPU backend, which holds no CUDA context and captures no
	// graphs, so it is budgeted differently.
	CUDA bool `json:"cuda"`
	// Accelerators is every GPU and accelerator on the PCI bus, whether or
	// not a driver llm-helper can size answers for it. It is how doctor can
	// say an AMD card is present but the machine is sized for the CPU.
	Accelerators []Accelerator `json:"accelerators,omitempty"`
	// MemLimitBytes is the cgroup memory limit this process runs under, 0
	// when there is none below system memory. It caps the system-memory pool.
	MemLimitBytes int64 `json:"mem_limit_bytes,omitempty"`
	// Metal reports that the configured vLLM has the vllm-metal plugin, so
	// it serves on the Apple Silicon GPU rather than the CPU backend.
	Metal bool `json:"metal"`
	// MetalWorkingSetBytes is the most unified memory macOS lets the GPU
	// wire, as MLX reports it. vllm-metal reads --gpu-memory-utilization as
	// a share of this, not of installed RAM. 0 when it could not be read.
	MetalWorkingSetBytes int64 `json:"metal_working_set_bytes"`
	MetalWorkingSetGB    int   `json:"metal_working_set_gb"`
	// MetalFixedKVCache is whether this vllm-metal honors
	// --kv-cache-memory-bytes. Without it, the cache takes what
	// --gpu-memory-utilization leaves, and a planned size would not hold.
	MetalFixedKVCache bool    `json:"metal_fixed_kv_cache"`
	BaselineGB        float64 `json:"baseline_gb"` // what the OS is holding; 0 when it could not be measured
	// InUseBytes is system memory in use when the probe ran, whatever is
	// holding it. Unlike BaselineGB it is reported however large it is.
	InUseBytes   int64  `json:"in_use_bytes,omitempty"`
	BudgetBytes  int64  `json:"budget_bytes"`  // recommended max model weights
	BudgetGB     int    `json:"budget_gb"`     // the same, for display
	BudgetReason string `json:"budget_reason"` // one-liner explaining the budget for the user
}

// Pool reports the memory weights are actually loaded into, and which pool
// that is. A discrete GPU uses its own VRAM; everything else uses system
// memory, as much of it as this process's cgroup allows. One definition,
// because a caller drawing installed RAM under a budget measured against
// VRAM is comparing two different machines.
func (h Hardware) Pool() (int64, string) {
	if h.Metal && h.MetalWorkingSetBytes > 0 {
		return h.MetalWorkingSetBytes, "metal"
	}
	if !h.Unified && h.GPUMemBytes > 0 {
		return h.GPUMemBytes, "vram"
	}
	if h.MemLimited() {
		return h.MemLimitBytes, "ram"
	}
	return h.TotalMemBytes, "ram"
}

// MemLimited reports that a cgroup limit, not installed RAM, bounds the
// system-memory pool.
func (h Hardware) MemLimited() bool {
	return h.MemLimitBytes > 0 && h.MemLimitBytes < h.TotalMemBytes
}

// GPUSummary names the accelerator in one line: "NVIDIA GB10", "4x NVIDIA
// H100 80GB HBM3", or each name when the GPUs differ. "" when there is none.
func (h Hardware) GPUSummary() string {
	if len(h.GPUs) <= 1 {
		return h.GPUName
	}
	names := make([]string, len(h.GPUs))
	same := true
	for i, g := range h.GPUs {
		names[i] = g.Name
		same = same && g.Name == h.GPUs[0].Name
	}
	if same {
		return strconv.Itoa(len(h.GPUs)) + "x " + names[0]
	}
	return strings.Join(names, " + ")
}

// TotalMemMB reports installed RAM in mebibytes, the unit the CPU-backend
// clamp works in because that is how vLLM's own memory readings arrive.
func (h Hardware) TotalMemMB() int { return int(h.TotalMemBytes / (1 << 20)) }

// decimalGB rounds bytes down to whole decimal gigabytes for display.
func decimalGB(b int64) int { return int(float64(b) / gigabyte) }

// GPU is one NVIDIA GPU as nvidia-smi lists it, and what llm-helper made of
// it.
type GPU struct {
	Index    int    `json:"index"`
	UUID     string `json:"uuid,omitempty"`
	Name     string `json:"name"`
	MemBytes int64  `json:"mem_bytes"` // 0 when nvidia-smi says [N/A], as it does on a GB10
	// ComputeCap is the CUDA compute capability, such as "12.1", and
	// AddressingMode how the GPU reaches system memory: "ATS" on a GPU
	// that shares it, "HMM" or "None" on a discrete card. Both are "" when
	// the driver is too old to say.
	ComputeCap     string `json:"compute_cap,omitempty"`
	AddressingMode string `json:"addressing_mode,omitempty"`
	HardwareID     string `json:"hardware_id,omitempty"`
	Memory         string `json:"memory,omitempty"`
	MemorySource   string `json:"memory_source,omitempty"`
	Arch           string `json:"arch,omitempty"`
}

// nvidiaSMIArgs asks for every GPU, one CSV row each. The UUID is there
// because CUDA_VISIBLE_DEVICES may name GPUs by it.
var nvidiaSMIArgs = []string{"--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"}

// nvidiaSMICapArgs asks what the first query cannot, in a query of its own:
// nvidia-smi fails the whole query on a field it does not know, so asking
// for these alongside the name would lose every GPU on an older driver.
// nvidiaSMICapFallback is for a driver that knows compute_cap but not
// addressing_mode.
var (
	nvidiaSMICapArgs     = []string{"--query-gpu=index,compute_cap,addressing_mode", "--format=csv,noheader,nounits"}
	nvidiaSMICapFallback = []string{"--query-gpu=index,compute_cap", "--format=csv,noheader,nounits"}
)

// smiCommand and smiTimeout are variables so a test can stand in a hung
// nvidia-smi.
var (
	smiCommand = "nvidia-smi"
	// smiTimeout bounds one call. A wedged driver makes nvidia-smi hang
	// rather than fail, and every command that sizes memory would hang
	// with it. It answers in tens of milliseconds when healthy.
	smiTimeout = 10 * time.Second
)

// runNVIDIASMI runs nvidia-smi, reporting false when it is missing, fails
// or does not answer in time.
func runNVIDIASMI(args ...string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), smiTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, smiCommand, args...)
	// A child that inherited the output pipe would otherwise keep Output
	// waiting after the process itself was killed.
	cmd.WaitDelay = time.Second
	out, err := cmd.Output()
	if err != nil {
		return "", false
	}
	return string(out), true
}

// parseNVIDIASMI reads nvidia-smi's rows. A row that does not parse is
// skipped rather than guessed at: one unreadable GPU must not hide the rest.
func parseNVIDIASMI(out string) []GPU {
	var gpus []GPU
	for line := range strings.SplitSeq(strings.TrimSpace(out), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) != 4 {
			continue
		}
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
		}
		idx, err := strconv.Atoi(parts[0])
		if err != nil || parts[2] == "" {
			continue
		}
		g := GPU{Index: idx, UUID: parts[1], Name: parts[2]}
		if mb, err := strconv.ParseInt(parts[3], 10, 64); err == nil && mb > 0 {
			g.MemBytes = mb << 20
		}
		gpus = append(gpus, g)
	}
	return gpus
}

// applyGPUCaps reads the capability query's rows (index, compute_cap and
// optionally addressing_mode) onto the GPUs they name. "[N/A]" is kept as
// unknown.
func applyGPUCaps(gpus []GPU, out string) {
	for line := range strings.SplitSeq(strings.TrimSpace(out), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
			if notAvailable(parts[i]) {
				parts[i] = ""
			}
		}
		idx, err := strconv.Atoi(parts[0])
		if err != nil {
			continue
		}
		for i := range gpus {
			if gpus[i].Index != idx {
				continue
			}
			gpus[i].ComputeCap = parts[1]
			if len(parts) > 2 {
				gpus[i].AddressingMode = parts[2]
			}
		}
	}
}

func notAvailable(s string) bool {
	return s == "[N/A]" || s == "N/A" || s == "[Not Supported]"
}

// readNVIDIA asks nvidia-smi for every GPU and what it can say about each.
func readNVIDIA() (gpus []GPU, ok bool) {
	out, ok := runNVIDIASMI(nvidiaSMIArgs...)
	if !ok {
		return nil, false
	}
	gpus = parseNVIDIASMI(out)
	if len(gpus) == 0 {
		return gpus, true
	}
	if caps, ok := runNVIDIASMI(nvidiaSMICapArgs...); ok {
		applyGPUCaps(gpus, caps)
	} else if caps, ok := runNVIDIASMI(nvidiaSMICapFallback...); ok {
		applyGPUCaps(gpus, caps)
	}
	return gpus, true
}

// visibleGPUs applies CUDA_VISIBLE_DEVICES the way the CUDA runtime reads it:
// a comma-separated list of indices or (prefixes of) UUIDs, in the order the
// process will number them, ending at the first entry that names no device.
// That makes "" and "-1" hide every GPU, which is how both are used.
//
// MIG instances are treated as an entry that names no device: nvidia-smi's
// GPU query does not list them, so their memory is unknown, and reporting
// the parent's would overstate it.
//
// Indices are matched against nvidia-smi's, which are in PCI bus order. CUDA
// numbers fastest-first unless CUDA_DEVICE_ORDER=PCI_BUS_ID, so on a box
// mixing GPU models an index can name a different card than it does here.
func visibleGPUs(all []GPU, value string, set bool) []GPU {
	if !set {
		return all
	}
	var out []GPU
	seen := map[int]bool{}
	for entry := range strings.SplitSeq(value, ",") {
		g, ok := lookupGPU(all, strings.TrimSpace(entry))
		if !ok || seen[g.Index] {
			break
		}
		seen[g.Index] = true
		out = append(out, g)
	}
	return out
}

func lookupGPU(all []GPU, entry string) (GPU, bool) {
	if entry == "" || strings.HasPrefix(entry, "MIG-") {
		return GPU{}, false
	}
	if n, err := strconv.Atoi(entry); err == nil {
		for _, g := range all {
			if g.Index == n {
				return g, true
			}
		}
		return GPU{}, false
	}
	// A UUID prefix counts only when it names exactly one GPU.
	var match []GPU
	for _, g := range all {
		if g.UUID != "" && strings.HasPrefix(g.UUID, entry) {
			match = append(match, g)
		}
	}
	if len(match) != 1 {
		return GPU{}, false
	}
	return match[0], true
}

var (
	visibleMu  sync.Mutex
	visibleVal string
	visibleSet bool
	visibleReg bool
)

// UseCUDAVisibleDevices records the CUDA_VISIBLE_DEVICES the server is started
// with, which comes from config.env or the profile as often as from this
// process. set false means the server gets none. Until this is called, Probe
// reads this process's environment.
func UseCUDAVisibleDevices(value string, set bool) {
	visibleMu.Lock()
	defer visibleMu.Unlock()
	visibleVal, visibleSet, visibleReg = value, set, true
}

func cudaVisibleDevices() (string, bool) {
	visibleMu.Lock()
	defer visibleMu.Unlock()
	if visibleReg {
		return visibleVal, visibleSet
	}
	return os.LookupEnv("CUDA_VISIBLE_DEVICES")
}

// facts are the raw readings Hardware is derived from. They are gathered
// apart from the derivation so another machine's readings, collected over
// SSH or saved as a report, produce the same answer that machine would.
type facts struct {
	os, arch           string
	system             string // DMI vendor and product, or a Mac's model
	totalMem, availMem int64
	memLimit           int64 // cgroup memory.max below system memory, else 0
	cpuBrand           string
	nvidiaDevice       bool
	smiOK              bool
	gpus               []GPU // as nvidia-smi listed them; meaningful only with smiOK
	pci                []PCIDevice
	tegra              bool
	tegraModel         string
	visible            string
	visibleSet         bool
}

// gatherFacts takes this machine's readings. Never fails -- missing pieces
// just degrade the result.
func gatherFacts() facts {
	f := facts{totalMem: totalMemBytes(), availMem: int64(AvailableMemMB()) << 20, nvidiaDevice: hasNVIDIADevice()}
	f.gpus, f.smiOK = readNVIDIA()
	platformFacts(&f)
	return f
}

// Probe inspects system memory and the machine's GPUs. Never fails --
// missing pieces just degrade the result.
func Probe() Hardware {
	f := gatherFacts()
	f.visible, f.visibleSet = cudaVisibleDevices()
	return f.hardware(currentMetal())
}

// hardware derives what this machine can serve from its readings. metal asks
// about vllm-metal, which only applies to the machine running this process,
// so a remote or saved reading passes nil.
func (f facts) hardware(metal *metalState) Hardware {
	hw := Hardware{TotalMemBytes: f.totalMem, MemLimitBytes: f.memLimit}
	hw.TotalMemGB = decimalGB(hw.TotalMemBytes)
	if !hw.MemLimited() {
		hw.MemLimitBytes = 0
	}

	// vLLM on a Jetson is a CUDA platform whether or not its nvidia-smi
	// lists the GPU.
	hw.CUDA = f.smiOK || f.nvidiaDevice || f.tegra
	f.describeGPUs(&hw)
	hw.Accelerators = accelerators(f.pci)

	// CUDA first: vllm-metal cannot serve on an NVIDIA machine, so there is
	// no reason to start an interpreter to ask.
	if !hw.CUDA {
		if m := metal; m.runtime() {
			hw.Metal = true
			mem := m.memoryInfo()
			hw.MetalFixedKVCache = mem.FixedKVCache
			if ws := mem.WorkingSetBytes; ws > 0 {
				hw.MetalWorkingSetBytes = ws
				hw.MetalWorkingSetGB = decimalGB(ws)
			}
		}
	}

	// Memory in use right now, which Budget takes as the OS baseline when it
	// looks like an idle reading and ignores when it does not.
	var used int64
	if avail := f.availMem; avail > 0 && hw.TotalMemBytes > avail {
		used = hw.TotalMemBytes - avail
		hw.InUseBytes = used
		if float64(used) < baselineCeiling*float64(hw.TotalMemBytes) {
			hw.BaselineGB = float64(used) / gigabyte
		}
	}

	hw.BudgetBytes, hw.BudgetReason = hw.Budget(used)
	hw.BudgetGB = decimalGB(hw.BudgetBytes)
	return hw
}
