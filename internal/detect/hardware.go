package detect

import (
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
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
	GPUName       string `json:"gpu_name"`        // e.g. "NVIDIA GB10", "" if no NVIDIA GPU visible
	GPUMemBytes   int64  `json:"gpu_mem_bytes"`   // VRAM reported by nvidia-smi; 0 if unknown/none
	GPUMemGB      int    `json:"gpu_mem_gb"`      // VRAM, for display
	// GPUs is every NVIDIA GPU the server can see, in CUDA order. GPUName
	// and GPUMemBytes describe the first, because that is the one a server
	// without --tensor-parallel-size loads onto.
	GPUs []GPU `json:"gpus,omitempty"`
	// GPUsHidden counts GPUs nvidia-smi listed that CUDA_VISIBLE_DEVICES
	// keeps from the server.
	GPUsHidden int  `json:"gpus_hidden,omitempty"`
	Unified    bool `json:"unified"` // GPU shares system memory (DGX Spark / GB10 / Grace)
	// CUDA reports an NVIDIA GPU vLLM could actually use. Without one it
	// runs its CPU backend, which holds no CUDA context and captures no
	// graphs, so it is budgeted differently.
	CUDA bool `json:"cuda"`
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
	BaselineGB        float64 `json:"baseline_gb"`   // what the OS is holding; 0 when it could not be measured
	BudgetBytes       int64   `json:"budget_bytes"`  // recommended max model weights
	BudgetGB          int     `json:"budget_gb"`     // the same, for display
	BudgetReason      string  `json:"budget_reason"` // one-liner explaining the budget for the user
}

// Pool reports the memory weights are actually loaded into, and which pool
// that is. A discrete GPU uses its own VRAM; everything else uses system
// memory. One definition, because a caller drawing installed RAM under a
// budget measured against VRAM is comparing two different machines.
func (h Hardware) Pool() (int64, string) {
	if h.Metal && h.MetalWorkingSetBytes > 0 {
		return h.MetalWorkingSetBytes, "metal"
	}
	if !h.Unified && h.GPUMemBytes > 0 {
		return h.GPUMemBytes, "vram"
	}
	return h.TotalMemBytes, "ram"
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

// unifiedNames are GPUs whose memory is the system's memory: Grace-Blackwell
// (GB10 in the DGX Spark, GB200, GB300), Grace-Hopper, and the integrated
// Jetson/Thor parts. Naming them individually beats inferring it, because
// nvidia-smi reports "[N/A]" for memory.total on GB10 and so tells us nothing.
var unifiedNames = regexp.MustCompile(`(?i)GB10|GB200|GB300|GH200|Grace|Orin|Thor|Tegra`)

// isUnified decides whether the GPU shares one memory pool with the CPU. It is
// the pure half of Probe, so the cases below can be tested without the machine.
//
// Only positive evidence counts. A GPU we could not describe is left alone: an
// NVIDIA device node says a GPU exists but nothing about its memory, and there
// is no cheap signal that would say more. Architecture looked like one, but a
// GB10 presents as an ordinary PCI display controller, so it cannot separate
// integrated from discrete on arm64 either. Guessing unified there would hand
// the 0.85 cap to a machine whose real need is the CPU backend's live clamp,
// which is the regression issue #13 describes.
func isUnified(gpuName string, gpuMemGB, totalMemGB int, nvidiaDevice bool) bool {
	if gpuName != "" && unifiedNames.MatchString(gpuName) {
		return true
	}
	// A pool the size of system memory is a shared pool -- but only when it
	// does not exceed it, since an 80 GB card in a 64 GB host is plainly
	// discrete memory rather than a share of the host's.
	return gpuMemGB > 0 && totalMemGB > 0 &&
		gpuMemGB <= totalMemGB &&
		float64(gpuMemGB) > 0.9*float64(totalMemGB)
}

// GPU is one NVIDIA GPU as nvidia-smi lists it.
type GPU struct {
	Index    int    `json:"index"`
	UUID     string `json:"uuid,omitempty"`
	Name     string `json:"name"`
	MemBytes int64  `json:"mem_bytes"` // 0 when nvidia-smi says [N/A], as it does on a GB10
}

// nvidiaSMIArgs asks for every GPU, one CSV row each. The UUID is there
// because CUDA_VISIBLE_DEVICES may name GPUs by it.
var nvidiaSMIArgs = []string{"--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"}

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
// SSH, produce the same answer this one would.
type facts struct {
	totalMem, availMem int64
	smiOut             string // nvidia-smi's rows; meaningful only with smiOK
	smiOK              bool
	nvidiaDevice       bool
	visible            string
	visibleSet         bool
}

// Probe inspects system memory and nvidia-smi (if present). Never fails --
// missing pieces just degrade the result.
func Probe() Hardware {
	f := facts{totalMem: totalMemBytes(), availMem: int64(AvailableMemMB()) << 20, nvidiaDevice: hasNVIDIADevice()}
	if out, err := exec.Command("nvidia-smi", nvidiaSMIArgs...).Output(); err == nil {
		f.smiOut, f.smiOK = string(out), true
	}
	f.visible, f.visibleSet = cudaVisibleDevices()
	return f.hardware(describeLocalGPU, currentMetal())
}

// hardware derives what this machine can serve from its readings. describe
// names a non-NVIDIA accelerator and metal asks about vllm-metal; both only
// apply to the machine running this process, so a remote reading passes nil.
func (f facts) hardware(describe func(*Hardware), metal *metalState) Hardware {
	hw := Hardware{TotalMemBytes: f.totalMem}
	hw.TotalMemGB = decimalGB(hw.TotalMemBytes)

	// Recorded before describe, which names an Apple SoC and would
	// otherwise make GPUName look like evidence of CUDA.
	hw.CUDA = f.smiOK || f.nvidiaDevice
	if f.smiOK {
		all := parseNVIDIASMI(f.smiOut)
		hw.GPUs = visibleGPUs(all, f.visible, f.visibleSet)
		hw.GPUsHidden = len(all) - len(hw.GPUs)
		if len(hw.GPUs) > 0 {
			hw.GPUName = hw.GPUs[0].Name
			hw.GPUMemBytes = hw.GPUs[0].MemBytes
			hw.GPUMemGB = decimalGB(hw.GPUMemBytes)
		}
	}

	hw.Unified = isUnified(hw.GPUName, hw.GPUMemGB, hw.TotalMemGB, f.nvidiaDevice)

	// No NVIDIA GPU, but the platform may still have an accelerator worth
	// naming (Apple Silicon shares one memory pool with the CPU).
	if hw.GPUName == "" && describe != nil {
		describe(&hw)
	}
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
		if float64(used) < baselineCeiling*float64(hw.TotalMemBytes) {
			hw.BaselineGB = float64(used) / gigabyte
		}
	}

	hw.BudgetBytes, hw.BudgetReason = hw.Budget(used)
	hw.BudgetGB = decimalGB(hw.BudgetBytes)
	return hw
}
