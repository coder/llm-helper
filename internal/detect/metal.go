package detect

import "sync"

// MetalMemUtilization is the --gpu-memory-utilization a new profile gets under
// vllm-metal. There the flag is a share of the Metal working set, which macOS
// already holds below installed RAM, so it can sit higher than the 0.85 that
// a DGX Spark needs. It stays a little under vllm-metal's own 0.92 default to
// leave room for the KV commit probe and a second process.
const MetalMemUtilization = 0.9

// metalWorkingSetGuess is the share of RAM taken as the working set when MLX
// could not report it. macOS commonly lets the GPU wire about three quarters
// of RAM; larger Macs get more, so this errs low.
const metalWorkingSetGuess = 0.75

// metalState is one registration of the vllm-metal probes. Hardware alone
// cannot answer whether a Mac serves on its GPU: the same Mac runs vLLM's CPU
// backend or the GPU depending on which vLLM is installed, and the two spend
// memory differently.
//
// The two questions are probed separately because their costs differ. Whether
// vllm_metal is importable takes a module lookup; the working set needs an MLX
// import, which only the commands that size memory should pay for. Each is
// asked at most once per registration, and its answer lands in this state, so
// a probe still running when a newer registration replaces it cannot
// overwrite the newer answer.
type metalState struct {
	installed func() bool
	memory    func() MetalMemory

	installedOnce sync.Once
	on            bool
	memoryOnce    sync.Once
	mem           MetalMemory
}

// MetalMemory is what the MLX probe learns about how vllm-metal spends memory.
type MetalMemory struct {
	// WorkingSetBytes is MLX's max_recommended_working_set_size in bytes, 0
	// when unknown.
	WorkingSetBytes int64
	// FixedKVCache is whether this vllm-metal honors --kv-cache-memory-bytes.
	FixedKVCache bool
}

var (
	metalMu sync.Mutex
	metal   *metalState
)

// UseMetalProbe registers how to ask the configured vLLM about vllm-metal:
// installed reports whether the plugin is importable, memory asks MLX and the
// plugin how memory is sized. Neither runs until a caller needs its answer,
// because each starts a Python interpreter. A nil installed clears the
// registration.
func UseMetalProbe(installed func() bool, memory func() MetalMemory) {
	metalMu.Lock()
	defer metalMu.Unlock()
	if installed == nil {
		metal = nil
		return
	}
	metal = &metalState{installed: installed, memory: memory}
}

// currentMetal returns the registration in force, or nil when there is none.
// Callers that need both answers take one snapshot so they agree.
func currentMetal() *metalState {
	metalMu.Lock()
	defer metalMu.Unlock()
	return metal
}

// runtime reports whether vllm_metal is importable. A nil state is false.
func (s *metalState) runtime() bool {
	if s == nil {
		return false
	}
	s.installedOnce.Do(func() { s.on = s.installed() })
	return s.on
}

// memoryInfo reads the Metal working set and KV sizing support, or a zero
// MetalMemory when vllm-metal is not installed or could not be asked. It is
// not probed without vllm-metal, since MLX is not there to ask.
func (s *metalState) memoryInfo() MetalMemory {
	if !s.runtime() || s.memory == nil {
		return MetalMemory{}
	}
	s.memoryOnce.Do(func() {
		s.mem = s.memory()
		s.mem.WorkingSetBytes = max(s.mem.WorkingSetBytes, 0)
	})
	return s.mem
}

// MetalRuntime reports whether the configured vLLM serves on the Apple Silicon
// GPU through vllm-metal. It never imports MLX, so it is cheap enough for
// 'start' and for choosing a profile's defaults.
func MetalRuntime() bool { return currentMetal().runtime() }
