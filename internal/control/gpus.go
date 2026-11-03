package control

import (
	"os"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/detect"
)

// CUDAVisible is the CUDA_VISIBLE_DEVICES a server started with these env
// files gets. files are in the order they are layered, so the last one that
// sets it wins; with none setting it, the server inherits this process's.
// An empty value is a real setting (no GPUs), which is why presence is
// checked rather than the value.
func CUDAVisible(files ...*config.EnvFile) (string, bool) {
	value, set := os.LookupEnv("CUDA_VISIBLE_DEVICES")
	for _, f := range files {
		if f != nil && f.Has("CUDA_VISIBLE_DEVICES") {
			value, set = f.Get("CUDA_VISIBLE_DEVICES"), true
		}
	}
	return value, set
}

// ServerCUDAVisible is the CUDA_VISIBLE_DEVICES the service runs with: the
// active profile's environment when one has been written, since a profile
// can set it, and config.env's otherwise.
func ServerCUDAVisible(cfg *config.Config) (string, bool) {
	files := []*config.EnvFile{cfg.File, cfg.Secrets}
	if active, err := ParseActiveEnv(cfg); err == nil {
		files = append(files, active)
	}
	return CUDAVisible(files...)
}

// UseServerGPUs tells detect which GPUs the service may use, so memory is
// sized for the GPU it will load onto. The CLI and the panel both call it,
// and the panel again after each switch, so they size the same machine.
func UseServerGPUs(cfg *config.Config) {
	detect.UseCUDAVisibleDevices(ServerCUDAVisible(cfg))
}
