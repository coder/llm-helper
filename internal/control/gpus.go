package control

import (
	"os"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
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

// ServerCUDAVisible is the CUDA_VISIBLE_DEVICES config.env gives a server.
// Each instance's profile can narrow it; callers sizing one profile layer
// its file on top with CUDAVisible.
func ServerCUDAVisible(cfg *config.Config) (string, bool) {
	return CUDAVisible(cfg.File, cfg.Secrets)
}

// UseServerGPUs tells detect which GPUs a server may use, so memory is sized
// for the GPU it will load onto. The CLI and the panel both call it.
func UseServerGPUs(cfg *config.Config) {
	detect.UseCUDAVisibleSource(func() (string, bool) { return ServerCUDAVisible(cfg) })
}
