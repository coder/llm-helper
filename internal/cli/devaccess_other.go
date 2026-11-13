//go:build !linux

package cli

import "github.com/coder/llm-helper/internal/detect"

// gpuDeviceWarnings has nothing to check off Linux, where no amdgpu device
// nodes exist.
func gpuDeviceWarnings([]detect.Accelerator) []string { return nil }
