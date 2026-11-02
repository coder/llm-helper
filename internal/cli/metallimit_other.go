//go:build !darwin

package cli

import (
	"errors"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/detect"
)

// runMetalLimit refuses off macOS. The command still exists here so the
// generated CLI reference is the same on every platform.
func runMetalLimit(metalLimitOpts) error {
	return errors.New("metal-limit is macOS only: it sets iogpu.wired_limit_mb, the cap on how much unified memory an Apple Silicon GPU may use")
}

// awaitWiredLimit is a no-op off macOS, where there is no wired limit.
func awaitWiredLimit(*config.Config) {}

// gatherWiredLimit reports nothing off macOS.
func gatherWiredLimit(*config.Config, detect.Hardware) *wiredLimit { return nil }
