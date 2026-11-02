//go:build !darwin

package cli

// useMetalProbe is a no-op off macOS, where vllm-metal cannot run.
func useMetalProbe(string) {}
