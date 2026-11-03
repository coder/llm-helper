//go:build !linux && !darwin

package toolchain

// Gather has nothing to check on platforms vLLM does not build on.
func Gather(string) []Check { return nil }

// Installer finds no package manager on platforms vLLM does not build on.
func Installer() (string, []string) { return "", nil }
