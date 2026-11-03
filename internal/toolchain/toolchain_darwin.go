package toolchain

import "github.com/coder/vllm-helper/internal/detect"

// Gather checks what a macOS source build of vLLM's CPU backend needs: it
// has no Apple Silicon wheels, so it is compiled locally against Apple Clang
// from the Command Line Tools. vllm-metal ships prebuilt wheels and Metal
// kernels, so a Mac serving on its GPU needs none of it. Triton and
// FlashInfer are CUDA-only and never enter the picture.
func Gather(string) []Check {
	if detect.MetalRuntime() {
		return nil
	}
	return []Check{
		{Name: "cmake", Package: "cmake", OK: hasBins("cmake")},
		{Name: "ninja", Package: "ninja", OK: hasBins("ninja")},
		// Clang ships with the Command Line Tools, which brew can't install.
		{Name: "clang", Hint: "xcode-select --install", OK: hasBins("clang")},
	}
}

// Installer is the command and leading args that install packages here.
func Installer() (string, []string) {
	if !hasBins("brew") {
		return "", nil
	}
	return "brew", []string{"install"}
}
