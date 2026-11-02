package cli

import "github.com/coder/vllm-helper/internal/detect"

// gatherToolchain checks what a macOS source build of vLLM's CPU backend
// needs: it has no Apple Silicon wheels, so it is compiled locally against
// Apple Clang from the Command Line Tools. vllm-metal ships prebuilt wheels and
// Metal kernels, so a Mac serving on its GPU needs none of it. Triton and
// FlashInfer are CUDA-only and never enter the picture.
func gatherToolchain() []toolchainCheck {
	if detect.MetalRuntime() {
		return nil
	}
	return []toolchainCheck{
		{Name: "cmake", Package: "cmake", OK: hasBins("cmake")},
		{Name: "ninja", Package: "ninja", OK: hasBins("ninja")},
		// Clang ships with the Command Line Tools, which brew can't install.
		{Name: "clang", Hint: "xcode-select --install", OK: hasBins("clang")},
	}
}

func toolchainInstaller() (string, []string) {
	if !hasBins("brew") {
		return "", nil
	}
	return "brew", []string{"install"}
}
