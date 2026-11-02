package cli

import (
	"os"
	"path/filepath"
)

// vllmCandidates are where the two vllm-metal installers put vllm. Neither is
// on PATH by default: Homebrew links /opt/homebrew/bin/vllm only when nothing
// else owns the name, and the install script's venv has to be activated.
func vllmCandidates() []string {
	out := []string{
		"/opt/homebrew/opt/vllm-metal/libexec/bin/vllm",
	}
	if home, err := os.UserHomeDir(); err == nil {
		out = append(out, filepath.Join(home, ".venv-vllm-metal", "bin", "vllm"))
	}
	return out
}

// vllmInstallHint is printed when setup finds no vllm.
var vllmInstallHint = []string{
	"WARNING: no 'vllm' executable found. On Apple Silicon, install vllm-metal,",
	"which runs vLLM on the GPU:",
	"",
	"  brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal",
	"  brew install vllm-project/vllm-metal/vllm-metal",
	"",
	"then run 'vllm-helper setup' again, or set VLLM_BIN in config.env to",
	"/opt/homebrew/opt/vllm-metal/libexec/bin/vllm.",
}
