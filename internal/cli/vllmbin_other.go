//go:build !darwin

package cli

// vllmCandidates is empty here: a pip or uv install is wherever the user put
// it, so PATH is the only place worth looking.
func vllmCandidates() []string { return nil }

// vllmInstallHint is printed when setup finds no vllm.
var vllmInstallHint = []string{
	"WARNING: no 'vllm' executable found on PATH. Install vLLM (e.g. in a",
	"venv: 'uv venv ~/venvs/vllm && uv pip install vllm'), then set VLLM_BIN",
	"in config.env to its full path.",
}
