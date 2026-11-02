package control

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/vllm"
)

// GGUF support left vLLM core in 0.27.1 and lives in an out-of-tree package.
// A stock install therefore cannot load a GGUF file at all -- the failure is
// not a bad flag or a missing tokenizer, it is that "gguf" is not a registered
// quantization method -- so it is worth naming precisely and offering to fix.
const GGUFPlugin = "vllm-gguf-plugin"

// GGUFInstallTimeout bounds the install. The plugin builds a CUDA extension
// from source, which is minutes on a fast machine and longer on a busy one.
const GGUFInstallTimeout = 20 * 60 // seconds

// GGUFCaveats are what someone should know before installing, and they are
// returned rather than printed so a terminal and a browser can both show them.
//
// Said before the prompt, not after: these are the things that would otherwise
// be discovered as a failed model load half an hour later.
// Two, not four. The earlier list repeated things the interface already shows
// -- which layouts cannot load is written against each quant in the list, and
// "prefer a native quant" is advice rather than a consequence of installing.
// A wall of warnings above a button is read as noise and pushes the button off
// the screen, which is the opposite of informed consent.
func GGUFCaveats() []string {
	return []string{
		"vLLM calls its own GGUF support experimental; a native quant (AWQ, GPTQ, FP8, NVFP4) is faster where one exists.",
		"It builds from source, so expect several minutes and a compiler.",
	}
}

// PythonForVLLM finds the interpreter that owns a vllm executable.
//
// Installing into whatever python is on PATH is the classic way to end up with
// the package present and still unavailable to the server, so this returns
// empty rather than falling back to a guess.
func PythonForVLLM(binPath string) string {
	return vllm.PythonBeside(binPath)
}

// GGUFInstallCommand is the command that would be run, for showing someone
// before they agree to it. Empty when the interpreter could not be found.
func GGUFInstallCommand(cfg *config.Config) string {
	python := PythonForVLLM(cfg.VLLMBin())
	if python == "" {
		return ""
	}
	return fmt.Sprintf("%s -m pip install %s", python, GGUFPlugin)
}

// InstallGGUF adds the plugin to the environment that owns the configured
// vLLM, writing pip's output to out.
//
// It does not restart anything. The plugin is read at engine start, so it
// takes effect on the next switch -- which means installing it can never
// disturb a model that is currently serving.
func InstallGGUF(ctx context.Context, cfg *config.Config, out io.Writer) error {
	python := PythonForVLLM(cfg.VLLMBin())
	if python == "" {
		return fmt.Errorf("couldn't find the interpreter next to %s -- install by hand:\n"+
			"  <your-vllm-venv>/bin/python -m pip install %s", cfg.VLLMBin(), GGUFPlugin)
	}
	cmd := exec.CommandContext(ctx, python, "-m", "pip", "install", GGUFPlugin)
	cmd.Stdout, cmd.Stderr = out, out
	if err := cmd.Run(); err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return fmt.Errorf("the install was still running after the time limit -- "+
				"it builds from source, so finish it in a terminal:\n  %s", GGUFInstallCommand(cfg))
		}
		return fmt.Errorf("pip install failed: %w", err)
	}
	return nil
}

// ForgetCaps drops the cached capability probe.
//
// Necessary after an install because the cache is keyed on vLLM's version,
// which adding a plugin does not change -- so without this the machine would
// keep reporting the capability it just gained as missing.
func ForgetCaps(dirs config.Dirs) {
	matches, _ := filepath.Glob(filepath.Join(dirs.State, "vllm-caps-*.json"))
	for _, m := range matches {
		_ = os.Remove(m)
	}
}
