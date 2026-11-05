package cli

import (
	"fmt"
	"os"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/vllm"
)

// vllmCaps asks the configured vLLM what it supports, caching under the state
// directory. A zero value means we couldn't ask, and every caller treats that
// as "don't validate" rather than "nothing is valid".
func vllmCaps(cfg *config.Config) vllm.Caps {
	if cfg == nil {
		return vllm.Caps{}
	}
	return vllm.Probe(cfg.VLLMBin(), cfg.Dirs.State)
}

// checkExtraArgs drops flags the installed vLLM can't accept and reports what
// went, to w. It returns the cleaned value.
//
// Nothing is substituted. A parser whose name merely looks similar is a
// different parser, and getting that wrong changes reasoning output in a way
// nobody notices, whereas a dropped flag is visible here and reversible with
// one 'set'.
func checkExtraArgs(extraArgs string, caps vllm.Caps) string {
	cleaned, warnings := vllm.Check(extraArgs, caps)
	for _, w := range warnings {
		fmt.Fprintf(os.Stderr, "warning: %s\n", w)
	}
	return cleaned
}

// warnExtraArgs reports problems without changing anything. Used where a
// profile is being served rather than written: someone explicitly asked for
// this model, and refusing to start it -- or quietly serving something other
// than what the profile says -- would both be worse than saying so.
func warnExtraArgs(p interface{ ExtraArgs() string }, caps vllm.Caps) {
	_, warnings := vllm.Check(p.ExtraArgs(), caps)
	if len(warnings) == 0 {
		return
	}
	for _, w := range warnings {
		fmt.Fprintf(os.Stderr, "warning: %s\n", w)
	}
	fmt.Fprintln(os.Stderr, "warning: the server will likely fail to start with these. Fix them with 'llm-helper set' or 'llm-helper edit'.")
}
