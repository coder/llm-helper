package plugin

import (
	"context"
	"fmt"
	"io"
	"runtime/debug"
	"slices"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
)

// registered is the built-in set, populated by each plugin's init. Plugins
// ship in the binary: this is a seam for keeping integrations separable and
// testable, not a mechanism for loading third-party code at runtime.
var registered []Plugin

// Register adds a plugin to the built-in set. Call it from an init function.
func Register(p Plugin) { registered = append(registered, p) }

// All returns every registered plugin.
func All() []Plugin { return slices.Clone(registered) }

// Commands collects the commands every plugin contributes.
func Commands() []*cobra.Command {
	var cmds []*cobra.Command
	for _, p := range registered {
		cmds = append(cmds, p.Commands()...)
	}
	return cmds
}

// Dispatch delivers an event to every configured plugin that asked for it.
//
// Failures are reported to w and otherwise ignored: an integration being
// down must not stop a model from loading. Plugins that did not ask for the
// event, or aren't configured, are never called -- so an unused integration
// cannot slow down a switch. A nil w discards the warnings.
func Dispatch(ctx context.Context, w io.Writer, e Event, s State) {
	if w == nil {
		w = io.Discard
	}
	for _, p := range registered {
		if !slices.Contains(p.Events(), e) || !p.Configured(s.Config) {
			continue
		}
		if err := handle(ctx, p, e, s); err != nil {
			fmt.Fprintf(w, "%s: %v\n", p.Name(), err)
		}
	}
}

// handle calls one plugin, turning a panic into an ordinary error.
//
// Plugins are compiled in, so a panic here is a bug in this repository
// rather than third-party misbehaviour. It still must not escape: activating
// a profile dispatches before the service restarts, so an unrecovered panic
// would stop the model from loading -- the one thing an integration is never
// allowed to do. Reporting it with a stack keeps that promise and still
// makes the bug obvious.
func handle(ctx context.Context, p Plugin, e Event, s State) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panicked handling %q: %v\n%s", e, r, debug.Stack())
		}
	}()
	return p.Handle(ctx, e, s)
}

// Configured returns the plugins the user has set up.
func Configured(cfg *config.Config) []Plugin {
	var out []Plugin
	for _, p := range registered {
		if p.Configured(cfg) {
			out = append(out, p)
		}
	}
	return out
}
