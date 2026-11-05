// Package plugin is the seam between vllm-helper's own commands and the
// integrations that mirror local state into somebody else's system -- Coder
// Agents today, potentially OpenWebUI or a local agent config later.
//
// A plugin owns three things: whatever configuration it needs, the commands
// it contributes to the CLI, and the lifecycle events it wants to hear about.
// Core never enumerates a plugin's config keys or hardcodes its command name,
// so adding one is a matter of writing it and registering it.
//
// Two rules the Coder integration established, which every plugin inherits:
// an unconfigured plugin is skipped rather than being an error (these are all
// optional), and a plugin that fails must not fail the local operation that
// triggered it. Switching models has to work on a laptop with no network.
package plugin

import (
	"context"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/profile"
)

// Event is a point in a profile's life that a plugin can react to.
type Event string

const (
	// EventActivating fires after the active profile is recorded but before
	// the service restarts, so a plugin can point elsewhere at the model
	// that is about to load.
	EventActivating Event = "activating"

	// EventReady fires once the endpoint answers /health. Only commands
	// that wait for the boot can raise it -- with --no-follow nothing is
	// watching, so a plugin must not depend on it for correctness.
	EventReady Event = "ready"

	// EventRemoved fires before a profile is deleted, while its state is
	// still readable.
	EventRemoved Event = "removed"
)

// State is what a plugin is told about the local machine when an event
// fires. Profile is nil for EventRemoved when the profile file is already
// gone, which is why Name is carried separately.
type State struct {
	Config  *config.Config
	Name    string
	Profile *profile.Profile

	// BaseURL is where this machine serves the OpenAI-compatible API, as
	// vllm-helper understands it. Plugins that register an endpoint (rather
	// than being told about one out of band, as Coder is) need it.
	BaseURL string
}

// Plugin is one integration.
type Plugin interface {
	// Name identifies the plugin in doctor output and state file names.
	Name() string

	// Configured reports whether the user has set this plugin up. An
	// unconfigured plugin is skipped entirely: no events, no doctor check.
	Configured(*config.Config) bool

	// Commands are contributed to the CLI whether or not the plugin is
	// configured, so `--help` documents them and running one can explain
	// what is missing.
	Commands() []*cobra.Command

	// Events lists what this plugin wants to hear about. Anything not
	// listed is never delivered, so an integration that has nothing to do
	// on a switch costs nothing.
	Events() []Event

	// Handle reacts to an event. An error is reported to the user as a
	// warning; it never fails the operation that triggered it.
	Handle(context.Context, Event, State) error

	// Check verifies the integration end to end for doctor -- credentials
	// and reachability, not just "the variables are set".
	Check(context.Context, *config.Config) error
}
