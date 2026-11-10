package plugin

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
)

type fake struct {
	name       string
	configured bool
	events     []Event
	err        error
	panics     bool
	got        []Event
}

func (f *fake) Name() string                                { return f.name }
func (f *fake) Configured(*config.Config) bool              { return f.configured }
func (f *fake) Commands() []*cobra.Command                  { return nil }
func (f *fake) Events() []Event                             { return f.events }
func (f *fake) Check(context.Context, *config.Config) error { return nil }

func (f *fake) Handle(_ context.Context, e Event, _ State) error {
	f.got = append(f.got, e)
	if f.panics {
		panic("held together with tape")
	}
	return f.err
}

// withRegistry swaps the built-in set for the duration of a test. The
// registry is package state, so these can't run in parallel.
func withRegistry(t *testing.T, plugins ...Plugin) {
	t.Helper()
	old := registered
	registered = plugins
	t.Cleanup(func() { registered = old })
}

func TestDispatchOnlyReachesPluginsThatOptedIn(t *testing.T) {
	interested := &fake{name: "interested", configured: true, events: []Event{EventStarting, EventRemoved}}
	// An integration with nothing to do on a switch shouldn't be called at
	// all -- that's what keeps an unused plugin off the critical path.
	uninterested := &fake{name: "uninterested", configured: true, events: []Event{EventRemoved}}
	unconfigured := &fake{name: "unconfigured", configured: false, events: []Event{EventStarting}}
	withRegistry(t, interested, uninterested, unconfigured)

	var out strings.Builder
	Dispatch(context.Background(), &out, EventStarting, State{})

	if len(interested.got) != 1 || interested.got[0] != EventStarting {
		t.Errorf("interested plugin saw %v, want one activating event", interested.got)
	}
	if len(uninterested.got) != 0 {
		t.Errorf("a plugin that didn't ask for activating saw %v", uninterested.got)
	}
	if len(unconfigured.got) != 0 {
		t.Errorf("an unconfigured plugin saw %v, want to be skipped entirely", unconfigured.got)
	}
	if out.String() != "" {
		t.Errorf("output = %q, want nothing when every plugin succeeded", out.String())
	}
}

// A failing integration must not stop the local operation, and the others
// still have to run: switching models works with the network unplugged.
func TestDispatchReportsFailuresAndKeepsGoing(t *testing.T) {
	broken := &fake{name: "broken", configured: true, events: []Event{EventStarting}, err: errors.New("deployment unreachable")}
	after := &fake{name: "after", configured: true, events: []Event{EventStarting}}
	withRegistry(t, broken, after)

	var out strings.Builder
	Dispatch(context.Background(), &out, EventStarting, State{})

	if len(after.got) != 1 {
		t.Error("a plugin registered after a failing one was skipped")
	}
	msg := out.String()
	if !strings.Contains(msg, "broken") || !strings.Contains(msg, "deployment unreachable") {
		t.Errorf("output = %q, want the plugin name and its error", msg)
	}
}

func TestConfiguredFiltersTheRegistry(t *testing.T) {
	withRegistry(t,
		&fake{name: "on", configured: true},
		&fake{name: "off", configured: false},
	)
	got := Configured(nil)
	if len(got) != 1 || got[0].Name() != "on" {
		t.Errorf("Configured() = %v, want only the configured plugin", got)
	}
}

// A panic is a bug in this repo rather than third-party misbehaviour, but it
// still must not escape: activating dispatches before the service restarts,
// so letting one through would stop the model from loading.
func TestDispatchSurvivesAPanickingPlugin(t *testing.T) {
	after := &fake{name: "after", configured: true, events: []Event{EventStarting}}
	withRegistry(t,
		&fake{name: "panicky", configured: true, events: []Event{EventStarting}, panics: true},
		after,
	)

	var out strings.Builder
	Dispatch(context.Background(), &out, EventStarting, State{})

	if len(after.got) != 1 {
		t.Error("a plugin registered after a panicking one was skipped")
	}
	msg := out.String()
	if !strings.Contains(msg, "panicky") || !strings.Contains(msg, "held together with tape") {
		t.Errorf("output = %q, want the plugin name and what it panicked with", msg)
	}
	if !strings.Contains(msg, "registry_internal_test.go") {
		t.Errorf("output = %q, want a stack pointing at the bug", msg)
	}
}

// Warnings have nowhere to go in a caller that doesn't want them, and that
// must not be a crash.
func TestDispatchToleratesANilWriter(t *testing.T) {
	p := &fake{name: "broken", configured: true, events: []Event{EventStarting}, err: errors.New("nope")}
	withRegistry(t, p)
	Dispatch(context.Background(), nil, EventStarting, State{})
	if len(p.got) != 1 {
		t.Error("the plugin was not called when the writer was nil")
	}
}
