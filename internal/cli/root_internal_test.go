package cli

import (
	"slices"
	"strings"
	"testing"
)

// The command set is the CLI's contract. Dispatching each one through Run
// can't cover setup or adopt, since both drive the machine the tests run on,
// but the registered tree can be read without executing anything.
func TestTheCommandSetIsPinned(t *testing.T) {
	// help is not here: cobra attaches it during Execute, not registration.
	// completion is not here either, because the root disables it.
	want := []string{
		"adopt", "coder-sync", "doctor", "edit", "inspect", "install-gguf", "list", "logs", "metal-limit",
		"new", "node", "pool", "recipes", "remove", "restart", "search", "serve", "set", "setup", "shard", "spark", "start",
		"status", "stop", "switch", "upgrade", "version",
	}

	var got []string
	for _, c := range newRootCmd().Commands() {
		got = append(got, c.Name())
	}
	slices.Sort(got)

	if !slices.Equal(got, want) {
		t.Errorf("command set drifted:\n got %v\nwant %v", got, want)
	}
}

// start is the service manager's ExecStart. It execs vLLM over the calling
// process, so listing it in help would invite someone to run it by hand.
func TestStartIsHiddenAndTheRestAreGrouped(t *testing.T) {
	for _, c := range newRootCmd().Commands() {
		switch c.Name() {
		case "start":
			if !c.Hidden {
				t.Error("start should stay out of the help listing")
			}
		case "version":
			// Sits with help under Additional Commands, not with the
			// commands you reach for to do something.
		default:
			if c.Hidden {
				t.Errorf("%s is hidden, so nothing in help mentions it", c.Name())
			}
			if c.GroupID == "" {
				t.Errorf("%s has no group, so it lands under Additional Commands", c.Name())
			}
		}
	}
}

// Every command's usage line has to name the binary, since cobra builds
// "vllm-helper <use>" from it and a stray "vllm-helper vllm-helper set" or a
// missing operand reads as a typo in the help.
func TestUsageLinesNameTheirOperands(t *testing.T) {
	wantOperands := map[string]string{
		"switch":     "<profile>",
		"new":        "<org/model-repo>",
		"search":     "<search terms>",
		"inspect":    "<org/model-repo>",
		"edit":       "<profile>",
		"remove":     "<profile>",
		"set":        "<profile> KEY=VALUE [KEY=VALUE ...]",
		"coder-sync": "<activate|register|remove|register-all|prune> [profile]",
	}
	for _, c := range newRootCmd().Commands() {
		want, ok := wantOperands[c.Name()]
		if !ok {
			continue
		}
		if got := strings.TrimPrefix(c.Use, c.Name()+" "); got != want {
			t.Errorf("%s usage = %q, want operands %q", c.Name(), c.Use, want)
		}
	}
}
