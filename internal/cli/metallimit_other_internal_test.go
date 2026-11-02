//go:build !darwin

package cli

import (
	"strings"
	"testing"
)

// The command exists everywhere so the CLI reference does not depend on the
// build platform, but it must say plainly that it cannot work here.
func TestMetalLimitIsMacOSOnly(t *testing.T) {
	t.Parallel()

	err := runMetalLimit(metalLimitOpts{set: "auto", yes: true})
	if err == nil || !strings.Contains(err.Error(), "macOS only") {
		t.Errorf("runMetalLimit = %v, want a macOS-only error", err)
	}
}
