package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A user-scope unit without lingering waits for a login, so a machine that
// reboots unattended serves nothing. Only the user scope is checked, and only
// on a machine that runs systemd.
func TestLingerWarnings(t *testing.T) {
	systemd := filepath.Join(t.TempDir(), "systemd")
	linger := filepath.Join(systemd, "linger")
	if err := os.MkdirAll(systemd, 0o755); err != nil {
		t.Fatal(err)
	}

	// No user lingers yet, so the folder itself is missing.
	w := lingerWarnings("user", "amd", linger)
	if len(w) != 1 || !strings.Contains(w[0], "sudo loginctl enable-linger amd") || !strings.Contains(w[0], "do not start at boot") {
		t.Errorf("user scope without lingering: %q", w)
	}
	if w := lingerWarnings("system", "amd", linger); len(w) != 0 {
		t.Errorf("system scope: %q", w)
	}

	if err := os.MkdirAll(linger, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(linger, "amd"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if w := lingerWarnings("user", "amd", linger); len(w) != 0 {
		t.Errorf("lingering on: %q", w)
	}
	if w := lingerWarnings("user", "other", linger); len(w) != 1 {
		t.Errorf("another user lingers, this one does not: %q", w)
	}
	if w := lingerWarnings("user", "amd", filepath.Join(t.TempDir(), "none", "linger")); len(w) != 0 {
		t.Errorf("no systemd: %q", w)
	}
}
