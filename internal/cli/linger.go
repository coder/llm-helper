package cli

import (
	"fmt"
	"os"
	"path/filepath"
)

// lingerWarnings warns when each model's unit is in user's systemd manager
// but user does not linger. Without lingering, systemd starts that manager at
// the user's first login and stops it at the last logout, so the models do
// not start at boot. lingerDir is /var/lib/systemd/linger, which holds a file
// per lingering user; a machine without its parent directory does not run
// systemd and gets no warning.
func lingerWarnings(scope, user, lingerDir string) []string {
	if scope != "user" || user == "" {
		return nil
	}
	if _, err := os.Stat(filepath.Dir(lingerDir)); err != nil {
		return nil
	}
	if _, err := os.Stat(filepath.Join(lingerDir, user)); err == nil {
		return nil
	}
	return []string{fmt.Sprintf(
		"SERVICE_SCOPE=user, but lingering is off for %s, so systemd runs the model units only while %s is logged in and they do not start at boot. "+
			"Turn it on with 'sudo loginctl enable-linger %s'.", user, user, user)}
}
