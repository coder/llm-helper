package cli

import (
	"os"
	"os/user"

	"github.com/coder/llm-helper/internal/config"
)

// serviceScopeWarnings checks that the configured service scope starts the
// models at boot on this machine.
func serviceScopeWarnings(cfg *config.Config) []string {
	name := os.Getenv("USER")
	if u, err := user.Current(); err == nil {
		name = u.Username
	}
	return lingerWarnings(cfg.ServiceScope(), name, "/var/lib/systemd/linger")
}
