//go:build !linux

package cli

import "github.com/coder/llm-helper/internal/config"

// serviceScopeWarnings has nothing to check off Linux: a LaunchAgent starts
// at login by design, and a headless Mac is told to use the system scope.
func serviceScopeWarnings(*config.Config) []string { return nil }
