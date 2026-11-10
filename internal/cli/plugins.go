package cli

import (
	// Integrations register themselves. Importing them here, rather than
	// from the plugin package, keeps the dependency pointing one way:
	// plugins may use the core, the core never imports a plugin.
	_ "github.com/coder/llm-helper/internal/plugin/coder"
	_ "github.com/coder/llm-helper/internal/plugin/spark"
)
