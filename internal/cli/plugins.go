package cli

import (
	"fmt"

	"github.com/coder/llm-helper/internal/config"

	// Integrations register themselves. Importing them here, rather than
	// from the plugin package, keeps the dependency pointing one way:
	// plugins may use the core, the core never imports a plugin.
	_ "github.com/coder/llm-helper/internal/plugin/coder"
	_ "github.com/coder/llm-helper/internal/plugin/spark"
)

// localBaseURL is where this machine serves the OpenAI-compatible API.
//
// It is deliberately localhost: that is the only address vllm-helper can
// vouch for, since vLLM binds one port and what reaches it from elsewhere
// depends on the network (a LAN address, Tailscale, or a WSL port proxy).
// A plugin that registers an endpoint somebody else has to reach should
// take that address from its own configuration rather than guess from here.
func localBaseURL(cfg *config.Config) string {
	return fmt.Sprintf("http://localhost:%s/v1", cfg.Port())
}
