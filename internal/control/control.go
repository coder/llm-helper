// Package control performs the operations that change or report what this
// machine is serving. It is the layer under both the CLI and the web panel,
// so the two cannot drift apart on what the work is.
package control

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/plugin"
	"github.com/coder/llm-helper/internal/profile"
)

// BaseURL is the address a single client such as the Coder provider should
// use: the router when it is installed, the first instance port otherwise.
func BaseURL(cfg *config.Config) string {
	port := cfg.Port()
	if rp := cfg.RouterPort(); rp != "" {
		port = rp
	}
	return fmt.Sprintf("http://localhost:%s/v1", port)
}

func state(cfg *config.Config, name string, p *profile.Profile, base string) plugin.State {
	live := LiveProfiles(cfg)
	return plugin.State{
		Config:  cfg,
		Name:    name,
		Profile: p,
		BaseURL: base,
		Live:    live,
		Default: DefaultInstance(cfg, live),
	}
}

// NotifyAndRestart tells the plugins an instance is starting, then calls
// restart. The instance file must already be written. Plugins run first so
// Coder's entry describes the model by the time it is up.
func NotifyAndRestart(ctx context.Context, cfg *config.Config, p *profile.Profile, inst *Instance, out io.Writer, restart func() error) error {
	plugin.Dispatch(ctx, out, plugin.EventStarting, state(cfg, p.Name, p, inst.BaseURL()))
	if err := restart(); err != nil {
		return fmt.Errorf("restarting service: %w", err)
	}
	return nil
}

// NotifyStopped tells the plugins an instance is gone. Call it after the
// instance file is removed.
func NotifyStopped(ctx context.Context, cfg *config.Config, name string, out io.Writer) {
	p, _ := profile.Load(cfg.Dirs, name)
	plugin.Dispatch(ctx, out, plugin.EventStopped, state(cfg, name, p, BaseURL(cfg)))
}

// NotifyDefault tells the plugins the default instance changed.
func NotifyDefault(ctx context.Context, cfg *config.Config, out io.Writer) {
	st := state(cfg, "", nil, BaseURL(cfg))
	if st.Default == "" {
		return
	}
	st.Name = st.Default
	st.Profile, _ = profile.Load(cfg.Dirs, st.Default)
	plugin.Dispatch(ctx, out, plugin.EventDefault, st)
}

// NotifyRemoved tells the plugins a profile is going away, so integrations can
// drop their copy of it. Called before the file is deleted, since plugins key
// off the name.
func NotifyRemoved(ctx context.Context, cfg *config.Config, name string, out io.Writer) {
	plugin.Dispatch(ctx, out, plugin.EventRemoved, state(cfg, name, nil, BaseURL(cfg)))
}

// ProbeHealth asks vLLM whether it is serving. Returns the HTTP status code
// (0 if nothing answered) and whether the endpoint is up.
//
// A closed port is the normal shape of a model load rather than a fault: vLLM
// binds only once the weights are in memory, so this reports 0 for minutes
// during an ordinary start.
func ProbeHealth(port string) (int, bool) {
	return ProbeHealthURL(fmt.Sprintf("http://localhost:%s", port))
}

// ProbeHealthURL is ProbeHealth for a server that is not on this machine, such
// as a recipe cluster whose head is another node.
func ProbeHealthURL(base string) (int, bool) { return ProbeHealthPath(base, "/health") }

// ProbeHealthPath is ProbeHealthURL for an engine whose health answers on
// another path. An empty path is /health.
func ProbeHealthPath(base, path string) (int, bool) {
	if path == "" {
		path = "/health"
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(strings.TrimSuffix(base, "/") + path)
	if err != nil {
		return 0, false
	}
	resp.Body.Close()
	return resp.StatusCode, resp.StatusCode == http.StatusOK
}

// LastMeaningfulLine picks the line most likely to say why the server died.
// A Python traceback ends with the exception, and vLLM's fatal startup errors
// are one-liners, so the last non-blank line is nearly always it.
func LastMeaningfulLine(logs string) string {
	lines := strings.Split(logs, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return line
		}
	}
	return ""
}
