// Package control performs the operations that change or report what this
// machine is serving. It is the layer under both the CLI and the web panel.
//
// The two interfaces exist to present this work differently -- one prints to a
// terminal, the other returns JSON to a browser -- and nothing about *what the
// work is* should differ between them. Keeping that here is not tidiness: an
// earlier second copy of WriteActiveEnv had already drifted into sorting the
// file, dropping empty values, and re-quoting values that were fine as
// written, so the same profile produced a different environment depending on
// which interface switched to it.
package control

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/plugin"
	"github.com/coder/vllm-helper/internal/profile"
	"github.com/coder/vllm-helper/internal/provider"
)

// BaseURL is where the model server answers on this machine.
func BaseURL(cfg *config.Config) string {
	return fmt.Sprintf("http://localhost:%s/v1", cfg.Port())
}

// WriteActiveEnv assembles state/active.env from config.env + secrets.env +
// the profile, and records the active profile name.
//
// Lines are copied through exactly as written, which is the whole contract.
// The file is sourced by the service, so re-serialising it would mean deciding
// how to quote values someone else already quoted; a value with spaces that
// round-trips through a quoting function is not guaranteed to produce the same
// argv. Dropping empty values would be a second such decision -- set-to-empty
// and unset are not the same thing to a program reading its environment.
//
// Together this is the pair that changes what is served: WriteActiveEnv puts
// the settings in place, NotifyAndRestart makes them live. Call them in that
// order; the gap between them is where a caller does its own reporting.
func WriteActiveEnv(cfg *config.Config, p *profile.Profile) error {
	// An unknown provider would otherwise surface only when the service tried
	// to start it, by which time the outgoing model is already down.
	if _, err := provider.Lookup(p.Provider()); err != nil {
		return fmt.Errorf("profile '%s': %w", p.Name, err)
	}
	if err := os.MkdirAll(cfg.Dirs.State, 0o755); err != nil {
		return err
	}
	var b strings.Builder
	appendFile := func(f *config.EnvFile) {
		if f == nil {
			return
		}
		for _, line := range f.Lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" || strings.HasPrefix(trimmed, "#") {
				continue
			}
			b.WriteString(line + "\n")
		}
	}
	appendFile(cfg.File)
	appendFile(cfg.Secrets)
	appendFile(p.File)
	// config.env's PROVIDER is the default for new profiles, not for this
	// one: a profile that names no provider is vLLM's. Without this line the
	// default would leak in through config.env's lines above and serve an
	// old vLLM profile with another server. Only written when there is a
	// default to override, so an install that never set one gets the file
	// it always got.
	if p.Provider() == "" && cfg.Provider() != "" {
		b.WriteString(provider.Key + "=" + provider.VLLM + "\n")
	}

	// Written and renamed rather than truncated in place: a crash halfway
	// through must not leave the service with half an environment.
	tmp := cfg.Dirs.ActiveEnvFile() + ".tmp"
	if err := os.WriteFile(tmp, []byte(b.String()), 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, cfg.Dirs.ActiveEnvFile()); err != nil {
		return err
	}
	return os.WriteFile(cfg.Dirs.ActiveProfileFile(), []byte(p.Name+"\n"), 0o644)
}

// NotifyAndRestart tells the plugins what is happening and restarts the
// service. Plugin output goes to out, so a terminal caller can stream it and
// a browser caller can capture it.
//
// restart is supplied by the caller because the two have different ideas of
// what a restart may cost: the CLI blocks on it, while the panel bounds it so
// an HTTP request cannot hang on systemd.
//
// Plugins run before the restart deliberately. Coder's model entry should
// describe the model that is coming up by the time it is up -- doing it after
// would leave a window where the picker offers something the server no longer
// serves.
func NotifyAndRestart(ctx context.Context, cfg *config.Config, p *profile.Profile, out io.Writer, restart func() error) error {
	plugin.Dispatch(ctx, out, plugin.EventActivating, plugin.State{
		Config:  cfg,
		Name:    p.Name,
		Profile: p,
		BaseURL: BaseURL(cfg),
	})
	if err := restart(); err != nil {
		return fmt.Errorf("restarting service: %w", err)
	}
	return nil
}

// NotifyRemoved tells the plugins a profile is going away, so integrations can
// drop their copy of it. Called before the file is deleted, since plugins key
// off the name.
func NotifyRemoved(ctx context.Context, cfg *config.Config, name string, out io.Writer) {
	plugin.Dispatch(ctx, out, plugin.EventRemoved, plugin.State{
		Config:  cfg,
		Name:    name,
		BaseURL: BaseURL(cfg),
	})
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
func ProbeHealthURL(base string) (int, bool) {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(strings.TrimSuffix(base, "/") + "/health")
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

// ParseActiveEnv reads back the environment the service was last started with.
func ParseActiveEnv(cfg *config.Config) (*config.EnvFile, error) {
	return config.ParseEnvFile(cfg.Dirs.ActiveEnvFile())
}
