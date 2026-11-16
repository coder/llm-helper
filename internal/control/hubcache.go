package control

import (
	"context"
	"os"
	"os/exec"
	"time"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/hfcache"
	"github.com/coder/llm-helper/internal/resolve"
)

// HubCache is the hub cache directory the server reads. The service sources
// config.env and secrets.env but not a login shell, so only those count.
func HubCache(cfg *config.Config) string {
	home, _ := os.UserHomeDir()
	var extra []string
	if p, err := engine.LookupEngine(cfg.Lookup(engine.Key, "")); err == nil && p.HubCacheVar() != "" {
		extra = append(extra, p.HubCacheVar())
	}
	return hfcache.Dir(func(k string) string { return cfg.Lookup(k, "") }, home, extra...)
}

// HubOffline reports whether HF_HUB_OFFLINE is on for the server, or for this
// process, which is what a command run by hand sees.
func HubOffline(cfg *config.Config) bool {
	return hfcache.Offline(func(k string) string { return cfg.Lookup(k, "") }) ||
		hfcache.Offline(hfcache.OSEnv)
}

// KeepOffline sets a new profile up to load from the cache when the Hub is
// not being used, asking the installed server what it accepts.
func KeepOffline(ctx context.Context, cfg *config.Config, prov engine.Provider, plan *resolve.Plan) {
	if hf.Offline() {
		plan.SetOffline(prov, func(args []string) string { return probeServer(ctx, cfg, prov, args) })
	}
}

// probeServer runs the installed server with args and returns what it
// printed, "" when it cannot be run.
func probeServer(ctx context.Context, cfg *config.Config, prov engine.Provider, args []string) string {
	bin := cfg.Lookup(prov.BinKey(), "")
	if bin == "" {
		bin, _ = exec.LookPath(prov.BinName())
	}
	if bin == "" {
		return ""
	}
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	out, _ := exec.CommandContext(ctx, bin, args...).CombinedOutput()
	return string(out)
}
