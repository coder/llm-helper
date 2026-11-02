// Package config resolves vllm-helper's directories (XDG layout, matching
// the bash implementation exactly) and loads the layered configuration:
// config.env + secrets.env + the active profile.
package config

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Dirs holds every path the tool reads or writes. Nothing is ever written
// into the checkout/install directory.
type Dirs struct {
	Config   string // ~/.config/vllm-helper
	State    string // ~/.local/state/vllm-helper
	Data     string // ~/.local/share/vllm-helper
	Profiles string // <Config>/models
}

func ResolveDirs() Dirs {
	home, _ := os.UserHomeDir()
	xdg := func(envOverride, xdgVar, fallback string) string {
		if v := os.Getenv(envOverride); v != "" {
			return v
		}
		base := os.Getenv(xdgVar)
		if base == "" {
			base = filepath.Join(home, fallback)
		}
		return filepath.Join(base, "vllm-helper")
	}
	d := Dirs{
		Config: xdg("VLLM_HELPER_CONFIG_DIR", "XDG_CONFIG_HOME", ".config"),
		State:  xdg("VLLM_HELPER_STATE_DIR", "XDG_STATE_HOME", filepath.Join(".local", "state")),
		Data:   xdg("VLLM_HELPER_DATA_DIR", "XDG_DATA_HOME", filepath.Join(".local", "share")),
	}
	d.Profiles = filepath.Join(d.Config, "models")
	return d
}

func (d Dirs) ConfigFile() string        { return filepath.Join(d.Config, "config.env") }
func (d Dirs) SecretsFile() string       { return filepath.Join(d.Config, "secrets.env") }
func (d Dirs) ActiveEnvFile() string     { return filepath.Join(d.State, "active.env") }
func (d Dirs) ActiveProfileFile() string { return filepath.Join(d.State, "active-profile") }
func (d Dirs) CoderIDsFile() string      { return filepath.Join(d.State, "coder-ids.env") }

// PluginStateFile is where an integration keeps its own state, namespaced by
// plugin name so two of them can't collide.
func (d Dirs) PluginStateFile(plugin string) string {
	return filepath.Join(d.State, plugin+"-ids.env")
}

// InstalledVersionFile records the version of the binary that last upgraded
// itself. Config and profiles outlive any one binary, so this is how a future
// release can tell what wrote what is already on disk.
func (d Dirs) InstalledVersionFile() string { return filepath.Join(d.State, "installed-version") }
func (d Dirs) ProfileFile(name string) string {
	return filepath.Join(d.Profiles, name+".env")
}

// Config is the loaded global configuration (config.env + secrets.env).
type Config struct {
	Dirs    Dirs
	File    *EnvFile // config.env
	Secrets *EnvFile // secrets.env (may be nil)
}

var ErrNotConfigured = errors.New("not configured")

// Load reads config.env (+ secrets.env if present). Returns
// ErrNotConfigured if config.env doesn't exist yet.
func Load() (*Config, error) {
	dirs := ResolveDirs()
	file, err := ParseEnvFile(dirs.ConfigFile())
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("%w: no config at %s -- run 'vllm-helper setup' first", ErrNotConfigured, dirs.ConfigFile())
	}
	if err != nil {
		return nil, err
	}
	cfg := &Config{Dirs: dirs, File: file}
	if secrets, err := ParseEnvFile(dirs.SecretsFile()); err == nil {
		cfg.Secrets = secrets
	}
	return cfg, nil
}

// get returns a value from secrets.env or config.env (secrets win), with a
// fallback default.
func (c *Config) get(key, def string) string {
	if c.Secrets != nil {
		if v := c.Secrets.Get(key); v != "" {
			return v
		}
	}
	if v := c.File.Get(key); v != "" {
		return v
	}
	return def
}

func (c *Config) Port() string         { return c.get("PORT", "8001") }
func (c *Config) VLLMBin() string      { return c.get("VLLM_BIN", "") }
func (c *Config) ServiceName() string  { return c.get("SERVICE_NAME", DefaultServiceName) }
func (c *Config) ServiceScope() string { return c.get("SERVICE_SCOPE", DefaultServiceScope) }

// Provider is the inference server that new profiles are made for, "" when
// unset. It is only a default: each profile records its own, and one that
// records none is vLLM's, as every profile made before providers existed is.
func (c *Config) Provider() string { return c.get("PROVIDER", "") }

// KVContextTarget is how many full-length requests a profile's KV cache is
// sized to hold. A property of the workload rather than the machine -- a
// single-user box wants fewer, a shared endpoint more -- so it is settable.
// 0 means unset, and callers apply their own default.
func (c *Config) KVContextTarget() int {
	n, err := strconv.Atoi(strings.TrimSpace(c.get("KV_CONTEXT_TARGET", "")))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// ModelBudgetGB is an override, empty when unset. The budget is a property of
// the machine -- one that changes when RAM is added or the desktop is turned
// off -- so the default is to measure it at each call rather than serve back a
// number frozen into config.env at install time.
func (c *Config) ModelBudgetGB() string { return c.get("MODEL_BUDGET_GB", "") }
func (c *Config) APIKey() string        { return c.get("VLLM_API_KEY", "") }

// UpdateChannel is where 'upgrade' looks for new builds: "stable" for tagged
// releases, "dev" for the build CI publishes from every commit on main.
func (c *Config) UpdateChannel() string { return c.get("UPDATE_CHANNEL", "stable") }

// GitHubToken is a token for reading releases. Only needed while the
// repository is private; public releases download without credentials.
func (c *Config) GitHubToken() string { return c.get("GITHUB_TOKEN", "") }

// HFToken returns a HuggingFace token from config/secrets ("" if unset).
// Needed only for gated or private models.
func (c *Config) HFToken() string { return c.get("HF_TOKEN", "") }

// Lookup returns any configuration key, from secrets.env or config.env
// (secrets win), or def when it is unset. Plugins use this for their own
// keys: the typed accessors below cover vllm-helper's own settings, and an
// integration shouldn't need a method here to read its credentials.
func (c *Config) Lookup(key, def string) string { return c.get(key, def) }

// HasAll reports whether every named key has a non-empty value. Integrations
// are configured all-or-nothing -- a URL without a token is not usable -- so
// this is how a plugin decides whether it is set up.
func (c *Config) HasAll(keys ...string) bool {
	for _, k := range keys {
		if c.get(k, "") == "" {
			return false
		}
	}
	return len(keys) > 0
}

func (c *Config) CoderURL() string        { return c.get("CODER_URL", "") }
func (c *Config) CoderToken() string      { return c.get("CODER_API_TOKEN", "") }
func (c *Config) CoderProviderID() string { return c.get("CODER_AI_PROVIDER_ID", "") }

// CoderOrganization names the organization the model entries hang off. Coder
// resolves the literal "default" server-side, which is right for the
// single-organization deployments that never had to know their own UUID --
// but a deployment with more than one has no default to fall back on, so it
// has to be sayable.
func (c *Config) CoderOrganization() string { return c.get("CODER_ORGANIZATION", "default") }
func (c *Config) CoderConfigured() bool {
	return c.CoderURL() != "" && c.CoderToken() != "" && c.CoderProviderID() != ""
}

// ActiveProfile returns the name of the currently active profile ("" if none).
func (c *Config) ActiveProfile() string {
	b, err := os.ReadFile(c.Dirs.ActiveProfileFile())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}
