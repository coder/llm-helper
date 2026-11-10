package control

import (
	"errors"
	"fmt"
	"io/fs"
	"net"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/profile"
)

// Keys llm-helper adds to an instance file. They follow the profile's lines,
// so they win when the file is sourced or parsed.
const (
	keySince   = "INSTANCE_SINCE"
	keyMemUtil = "INSTANCE_MEM_UTIL"
	keyKVCache = "INSTANCE_KV_CACHE"
)

// InstanceOpts are the per-instance settings that are not in the profile.
type InstanceOpts struct {
	Port    string
	MemUtil string
	KVCache string
}

// Instance is one running model: a profile, its port and the environment its
// server starts with.
type Instance struct {
	Name  string
	Port  string
	Since time.Time
	Env   *config.EnvFile
}

func (i *Instance) Opts() InstanceOpts {
	return InstanceOpts{Port: i.Port, MemUtil: i.Env.Get(keyMemUtil), KVCache: i.Env.Get(keyKVCache)}
}

// BaseURL is where the instance's OpenAI-compatible API answers.
func (i *Instance) BaseURL() string { return "http://localhost:" + i.Port + "/v1" }

// HealthPath is the engine's health path, "" (meaning /health) when unknown.
func (i *Instance) HealthPath() string {
	e, err := engine.LookupEngine(i.Env.Get(engine.Key))
	if err != nil {
		return ""
	}
	return e.Health.Path
}

func (i *Instance) Probe() (int, bool) {
	return ProbeHealthPath("http://localhost:"+i.Port, i.HealthPath())
}

// WriteInstanceEnv assembles the instance file from config.env, secrets.env
// and the profile, followed by the instance's own settings. Lines are copied
// as written; the service sources the file.
func WriteInstanceEnv(cfg *config.Config, p *profile.Profile, o InstanceOpts) error {
	if err := profile.ValidInstanceName(p.Name); err != nil {
		return err
	}
	if _, err := engine.Lookup(p.Provider()); err != nil {
		return fmt.Errorf("profile '%s': %w", p.Name, err)
	}
	if o.Port == "" {
		return fmt.Errorf("instance '%s': no port", p.Name)
	}
	if err := os.MkdirAll(cfg.Dirs.InstancesDir(), 0o755); err != nil {
		return err
	}
	since := strconv.FormatInt(time.Now().Unix(), 10)
	if old, err := LoadInstance(cfg, p.Name); err == nil && old.Env.Get(keySince) != "" {
		since = old.Env.Get(keySince)
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
	// A profile that names no provider is vLLM's, whatever config.env's
	// default for new profiles says.
	if p.Provider() == "" && cfg.Provider() != "" {
		b.WriteString(engine.Key + "=" + engine.VLLM + "\n")
	}
	line := func(k, v string) { b.WriteString(k + "=" + config.Quote(v) + "\n") }
	line(keySince, since)
	if o.MemUtil != "" {
		line(keyMemUtil, o.MemUtil)
		line("GPU_MEMORY_UTILIZATION", o.MemUtil)
	}
	if o.KVCache != "" {
		line(keyKVCache, o.KVCache)
		line("KV_CACHE_MEMORY_BYTES", o.KVCache)
	}
	line("PORT", o.Port)

	path := cfg.Dirs.InstanceEnvFile(p.Name)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(b.String()), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadInstance reads one instance file. A missing file is fs.ErrNotExist.
func LoadInstance(cfg *config.Config, name string) (*Instance, error) {
	env, err := config.ParseEnvFile(cfg.Dirs.InstanceEnvFile(name))
	if err != nil {
		return nil, err
	}
	inst := &Instance{Name: name, Port: env.Get("PORT"), Env: env}
	if n, err := strconv.ParseInt(env.Get(keySince), 10, 64); err == nil {
		inst.Since = time.Unix(n, 0)
	}
	return inst, nil
}

// ListInstances returns every instance, oldest first.
func ListInstances(cfg *config.Config) ([]*Instance, error) {
	entries, err := os.ReadDir(cfg.Dirs.InstancesDir())
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []*Instance
	for _, e := range entries {
		name, ok := strings.CutSuffix(e.Name(), ".env")
		if !ok || e.IsDir() {
			continue
		}
		inst, err := LoadInstance(cfg, name)
		if err != nil {
			continue
		}
		out = append(out, inst)
	}
	sort.SliceStable(out, func(a, b int) bool {
		if !out[a].Since.Equal(out[b].Since) {
			return out[a].Since.Before(out[b].Since)
		}
		return out[a].Name < out[b].Name
	})
	return out, nil
}

func RemoveInstance(cfg *config.Config, name string) error {
	err := os.Remove(cfg.Dirs.InstanceEnvFile(name))
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	return err
}

// LiveProfiles names the profiles that have an instance, oldest first.
func LiveProfiles(cfg *config.Config) []string {
	list, _ := ListInstances(cfg)
	names := make([]string, len(list))
	for i, inst := range list {
		names[i] = inst.Name
	}
	return names
}

// DefaultInstance is the instance Coder should offer first: DEFAULT_PROFILE
// when it is running, the oldest instance otherwise.
func DefaultInstance(cfg *config.Config, live []string) string {
	if d := cfg.DefaultProfile(); d != "" {
		for _, n := range live {
			if n == d {
				return d
			}
		}
	}
	if len(live) > 0 {
		return live[0]
	}
	return ""
}

// portFree is swapped by tests.
var portFree = func(port string) bool {
	l, err := net.Listen("tcp", net.JoinHostPort("", port))
	if err != nil {
		return false
	}
	l.Close()
	return true
}

// AllocatePort picks the port for name: want when given, the port it already
// has, or the first port from PORT up that no other instance holds and
// nothing listens on.
func AllocatePort(cfg *config.Config, name, want string) (string, error) {
	taken := map[string]string{}
	list, err := ListInstances(cfg)
	if err != nil {
		return "", err
	}
	for _, inst := range list {
		if inst.Name != name {
			taken[inst.Port] = inst.Name
		}
	}
	if rp := cfg.RouterPort(); rp != "" {
		taken[rp] = "the router"
	}
	if want != "" {
		if _, err := strconv.Atoi(want); err != nil {
			return "", fmt.Errorf("port %q is not a number", want)
		}
		if owner, ok := taken[want]; ok {
			return "", fmt.Errorf("port %s is used by %s", want, owner)
		}
		return want, nil
	}
	for _, inst := range list {
		if inst.Name == name && inst.Port != "" {
			return inst.Port, nil
		}
	}
	start, err := strconv.Atoi(cfg.Port())
	if err != nil {
		return "", fmt.Errorf("PORT %q is not a number", cfg.Port())
	}
	for p := start; p < start+100 && p <= 65535; p++ {
		s := strconv.Itoa(p)
		if _, ok := taken[s]; ok {
			continue
		}
		if portFree(s) {
			return s, nil
		}
	}
	return "", fmt.Errorf("no free port in %d-%d", start, start+99)
}

// CheckRouterPort reports why port can't take the router, nil when it can.
func CheckRouterPort(cfg *config.Config, port string) error {
	if n, err := strconv.Atoi(port); err != nil || n < 1 || n > 65535 {
		return fmt.Errorf("port %q is not a valid port", port)
	}
	list, err := ListInstances(cfg)
	if err != nil {
		return err
	}
	for _, inst := range list {
		if inst.Port == port {
			return fmt.Errorf("port %s is used by %s", port, inst.Name)
		}
	}
	if port != cfg.RouterPort() && !portFree(port) {
		return fmt.Errorf("port %s is already in use", port)
	}
	return nil
}
