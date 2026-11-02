// Package provider describes the inference servers a profile can run on.
//
// A provider only decides how a profile becomes a command line: which
// executable, which flags, and what that server can be trusted to do. The rest
// (profiles, switching, the service unit, the health probe, Coder sync) is the
// same for every provider, which is what keeps adding one small.
package provider

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// Key is the config.env and profile setting that picks a provider. A profile's
// value wins over config.env's, because active.env lists the profile last.
const Key = "PROVIDER"

// Names of the providers that exist.
const (
	VLLM     = "vllm"
	MLX      = "mlx"
	LlamaCPP = "llamacpp"
)

// Spec is the part of a profile, merged over config.env, that a provider turns
// into a command line.
type Spec struct {
	Bin            string
	Model          string
	Port           string
	APIKey         string
	MaxModelLen    string
	MemUtilization string
	KVCacheBytes   string
	ExtraArgs      string
}

// Capabilities are the differences between providers that the rest of the
// tool has to act on.
type Capabilities struct {
	// APIKey reports whether the server enforces VLLM_API_KEY itself. Without
	// it, anything that can reach the port can use the endpoint.
	APIKey bool
	// VLLMArgs reports whether a profile's settings are vllm serve's. Only
	// then do model-card flags, parser validation against the installed vLLM,
	// KV cache sizing and the memory-utilization fit apply.
	VLLMArgs bool
	// GGUFQuant reports whether the server loads one GGUF file out of a repo
	// that holds several, named by appending its quant to the model:
	// "org/model-GGUF:Q8_0". Only then does choosing a quant mean anything.
	GGUFQuant bool
}

// Provider is one inference server.
type Provider interface {
	// Name is the PROVIDER value that selects it.
	Name() string
	// BinKey is the config key holding the full path to the server
	// executable, and BinName is that executable's name on PATH.
	BinKey() string
	BinName() string
	Capabilities() Capabilities
	// Command is the argv that serves s, starting with s.Bin. It assumes s
	// passed Validate.
	Command(s Spec) ([]string, error)
}

// EnvProvider is a Provider that takes some settings from the server's
// environment instead of its command line. A secret belongs there: any user
// on the machine can read another process's argv with ps, but not its
// environment.
type EnvProvider interface {
	Provider
	// Env is the KEY=value entries that serve s, added to the server's
	// environment over any inherited entry of the same name.
	Env(s Spec) []string
}

var registry = map[string]Provider{
	VLLM:     vllm{},
	MLX:      mlx{},
	LlamaCPP: llamaCPP{},
}

// Lookup returns the provider a PROVIDER value names. Empty means vLLM, which
// is what every install made before providers existed is running.
func Lookup(name string) (Provider, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		name = VLLM
	}
	p, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown %s %q: use one of %s", Key, name, strings.Join(Names(), ", "))
	}
	return p, nil
}

// Names lists the providers, sorted.
func Names() []string {
	out := make([]string, 0, len(registry))
	for n := range registry {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Validate reports what would stop s being served, naming the setting to fix.
func Validate(p Provider, s Spec) error {
	if s.Bin == "" {
		return fmt.Errorf("%s is not set in config.env", p.BinKey())
	}
	if !Executable(s.Bin) {
		return fmt.Errorf("%s (%s) does not exist or is not executable", p.BinKey(), s.Bin)
	}
	if s.Model == "" {
		return fmt.Errorf("MODEL is not set")
	}
	if s.Port == "" {
		return fmt.Errorf("PORT is not set")
	}
	return nil
}

// Executable reports whether path names a file with an execute bit set.
func Executable(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	return err == nil && info.Mode()&0o111 != 0
}
