// Package engine describes the inference servers a profile or recipe can
// run on, each defined as data in its own file under catalogue/engines/.
//
// An engine only decides how a profile becomes a command line: which
// executable, which flags, which environment, and what that server can be
// trusted to do. The rest (profiles, switching, the service unit, the health
// probe, Coder sync) is the same for every engine, which is what keeps adding
// one a matter of writing a definition. A machine owner can add engines
// without a rebuild by dropping a file into engines.d in the config
// directory; see UseLocal.
//
// The data is parsed strictly. An unknown key, a duplicate id, or a
// placeholder a template cannot fill is an error, so a typo fails when the
// definition loads rather than reaching the server as an empty argument.
package engine

import (
	"fmt"
	"os"
	"strings"
)

// Key is the config.env and profile setting that picks a provider. A profile's
// value wins over config.env's, because an instance's env lists the profile last.
const Key = "PROVIDER"

// Ids of shipped engines that Go code names: vLLM is the default, and the
// others are what setup and the tests refer to.
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
	// Params set the engine's own params by name; any left unset take the
	// engine's default.
	Params map[string]string
}

// Provider is one inference server. Engine is the only implementation; the
// interface keeps callers from depending on the definition's layout.
type Provider interface {
	// Name is the PROVIDER value that selects it.
	Name() string
	// BinKey is the config key holding the full path to the server
	// executable, and BinName is that executable's name on PATH.
	BinKey() string
	BinName() string
	Capabilities() Capabilities
	// HybridNote is what inspect says about serving a hybrid model, if
	// anything.
	HybridNote() string
	// Has reports whether the engine can run in a runtime.
	Has(runtime string) bool
	// InstallHint says how to get the server on an OS, by GOOS.
	InstallHint(goos string) string
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

// Lookup returns the provider a PROVIDER value names. Empty means vLLM, which
// is what every install made before providers existed is running.
func Lookup(name string) (Provider, error) {
	e, err := LookupEngine(name)
	if err != nil {
		return nil, err
	}
	return e, nil
}

// LookupEngine is Lookup returning the definition itself, for the callers
// that render containers or list engines.
func LookupEngine(name string) (Engine, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		name = VLLM
	}
	e, ok := Default().Get(name)
	if !ok {
		return Engine{}, fmt.Errorf("unknown %s %q: use one of %s", Key, name, strings.Join(Names(), ", "))
	}
	return e, nil
}

// Names lists the engines, sorted.
func Names() []string {
	var out []string
	for _, e := range Default().Engines() {
		out = append(out, e.ID)
	}
	return out
}

// ProcessNames lists the engines that can run on the host, which are the
// ones a profile's PROVIDER can name for the service unit, sorted.
func ProcessNames() []string {
	var out []string
	for _, e := range Default().Engines() {
		if e.Has(RuntimeProcess) {
			out = append(out, e.ID)
		}
	}
	return out
}

// Validate reports what would stop s being served, naming the setting to fix.
func Validate(p Provider, s Spec) error {
	if !p.Has(RuntimeProcess) {
		return fmt.Errorf("%s runs only in a container, so the service cannot serve it; launch it with 'llm-helper recipes launch'", p.Name())
	}
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
