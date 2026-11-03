package recipe

import (
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/coder/vllm-helper/internal/hardware"
	"github.com/coder/vllm-helper/internal/provider"
)

// Placeholders a recipe may reference in its env values. They are discovered on
// each node at launch time rather than stored in the recipe: an interface name
// or address is a property of the machine, not of the model.
const (
	PlaceholderIface  = "IFACE_NAME"
	PlaceholderIBIf   = "IB_IF"
	PlaceholderNodeIP = "NODE_IP"
)

// Placeholders returns the placeholder names this recipe references, sorted, so
// a launcher knows what it has to discover before it can expand the env.
func (r TestedRecipe) Placeholders() []string {
	seen := map[string]bool{}
	for _, v := range r.Env {
		if name, ok := placeholderName(v); ok {
			seen[name] = true
		}
	}
	out := make([]string, 0, len(seen))
	for n := range seen {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// NodeEnv resolves the recipe env for one node. Only a whole value of the form
// "$NAME" is a placeholder; there is no general interpolation, so an unset or
// misspelled name fails here rather than reaching vLLM as an empty string and
// silently falling back to the wrong interface.
func (r TestedRecipe) NodeEnv(discovered map[string]string) (map[string]string, error) {
	out := make(map[string]string, len(r.Env))
	for k, v := range r.Env {
		name, ok := placeholderName(v)
		if !ok {
			out[k] = v
			continue
		}
		got := discovered[name]
		if got == "" {
			return nil, fmt.Errorf("%s=$%s but %s was not discovered on this node", k, name, name)
		}
		out[k] = got
	}
	return out, nil
}

func placeholderName(v string) (string, bool) {
	if !strings.HasPrefix(v, "$") || strings.ContainsAny(v, " \t") {
		return "", false
	}
	return strings.TrimPrefix(v, "$"), true
}

// ServeArgs is the recipe's own args, which the engine's role templates
// place, with nothing overridden. It needs no arch check: the shipped args
// are held to that invariant by TestTestedRecipesHoldTheArchInvariant.
func (r TestedRecipe) ServeArgs() []string {
	args := append([]string{}, r.Args...)
	// Host and port are the launcher's business, not the model's, which is why
	// the recipe carries "port" as a field instead of inside args. The engine
	// says how its server takes them.
	return dedupeArgs(append(args, r.Engine().LaunchArgs(r.Port, r.Params)...))
}

// ServeArgsWith layers local overrides on top of the recipe, last-wins per
// flag. This is the precedence vLLM itself documents for a command line over a
// config file: the recipe is the config layer, an override is the command line.
// Honouring an override is what keeps the server and what Coder advertises in
// agreement -- refusing the edit only moved the disagreement somewhere else.
//
// Honouring is not the same as trusting: the merged argv is checked against
// the flags known to abort on the recipe's hardware, so an override cannot
// carry an SM100-only kernel onto a GB10. Those flags are vLLM's, so only an
// engine that takes vLLM's args is checked.
func (r TestedRecipe) ServeArgsWith(overrides []string) ([]string, error) {
	merged := dedupeArgs(append(r.ServeArgs(), overrides...))
	if !r.Engine().Capabilities().VLLMArgs {
		return merged, nil
	}
	if t, ok := r.Target(); ok {
		if err := checkForbidden(merged, t); err != nil {
			return nil, fmt.Errorf("%s: %w", r.ID, err)
		}
	}
	return merged, nil
}

// Target is the hardware a recipe was verified on, when that hardware has
// arch rules. The verified hardware is the only thing a recipe says about
// where it runs, and launching it elsewhere is not something it claims.
//
// Verified.Hardware starts with a hardware definition id, optionally followed
// by a count ("dgx_spark_gb10 x2").
func (r TestedRecipe) Target() (Target, bool) {
	reg := hardware.Default()
	d, ok := reg.ByID(r.HardwareID())
	if !ok || d.Arch == "" {
		return Target{}, false
	}
	a, ok := reg.Architecture(d.Arch)
	if !ok {
		return Target{}, false
	}
	return Target{Hardware: d.ID, Generation: a.Generation, Arch: a.Arch}, true
}

// HardwareID is the hardware definition id a recipe requires, or else the
// one it records as the hardware it was verified on, or "" when it records
// none.
func (r TestedRecipe) HardwareID() string {
	if r.Requires != "" {
		return r.Requires
	}
	if f := strings.Fields(r.Verified.Hardware); len(f) > 0 {
		return f[0]
	}
	return ""
}

// MergeArgs layers argument lists last-wins per flag, the same precedence a
// launch applies, so callers can ask what a set of layers resolves to.
func MergeArgs(layers ...[]string) []string {
	var all []string
	for _, l := range layers {
		all = append(all, l...)
	}
	return dedupeArgs(all)
}

// FlagValue reports whether a flag is present in an arg list and what value it
// carries, so a caller can tell an override ("this disagrees with the recipe")
// from a restatement. A flag that takes no value reports an empty one.
func FlagValue(args []string, flag string) (string, bool) {
	for _, p := range splitArgs(args) {
		if p.flag == flag {
			return p.value, true
		}
	}
	return "", false
}

// StripRestatements drops overrides that say exactly what the base already
// says. The merged result is the same either way, so this only affects the
// claim -- and a profile that repeats the recipe verbatim has not moved away
// from what we measured. The first launcher wrote the recipe's own flags into
// EXTRA_ARGS, so real profiles do repeat it verbatim.
//
// Overrides are resolved last-wins first. Comparing each token on its own
// would drop a final restatement and let an earlier, overridden value take
// effect: "--max-num-seqs 16 --max-num-seqs 8" over a base of 8 means 8.
func StripRestatements(base, overrides []string) []string {
	var out []string
	for _, p := range splitArgs(dedupeArgs(overrides)) {
		if v, ok := FlagValue(base, p.flag); ok && v == p.value {
			continue
		}
		out = append(out, p.render()...)
	}
	return out
}

// OverrideFlags names the flags a set of overrides touches, for reporting how
// far a launch has moved from the configuration we measured.
func OverrideFlags(overrides []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range splitArgs(overrides) {
		if p.flag == "" || seen[p.flag] {
			continue
		}
		seen[p.flag] = true
		out = append(out, p.flag)
	}
	sort.Strings(out)
	return out
}

// Rank is what a launch knows about one node of a recipe.
type Rank struct {
	Rank       int
	MasterAddr string
	MasterPort int
	// Discovered fills the recipe env's $NAME placeholders on this node.
	Discovered map[string]string
	// HomeDir is the node user's home, which the engine's mounts are under.
	HomeDir string
	// ServeArgs are the recipe's args with local overrides merged.
	ServeArgs []string
	// APIKey is passed only if the engine's container checks one.
	APIKey string
	// Container is the container's name on the node.
	Container string
}

// RankSpec is the container one rank runs: the engine's role for that rank,
// with the recipe's args, env, params and pinned models filled in. Rank 0
// serves the API; what the others run is the engine's to say.
func (r TestedRecipe) RankSpec(k Rank) (ContainerSpec, error) {
	e := r.Engine()
	role, err := e.RoleFor(k.Rank)
	if err != nil {
		return ContainerSpec{}, err
	}
	entry, args, env := e.RenderRole(role, provider.RoleValues{
		Model: r.Model, Port: r.Port, Rank: k.Rank, Nodes: r.Nodes,
		MasterAddr: k.MasterAddr, MasterPort: k.MasterPort,
		Args: k.ServeArgs, Params: r.Params, Models: r.ModelRefs(),
	})
	nodeEnv, err := r.NodeEnv(k.Discovered)
	if err != nil {
		return ContainerSpec{}, err
	}
	// The recipe's env wins over the engine's: the recipe is what was
	// measured.
	for key, v := range nodeEnv {
		env[key] = v
	}
	mounts, err := e.RenderMounts(k.HomeDir, r.Params)
	if err != nil {
		return ContainerSpec{}, err
	}
	spec := ContainerSpec{
		Name: k.Container, Image: r.Image, Entrypoint: entry,
		Flags: e.Container.Flags, Mounts: mounts, Env: env, Args: args,
	}
	if keyEnv := e.ContainerKeyEnv(); keyEnv != "" && !role.NoAPIKey {
		spec.KeyEnv, spec.APIKey = keyEnv, k.APIKey
	}
	return spec, nil
}

// LaunchOrder is the order the ranks start in. Workers usually have to be
// listening before the head starts its rendezvous.
func (r TestedRecipe) LaunchOrder() []int {
	out := make([]int, r.Nodes)
	for i := range out {
		out[i] = i
	}
	if c := r.Engine().Container; c == nil || c.LaunchOrder == "workers_first" {
		slices.Reverse(out)
	}
	return out
}

// ContextLimit is the context window the recipe's own args set, through the
// engine's context flag, or else the context it was verified at, or "" when
// it says neither.
func (r TestedRecipe) ContextLimit() string {
	if c := r.Engine().Container; c != nil && c.ContextFlag != "" {
		if v, ok := FlagValue(r.Args, c.ContextFlag); ok && v != "" {
			return v
		}
	}
	if r.Verified.Context > 0 {
		return strconv.Itoa(r.Verified.Context)
	}
	return ""
}

// HealthPath is where the head answers once it serves.
func (r TestedRecipe) HealthPath() string { return r.Engine().Health.Path }

// StagingScript reports which of the recipe's pinned models are missing from
// a node's Hugging Face cache, one repo@revision per line, so a launch can
// refuse before it stops anything. Downloading on every launch would be a
// surprise of hundreds of gigabytes; the hint says how to do it once.
func (r TestedRecipe) StagingScript() string {
	cache := r.Engine().HFCacheHostPath(r.Params)
	if cache == "" || len(r.Models) == 0 {
		return ""
	}
	var b strings.Builder
	for _, m := range r.Models {
		dir := provider.SnapshotPath(cache, provider.ModelRef{ID: m.ID, Revision: m.Revision})
		fmt.Fprintf(&b, "[ -d \"$HOME\"/%s ] || echo %s\n", ShellQuote(dir), ShellQuote(m.ID+"@"+m.Revision))
	}
	return b.String()
}

// NeedsCluster reports whether the recipe must be launched across more than one
// node, so a launcher can refuse early instead of half-starting a shard.
func (r TestedRecipe) NeedsCluster() bool { return r.Topology == "shard" && r.Nodes > 1 }
