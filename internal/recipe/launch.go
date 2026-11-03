package recipe

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/coder/vllm-helper/internal/hardware"
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

// ServeArgs is the argv after "vllm serve <model>", with nothing overridden.
// It needs no arch check: the shipped args are held to that invariant by
// TestTestedRecipesHoldTheArchInvariant.
func (r TestedRecipe) ServeArgs() []string {
	args := append([]string{}, r.Args...)
	// Host and port are the launcher's business, not the model's, which is why
	// the recipe carries "port" as a field instead of inside args.
	return dedupeArgs(append(args, "--host", "0.0.0.0", "--port", strconv.Itoa(r.Port)))
}

// ServeArgsWith layers local overrides on top of the recipe, last-wins per
// flag. This is the precedence vLLM itself documents for a command line over a
// config file: the recipe is the config layer, an override is the command line.
// Honouring an override is what keeps the server and what Coder advertises in
// agreement -- refusing the edit only moved the disagreement somewhere else.
//
// Honouring is not the same as trusting: the merged argv is checked against
// the flags known to abort on the recipe's hardware, so an override cannot
// carry an SM100-only kernel onto a GB10.
func (r TestedRecipe) ServeArgsWith(overrides []string) ([]string, error) {
	merged := dedupeArgs(append(r.ServeArgs(), overrides...))
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

// HardwareID is the hardware definition id a recipe records as the hardware it was
// verified on, or "" when it records none.
func (r TestedRecipe) HardwareID() string {
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

// DistArgs adds the multi-node flags for one rank. Rank 0 is the head that
// serves the API; every other rank runs headless.
func (r TestedRecipe) DistArgs(rank int, masterAddr string, masterPort int) []string {
	args := []string{
		"--nnodes", strconv.Itoa(r.Nodes),
		"--node-rank", strconv.Itoa(rank),
		"--master-addr", masterAddr,
		"--master-port", strconv.Itoa(masterPort),
	}
	if rank != 0 {
		args = append(args, "--headless")
	}
	return args
}

// NeedsCluster reports whether the recipe must be launched across more than one
// node, so a launcher can refuse early instead of half-starting a shard.
func (r TestedRecipe) NeedsCluster() bool { return r.Topology == "shard" && r.Nodes > 1 }
