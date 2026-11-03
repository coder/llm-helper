package recipe

import (
	"fmt"
	"strings"

	"github.com/coder/vllm-helper/internal/hardware"
)

// Target is the machine a recipe is resolved for. The hardware definitions
// supply these; the catalog never infers them.
type Target struct {
	Hardware   string // exact hardware definition id, e.g. "dgx_spark_gb10"
	Generation string // e.g. "blackwell"
	Arch       string // e.g. "sm121", whose forbidden args are checked
}

type argPair struct {
	flag, value string
	hasValue    bool
	inline      bool // written as --flag=value, which must round-trip
}

// splitArgs groups a flat arg list into flags and their values. Values are
// either inline (--flag=v) or the following token.
func splitArgs(in []string) []argPair {
	var out []argPair
	for i := 0; i < len(in); i++ {
		tok := in[i]
		if !strings.HasPrefix(tok, "-") {
			out = append(out, argPair{flag: tok})
			continue
		}
		if k, val, ok := strings.Cut(tok, "="); ok {
			out = append(out, argPair{flag: k, value: val, hasValue: true, inline: true})
			continue
		}
		if i+1 < len(in) && !strings.HasPrefix(in[i+1], "--") {
			out = append(out, argPair{flag: tok, value: in[i+1], hasValue: true})
			i++
			continue
		}
		out = append(out, argPair{flag: tok})
	}
	return out
}

// render turns a pair back into argv tokens, keeping --flag=value inline.
func (p argPair) render() []string {
	switch {
	case p.inline:
		return []string{p.flag + "=" + p.value}
	case p.hasValue:
		return []string{p.flag, p.value}
	default:
		return []string{p.flag}
	}
}

// dedupeArgs keeps the last occurrence of each flag, at its last position.
func dedupeArgs(in []string) []string {
	pairs := splitArgs(in)
	last := make(map[string]int, len(pairs))
	for i, p := range pairs {
		last[p.flag] = i
	}
	var out []string
	for i, p := range pairs {
		if last[p.flag] != i {
			continue
		}
		out = append(out, p.render()...)
	}
	return out
}

// checkForbidden fails closed if resolved args carry a flag the target's
// architecture is known to abort on. The list lives with the architecture in
// the hardware definitions, so every machine built on it inherits it.
func checkForbidden(args []string, t Target) error {
	return CheckForbidden(args, t.Arch)
}

// CheckForbidden fails if args carry a flag and value that the architecture
// arch (for example "sm121") is known to abort on. An unknown or empty arch
// has no rules, so it passes: the check only refuses what is known to crash.
// Anything that builds a vLLM command line for a known machine, not only a
// recipe launch, should run it.
func CheckForbidden(args []string, arch string) error {
	if arch == "" {
		return nil
	}
	a, ok := hardware.Default().Architecture(arch)
	if !ok || len(a.ForbiddenArgs) == 0 {
		return nil
	}
	for _, p := range splitArgs(args) {
		for _, f := range a.ForbiddenArgs {
			if p.flag == f.Flag && p.value == f.Value {
				return fmt.Errorf("the args contain %s %s, which is unsupported on %s", f.Flag, f.Value, arch)
			}
		}
	}
	return nil
}
