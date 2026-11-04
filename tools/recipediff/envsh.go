package main

import (
	"regexp"
	"sort"
	"strings"
)

// envSh is a shell config file of KEY="${KEY:-default}" lines, such as a
// TensorFold launcher's scripts/config.sh. It is read, never run: only the
// assignments at the top level whose default is a literal, or names another
// such key, are understood. A default the script computes, or one set inside
// a block, is not a value anyone can compare against, so it is left out
// rather than guessed.
type envSh struct {
	// Values are the resolved defaults by key, exported or not.
	Values map[string]string
	// Exported are the keys the script exports, which reach the server's
	// environment; the rest configure the launcher.
	Exported map[string]bool
	// Text is the file itself, which the pinned models are looked up in.
	Text string
}

var (
	assignRE = regexp.MustCompile(`([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"`)
	// ${NAME:-x} and ${NAME-x}: x is what the key is when the environment
	// does not set it.
	defaultRE = regexp.MustCompile(`^\$\{[A-Za-z_][A-Za-z0-9_]*:?-(.*)\}$`)
	refRE     = regexp.MustCompile(`^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$`)
)

func parseEnvSh(b []byte) envSh {
	out := envSh{Values: map[string]string{}, Exported: map[string]bool{}, Text: string(b)}
	for _, line := range strings.Split(string(b), "\n") {
		// Indented lines are inside a function, a loop or a condition, whose
		// assignments may never run.
		if line == "" || line[0] == ' ' || line[0] == '\t' || line[0] == '#' {
			continue
		}
		rest, exported := strings.CutPrefix(line, "export ")
		// Only a run of assignments from the start of the line counts: one
		// after a [[ test ]] && is conditional, and one after a # is prose.
		end := 0
		for _, idx := range assignRE.FindAllStringSubmatchIndex(rest, -1) {
			if strings.Trim(rest[end:idx[0]], "; ") != "" {
				break
			}
			end = idx[1]
			m := []string{rest[idx[0]:idx[1]], rest[idx[2]:idx[3]], rest[idx[4]:idx[5]]}
			key := m[1]
			if strings.HasPrefix(key, "_") {
				continue
			}
			if v, ok := out.resolve(m[2]); ok {
				out.Values[key] = v
			} else {
				delete(out.Values, key)
			}
			if exported {
				out.Exported[key] = true
			}
		}
	}
	return out
}

// resolve reads one assignment's value: a literal, a default, or a reference
// to a key already read.
func (s envSh) resolve(v string) (string, bool) {
	for {
		m := defaultRE.FindStringSubmatch(v)
		if m == nil {
			break
		}
		v = m[1]
	}
	if m := refRE.FindStringSubmatch(v); m != nil {
		name := m[1] + m[2]
		val, ok := s.Values[name]
		return val, ok
	}
	if strings.ContainsAny(v, "$`") {
		return "", false
	}
	return v, true
}

// compareEnvSh reports where an entry disagrees with an env-sh upstream.
//
// The server's environment is compared key by key, both ways: what the
// script exports and the entry leaves out or changes, and what the entry
// sets that the script has no default for. A config key that is not
// exported configures upstream's launcher, which this tool cannot map onto
// the entry's own args, so only its declared upstream value is held to the
// file: a declaration that names a value upstream no longer has is stale.
func compareEnvSh(e entry, up envSh) []finding {
	declared := map[string]declaration{}
	byName := map[string]bool{}
	for _, d := range e.Source.Deviations {
		declared[d.Kind+" "+d.Setting] = declaration{upstream: d.Upstream, ours: d.Ours}
		byName[d.Setting] = true
	}
	var out []finding
	add := func(f finding) {
		switch d, ok := declared[f.Kind+" "+f.Setting]; {
		case ok:
			f.Declared = d.fits(f)
			f.Misdeclared = !f.Declared
		case byName[f.Setting]:
			f.Misdeclared = true
		}
		out = append(out, f)
	}

	for k := range up.Exported {
		upVal, known := up.Values[k]
		if !known {
			continue
		}
		ourVal, present := e.Env[k]
		switch {
		case !present:
			add(finding{Kind: "env", Setting: k, Label: k, Upstream: upVal})
		case ourVal != upVal:
			add(finding{Kind: "env", Setting: k, Label: k, Upstream: upVal, Ours: ourVal})
		}
	}
	for k, ourVal := range e.Env {
		if _, known := up.Values[k]; !known || !up.Exported[k] {
			add(finding{Kind: "env", Setting: k, Label: k, Ours: ourVal, Info: true})
		}
	}
	for _, d := range e.Source.Deviations {
		if d.Kind != "config" {
			continue
		}
		upVal, known := up.Values[d.Setting]
		f := finding{Kind: "config", Setting: d.Setting, Label: d.Setting, Upstream: upVal, Ours: d.Ours}
		if !known {
			f.Upstream = "(no literal default)"
		}
		f.Declared = known && upVal == d.Upstream
		f.Misdeclared = !f.Declared
		out = append(out, f)
	}
	sortFindings(out)
	return out
}

// missingModels are the entry's pinned models whose id or revision the
// upstream file does not mention: a sign the entry cites the wrong file, or
// that upstream moved its pins.
func (s envSh) missingModels(e entry) []string {
	var out []string
	for _, m := range e.Models {
		if !strings.Contains(s.Text, m.ID) || !strings.Contains(s.Text, m.Revision) {
			out = append(out, m.ID+"@"+m.Revision)
		}
	}
	return out
}

func sortFindings(out []finding) {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].Label < out[j].Label
	})
}

// diffEnvSh reports what upstream altered between the commit we pinned and
// its newest revision of the same file: the server's environment as env,
// the launcher's own keys as config.
func diffEnvSh(pinned, newest envSh) []change {
	split := func(s envSh) (env, config map[string]string) {
		env, config = map[string]string{}, map[string]string{}
		for k, v := range s.Values {
			if s.Exported[k] {
				env[k] = v
			} else {
				config[k] = v
			}
		}
		return env, config
	}
	pe, pc := split(pinned)
	ne, nc := split(newest)
	return append(diffMaps("config", pc, nc), diffMaps("env", pe, ne)...)
}
