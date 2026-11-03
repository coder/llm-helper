package main

import (
	"sort"
	"strings"
)

// entry mirrors the fields of a tested recipe this tool compares. It is a
// local copy rather than an import so the tool stays a separate module with
// its own dependency, and so a schema change here cannot break the CLI.
type entry struct {
	ID     string            `json:"id"`
	Model  string            `json:"model"`
	Args   []string          `json:"args"`
	Env    map[string]string `json:"env"`
	Models []struct {
		ID       string `json:"id"`
		Revision string `json:"revision"`
	} `json:"models"`
	Source struct {
		Repo       string      `json:"repo"`
		Path       string      `json:"path"`
		Commit     string      `json:"commit"`
		Format     string      `json:"format"`
		Deviations []deviation `json:"deviations"`
	} `json:"source"`
}

// deviation mirrors recipe.Deviation: what an entry claims about one setting.
type deviation struct {
	Kind     string `json:"kind"`
	Setting  string `json:"setting"`
	Upstream string `json:"upstream"`
	Ours     string `json:"ours"`
	Why      string `json:"why"`
}

type tested struct {
	Recipes []entry `json:"recipes"`
}

// finding is one place an entry and its upstream disagree.
type finding struct {
	Kind        string // flag, env, mod or config
	Setting     string // the deviation key this would need to declare
	Label       string // how it reads in the report
	Upstream    string
	Ours        string
	Declared    bool
	Misdeclared bool // declared, but the declaration describes a different move
	Info        bool // reported, but not a conflict
}

// maxLiteralValue is how long a value may be before a declaration is allowed
// to summarise it. Upstream carries whole JSON blobs for --speculative-config
// and --reasoning-config, which no one can usefully restate verbatim; a short
// value like 8 or auto has no such excuse and must be stated literally.
const maxLiteralValue = 32

// declaration is what an entry claims about one setting.
type declaration struct {
	upstream string
	ours     string
}

// fits reports whether a declaration actually describes the finding it
// silences. Keying on the setting name alone made every declaration
// unfalsifiable: the values were parsed and never read, so a stale claim went
// on hiding a difference that had since moved.
func (d declaration) fits(f finding) bool {
	// The direction must agree whatever the values are. A drop cannot be
	// declared as a change, nor a change as a drop.
	if (d.upstream == "") != (f.Upstream == "") {
		return false
	}
	if (d.ours == "") != (f.Ours == "") {
		return false
	}
	// Our own value is ours to state exactly; the args test beside it holds
	// the same claim to the same standard.
	if d.ours != f.Ours {
		return false
	}
	if len(f.Upstream) > maxLiteralValue {
		return true
	}
	return d.upstream == f.Upstream
}

// compare reports every disagreement. It never resolves one: a difference we
// meant to make must be declared in source.deviations, and anything else is
// something nobody has looked at yet.
func compare(e entry, up eugrRecipe) []finding {
	// Keyed on the kind as well as the name, because the two namespaces
	// overlap in spirit if not in spelling: an env key and a flag are
	// different settings even when they describe the same behaviour.
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
			// Declared under the wrong kind, which is how the env keys were
			// written before the kind existed. Say so rather than reporting
			// it as something nobody has looked at.
			f.Misdeclared = true
		}
		out = append(out, f)
	}

	ours := argFlags(e.Args)
	upFlags := up.Flags()
	// The launcher supplies these, so their absence from an entry is by
	// construction rather than a decision anyone made.
	supplied := map[string]bool{"--host": true, "--port": true}

	for name, upVal := range upFlags {
		if supplied[name] {
			continue
		}
		ourVal, present := ours[name]
		switch {
		case !present:
			add(finding{Kind: "flag", Setting: settingOf(name), Label: name, Upstream: upVal, Ours: ""})
		case ourVal != upVal:
			add(finding{Kind: "flag", Setting: settingOf(name), Label: name, Upstream: upVal, Ours: ourVal})
		}
	}
	for name, ourVal := range ours {
		if _, present := upFlags[name]; !present {
			add(finding{Kind: "flag", Setting: settingOf(name), Label: name, Upstream: "", Ours: ourVal})
		}
	}

	for _, m := range up.Mods {
		add(finding{Kind: "mod", Setting: m, Label: m, Upstream: "applied", Ours: ""})
	}

	for k, upVal := range up.Env {
		ourVal, present := e.Env[k]
		switch {
		case !present:
			add(finding{Kind: "env", Setting: k, Label: k, Upstream: upVal, Ours: ""})
		case ourVal != upVal:
			add(finding{Kind: "env", Setting: k, Label: k, Upstream: upVal, Ours: ourVal})
		}
	}
	// Env we add is cluster plumbing that upstream's runner sets outside the
	// recipe, so it is reported but does not count as disagreement.
	for k, ourVal := range e.Env {
		if _, present := up.Env[k]; !present {
			add(finding{Kind: "env", Setting: k, Label: k, Upstream: "", Ours: ourVal, Info: true})
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].Label < out[j].Label
	})
	return out
}

// settingOf is the deviation key for a flag: --max-num-seqs is max_num_seqs,
// which is also what upstream calls it in defaults.
func settingOf(flag string) string {
	return strings.ReplaceAll(strings.TrimPrefix(flag, "--"), "-", "_")
}

// argFlags reads our argv the same way the upstream template is read.
func argFlags(args []string) map[string]string {
	out := map[string]string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !strings.HasPrefix(a, "--") {
			continue
		}
		if name, value, ok := strings.Cut(a, "="); ok {
			out[name] = value
			continue
		}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
			out[a] = args[i+1]
			i++
			continue
		}
		out[a] = ""
	}
	return out
}

// change is one setting that moved between two revisions of the same upstream
// file. This is the "have we fallen behind" signal: we once ran DeepSeek for a
// week without speculative decoding because nothing watched for it.
type change struct {
	Kind string
	Key  string
	Was  string
	Now  string
}

// diffUpstream reports what upstream altered between the commit we pinned and
// its newest revision of the same file.
func diffUpstream(pinned, newest eugrRecipe) []change {
	mods := func(list []string) map[string]string {
		m := map[string]string{}
		for _, v := range list {
			m[v] = "applied"
		}
		return m
	}
	var out []change
	out = append(out, diffMaps("flag", pinned.Flags(), newest.Flags())...)
	out = append(out, diffMaps("env", pinned.Env, newest.Env)...)
	out = append(out, diffMaps("mod", mods(pinned.Mods), mods(newest.Mods))...)
	return out
}

// diffMaps reports the keys of one kind that were removed, added or changed.
func diffMaps(kind string, was, now map[string]string) []change {
	keys := func(m map[string]string) []string {
		var ks []string
		for k := range m {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		return ks
	}
	var out []change
	for _, k := range keys(was) {
		if n, ok := now[k]; !ok {
			out = append(out, change{kind, k, was[k], "(removed)"})
		} else if n != was[k] {
			out = append(out, change{kind, k, was[k], n})
		}
	}
	for _, k := range keys(now) {
		if _, ok := was[k]; !ok {
			out = append(out, change{kind, k, "(absent)", now[k]})
		}
	}
	return out
}

// undeclared counts the findings that need a decision.
func undeclared(fs []finding) int {
	n := 0
	for _, f := range fs {
		if !f.Declared && !f.Info {
			n++
		}
	}
	return n
}
