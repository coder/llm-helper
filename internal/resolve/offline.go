package resolve

import (
	"fmt"
	"slices"
	"strings"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/engine"
)

// SetOffline keeps the profile's server off the network, the way its engine
// declares: environment lines, and arguments the installed server is asked
// about first. probe runs the server with the given arguments and returns its
// output.
func (p *Plan) SetOffline(prov engine.Provider, probe func(args []string) string) {
	off := prov.OfflineSetting()
	if off == nil {
		p.Notes = append(p.Notes, fmt.Sprintf("%s has no offline setting; set what it needs in the profile.", prov.Name()))
		return
	}
	for _, line := range off.EnvLines() {
		key, _, _ := strings.Cut(line, "=")
		if !slices.ContainsFunc(p.Opts.Env, func(l string) bool { return strings.HasPrefix(l, key+"=") }) {
			p.Opts.Env = append(p.Opts.Env, line)
		}
	}
	if len(off.Args) == 0 {
		return
	}
	var out string
	if len(off.ProbeArgs) > 0 && probe != nil {
		out = probe(off.ProbeArgs)
	}
	if !off.ArgsAccepted(out) {
		p.Notes = append(p.Notes, fmt.Sprintf("The installed %s does not list %s, so it was not added; a newer build can load a cached model offline.", prov.Name(), off.Args[0]))
		return
	}
	if words, err := config.SplitArgs(p.Opts.ExtraArgs); err == nil && slices.Contains(words, off.Args[0]) {
		return
	}
	p.Opts.ExtraArgs = strings.TrimSpace(p.Opts.ExtraArgs + " " + strings.Join(off.Args, " "))
}
