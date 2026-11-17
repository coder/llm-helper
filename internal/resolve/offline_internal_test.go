package resolve

import (
	"slices"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/engine"
)

func offlinePlan(t *testing.T, id, extra string, probeOut string) (Plan, []string) {
	t.Helper()
	prov, err := engine.Lookup(id)
	if err != nil {
		t.Fatal(err)
	}
	p := Plan{}
	p.Opts.ExtraArgs = extra
	var probed []string
	p.SetOffline(prov, func(args []string) string {
		probed = args
		return probeOut
	})
	return p, probed
}

func TestSetOfflineUsesWhatEachEngineDeclares(t *testing.T) {
	for _, id := range engine.ProcessNames() {
		prov, _ := engine.Lookup(id)
		off := prov.OfflineSetting()
		if off == nil {
			continue
		}
		help := strings.Join(off.Args, " ") + " listed"
		p, _ := offlinePlan(t, id, "--port 1", help)
		for _, line := range off.EnvLines() {
			if !slices.Contains(p.Opts.Env, line) {
				t.Errorf("%s: env %q missing from %v", id, line, p.Opts.Env)
			}
		}
		for _, a := range off.Args {
			if !strings.Contains(p.Opts.ExtraArgs, a) {
				t.Errorf("%s: %q missing from %q", id, a, p.Opts.ExtraArgs)
			}
		}
		if !strings.HasPrefix(p.Opts.ExtraArgs, "--port 1") {
			t.Errorf("%s: existing args lost: %q", id, p.Opts.ExtraArgs)
		}
	}
}

func TestSetOfflineSetsHFHubOfflineForPythonEngines(t *testing.T) {
	for _, id := range []string{"vllm", "mlx"} {
		p, probed := offlinePlan(t, id, "", "")
		if !slices.Contains(p.Opts.Env, "HF_HUB_OFFLINE=1") {
			t.Errorf("%s: env = %v", id, p.Opts.Env)
		}
		if p.Opts.ExtraArgs != "" || probed != nil {
			t.Errorf("%s: args %q probed %v", id, p.Opts.ExtraArgs, probed)
		}
	}
}

func TestSetOfflineAddsTheFlagOnlyWhenTheServerListsIt(t *testing.T) {
	p, probed := offlinePlan(t, "llamacpp", "-ngl 99", "  --offline   work offline\n  --port N")
	if p.Opts.ExtraArgs != "-ngl 99 --offline" || !slices.Equal(probed, []string{"--help"}) {
		t.Errorf("args %q probed %v", p.Opts.ExtraArgs, probed)
	}

	p, _ = offlinePlan(t, "llamacpp", "-ngl 99", "  --port N")
	if p.Opts.ExtraArgs != "-ngl 99" {
		t.Errorf("flag added to a server that does not list it: %q", p.Opts.ExtraArgs)
	}
	if len(p.Notes) != 1 || !strings.Contains(p.Notes[0], "--offline") {
		t.Errorf("notes = %v", p.Notes)
	}
}

func TestSetOfflineDoesNotRepeatWhatIsThere(t *testing.T) {
	p, _ := offlinePlan(t, "llamacpp", "--offline", "--offline")
	if p.Opts.ExtraArgs != "--offline" {
		t.Errorf("args = %q", p.Opts.ExtraArgs)
	}
	prov, _ := engine.Lookup("vllm")
	q := Plan{}
	q.Opts.Env = []string{"HF_HUB_OFFLINE=0"}
	q.SetOffline(prov, nil)
	if len(q.Opts.Env) != 1 || q.Opts.Env[0] != "HF_HUB_OFFLINE=0" {
		t.Errorf("env = %v; a setting already in the profile must win", q.Opts.Env)
	}
}
