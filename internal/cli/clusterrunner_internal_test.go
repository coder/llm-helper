package cli

import (
	"strconv"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/nodes"
)

// status reads this vocabulary, so a mistranslation here is what makes a
// healthy cluster look broken or a crash-looping one look idle.
func TestContainerStateSpeaksTheInitVocabulary(t *testing.T) {
	cases := []struct {
		status string
		code   int
		want   string
		why    string
	}{
		{"running", 0, "active", "serving"},
		{"created", 0, "activating", "starting, not yet up"},
		{"restarting", 0, "activating", "docker is bringing it back"},
		{"", 0, "inactive", "no container at all"},
		{"exited", 0, "inactive", "stopped on purpose is not a failure"},
		{"exited", 137, "failed", "SIGKILL, which is how an OOM shows up"},
		{"dead", 1, "failed", "docker could not clean it up after a fault"},
		{"paused", 0, "paused", "unknown states pass through rather than lie"},
	}
	for _, c := range cases {
		if got := containerState(c.status, c.code); got != c.want {
			t.Errorf("containerState(%q, %d) = %q, want %q (%s)", c.status, c.code, got, c.want, c.why)
		}
	}
}

// useRegistry points the node registry at a temp dir holding these nodes.
func useRegistry(t *testing.T, list ...nodes.Node) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("VLLM_HELPER_CONFIG_DIR", dir)
	reg, err := nodes.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range list {
		reg.Add(n)
	}
	if err := reg.Save(); err != nil {
		t.Fatal(err)
	}
}

// A recorded node list is not enough: a profile naming a node since removed
// from the registry passed the old check, so activation published it to
// Coder before the launch refused it.
func TestPreflightRefusesANodeTheRegistryLost(t *testing.T) {
	r := testRecipe(t)
	useRegistry(t, nodes.Node{Name: "a", Host: "http://127.0.0.1:8001"})
	cfg := writeProfile(t, "")
	c := &clusterRunner{recipe: r, opts: recipeLaunchOptions{nodeNames: []string{"a", "gone"}}}
	err := c.preflight(cfg)
	if err == nil || !strings.Contains(err.Error(), `"gone"`) {
		t.Fatalf("preflight = %v, want a refusal naming the missing node", err)
	}
}

// An override the hardware cannot run has to be refused before Coder hears
// about the switch, the same as a missing node.
func TestPreflightRefusesAFatalOverride(t *testing.T) {
	r := testRecipe(t)
	if _, ok := r.Target(); !ok {
		t.Skipf("%s records no verified hardware", r.ID)
	}
	useRegistry(t,
		nodes.Node{Name: "a", Host: "http://127.0.0.1:8001"},
		nodes.Node{Name: "b", Host: "http://10.0.0.2:8001"})
	cfg := writeProfile(t, "MODEL="+r.Model+"\nEXTRA_ARGS=\"--moe-backend deep_gemm_mega_moe\"\nRECIPE="+r.ID+"\n")
	c := &clusterRunner{recipe: r, opts: recipeLaunchOptions{nodeNames: []string{"a", "b"}}}
	if err := c.preflight(cfg); err == nil {
		t.Fatal("preflight accepted an override GB10 is known to abort on")
	}
}

// The watchdog probes this URL, and the head can be another machine: asking
// localhost about a remote head reported a healthy cluster as down and had
// the watchdog relaunch it every few minutes.
func TestEndpointURLFollowsTheHead(t *testing.T) {
	r := testRecipe(t)
	port := strconv.Itoa(r.Port)
	useRegistry(t,
		nodes.Node{Name: "here", Host: "http://127.0.0.1:8001"},
		nodes.Node{Name: "there", Host: "http://10.99.0.7:8001", SSH: "serve@10.99.0.7"},
		// An ~/.ssh/config alias resolves for ssh only; the endpoint is HTTP.
		nodes.Node{Name: "aliased", Host: "http://10.99.0.8:8001", SSH: "gpu-worker"})
	for _, c := range []struct {
		head, want string
	}{
		{"here", "http://localhost:" + port},
		{"there", "http://10.99.0.7:" + port},
		{"aliased", "http://10.99.0.8:" + port},
	} {
		run := &clusterRunner{recipe: r, opts: recipeLaunchOptions{nodeNames: []string{c.head, "here"}}}
		got, err := run.endpointURL()
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Errorf("head %s: endpoint = %s, want %s", c.head, got, c.want)
		}
	}
}

// Switching runners has to stop the one being left: the incoming runner only
// replaces its own containers or unit, so a recipe kept the GPUs after a
// switch to a native profile and vice versa.
func TestOutgoingNeedsStop(t *testing.T) {
	cases := []struct {
		name     string
		out, in  string
		cluster  bool
		state    string
		wantStop bool
	}{
		{"same unit restarts itself", "native:vllm.service", "native:vllm.service", false, "active", false},
		{"same cluster replaces its own containers", "cluster:vllm-node@a,b", "cluster:vllm-node@a,b", true, "", false},
		{"native to recipe stops the unit", "native:vllm.service", "cluster:vllm-node@a,b", false, "active", true},
		{"a loading unit is still holding the GPUs", "native:vllm.service", "cluster:vllm-node@a,b", false, "activating", true},
		{"an idle unit is left alone", "native:vllm.service", "cluster:vllm-node@a,b", false, "inactive", false},
		// Workers can outlive a crashed head, so the head's state says
		// nothing about whether there is anything to stop.
		{"recipe to native always stops the cluster", "cluster:vllm-node@a,b", "native:vllm.service", true, "", true},
		{"a recipe on other nodes stops the old one", "cluster:vllm-node@a,b", "cluster:vllm-node@c,d", true, "", true},
	}
	for _, c := range cases {
		if got := outgoingNeedsStop(c.out, c.in, c.cluster, c.state); got != c.wantStop {
			t.Errorf("%s: outgoingNeedsStop = %v, want %v", c.name, got, c.wantStop)
		}
	}
}

// A restart relaunches the recipe the profile already describes. It used to
// print the --no-activate warning that the profile and Coder "still describe
// whatever ran before", which is false there; the warning stays for a user who
// actually passed --no-activate.
func TestRestartDoesNotWarnAboutSkippedActivation(t *testing.T) {
	c := &clusterRunner{opts: recipeLaunchOptions{nodeNames: []string{"a"}}}
	o := c.restartOpts()
	if !o.skipCoder {
		t.Fatal("restart must not re-sync the profile and Coder")
	}
	if note := skippedActivationNote(o); note != "" {
		t.Errorf("restart printed %q", note)
	}
	if note := skippedActivationNote(recipeLaunchOptions{skipCoder: true}); note == "" {
		t.Error("--no-activate lost its warning")
	}
}

// Each node's own recorded architecture is held to the resolved args, and a
// node that matched different hardware than the recipe was verified on is
// named rather than silently treated as tested.
func TestCheckRecipeMembersUsesEachNodesHardware(t *testing.T) {
	r := testRecipe(t)
	want := r.HardwareID()
	if want == "" {
		t.Skipf("%s records no verified hardware", r.ID)
	}
	same := recipeMember{node: nodes.Node{Name: "a", Caps: nodes.Capabilities{HardwareID: want, Arch: "sm121"}}}
	other := recipeMember{node: nodes.Node{Name: "b", Caps: nodes.Capabilities{HardwareID: "nvidia_grace_hopper", Arch: "sm90"}}}
	var w strings.Builder
	if err := checkRecipeMembers(&w, r, []recipeMember{same, other}, r.ServeArgs()); err != nil {
		t.Fatalf("checkRecipeMembers = %v, want the shipped args allowed", err)
	}
	if !strings.Contains(w.String(), "node b is nvidia_grace_hopper") || strings.Contains(w.String(), "node a") {
		t.Fatalf("warnings = %q, want only node b named", w.String())
	}
	fatal := append(r.ServeArgs(), "--moe-backend", "deep_gemm_mega_moe")
	err := checkRecipeMembers(&w, r, []recipeMember{other, same}, fatal)
	if err == nil || !strings.Contains(err.Error(), "node a") {
		t.Fatalf("checkRecipeMembers = %v, want a refusal naming node a", err)
	}
}
