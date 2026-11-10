package cli

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hardware"
	"github.com/coder/llm-helper/internal/nodes"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/recipe"
)

// discoverScript reports the cluster interface, both RoCE devices and this
// node's address on that interface. It runs on the node rather than being
// configured because these are properties of the machine; hardcoding them
// would bake one deployment into a shipped tool.
const discoverScript = `set -u
iface="${VH_IFACE:-}"
if [ -z "$iface" ]; then
  iface=$(ibdev2netdev 2>/dev/null | awk '$NF=="(Up)"{print $5; exit}')
fi
ibif=$(ibdev2netdev 2>/dev/null | awk '$NF=="(Up)"{printf "%s,", $1}' | sed 's/,$//')
ip=$(ip -o -4 addr show dev "$iface" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
printf 'IFACE_NAME=%s\nIB_IF=%s\nNODE_IP=%s\nHOME_DIR=%s\n' "$iface" "$ibif" "$ip" "$HOME"
`

type recipeLaunchOptions struct {
	nodeNames  []string
	iface      string
	masterPort int
	container  string
	sshKey     string
	dryRun     bool
	skipCoder  bool
	// relaunch marks a restart of an existing instance, not --no-activate.
	relaunch bool
}

func newRecipesLaunchCmd() *cobra.Command {
	var (
		o recipeLaunchOptions
		w waitFlags
	)
	c := &cobra.Command{
		Use:   "launch <id>",
		Short: "Launch a tested recipe as an instance, and tell Coder about it",
		Long: `Launch a tested recipe as an instance, and tell Coder about it.

Like up and restart, it then watches the load until the head node answers
its engine's health check: following the log on a terminal, waiting quietly
otherwise. Pass
--no-follow to return as soon as the containers have started.

To bring a running instance back up, use "llm-helper restart <profile>":
it drives whichever runner that instance belongs to.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, a []string) error {
			if err := runRecipeLaunch(cmd.Context(), a[0], o); err != nil || o.dryRun {
				return err
			}
			return waitForLaunch(a[0], o, w.opts())
		},
	}
	w.register(c)
	f := c.Flags()
	f.StringSliceVar(&o.nodeNames, "nodes", nil, "registry node names in rank order; rank 0 serves the API")
	f.StringVar(&o.iface, "iface", "", "cluster interface to use instead of the discovered one")
	f.IntVar(&o.masterPort, "master-port", defaultRecipeMasterPort, "rendezvous port for the ranks")
	f.StringVar(&o.container, "container-name", defaultRecipeContainer, "container name on each node")
	f.StringVar(&o.sshKey, "ssh-key", "", "private key for reaching non-head nodes")
	f.BoolVar(&o.dryRun, "dry-run", false, "print what would run and change nothing")
	f.BoolVar(&o.skipCoder, "no-activate", false, "launch only; record no instance and leave Coder alone")
	return c
}

func newRecipesStopCmd() *cobra.Command {
	var o recipeLaunchOptions
	c := &cobra.Command{
		Use:   "stop <id>",
		Short: "Stop the containers a tested recipe started",
		Args:  cobra.ExactArgs(1),
		RunE:  func(cmd *cobra.Command, a []string) error { return runRecipeStop(a[0], o) },
	}
	f := c.Flags()
	f.StringSliceVar(&o.nodeNames, "nodes", nil, "registry node names")
	f.StringVar(&o.container, "container-name", defaultRecipeContainer, "container name on each node")
	f.StringVar(&o.sshKey, "ssh-key", "", "private key for reaching non-head nodes")
	f.BoolVar(&o.dryRun, "dry-run", false, "print what would run and change nothing")
	return c
}

// Defaults shared by the launch flags and the cluster runner, so a runner
// built from a profile that predates those keys lands on the same values the
// launch would have used.
const (
	defaultRecipeContainer  = "vllm-node"
	defaultRecipeMasterPort = 29501
)

// recipeMember is one node plus what we discovered about it.
type recipeMember struct {
	node       nodes.Node
	rank       int
	discovered map[string]string
}

func resolveRecipeMembers(r recipe.TestedRecipe, o recipeLaunchOptions) ([]recipeMember, error) {
	reg, err := nodes.Load()
	if err != nil {
		return nil, err
	}
	names := o.nodeNames
	if len(names) == 0 {
		return nil, fmt.Errorf("--nodes is required: %s needs %d node(s), and rank order matters", r.ID, r.Nodes)
	}
	if len(names) != r.Nodes {
		return nil, fmt.Errorf("%s is verified on %d node(s) but %d were given; a tested recipe is only tested at its own topology", r.ID, r.Nodes, len(names))
	}
	out := make([]recipeMember, 0, len(names))
	for i, name := range names {
		n, ok := reg.Get(name)
		if !ok {
			return nil, fmt.Errorf("no node %q in the registry; add it with 'llm-helper node add'", name)
		}
		out = append(out, recipeMember{node: n, rank: i})
	}
	return out, nil
}

// checkRecipeMembers holds the resolved args to each node's own hardware, not
// only to the hardware the recipe was verified on: a node whose recorded
// architecture is known to abort on one of the flags is refused. A node that
// matched a different hardware definition than the recipe was verified on is
// allowed, since a recipe is evidence and not a whitelist, but it is named on
// w so nobody mistakes the launch for a tested one.
//
// The forbidden flags are vLLM's, so only an engine that takes vLLM's args is
// held to them. An engine that names the GPU vendors it runs on refuses a node
// whose recorded hardware is another vendor's.
func checkRecipeMembers(w io.Writer, r recipe.TestedRecipe, members []recipeMember, args []string) error {
	want := r.HardwareID()
	engine := r.Engine()
	for _, m := range members {
		c := m.node.Caps
		if engine.Capabilities().VLLMArgs {
			if err := recipe.CheckForbidden(args, c.Arch); err != nil {
				return fmt.Errorf("%s on node %s: %w", r.ID, m.node.Name, err)
			}
		}
		if v := nodeVendor(m.node); v != "" && len(engine.GPUVendors) > 0 && !slices.Contains(engine.GPUVendors, v) {
			return fmt.Errorf("%s runs on %s, which supports %s GPUs, but node %s is %s", r.ID, engine.ID, strings.Join(engine.GPUVendors, ", "), m.node.Name, v)
		}
		if want != "" && c.HardwareID != "" && c.HardwareID != want {
			fmt.Fprintf(w, "warning: node %s is %s, but %s was verified on %s\n", m.node.Name, c.HardwareID, r.ID, r.Verified.Hardware)
		}
	}
	return nil
}

// nodeVendor is the GPU vendor of the hardware definition a node matched, or
// "" when it matched none.
func nodeVendor(n nodes.Node) string {
	d, ok := hardware.Default().ByID(n.Caps.HardwareID)
	if !ok {
		return ""
	}
	return d.Match.Vendor
}

// nodeCommand builds the command that runs a script on a member: directly
// when the node is this machine, over ssh with the script on stdin otherwise.
// Locality is a property of the node, not of its rank -- assuming the head was
// wherever the CLI happened to run meant a container with the same name on
// another node could be inspected and reported as the head's.
func nodeCommand(m recipeMember, key, script string, env []string) (*exec.Cmd, error) {
	var cmd *exec.Cmd
	if m.node.IsLocal() {
		cmd = exec.Command("bash", "-s")
	} else {
		target, err := sshTarget(m.node)
		if err != nil {
			return nil, err
		}
		args := []string{"-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"}
		if key != "" {
			args = append(args, "-i", key)
		}
		args = append(args, target, "bash", "-s")
		cmd = exec.Command("ssh", args...)
	}
	cmd.Stdin = strings.NewReader(script + "\n")
	cmd.Env = append(os.Environ(), env...)
	return cmd, nil
}

// nodeScript runs a script on a member and returns stdout. Callers that need
// to stream and kill the command build it with nodeCommand instead.
func nodeScript(m recipeMember, key, script string, env []string) (string, error) {
	cmd, err := nodeCommand(m, key, script, env)
	if err != nil {
		return "", err
	}
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return out.String(), fmt.Errorf("%s: %w: %s", m.node.Name, err, strings.TrimSpace(errb.String()))
	}
	return out.String(), nil
}

func parseKeyValues(s string) map[string]string {
	out := map[string]string{}
	for _, line := range strings.Split(s, "\n") {
		k, v, ok := strings.Cut(strings.TrimSpace(line), "=")
		if ok && k != "" {
			out[k] = v
		}
	}
	return out
}

func runRecipeLaunch(ctx context.Context, id string, o recipeLaunchOptions) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	tested, err := recipe.LoadTested()
	if err != nil {
		return err
	}
	r, ok := tested.Get(id)
	if !ok {
		return fmt.Errorf("no tested recipe %q", id)
	}
	// Any real launch is someone asking for the cluster to run, so it lifts a
	// deliberate stop before anything on the nodes can fail. Lifting it only on
	// success would let a relaunch that failed early -- a worker still booting
	// -- leave the unit's ExecStop marker in place and the watchdog off for good.
	if !o.dryRun {
		clearStopped(cfg, r.ID)
	}
	if r.Image == "" {
		return fmt.Errorf("%s pins no image, so there is nothing to launch", r.ID)
	}
	local, err := recipeOverrides(cfg, r)
	if err != nil {
		return err
	}
	// Resolve the argv before touching any node, so an override the hardware
	// cannot run is refused while whatever is serving is still serving.
	serveArgs, err := r.ServeArgsWith(local.args)
	if err != nil {
		return err
	}
	members, err := resolveRecipeMembers(r, o)
	if err != nil {
		return err
	}
	if err := checkRecipeMembers(os.Stderr, r, members, serveArgs); err != nil {
		return err
	}
	if err := checkRecipeClaim(cfg, r, o); err != nil {
		return err
	}
	head := members[0].node.Name

	var denv []string
	if o.iface != "" {
		denv = append(denv, "VH_IFACE="+o.iface)
	}
	for i := range members {
		envScript := discoverScript
		if o.iface != "" {
			envScript = "VH_IFACE=" + recipe.ShellQuote(o.iface) + "\n" + discoverScript
		}
		out, err := nodeScript(members[i], o.sshKey, envScript, denv)
		if err != nil {
			return fmt.Errorf("discover on %s: %w", members[i].node.Name, err)
		}
		members[i].discovered = parseKeyValues(out)
		for _, need := range r.Placeholders() {
			if members[i].discovered[need] == "" {
				return fmt.Errorf("%s: could not discover %s (recipe %s needs it); pass --iface if autodiscovery is wrong", members[i].node.Name, need, r.ID)
			}
		}
	}
	if err := checkStaged(r, members, o.sshKey); err != nil {
		return err
	}

	masterAddr := members[0].discovered[recipe.PlaceholderNodeIP]
	if masterAddr == "" {
		return fmt.Errorf("head node %s has no address on the cluster interface", head)
	}

	engine := r.Engine()
	specs := make([]recipe.ContainerSpec, len(members))
	roles := make([]string, len(members))
	for i, m := range members {
		spec, err := r.RankSpec(recipe.Rank{
			Rank:       m.rank,
			MasterAddr: masterAddr,
			MasterPort: o.masterPort,
			Discovered: m.discovered,
			HomeDir:    m.discovered["HOME_DIR"],
			ServeArgs:  serveArgs,
			APIKey:     cfg.APIKey(),
			Container:  o.container,
		})
		if err != nil {
			return fmt.Errorf("%s: %w", m.node.Name, err)
		}
		specs[i] = spec
		role, err := engine.RoleFor(m.rank)
		if err != nil {
			return err
		}
		roles[i] = role.Name
		if role.ServesHTTP && m.rank == 0 {
			roles[i] += ", serves the API"
		}
	}

	fmt.Printf("%s  %s\n", r.DisplayName, r.Summary())
	fmt.Printf("  engine     %s\n", engine.ID)
	fmt.Printf("  image      %s\n", r.Image)
	fmt.Printf("  master     %s:%d\n", masterAddr, o.masterPort)
	if flags := recipe.OverrideFlags(local.args); len(flags) > 0 {
		fmt.Println("  overrides  " + strings.Join(flags, " ") + " from profile " + local.profile.Name)
		fmt.Println("  claim      recipe plus local overrides -- the verified numbers no longer describe this launch")
	}
	switch {
	case engine.ContainerKeyEnv() == "":
		fmt.Println("  auth       none -- " + engine.ID + " checks no API key, so front it with a proxy that does, or keep it on loopback")
	case cfg.APIKey() == "":
		fmt.Println("  auth       none -- VLLM_API_KEY is unset, so anyone who can reach the port can use it")
	default:
		fmt.Println("  auth       VLLM_API_KEY from secrets.env")
	}
	for i, m := range members {
		fmt.Printf("  rank %d     %s (%s) iface=%s ib=%s ip=%s\n", m.rank, m.node.Name, roles[i],
			m.discovered[recipe.PlaceholderIface], m.discovered[recipe.PlaceholderIBIf], m.discovered[recipe.PlaceholderNodeIP])
	}
	fmt.Println()

	if o.dryRun {
		for i, m := range members {
			cmdline, err := specs[i].DisplayCommand()
			if err != nil {
				return err
			}
			fmt.Printf("# %s (rank %d)\n%s\n\n", m.node.Name, m.rank, cmdline)
		}
		fmt.Println("dry run: nothing was started and no profile or Coder state changed")
		return nil
	}

	// The engine says the order the ranks start in: usually workers first,
	// since they must be listening before the head starts its rendezvous.
	//
	// Stand the watchdog off for the length of the load: the endpoint is
	// deliberately down while a model loads, and a timer that reads that as
	// an outage will relaunch the previous profile over the top of this one.
	// Written only now, so a launch refused above, or a dry run, does not
	// silence it.
	markLaunching(cfg, r.ID, launchStandOff)
	for _, i := range r.LaunchOrder() {
		m := members[i]
		if _, err := nodeScript(m, o.sshKey, "docker rm -f "+recipe.ShellQuote(o.container)+" >/dev/null 2>&1 || true", nil); err != nil {
			return fmt.Errorf("clear old container on %s: %w", m.node.Name, err)
		}
		cmdline, err := specs[i].ShellCommand()
		if err != nil {
			return err
		}
		if _, err := nodeScript(m, o.sshKey, cmdline, nil); err != nil {
			return fmt.Errorf("start rank %d on %s: %w", m.rank, m.node.Name, err)
		}
		fmt.Printf("started rank %d on %s\n", m.rank, m.node.Name)
	}

	if o.skipCoder {
		if note := skippedActivationNote(o); note != "" {
			fmt.Println(note)
		}
		return nil
	}
	return activateRecipeProfile(ctx, cfg, r, local, o)
}

// waitForLaunch watches a launch until its endpoint answers, as up and restart
// do. The runner is built from the launch itself, so a --no-activate launch is
// watched too.
func waitForLaunch(id string, o recipeLaunchOptions, w watchOpts) error {
	if !w.follow && !w.wait {
		fmt.Println(notWaitingNote(o))
		return nil
	}
	tested, err := recipe.LoadTested()
	if err != nil {
		return err
	}
	r, ok := tested.Get(id)
	if !ok {
		return fmt.Errorf("no tested recipe %q", id)
	}
	t, err := bootTargetFor(&clusterRunner{recipe: r, opts: o}, nil)
	if err != nil {
		return err
	}
	fmt.Println()
	return watchTarget(t, r.ID, w)
}

// notWaitingNote says where to look after a --no-follow launch.
func notWaitingNote(o recipeLaunchOptions) string {
	if o.skipCoder {
		return "Not waiting for the model to load. No instance records this launch, so follow it with 'docker logs -f " + o.container + "' on the head node."
	}
	return "Not waiting for the model to load. Check progress with 'llm-helper status'."
}

// skippedActivationNote is what a --no-activate launch tells the user.
func skippedActivationNote(o recipeLaunchOptions) string {
	if o.relaunch {
		return ""
	}
	return "skipped recording an instance, so status and Coder do not list this launch"
}

// recipeOverrides reads the local override layer for a recipe: the profile of
// the same name. Editing that profile is allowed, and has to reach argv -- the
// Coder sync already reads MAX_MODEL_LEN and DISPLAY_NAME from it, so an
// ignored edit would advertise one server while another one runs.
//
// The knobs are vllm serve's flags, so an engine that does not take vLLM's
// args is reached through EXTRA_ARGS and its context flag only.
func recipeOverrides(cfg *config.Config, r recipe.TestedRecipe) (overrideLayer, error) {
	if _, err := os.Stat(cfg.Dirs.ProfileFile(r.ID)); err != nil {
		return overrideLayer{}, nil
	}
	p, err := profile.Load(cfg.Dirs, r.ID)
	if err != nil {
		return overrideLayer{}, err
	}
	if m := strings.TrimSpace(p.Model()); m != "" && m != r.Model {
		return overrideLayer{}, fmt.Errorf("profile %s names model %s but recipe %s serves %s; launch runs the recipe's model, so keep a different model in its own profile", p.Name, m, r.ID, r.Model)
	}
	var knobs []string
	knobFlags := []struct{ key, flag string }{
		{"MAX_MODEL_LEN", "--max-model-len"},
		{"GPU_MEMORY_UTILIZATION", "--gpu-memory-utilization"},
		{"KV_CACHE_MEMORY_BYTES", "--kv-cache-memory-bytes"},
	}
	engine := r.Engine()
	if !engine.Capabilities().VLLMArgs {
		knobFlags = nil
		if c := engine.Container; c != nil && c.ContextFlag != "" {
			knobFlags = append(knobFlags, struct{ key, flag string }{"MAX_MODEL_LEN", c.ContextFlag})
		}
	}
	for _, k := range knobFlags {
		if v := strings.TrimSpace(p.File.Get(k.key)); v != "" {
			knobs = append(knobs, k.flag, v)
		}
	}
	var extra []string
	if raw := strings.TrimSpace(p.ExtraArgs()); raw != "" {
		words, err := config.SplitArgs(raw)
		if err != nil {
			return overrideLayer{}, fmt.Errorf("profile %s EXTRA_ARGS: %w", p.Name, err)
		}
		extra = words
	}
	// Only a disagreement is an override: a profile that restates the recipe
	// -- which the first launcher wrote out in full -- is still untouched.
	//
	// Knobs and EXTRA_ARGS land in one argv, EXTRA_ARGS last, so they are
	// resolved together: an EXTRA_ARGS value that happens to match the recipe
	// still overrides a knob, and stripping it on its own would hand the
	// launch the knob's value instead. For the same reason the EXTRA_ARGS
	// written back is compared against the recipe with the knobs applied: a
	// token is only dropped when the next launch would not change without it.
	base := r.ServeArgs()
	extraOnly := recipe.StripRestatements(recipe.MergeArgs(base, knobs), extra)
	layer := overrideLayer{
		profile: p,
		args:    recipe.StripRestatements(base, append(append([]string{}, knobs...), extra...)),
		extra:   extraOnly,
		pruned:  len(extraOnly) < len(extra),
	}
	return layer, nil
}

// overrideLayer is what the local profile contributes on top of a recipe.
// extra is kept apart from args because it is what gets written back to
// EXTRA_ARGS: a token that restates the recipe carries no information, and
// leaving it there would silently pin this profile to today's recipe if the
// recipe's value ever changed.
type overrideLayer struct {
	profile *profile.Profile
	args    []string
	extra   []string
	pruned  bool
}

// recipeForProfile finds the recipe a generated profile was launched from.
func recipeForProfile(p *profile.Profile) (recipe.TestedRecipe, bool) {
	if !p.Generated() {
		return recipe.TestedRecipe{}, false
	}
	t, err := recipe.LoadTested()
	if err != nil {
		return recipe.TestedRecipe{}, false
	}
	return t.Get(p.Recipe())
}

// overrideNote describes how far the local profile has moved from the recipe,
// for the places that would otherwise present the verified numbers as if they
// still described this machine. Best effort: no config means nothing to say.
func overrideNote(r recipe.TestedRecipe) string {
	cfg, err := loadConfig()
	if err != nil {
		return ""
	}
	local, err := recipeOverrides(cfg, r)
	if err != nil || local.profile == nil || len(local.args) == 0 {
		return ""
	}
	return "profile " + local.profile.Name + " overrides " + strings.Join(recipe.OverrideFlags(local.args), " ")
}

// recipeFlag reads a value out of the recipe's own flags, so a generated
// profile starts out agreeing with the recipe instead of quietly overriding it.
func recipeFlag(r recipe.TestedRecipe, flag, fallback string) string {
	if v, ok := recipe.FlagValue(r.Args, flag); ok && v != "" {
		return v
	}
	return fallback
}

// newRecipeProfile writes the profile a recipe's first launch is recorded
// as. It names the engine whenever that is not vLLM.
func newRecipeProfile(dirs config.Dirs, r recipe.TestedRecipe, limit string) (*profile.Profile, error) {
	opts := profile.CreateOpts{
		Repo:           r.Model,
		Name:           r.ID,
		MaxModelLen:    limit,
		MemUtilization: recipeFlag(r, "--gpu-memory-utilization", ""),
		Status:         "tested recipe " + r.ID + ", runs in a container",
		Source:         "recipes launch",
	}
	// Another engine's profile carries no vLLM sizing keys, only the
	// context, which its context flag takes.
	e := r.Engine()
	if !e.Capabilities().VLLMArgs {
		opts.Provider = e.ID
	}
	p, err := profile.Create(dirs, opts)
	if err != nil {
		return nil, err
	}
	if opts.Provider != "" {
		p.File.Set("MAX_MODEL_LEN", limit)
	}
	// An engine that takes vLLM's args keeps the sizing keys, and is still
	// named.
	if e.ID != engine.VLLM {
		p.File.Set(engine.Key, e.ID)
	}
	return p, nil
}

// checkRecipeClaim refuses a launch whose port or container another instance
// owns, before anything on the nodes is removed.
func checkRecipeClaim(cfg *config.Config, r recipe.TestedRecipe, o recipeLaunchOptions) error {
	if _, err := control.AllocatePort(cfg, r.ID, strconv.Itoa(r.Port)); err != nil {
		return err
	}
	list, err := control.ListInstances(cfg)
	if err != nil {
		return err
	}
	for _, inst := range list {
		if inst.Name == r.ID {
			continue
		}
		c := clusterRunnerForName(cfg, inst.Name)
		if c == nil || c.opts.container != o.container {
			continue
		}
		for _, n := range o.nodeNames {
			if slices.Contains(c.opts.nodeNames, n) {
				return fmt.Errorf("container %s on %s belongs to %s; stop it with '%s down %s' or pass --container-name", o.container, n, inst.Name, cmdName, inst.Name)
			}
		}
	}
	return nil
}

// activateRecipeProfile records the launch as a profile and an instance and
// tells Coder, so all three describe what is actually serving.
func activateRecipeProfile(ctx context.Context, cfg *config.Config, r recipe.TestedRecipe, local overrideLayer, o recipeLaunchOptions) error {
	// Coder advertises the context the server is configured to serve.
	limit := r.ContextLimit()
	if limit == "" {
		return fmt.Errorf("%s sets no context limit in its args and records no verified context, so there is no context limit to advertise", r.ID)
	}
	p := local.profile
	// An existing profile is updated in place, never replaced: its knobs and
	// EXTRA_ARGS are the override layer this launch just honoured, so
	// overwriting them would discard the user's edit after acting on it.
	if p == nil {
		created, err := newRecipeProfile(cfg.Dirs, r, limit)
		if err != nil {
			return err
		}
		p = created
	}
	p.File.Set(profile.RecipeKey, r.ID)
	// Record how this launch reached the nodes, so status, logs and restart can
	// act on the same cluster without being handed the topology again.
	p.File.Set(profile.NodesKey, strings.Join(o.nodeNames, ","))
	p.File.Set(profile.ContainerKey, o.container)
	p.File.Set(profile.MasterPortKey, strconv.Itoa(o.masterPort))
	if o.sshKey != "" {
		p.File.Set(profile.SSHKeyKey, o.sshKey)
	}
	// Written back with restatements removed: those tokens change nothing now,
	// and left in place they would override a future recipe value.
	p.File.Set("EXTRA_ARGS", strings.Join(local.extra, " "))
	if r.DisplayName != "" {
		p.File.Set("DISPLAY_NAME", r.DisplayName)
	}
	if err := p.File.Write(cfg.Dirs.ProfileFile(p.Name), 0o644); err != nil {
		return err
	}
	port, err := control.AllocatePort(cfg, p.Name, strconv.Itoa(r.Port))
	if err != nil {
		return err
	}
	if err := control.WriteInstanceEnv(cfg, p, control.InstanceOpts{Port: port}); err != nil {
		return err
	}
	inst, err := control.LoadInstance(cfg, p.Name)
	if err != nil {
		return err
	}
	// The containers are already up; the notify half is what syncs Coder.
	if err := control.NotifyAndRestart(ctx, cfg, p, inst, os.Stderr, func() error { return nil }); err != nil {
		return err
	}
	fmt.Printf("%s is now serving\n", p.Name)
	if local.pruned {
		fmt.Println("EXTRA_ARGS: dropped the flags that only restated the recipe.")
	}
	if len(local.args) > 0 {
		fmt.Printf("Coder is advertising %s tokens, which is what this launch serves.\n", p.MaxModelLen())
	}
	return nil
}

// checkStaged refuses a launch whose pinned models are not already in every
// node's Hugging Face cache, before anything serving is stopped. A container
// that downloaded on start would hide hundreds of gigabytes behind a launch,
// and one running offline would fail minutes in.
func checkStaged(r recipe.TestedRecipe, members []recipeMember, key string) error {
	script := r.StagingScript()
	if script == "" {
		return nil
	}
	cache := r.Engine().HFCacheHostPath(r.Params)
	for _, m := range members {
		out, err := nodeScript(m, key, script, nil)
		if err != nil {
			return fmt.Errorf("check staged models on %s: %w", m.node.Name, err)
		}
		missing := strings.Fields(out)
		if len(missing) == 0 {
			continue
		}
		var hint []string
		for _, ref := range missing {
			id, rev, _ := strings.Cut(ref, "@")
			hint = append(hint, fmt.Sprintf("  HF_HOME=\"$HOME/%s\" hf download %s --revision %s", cache, id, rev))
		}
		return fmt.Errorf("%s needs %s on node %s, which is missing or incomplete in its Hugging Face cache; download it there once with:\n%s",
			r.ID, strings.Join(missing, ", "), m.node.Name, strings.Join(hint, "\n"))
	}
	return nil
}

func runRecipeStop(id string, o recipeLaunchOptions) error {
	tested, err := recipe.LoadTested()
	if err != nil {
		return err
	}
	if _, ok := tested.Get(id); !ok {
		return fmt.Errorf("no tested recipe %q", id)
	}
	if len(o.nodeNames) == 0 {
		return fmt.Errorf("--nodes is required")
	}
	reg, err := nodes.Load()
	if err != nil {
		return err
	}
	// Tell the watchdog first: a check landing mid-teardown would otherwise
	// see the endpoint gone and start the cluster again.
	if !o.dryRun {
		if cfg, err := config.Load(); err == nil {
			markStopped(cfg, id)
		}
	}
	cmdline := "docker rm -f " + recipe.ShellQuote(o.container)
	for i, name := range o.nodeNames {
		n, ok := reg.Get(name)
		if !ok {
			return fmt.Errorf("no node %q in the registry", name)
		}
		m := recipeMember{node: n, rank: i}
		if o.dryRun {
			fmt.Printf("# %s\n%s\n", name, cmdline)
			continue
		}
		out, err := nodeScript(m, o.sshKey, stopScript(o.container), nil)
		if err != nil {
			return fmt.Errorf("stop on %s: %w", name, err)
		}
		msg, err := stopReport(o.container, name, out)
		if err != nil {
			return err
		}
		fmt.Println(msg)
	}
	return nil
}

// stopScript removes the container and says whether there was one. Swallowing
// every failure reported "stopped" for a container that was never there, and
// equally for a docker that could not be reached, which left GPUs held.
func stopScript(container string) string {
	q := recipe.ShellQuote(container)
	return `out=$(docker rm -f ` + q + ` 2>&1); rc=$?; ` +
		`case "$out" in *"No such container"*) echo absent ;; ` +
		`*) if [ "$rc" -eq 0 ]; then echo removed; else printf 'error: %s\n' "$out"; fi ;; esac`
}

// stopReport turns stopScript's answer into what the user is told.
func stopReport(container, node, out string) (string, error) {
	out = strings.TrimSpace(out)
	switch {
	case out == "removed":
		return fmt.Sprintf("stopped %s on %s", container, node), nil
	case out == "absent":
		return fmt.Sprintf("no %s container on %s; nothing to stop", container, node), nil
	default:
		return "", fmt.Errorf("stop %s on %s: %s", container, node, strings.TrimSpace(strings.TrimPrefix(out, "error:")))
	}
}
