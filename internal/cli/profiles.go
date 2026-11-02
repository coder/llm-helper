package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/catalog"
	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/control"
	"github.com/coder/vllm-helper/internal/hf"
	"github.com/coder/vllm-helper/internal/profile"
	"github.com/coder/vllm-helper/internal/provider"
	"github.com/coder/vllm-helper/internal/vllm"
)

// profileInfo is the machine-readable shape of one saved profile.
type profileInfo struct {
	Name        string `json:"name"`
	Model       string `json:"model"`
	MaxModelLen string `json:"max_model_len"`
	GPUMemUtil  string `json:"gpu_memory_utilization"`
	ExtraArgs   string `json:"extra_args"`
	Status      string `json:"status"`
	Active      bool   `json:"active"`
}

func newListCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:     "list",
		Short:   "List saved model profiles",
		GroupID: groupModels,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runList(jsonOut) },
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print profiles as JSON")
	return c
}

func runList(jsonOut bool) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	if jsonOut {
		profiles, _ := profile.List(cfg.Dirs)
		active := cfg.ActiveProfile()
		out := make([]profileInfo, 0, len(profiles))
		for _, p := range profiles {
			out = append(out, profileInfo{
				Name:        p.Name,
				Model:       p.Model(),
				MaxModelLen: p.MaxModelLen(),
				GPUMemUtil:  p.File.Get("GPU_MEMORY_UTILIZATION"),
				ExtraArgs:   p.ExtraArgs(),
				Status:      p.Status(),
				Active:      p.Name == active,
			})
		}
		return printJSON(out)
	}
	printProfiles(cfg)
	return nil
}

func printProfiles(cfg *config.Config) {
	profiles, _ := profile.List(cfg.Dirs)
	fmt.Printf("Available profiles (%s/*.env):\n", cfg.Dirs.Profiles)
	if len(profiles) == 0 {
		fmt.Println("  (none yet -- add one with 'vllm-helper new <org/model>')")
		return
	}
	active := cfg.ActiveProfile()
	for _, p := range profiles {
		marker := ""
		if p.Name == active {
			marker = " [active]"
		}
		// A recipe profile with edits is no longer the thing we measured, so
		// say so wherever it is listed.
		if r, ok := recipeForProfile(p); ok {
			if local, err := recipeOverrides(cfg, r); err == nil && len(local.args) > 0 {
				marker = " (+ local overrides)" + marker
			}
		}
		fmt.Printf("  %-24s %s%s\n", p.Name, p.Status(), marker)
	}
}

// waitFlags are shared by switch/new/restart: how long to stay attached
// while the model loads.
type waitFlags struct {
	noFollow bool
	wait     bool
	timeout  time.Duration
}

func (w *waitFlags) register(c *cobra.Command) {
	c.Flags().BoolVar(&w.noFollow, "no-follow", false, "return as soon as the server is started instead of watching the boot")
	c.Flags().BoolVar(&w.wait, "wait", false, "block until the endpoint is healthy, without streaming logs")
	c.Flags().DurationVar(&w.timeout, "timeout", 0, "give up waiting after this long (default: wait forever)")
}

func (w *waitFlags) opts() watchOpts {
	opts := watchOpts{timeout: w.timeout}
	switch {
	case w.noFollow:
		// fire-and-forget
	case w.wait:
		opts.wait = true
	default:
		// Interactive terminals get the log stream; pipes and scripts
		// get quiet blocking (--wait behavior).
		opts.follow = stdoutIsTTY()
		opts.wait = true
	}
	return opts
}

func newSwitchCmd() *cobra.Command {
	var w waitFlags
	c := &cobra.Command{
		Use:     "switch <profile>",
		Short:   "Switch to a saved profile, restarting the server",
		GroupID: groupModels,
		// Not ExactArgs: with no profile named, runSwitch lists what's
		// available, which beats cobra's count of missing arguments.
		Args: cobra.MaximumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			return runSwitch(args, w.opts())
		},
	}
	w.register(c)
	return c
}

func runSwitch(args []string, opts watchOpts) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	if len(args) < 1 {
		printProfiles(cfg)
		fmt.Println("\nusage: vllm-helper switch <profile> [--wait|--no-follow] [--timeout 30m]")
		return errors.New("missing profile name")
	}
	p, err := profile.Load(cfg.Dirs, args[0])
	if err != nil {
		printProfiles(cfg)
		return err
	}
	return activateProfile(cfg, p, opts)
}

func newNewCmd() *cobra.Command {
	var w waitFlags
	var quant string
	c := &cobra.Command{
		Use:     "new <org/model-repo>",
		Short:   "Pull a new HuggingFace model, save it as a profile, and switch to it",
		GroupID: groupModels,
		Args:    cobra.MaximumNArgs(1),
		Example: "  vllm-helper new Qwen/Qwen3-32B\n" +
			"  vllm-helper new bartowski/Qwen_Qwen3-32B-GGUF --quant Q8_0",
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) < 1 {
				return errors.New("usage: vllm-helper new <org/model-repo> [--quant NAME] [--wait|--no-follow] [--timeout 30m]")
			}
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			choice, err := resolveQuant(cmd.Context(), cfg, args[0], quant)
			if err != nil {
				return err
			}
			return pullAndSwitch(cmd.Context(), cfg, args[0], choice, w.opts())
		},
	}
	c.Flags().StringVar(&quant, "quant", "",
		"which quantization to serve, for repos that ship several (see 'vllm-helper inspect')")
	w.register(c)
	return c
}

// resolveQuant turns a --quant name into a choice, refusing anything this
// machine could not actually serve.
//
// Refusing here rather than at startup matters: activating a profile restarts
// the service, so a model that cannot load takes the running one down with it
// and leaves the machine serving nothing.
func resolveQuant(ctx context.Context, cfg *config.Config, repo, want string) (*catalog.Choice, error) {
	if want == "" {
		return nil, nil
	}
	prov, err := provider.Lookup(cfg.Provider())
	if err != nil {
		return nil, err
	}
	if !prov.Capabilities().GGUFQuant {
		return nil, fmt.Errorf("--quant picks one GGUF file out of a repo, but new profiles here are served by %s, which does not load GGUF; drop --quant and name a repo of that quant instead", prov.Name())
	}
	if !prov.Capabilities().VLLMArgs {
		return catalog.ChooseGGUFQuant(ctx, repo, want)
	}
	m, err := hf.Lookup(ctx, repo)
	if err != nil {
		return nil, fmt.Errorf("looking up %s: %w", repo, err)
	}
	d := inspectModel(ctx, m, cfg)
	v, ok := catalog.FindVariant(d.All, want)
	if !ok {
		// A repo holding one copy has nothing to choose between, and listing
		// its single placeholder label back ("it has: default") reads as a
		// typo on the user's part rather than a category error.
		if len(d.All) < 2 {
			return nil, fmt.Errorf(
				"%s ships a single set of weights, so there is no quant to choose -- drop --quant, "+
					"or look for a repo of quantized builds", repo)
		}
		have := make([]string, 0, len(d.All))
		for _, x := range d.All {
			have = append(have, x.Label)
		}
		return nil, fmt.Errorf("%s has no quant called %q -- it has: %s",
			repo, want, strings.Join(have, ", "))
	}
	if !v.Servable() {
		return nil, fmt.Errorf("%s is %s", v.Label, v.Blocked)
	}
	// Refused here rather than at startup: activating a profile restarts the
	// service, so a model that cannot load takes the running one down with it.
	if v.GGUF && d.ArchBlocked != "" {
		return nil, fmt.Errorf("%s can't be served as GGUF -- it is %s", repo, d.ArchBlocked)
	}
	if v.GGUF && !d.GGUFReady {
		return nil, errors.New(ggufMissingMessage(cfg))
	}
	return &catalog.Choice{Variant: v, BaseModel: d.BaseModel}, nil
}

// pullAndSwitch is `new` minus the flag parsing, so `search` can funnel
// into it with its own options.
func pullAndSwitch(ctx context.Context, cfg *config.Config, repo string, choice *catalog.Choice, opts watchOpts) error {
	name := profile.NameFromRepo(repo)
	if choice != nil {
		if _, ok := choice.Variant.Selector(); ok {
			name += "-" + strings.ToLower(choice.Variant.Label)
		}
	}

	p, err := profile.Load(cfg.Dirs, name)
	if err == nil {
		fmt.Printf("Profile '%s' already exists for %s -- using its saved settings.\n", name, repo)
		fmt.Printf("(change it with 'vllm-helper edit %s' if it needs different flags)\n", name)
		return activateProfile(cfg, p, opts)
	}

	fmt.Println("Pulling new model:", repo)
	prov, err := provider.Lookup(cfg.Provider())
	if err != nil {
		return err
	}
	// Only asked for when it is used: probing runs the vLLM binary.
	var caps vllm.Caps
	if prov.Capabilities().VLLMArgs {
		fmt.Println("Checking the model's README for known vLLM serving flags, and its")
		fmt.Println("chat template for known crash patterns...")
		caps = vllmCaps(cfg)
	}
	plan := catalog.PrepareFor(ctx, cfg.Dirs, prov, repo, choice, caps, cfg.KVContextTarget())
	for _, n := range plan.Notes {
		fmt.Println(n)
	}
	createOpts := plan.Opts
	fmt.Printf("Saving it as profile '%s' for next time.\n", name)
	p, err = profile.Create(cfg.Dirs, createOpts)
	if err != nil {
		return err
	}
	return activateProfile(cfg, p, opts)
}

func newRemoveCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "remove <profile>",
		Short:   "Delete a profile, and its Coder model entry when sync is configured",
		GroupID: groupModels,
		Args:    cobra.MaximumNArgs(1),
		RunE:    func(_ *cobra.Command, args []string) error { return runRemove(args) },
	}
}

func runRemove(args []string) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	if len(args) < 1 {
		printProfiles(cfg)
		return errors.New("usage: vllm-helper remove <profile>")
	}
	name := args[0]
	if _, err := profile.Load(cfg.Dirs, name); err != nil {
		return err
	}
	if name == cfg.ActiveProfile() {
		return fmt.Errorf("'%s' is the currently active model -- switch to something else first", name)
	}
	control.NotifyRemoved(context.Background(), cfg, name, os.Stderr)
	if err := profile.Remove(cfg.Dirs, name); err != nil {
		return err
	}
	fmt.Println("Removed profile:", name)
	return nil
}

// activateProfile is the core of switch/new: assemble the runtime env,
// record the active profile, sync Coder, restart the service, and watch
// the boot according to opts.
func activateProfile(cfg *config.Config, p *profile.Profile, opts watchOpts) error {
	// WriteActiveEnv records the active profile as its last step, so the restart
	// below resolves the runner for the profile being switched to rather than
	// the one being left. That ordering is what lets a recipe profile go through
	// here at all: it used to be refused because the restart could only ever
	// mean the init system.
	//
	// Check the runner can act BEFORE publishing anything. The Coder notify
	// happens before the restart, so a failure after this point leaves Coder
	// advertising a model that is not being served -- switching to a recipe
	// whose profile had no recorded topology did exactly that.
	if p.Generated() {
		c, err := clusterRunnerFor(cfg, p)
		if err != nil {
			return err
		}
		if err := c.preflight(cfg); err != nil {
			return err
		}
	}
	// Captured before WriteActiveEnv changes which runner svc resolves to.
	outgoing := svc(cfg)
	if err := control.WriteActiveEnv(cfg, p); err != nil {
		return err
	}
	incoming := svc(cfg)
	// Warn, never refuse: profiles get hand-edited and vLLM gets upgraded
	// underneath one that used to work, but the user asked for this model
	// and it is not this command's place to overrule that.
	if prov, err := profileProvider(p); err == nil && prov.Capabilities().VLLMArgs {
		warnExtraArgs(p, vllmCaps(cfg))
	}

	fmt.Println("Switched to profile:", p.Name)
	fmt.Println()
	for i, line := range p.File.LeadingComments() {
		if i >= 8 {
			break
		}
		fmt.Println(line)
	}

	fmt.Println()
	// The banner belongs to the restart rather than before it, so that a
	// plugin taking its time doesn't leave "Restarting..." on screen with
	// nothing happening yet.
	restart := func() error {
		if err := stopOutgoing(outgoing, incoming); err != nil {
			return err
		}
		fmt.Println()
		fmt.Println("Reloading the model (this can take several minutes, longer if it")
		fmt.Println("isn't cached on disk yet)...")
		return incoming.Restart()
	}
	if err := control.NotifyAndRestart(context.Background(), cfg, p, os.Stderr, restart); err != nil {
		return err
	}

	if !opts.follow && !opts.wait {
		fmt.Println()
		fmt.Println("Not waiting for the model to load. Check progress with 'vllm-helper status'")
		fmt.Println("or 'vllm-helper logs'.")
		return nil
	}
	fmt.Println()
	return watchBoot(cfg, p.Name, opts)
}

// statusReport is the machine-readable shape of `status`.
type statusReport struct {
	Service    string `json:"service"`
	Profile    string `json:"profile"`
	Model      string `json:"model"`
	Port       string `json:"port"`
	Healthy    bool   `json:"endpoint_healthy"`
	HTTPStatus int    `json:"http_status"`
	Crashed    bool   `json:"crashed"`
	Restarts   int    `json:"restarts"`
	FailReason string `json:"fail_reason,omitempty"`
	LastLog    string `json:"last_log,omitempty"`
}

func gatherStatus(cfg *config.Config) statusReport {
	health := svc(cfg).Health()
	r := statusReport{
		Service:    health.State,
		Profile:    cfg.ActiveProfile(),
		Model:      "unknown",
		Port:       cfg.Port(),
		Crashed:    health.Crashed,
		Restarts:   health.NRestarts,
		FailReason: health.Result,
	}
	if r.Profile == "" {
		r.Profile = "unknown"
	}
	if active, err := config.ParseEnvFile(cfg.Dirs.ActiveEnvFile()); err == nil {
		if m := active.Get("MODEL"); m != "" {
			r.Model = m
		}
	}
	r.HTTPStatus, r.Healthy = control.ProbeHealth(cfg.Port())
	if r.Crashed {
		// Only worth the journalctl call when there's a failure to explain.
		r.LastLog = control.LastMeaningfulLine(svc(cfg).LastLogs(40))
	}
	return r
}

func newStatusCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:     "status",
		Short:   "Server state, active model, and whether the endpoint responds",
		GroupID: groupServer,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runStatus(jsonOut) },
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print status as JSON")
	return c
}

func runStatus(jsonOut bool) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	r := gatherStatus(cfg)
	if jsonOut {
		return printJSON(r)
	}
	fmt.Printf("%-9s %s\n", svc(cfg).Label()+":", r.Service)
	fmt.Printf("profile:  %s\n", r.Profile)
	fmt.Printf("model:    %s\n", r.Model)
	if r.Healthy {
		fmt.Printf("endpoint: healthy (http %d on port %s)\n", r.HTTPStatus, r.Port)
		return nil
	}

	// An unresponsive endpoint means one of two very different things, and
	// telling someone to wait for a server that keeps dying wastes their
	// afternoon. Say which one it is.
	if r.Crashed {
		fmt.Printf("endpoint: not responding (http %03d on port %s) -- the service is failing to start, not loading\n", r.HTTPStatus, r.Port)
		detail := r.FailReason
		if r.Restarts > 0 {
			detail = fmt.Sprintf("%s, %d restarts", detail, r.Restarts)
			detail = strings.TrimPrefix(detail, ", ")
		}
		if detail != "" {
			fmt.Printf("failure:  %s\n", detail)
		}
		if r.LastLog != "" {
			fmt.Printf("last log: %s\n", r.LastLog)
		}
	} else {
		// "still loading" is wrong when nothing is running: a stopped model is
		// waiting for someone to start it, not making progress.
		switch r.Service {
		case "inactive", "unknown":
			fmt.Printf("endpoint: not responding (http %03d on port %s) -- nothing is running; start it with 'vllm-helper restart'\n", r.HTTPStatus, r.Port)
		default:
			fmt.Printf("endpoint: not responding (http %03d on port %s) -- still loading, or check the logs\n", r.HTTPStatus, r.Port)
		}
	}
	if hint := svc(cfg).LogHint(); hint != "" {
		fmt.Printf("logs:     %s\n", hint)
	}
	return nil
}
