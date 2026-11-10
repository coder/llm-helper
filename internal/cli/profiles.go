package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/resolve"
	"github.com/coder/llm-helper/internal/vllm"
)

// profileInfo is the machine-readable shape of one saved profile.
type profileInfo struct {
	Name        string `json:"name"`
	Model       string `json:"model"`
	MaxModelLen string `json:"max_model_len"`
	GPUMemUtil  string `json:"gpu_memory_utilization"`
	ExtraArgs   string `json:"extra_args"`
	Status      string `json:"status"`
	Thinking    string `json:"thinking,omitempty"`
	Running     bool   `json:"running"`
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
		live := liveSet(cfg)
		out := make([]profileInfo, 0, len(profiles))
		for _, p := range profiles {
			out = append(out, profileInfo{
				Name:        p.Name,
				Model:       p.Model(),
				MaxModelLen: p.MaxModelLen(),
				GPUMemUtil:  p.File.Get("GPU_MEMORY_UTILIZATION"),
				ExtraArgs:   p.ExtraArgs(),
				Status:      p.Status(),
				Thinking:    p.Thinking(),
				Running:     live[p.Name] != "",
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
		fmt.Println("  (none yet -- add one with 'llm-helper new <org/model>')")
		return
	}
	live := liveSet(cfg)
	for _, p := range profiles {
		marker := ""
		if p.Thinking() == resolve.ThinkingOn {
			marker = " [thinking]"
		}
		if port := live[p.Name]; port != "" {
			marker += " [running :" + port + "]"
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

// waitFlags are shared by up/new/restart: how long to stay attached
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

func newNewCmd() *cobra.Command {
	var w waitFlags
	var quant, thinking string
	c := &cobra.Command{
		Use:     "new <org/model-repo>",
		Short:   "Pull a new HuggingFace model, save it as a profile, and start it",
		GroupID: groupModels,
		Args:    cobra.MaximumNArgs(1),
		Example: "  llm-helper new Qwen/Qwen3-32B\n" +
			"  llm-helper new bartowski/Qwen_Qwen3-32B-GGUF --quant Q8_0\n" +
			"  llm-helper new Qwen/Qwen3-0.6B --thinking off",
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) < 1 {
				return errors.New("usage: llm-helper new <org/model-repo> [--quant NAME] [--thinking on|off] [--wait|--no-follow] [--timeout 30m]")
			}
			if thinking != "" && thinking != resolve.ThinkingOn && thinking != resolve.ThinkingOff {
				return fmt.Errorf("--thinking must be %s or %s, not %q", resolve.ThinkingOn, resolve.ThinkingOff, thinking)
			}
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			choice, err := resolveQuant(cmd.Context(), cfg, args[0], quant)
			if err != nil {
				return err
			}
			return pullAndUp(cmd.Context(), cfg, args[0], choice, thinking, w.opts())
		},
	}
	c.Flags().StringVar(&quant, "quant", "",
		"which quantization to serve, for repos that ship several (see 'llm-helper inspect')")
	c.Flags().StringVar(&thinking, "thinking", "",
		"on or off: whether a thinking model reasons before it answers (default: the model's own)")
	w.register(c)
	return c
}

// resolveQuant turns a --quant name into a choice, refusing anything this
// machine could not actually serve.
//
// Refused here rather than at startup, so a model that cannot load never gets
// as far as a service.
func resolveQuant(ctx context.Context, cfg *config.Config, repo, want string) (*resolve.Choice, error) {
	if want == "" {
		return nil, nil
	}
	prov, err := engine.Lookup(cfg.Provider())
	if err != nil {
		return nil, err
	}
	if !prov.Capabilities().GGUFQuant {
		return nil, fmt.Errorf("--quant picks one GGUF file out of a repo, but new profiles here are served by %s, which does not load GGUF; drop --quant and name a repo of that quant instead", prov.Name())
	}
	if !prov.Capabilities().VLLMArgs {
		return resolve.ChooseGGUFQuant(ctx, repo, want)
	}
	m, err := hf.Lookup(ctx, repo)
	if err != nil {
		return nil, fmt.Errorf("looking up %s: %w", repo, err)
	}
	d := inspectModel(ctx, m, cfg)
	v, ok := resolve.FindVariant(d.All, want)
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
	if v.GGUF && d.ArchBlocked != "" {
		return nil, fmt.Errorf("%s can't be served as GGUF -- it is %s", repo, d.ArchBlocked)
	}
	if v.GGUF && !d.GGUFReady {
		return nil, errors.New(ggufMissingMessage(cfg))
	}
	return &resolve.Choice{Variant: v, BaseModel: d.BaseModel}, nil
}

// pullAndUp is `new` minus the flag parsing, so `search` can funnel
// into it with its own options.
func pullAndUp(ctx context.Context, cfg *config.Config, repo string, choice *resolve.Choice, thinking string, opts watchOpts) error {
	name := profile.NameFromRepo(repo)
	if choice != nil {
		if _, ok := choice.Variant.Selector(); ok {
			name += "-" + strings.ToLower(choice.Variant.Label)
		}
	}

	p, err := profile.Load(cfg.Dirs, name)
	if err == nil {
		if thinking != "" {
			return fmt.Errorf("profile '%s' already exists, so --thinking has nothing to set; remove it first, or edit its EXTRA_ARGS", name)
		}
		fmt.Printf("Profile '%s' already exists for %s -- using its saved settings.\n", name, repo)
		fmt.Printf("(change it with 'llm-helper edit %s' if it needs different flags)\n", name)
		return upProfile(cfg, p, upOpts{}, opts)
	}

	fmt.Println("Pulling new model:", repo)
	prov, err := engine.Lookup(cfg.Provider())
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
	plan := resolve.PrepareFor(ctx, cfg.Dirs, prov, repo, choice, caps, cfg.KVContextTarget())
	if err := plan.SetThinking(thinking, prov, caps); err != nil {
		return err
	}
	for _, n := range plan.Notes {
		fmt.Println(n)
	}
	createOpts := plan.Opts
	fmt.Printf("Saving it as profile '%s' for next time.\n", name)
	p, err = profile.Create(cfg.Dirs, createOpts)
	if err != nil {
		return err
	}
	return upProfile(cfg, p, upOpts{}, opts)
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
		return errors.New("usage: llm-helper remove <profile>")
	}
	name := args[0]
	if _, err := profile.Load(cfg.Dirs, name); err != nil {
		return err
	}
	if liveSet(cfg)[name] != "" {
		return fmt.Errorf("'%s' is running -- stop it first with 'llm-helper down %s'", name, name)
	}
	control.NotifyRemoved(context.Background(), cfg, name, os.Stderr)
	if err := profile.Remove(cfg.Dirs, name); err != nil {
		return err
	}
	fmt.Println("Removed profile:", name)
	return nil
}

// liveSet maps each running profile to its port.
func liveSet(cfg *config.Config) map[string]string {
	out := map[string]string{}
	list, _ := control.ListInstances(cfg)
	for _, inst := range list {
		out[inst.Name] = inst.Port
	}
	return out
}

// statusReport is the machine-readable shape of `status`, one per instance.
type statusReport struct {
	Service    string `json:"service"`
	Profile    string `json:"profile"`
	Engine     string `json:"engine,omitempty"`
	Model      string `json:"model"`
	Port       string `json:"port"`
	Default    bool   `json:"default,omitempty"`
	Healthy    bool   `json:"endpoint_healthy"`
	HTTPStatus int    `json:"http_status"`
	Crashed    bool   `json:"crashed"`
	Restarts   int    `json:"restarts"`
	FailReason string `json:"fail_reason,omitempty"`
	LastLog    string `json:"last_log,omitempty"`
	Thinking   string `json:"thinking,omitempty"`
	MemUtil    string `json:"mem_util,omitempty"`
}

func gatherStatus(cfg *config.Config, inst *control.Instance, def string) statusReport {
	run := instanceRunner(cfg, inst.Name)
	health := run.Health()
	r := statusReport{
		Service:    health.State,
		Profile:    inst.Name,
		Engine:     inst.Env.Get(engine.Key),
		Model:      inst.Env.Get("MODEL"),
		Port:       inst.Port,
		Default:    inst.Name == def,
		Crashed:    health.Crashed,
		Restarts:   health.NRestarts,
		FailReason: health.Result,
		MemUtil:    inst.Env.Get("GPU_MEMORY_UTILIZATION"),
	}
	if r.Engine == "" {
		r.Engine = engine.VLLM
	}
	if r.Model == "" {
		r.Model = "unknown"
	}
	if p, err := profile.Load(cfg.Dirs, inst.Name); err == nil {
		r.Thinking = p.Thinking()
	}
	if c, ok := run.(*clusterRunner); ok {
		r.HTTPStatus, r.Healthy = c.probeHealth()
	} else {
		r.HTTPStatus, r.Healthy = inst.Probe()
	}
	if r.Crashed {
		r.LastLog = control.LastMeaningfulLine(run.LastLogs(40))
	}
	return r
}

func newStatusCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:     "status [profile]",
		Short:   "Running models, and whether each endpoint responds",
		GroupID: groupServer,
		Args:    cobra.MaximumNArgs(1),
		RunE:    func(_ *cobra.Command, args []string) error { return runStatus(args, jsonOut) },
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print status as JSON")
	return c
}

func runStatus(args []string, jsonOut bool) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	var list []*control.Instance
	if len(args) > 0 {
		inst, err := pickInstance(cfg, args)
		if err != nil {
			return err
		}
		list = []*control.Instance{inst}
	} else if list, err = control.ListInstances(cfg); err != nil {
		return err
	}
	def := control.DefaultInstance(cfg, control.LiveProfiles(cfg))
	reports := make([]statusReport, len(list))
	for i, inst := range list {
		reports[i] = gatherStatus(cfg, inst, def)
	}
	if jsonOut {
		if len(args) > 0 {
			return printJSON(reports[0])
		}
		return printJSON(reports)
	}
	switch len(reports) {
	case 0:
		if legacyActive(cfg) {
			fmt.Printf("A model was started by an older release; run '%s migrate' to manage it.\n", cmdName)
			return nil
		}
		fmt.Println(errNoInstances.Error())
		return nil
	case 1:
		printStatusDetail(cfg, reports[0])
		return nil
	}
	fmt.Printf("%-24s %-10s %-6s %-10s %-6s %s\n", "PROFILE", "ENGINE", "PORT", "STATE", "MEMORY", "")
	for _, r := range reports {
		state := "loading"
		switch {
		case r.Healthy:
			state = "healthy"
		case r.Crashed:
			state = "failing"
		case r.Service == "inactive" || r.Service == "unknown":
			state = "stopped"
		}
		mark := ""
		if r.Default {
			mark = "default"
		}
		mem := r.MemUtil
		if mem == "" {
			mem = "-"
		}
		fmt.Printf("%-24s %-10s %-6s %-10s %-6s %s\n", r.Profile, r.Engine, r.Port, state, mem, mark)
	}
	fmt.Printf("\nDetails for one: %s status <profile>\n", cmdName)
	return nil
}

func printStatusDetail(cfg *config.Config, r statusReport) {
	run := instanceRunner(cfg, r.Profile)
	fmt.Printf("%-9s %s\n", run.Label()+":", r.Service)
	def := ""
	if r.Default {
		def = " (default)"
	}
	fmt.Printf("profile:  %s%s\n", r.Profile, def)
	fmt.Printf("model:    %s\n", r.Model)
	switch r.Thinking {
	case resolve.ThinkingOn:
		fmt.Println("thinking: on -- replies begin after a reasoning pass")
	case resolve.ThinkingOff:
		fmt.Println("thinking: off")
	}
	if r.Healthy {
		fmt.Printf("endpoint: healthy (http %d on port %s)\n", r.HTTPStatus, r.Port)
		return
	}
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
		switch r.Service {
		case "inactive", "unknown":
			fmt.Printf("endpoint: not responding (http %03d on port %s) -- nothing is running; start it with 'llm-helper restart %s'\n", r.HTTPStatus, r.Port, r.Profile)
		default:
			fmt.Printf("endpoint: not responding (http %03d on port %s) -- still loading, or check the logs\n", r.HTTPStatus, r.Port)
		}
	}
	if hint := run.LogHint(); hint != "" {
		fmt.Printf("logs:     %s\n", hint)
	}
}
