package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hardware"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/modeltype"
	"github.com/coder/llm-helper/internal/plugin"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/recipe"
	"github.com/coder/llm-helper/internal/service"
	"github.com/coder/llm-helper/internal/toolchain"
)

// doctorReport is everything doctor learns, in one machine-readable shape.
type doctorReport struct {
	Hardware    detect.Hardware    `json:"hardware"`
	Config      *doctorConfig      `json:"config"`
	ConfigError string             `json:"config_error,omitempty"`
	Servers     []detect.Server    `json:"servers"`
	Containers  []detect.Container `json:"containers"`
	Endpoints   []detect.Endpoint  `json:"endpoints"`
	Toolchain   []toolchain.Check  `json:"toolchain"`
	Coder       coderStatus        `json:"coder_sync"`
	HFToken     hfTokenStatus      `json:"hf_token"`
	// WiredLimit is the macOS GPU wired memory cap, nil unless the Mac
	// serves on the GPU through vllm-metal.
	WiredLimit *wiredLimit `json:"wired_limit,omitempty"`
	Warnings   []string    `json:"warnings"`
}

type doctorConfig struct {
	Path      string `json:"path"`
	Port      string `json:"port"`
	VLLMBin   string `json:"vllm_bin"`
	VLLMBinOK bool   `json:"vllm_bin_ok"`
	// Provider is what new profiles are made for. Its server is reported
	// beside VLLM_BIN when it is not vLLM, since that is then the binary
	// whose absence stops a new profile serving.
	Provider        string           `json:"provider"`
	ProviderBin     string           `json:"provider_bin,omitempty"`
	ProviderBinOK   bool             `json:"provider_bin_ok,omitempty"`
	ProviderError   string           `json:"provider_error,omitempty"`
	Instances       []doctorInstance `json:"instances"`
	LegacyProfile   string           `json:"legacy_profile,omitempty"`
	BudgetOverride  string           `json:"budget_override,omitempty"`
	CoderConfigured bool             `json:"coder_sync_configured"`
}

type doctorInstance struct {
	Profile  string `json:"profile"`
	Model    string `json:"model"`
	Port     string `json:"port"`
	Service  string `json:"service"`
	Scope    string `json:"scope,omitempty"`
	State    string `json:"state"`
	Crashed  bool   `json:"crashed"`
	Restarts int    `json:"restarts"`
}

type coderStatus struct {
	Configured bool   `json:"configured"`
	URL        string `json:"url,omitempty"`
	Reachable  bool   `json:"reachable"`
	Error      string `json:"error,omitempty"`
}

type hfTokenStatus struct {
	Present bool   `json:"present"`
	Source  string `json:"source,omitempty"`
	Account string `json:"account,omitempty"`
	Error   string `json:"error,omitempty"`
}

// cmdDoctor prints everything we can detect about this machine and flags
// drift between config and reality.
func newDoctorCmd() *cobra.Command {
	var jsonOut, yes bool
	c := &cobra.Command{
		Use:     "doctor",
		Short:   "Diagnose hardware, running servers, integrations, and config drift",
		GroupID: groupMachine,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runDoctor(jsonOut, yes) },
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print the full report as JSON (no prompts)")
	c.Flags().BoolVar(&yes, "yes", false, "install missing toolchain packages without asking")
	return c
}

func runDoctor(jsonOut, yes bool) error {

	r := gatherDoctor()
	if jsonOut {
		return printJSON(r)
	}
	renderDoctor(r, yes)
	return nil
}

func gatherDoctor() doctorReport {
	r := doctorReport{Hardware: detect.Probe()}

	cfg, err := loadConfig()
	if err != nil {
		r.ConfigError = err.Error()
		cfg = nil
	} else {
		bin := cfg.VLLMBin()
		binOK := false
		if bin != "" {
			if info, err := os.Stat(bin); err == nil && info.Mode()&0o111 != 0 {
				binOK = true
			}
		}
		r.Config = &doctorConfig{
			Path:            cfg.Dirs.ConfigFile(),
			Port:            cfg.Port(),
			VLLMBin:         bin,
			VLLMBinOK:       binOK,
			Instances:       gatherDoctorInstances(cfg),
			BudgetOverride:  cfg.ModelBudgetGB(),
			CoderConfigured: cfg.CoderConfigured(),
		}
		if legacyActive(cfg) {
			r.Config.LegacyProfile = cfg.LegacyActiveProfile()
			r.Warnings = append(r.Warnings, fmt.Sprintf(
				"%s still runs under the single-model service from an older release. Run '%s migrate' to move it to an instance.",
				r.Config.LegacyProfile, cmdName))
		}
		if prov, err := engine.Lookup(cfg.Provider()); err != nil {
			r.Config.Provider = cfg.Provider()
			r.Config.ProviderError = err.Error()
		} else {
			r.Config.Provider = prov.Name()
			if prov.Name() != engine.VLLM {
				r.Config.ProviderBin = cfg.Lookup(prov.BinKey(), "")
				r.Config.ProviderBinOK = engine.Executable(r.Config.ProviderBin)
			}
		}
	}

	r.Servers = detect.ScanProcesses()
	r.Containers = detect.ScanContainers()

	apiKey, extraPort := "", ""
	if cfg != nil {
		apiKey, extraPort = cfg.APIKey(), cfg.Port()
	}
	r.Endpoints = detect.ProbePorts(detect.DefaultPorts(extraPort), apiKey)

	vllmBin := ""
	if cfg != nil {
		vllmBin = cfg.VLLMBin()
	}
	r.Toolchain = toolchain.Gather(vllmBin)
	r.Coder = gatherCoderStatus(cfg)
	r.HFToken = gatherHFToken(cfg)
	r.WiredLimit = gatherWiredLimit(cfg, r.Hardware)
	running := 0
	if r.Config != nil {
		running = len(r.Config.Instances)
	}
	r.Warnings = append(r.Warnings, hardwareWarnings(r.Hardware, running)...)

	// Drift checks: the part a human wouldn't spot at a glance.
	if cfg != nil {
		byPort := map[string]doctorInstance{}
		units := map[string]bool{cfg.ServiceName(): true}
		for _, in := range r.Config.Instances {
			byPort[in.Port] = in
			units[in.Service] = true
		}
		for _, s := range r.Servers {
			if s.InContainer {
				continue
			}
			if in, ok := byPort[s.Port]; ok && in.Model != "" && s.Model != "" && s.Model != in.Model {
				r.Warnings = append(r.Warnings, fmt.Sprintf(
					"%s should serve %s on port %s, but %s is there -- was the server started outside llm-helper? Run 'llm-helper adopt'.",
					in.Profile, in.Model, s.Port, s.Model))
			}
			if s.SystemdUnit != "" && !units[s.SystemdUnit] {
				r.Warnings = append(r.Warnings, fmt.Sprintf(
					"running server belongs to %s unit %s, which llm-helper does not manage -- run 'llm-helper adopt' to take it over.",
					config.ServiceManager, s.SystemdUnit))
			}
		}
	}
	return r
}

// gatherDoctorInstances names what actually serves each instance: a recipe's
// cluster, or the instance's unit.
func gatherDoctorInstances(cfg *config.Config) []doctorInstance {
	list, err := control.ListInstances(cfg)
	if err != nil {
		return nil
	}
	out := make([]doctorInstance, 0, len(list))
	for _, inst := range list {
		run := instanceRunner(cfg, inst.Name)
		h := run.Health()
		d := doctorInstance{
			Profile:  inst.Name,
			Model:    inst.Env.Get("MODEL"),
			Port:     inst.Port,
			Service:  service.InstanceName(inst.Name),
			Scope:    cfg.ServiceScope(),
			State:    h.State,
			Crashed:  h.Crashed,
			Restarts: h.NRestarts,
		}
		if c, ok := run.(*clusterRunner); ok {
			d.Service, d.Scope = c.Label(), ""
		}
		out = append(out, d)
	}
	return out
}

// gatherCoderStatus goes beyond "the variables are set": it calls the API
// so an expired token or wrong URL shows up here instead of on the next
// up.
func gatherCoderStatus(cfg *config.Config) coderStatus {
	st := coderStatus{}
	if cfg == nil {
		return st
	}
	// Asking the registry rather than the Coder package keeps doctor from
	// importing an integration directly. The URL still comes from config:
	// it's what the user would need to fix, and only doctor reports it.
	p := pluginNamed("coder")
	if p == nil || !p.Configured(cfg) {
		return st
	}
	st.Configured = true
	st.URL = cfg.CoderURL()
	if err := p.Check(context.Background(), cfg); err != nil {
		st.Error = err.Error()
	} else {
		st.Reachable = true
	}
	return st
}

// pluginNamed finds a registered integration, or nil when it isn't built in.
func pluginNamed(name string) plugin.Plugin {
	for _, p := range plugin.All() {
		if p.Name() == name {
			return p
		}
	}
	return nil
}

// gatherHFToken reports whether a HuggingFace token is available (gated
// models can't be pulled without one) and whether the Hub accepts it.
func gatherHFToken(cfg *config.Config) hfTokenStatus {
	st := hfTokenStatus{}
	token := ""
	if cfg != nil && cfg.HFToken() != "" {
		token, st.Source = cfg.HFToken(), "config"
	} else {
		for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"} {
			if v := os.Getenv(k); v != "" {
				token, st.Source = v, "environment ("+k+")"
				break
			}
		}
	}
	if token == "" {
		return st
	}
	st.Present = true
	account, err := hf.WhoAmI(context.Background(), token)
	if err != nil {
		st.Error = err.Error()
	} else {
		st.Account = account
	}
	return st
}

// hardwareWarnings is what doctor has to say about the machine itself: the
// cases where the budget it prints is right by its own rules but would
// still mislead someone choosing a model.
func hardwareWarnings(hw detect.Hardware, running int) []string {
	var w []string
	for _, err := range hardware.LocalErrors() {
		w = append(w, "a local hardware definition was skipped: "+err.Error())
	}
	for _, err := range engine.LocalErrors() {
		w = append(w, "a local engine definition was skipped: "+err.Error())
	}
	for _, err := range recipe.LocalErrors() {
		w = append(w, "a local recipe was skipped: "+err.Error())
	}
	for _, err := range modeltype.LocalErrors() {
		w = append(w, "a local model type was skipped: "+err.Error())
	}
	w = append(w, overlapWarnings(hardware.Default(), hw)...)
	if hw.AllGPUsHidden() {
		w = append(w, fmt.Sprintf(
			"CUDA_VISIBLE_DEVICES hides all %d GPU(s) from the server, and vLLM will not start without one. Unset it, or name a GPU in config.env or the profile.",
			hw.GPUsHidden))
	}
	switch {
	case hw.Unified && hw.Busy() && running > 0:
		w = append(w, fmt.Sprintf(
			"%d GB of this machine's shared memory is in use, including %d running instance(s). '%s up' checks a new model against the instances, but not against anything else holding memory (another server, a container).",
			int(float64(hw.InUseBytes)/1e9), running, cmdName))
	case hw.Unified && hw.Busy():
		w = append(w, fmt.Sprintf(
			"%d GB of this machine's shared memory is in use. The budget assumes it is free when a model starts, and vLLM sizes its cache from what is free then, so anything still holding it (another server, a container) takes room from the next model.",
			int(float64(hw.InUseBytes)/1e9)))
	}
	if hw.NVIDIASMIFailed {
		w = append(w, "the NVIDIA driver's device node is present, but nvidia-smi does not answer, so the GPU's name, memory and architecture are unknown and flags that abort on it are not refused. This usually follows a driver upgrade without a reboot; reboot, or reinstall the driver.")
	}
	if hw.CUDA || hw.Metal {
		return w
	}
	// Sized for the CPU backend with a GPU on the bus: say why, because
	// otherwise the budget reads as the machine's limit.
	for _, a := range hw.Accelerators {
		switch {
		case a.Vendor == "nvidia":
			w = append(w, fmt.Sprintf(
				"an NVIDIA GPU is on the PCI bus (%s), but no NVIDIA driver answers, so this machine is sized for vLLM's CPU backend. Install or load the NVIDIA driver to serve on it.",
				a.Describe()))
		case a.Kind == "accelerator" || a.VRAMBytes > 0:
			w = append(w, fmt.Sprintf(
				"%s is present, but llm-helper only sizes NVIDIA and Apple GPUs, so this machine is sized for vLLM's CPU backend.",
				a.Describe()))
		}
	}
	return w
}

// overlapWarnings names the definitions that claim one of this machine's
// GPUs when more than one does. The shipped ones are tested never to
// overlap, so the others come from hardware.d, and only the first is used.
// Every GPU nvidia-smi lists is checked, and the one the machine is named
// after too, which is how an Apple SoC or a Jetson without nvidia-smi is
// seen.
func overlapWarnings(reg *hardware.Registry, hw detect.Hardware) []string {
	observed := []hardware.Observed{{Vendor: hw.GPUVendor, GPUName: hw.GPUName}}
	for _, g := range hw.GPUs {
		observed = append(observed, hardware.Observed{Vendor: "nvidia", GPUName: g.Name})
	}
	var w []string
	seen := map[hardware.Observed]bool{}
	for _, o := range observed {
		if seen[o] {
			continue
		}
		seen[o] = true
		m := reg.Matches(o)
		if len(m) < 2 {
			continue
		}
		var others []string
		for _, d := range m[1:] {
			src := d.Source
			if src == "" {
				src = "shipped"
			}
			others = append(others, fmt.Sprintf("%s (%s)", d.ID, src))
		}
		w = append(w, fmt.Sprintf("%s matches more than one hardware definition. llm-helper uses %s and ignores %s; narrow the gpu_name of the one that is wrong.",
			o.GPUName, m[0].ID, strings.Join(others, ", ")))
	}
	return w
}

// hardwareLine says which hardware definition recognised the GPU, its
// architecture, and what decided its memory kind when no definition did.
func hardwareLine(hw detect.Hardware) string {
	var parts []string
	if hw.HardwareName != "" {
		parts = append(parts, fmt.Sprintf("%s [%s]", hw.HardwareName, hw.HardwareID))
	}
	if hw.Arch != "" {
		parts = append(parts, "arch "+hw.Arch)
	}
	switch hw.MemorySource {
	case detect.MemoryFromAddressing:
		parts = append(parts, "unified because it has no memory of its own and addresses system memory (ATS)")
	case detect.MemoryFromSize:
		parts = append(parts, "unified because it reports nearly all of system memory as its own")
	case detect.MemoryFromTegra:
		parts = append(parts, "unified because this is a Jetson")
	}
	return strings.Join(parts, ", ")
}

func renderDoctor(r doctorReport, yes bool) {
	hw := r.Hardware
	fmt.Println("HARDWARE")
	fmt.Printf("  memory:     %d GB total\n", hw.TotalMemGB)
	if hw.GPUName != "" {
		kind := "discrete VRAM"
		if hw.Unified {
			kind = "unified memory"
		}
		fmt.Printf("  gpu:        %s (%s)\n", hw.GPUSummary(), kind)
		if line := hardwareLine(hw); line != "" {
			fmt.Printf("  hardware:   %s\n", line)
		}
		if hw.GPUsHidden > 0 {
			fmt.Printf("              %d more hidden from the server by CUDA_VISIBLE_DEVICES\n", hw.GPUsHidden)
		}
		newProfiles := engine.VLLM
		if r.Config != nil {
			newProfiles = r.Config.Provider
		}
		if line := appleBackend(hw, newProfiles); line != "" {
			fmt.Printf("  backend:    %s\n", line)
		}
		if r.WiredLimit != nil {
			for _, line := range doctorWiredLimitLines(*r.WiredLimit, hw.MetalWorkingSetGB) {
				fmt.Println(line)
			}
		}
	} else if hw.GPUsHidden > 0 {
		fmt.Printf("  gpu:        %d NVIDIA GPU(s) present, but CUDA_VISIBLE_DEVICES hides all of them from the server\n", hw.GPUsHidden)
	} else {
		fmt.Println("  gpu:        no NVIDIA GPU visible (nvidia-smi missing or failed)")
		for _, a := range hw.Accelerators {
			fmt.Printf("              on the PCI bus: %s\n", a.Describe())
		}
	}
	if hw.MemLimited() {
		fmt.Printf("  limit:      %d GB, this process's cgroup memory limit\n", int(float64(hw.MemLimitBytes)/1e9))
	}
	if hw.BaselineGB > 0 {
		fmt.Printf("  in use:     %.1f GB by the OS and everything else running\n", hw.BaselineGB)
	}
	fmt.Printf("  budget:     ~%d GB of model weights (%s)\n", hw.BudgetGB, hw.BudgetReason)
	// An override silently replaces the figure above everywhere else, so
	// showing only the measured one would make doctor disagree with search.
	if r.Config != nil && r.Config.BudgetOverride != "" {
		fmt.Printf("              overridden to %s GB by MODEL_BUDGET_GB in config.env\n", r.Config.BudgetOverride)
	}
	fmt.Println()

	fmt.Println("CONFIG")
	if r.Config == nil {
		fmt.Printf("  %s\n\n", r.ConfigError)
	} else {
		c := r.Config
		fmt.Printf("  config:     %s\n", c.Path)
		fmt.Printf("  port:       %s\n", c.Port)
		// Only when there is something to say: an install that never chose a
		// provider reads as it always did.
		switch {
		case c.ProviderError != "":
			fmt.Printf("  provider:   %s\n", c.ProviderError)
		case c.Provider != engine.VLLM:
			p, _ := engine.Lookup(c.Provider)
			fmt.Printf("  provider:   %s (new profiles)\n", c.Provider)
			// "server", not "<provider> bin": the line above names the provider,
			// and a longer name would push the value out of the column.
			switch {
			case c.ProviderBin == "":
				fmt.Printf("  server:     NOT SET -- set %s in config.env\n", p.BinKey())
			case !c.ProviderBinOK:
				fmt.Printf("  server:     %s (MISSING or not executable)\n", c.ProviderBin)
			default:
				fmt.Printf("  server:     %s\n", c.ProviderBin)
			}
			if !p.Capabilities().APIKey {
				fmt.Printf("              %s checks no API key: keep it on 127.0.0.1 unless the network is trusted\n", p.BinName())
			}
		}
		switch {
		case c.VLLMBin == "" && c.Provider != engine.VLLM:
			// Not a fault: vLLM is optional once new profiles go elsewhere.
		case c.VLLMBin == "":
			fmt.Println("  vllm bin:   NOT SET -- set VLLM_BIN in config.env")
		case !c.VLLMBinOK:
			fmt.Printf("  vllm bin:   %s (MISSING or not executable)\n", c.VLLMBin)
		default:
			fmt.Printf("  vllm bin:   %s\n", c.VLLMBin)
		}
		if len(c.Instances) == 0 {
			fmt.Println("  models:     none running")
		}
		for i, in := range c.Instances {
			state := in.State
			if in.Crashed {
				state += ", failing to start"
				if in.Restarts > 0 {
					state += fmt.Sprintf(" (%d restarts)", in.Restarts)
				}
			}
			label := "  models:    "
			if i > 0 {
				label = "             "
			}
			unit := in.Service
			if in.Scope != "" {
				unit += " (" + in.Scope + " scope)"
			}
			fmt.Printf("%s %s on port %s, %s -- %s\n", label, in.Profile, in.Port, unit, state)
		}
		fmt.Println()
	}

	fmt.Println("RUNNING SERVERS")
	if len(r.Servers) == 0 && len(r.Containers) == 0 {
		fmt.Println("  no vLLM processes or containers found")
	}
	for _, s := range r.Servers {
		fmt.Printf("  pid %d: %s on port %s", s.PID, orUnknown(s.Model), s.Port)
		switch {
		case s.InContainer:
			fmt.Printf(" (inside a %s container)", s.ContainerHint)
		case s.SystemdUnit != "":
			fmt.Printf(" (%s: %s)", config.ServiceManager, s.SystemdUnit)
		default:
			fmt.Printf(" (bare process -- not service-managed)")
		}
		fmt.Println()
		if s.ExtraArgs != "" {
			fmt.Printf("      flags: %s\n", s.ExtraArgs)
		}
	}
	for _, c := range r.Containers {
		fmt.Printf("  %s container %s: image %s, ports %s\n", c.Runtime, c.Name, c.Image, orUnknown(c.Ports))
	}
	fmt.Println()

	fmt.Println("ENDPOINTS (localhost probe)")
	if len(r.Endpoints) == 0 {
		fmt.Println("  no OpenAI-compatible endpoints responding")
	}
	for _, ep := range r.Endpoints {
		desc := []string{}
		if ep.Healthy {
			desc = append(desc, "healthy")
		}
		if ep.NeedsAuth {
			desc = append(desc, "requires auth (key not accepted or not configured)")
		}
		if len(ep.Models) > 0 {
			desc = append(desc, "serving "+strings.Join(ep.Models, ", "))
		}
		fmt.Printf("  :%s  %s\n", ep.Port, strings.Join(desc, "; "))
	}
	fmt.Println()

	fmt.Println("INTEGRATIONS")
	switch {
	case !r.Coder.Configured:
		fmt.Println("  coder sync: not configured (optional)")
	case r.Coder.Reachable:
		fmt.Printf("  coder sync: OK -- %s answered with this token\n", r.Coder.URL)
	default:
		fmt.Printf("  coder sync: CONFIGURED BUT FAILING -- %s\n", r.Coder.Error)
		fmt.Println("              (check CODER_URL and that CODER_API_TOKEN hasn't expired)")
	}
	switch {
	case !r.HFToken.Present:
		fmt.Println("  hf token:   not set (fine for public models; gated models like Llama")
		fmt.Println("              need HF_TOKEN in config.env or the environment)")
	case r.HFToken.Error != "":
		fmt.Printf("  hf token:   set (%s) but the Hub rejected it: %s\n", r.HFToken.Source, r.HFToken.Error)
	default:
		fmt.Printf("  hf token:   OK -- authenticated as %q (%s)\n", r.HFToken.Account, r.HFToken.Source)
	}
	fmt.Println()

	renderToolchain(r.Toolchain, yes)

	if len(r.Warnings) > 0 {
		fmt.Println()
		fmt.Println("WARNINGS")
		for _, w := range r.Warnings {
			fmt.Println("  ! " + w)
		}
	}
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

func renderToolchain(checks []toolchain.Check, yes bool) {
	// Nothing to build on this machine, e.g. vllm-metal's prebuilt wheels.
	if len(checks) == 0 {
		return
	}
	fmt.Println("BUILD TOOLCHAIN (for vLLM/FlashInfer/Triton JIT kernels)")
	var hints []string
	for _, c := range checks {
		state := "ok"
		if !c.OK {
			state = "MISSING"
			if c.Hint != "" {
				hints = append(hints, c.Hint)
			}
		}
		fmt.Printf("  %-12s %s\n", c.Name, state)
	}
	for _, h := range hints {
		fmt.Printf("  fix: %s\n", h)
	}
	missing := toolchain.Missing(checks)
	if len(missing) == 0 {
		return
	}
	installer, args := toolchain.Installer()
	if installer == "" {
		fmt.Printf("  install the missing packages with your package manager: %s\n", strings.Join(missing, " "))
		return
	}
	fmt.Println()
	fmt.Printf("  fix: %s %s\n", strings.Join(append([]string{installer}, args...), " "), strings.Join(missing, " "))
	if !yes && !confirm("  Run this now?", false) {
		return
	}
	if err := toolchain.Install(missing, stdinIsTTY()); err != nil {
		fmt.Fprintln(os.Stderr, "  install failed:", err)
		if installer == "sudo" && !stdinIsTTY() {
			fmt.Fprintln(os.Stderr, "  (no terminal for sudo -- run the fix above yourself)")
		}
	}
}

// cmdAdopt detects a running vLLM server and takes it over: saves its
// command line as a profile, points config at its port/binary/unit, and
// marks it active -- WITHOUT restarting anything.
func newAdoptCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "adopt",
		Short:   "Detect an already-running vLLM server and save it as a managed profile",
		GroupID: groupMachine,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runAdopt() },
	}
}

func runAdopt() error {
	cfg, err := loadConfig()
	if err != nil {
		if errors.Is(err, config.ErrNotConfigured) {
			return errors.New("no config yet -- run 'llm-helper setup' (it offers adoption as part of first-time setup)")
		}
		return err
	}
	return adoptInto(cfg)
}

// adoptInto runs the adoption flow against an existing config. Shared by
// `adopt` and `setup`.
func adoptInto(cfg *config.Config) error {
	servers := detect.ScanProcesses()
	containers := detect.ScanContainers()

	var host []detect.Server
	for _, s := range servers {
		if !s.InContainer {
			host = append(host, s)
		}
	}

	if len(host) == 0 {
		if len(containers) > 0 {
			fmt.Println("Found vLLM running in containers (not adoptable as a host service yet):")
			for _, c := range containers {
				fmt.Printf("  %s container %s (image %s)\n", c.Runtime, c.Name, c.Image)
				if args := detect.InspectArgs(c.Runtime, c.ID); len(args) > 0 {
					fmt.Printf("      args: %s\n", strings.Join(args, " "))
				}
			}
			fmt.Println()
			fmt.Println("Container-managed switching is on the roadmap. For now llm-helper can")
			fmt.Println("still search/size models for you, and 'llm-helper doctor' will keep")
			fmt.Println("showing what's running.")
			return nil
		}
		fmt.Println("No running vLLM server found to adopt.")
		fmt.Println("Use 'llm-helper up <profile>' to start one, or 'llm-helper doctor' to see what was scanned.")
		return nil
	}

	s := host[0]
	if len(host) > 1 {
		fmt.Printf("Found %d running vLLM servers; adopting the one on port %s (pid %d).\n\n", len(host), s.Port, s.PID)
	}

	fmt.Println("Found a running vLLM server:")
	fmt.Printf("  model:  %s\n", orUnknown(s.Model))
	fmt.Printf("  port:   %s\n", s.Port)
	if s.MaxModelLen != "" {
		fmt.Printf("  ctx:    %s\n", s.MaxModelLen)
	}
	if s.MemUtilization != "" {
		fmt.Printf("  mem:    %s\n", s.MemUtilization)
	}
	if s.ExtraArgs != "" {
		fmt.Printf("  flags:  %s\n", s.ExtraArgs)
	}
	if s.SystemdUnit != "" {
		fmt.Printf("  unit:   %s\n", s.SystemdUnit)
	} else {
		fmt.Println("  unit:   none (bare process)")
	}
	if s.VLLMBin != "" {
		fmt.Printf("  binary: %s\n", s.VLLMBin)
	}
	fmt.Println()
	fmt.Println("Adopting saves this as a profile llm-helper can run. The running server")
	fmt.Println("is NOT restarted or interrupted.")
	if !confirm("Adopt this setup?", true) {
		fmt.Println("Left everything as-is.")
		return nil
	}

	if s.Model == "" {
		return errors.New("couldn't determine the model from the process command line -- adoption needs it")
	}

	// Profile: reuse an existing one for this model or create from the
	// observed flags.
	name := profile.NameFromRepo(s.Model)
	if _, err := profile.Load(cfg.Dirs, name); err != nil {
		_, err = profile.Create(cfg.Dirs, profile.CreateOpts{
			Repo:           s.Model,
			MaxModelLen:    s.MaxModelLen,
			MemUtilization: s.MemUtilization,
			ExtraArgs:      s.ExtraArgs,
			Status:         "ADOPTED from a running server -- flags taken from its live command line",
			Source:         "llm-helper adopt",
		})
		if err != nil {
			return err
		}
		fmt.Printf("Saved profile '%s' from the running command line.\n", name)
	} else {
		fmt.Printf("Profile '%s' already exists for this model -- keeping its saved settings.\n", name)
	}

	// Config: follow the server's reality.
	changed := false
	if cfg.Port() != s.Port {
		cfg.File.Set("PORT", s.Port)
		changed = true
		fmt.Printf("Config: PORT -> %s (matching the running server)\n", s.Port)
	}
	if s.VLLMBin != "" && cfg.VLLMBin() != s.VLLMBin {
		cfg.File.Set("VLLM_BIN", s.VLLMBin)
		changed = true
		fmt.Printf("Config: VLLM_BIN -> %s\n", s.VLLMBin)
	}
	if changed {
		if err := cfg.File.Write(cfg.Dirs.ConfigFile(), 0o644); err != nil {
			return err
		}
	}

	fmt.Println()
	if s.SystemdUnit != "" {
		fmt.Printf("To let llm-helper run it, stop and disable %s unit %s,\n", config.ServiceManager, s.SystemdUnit)
	} else {
		fmt.Printf("To let llm-helper run it, stop the bare process (pid %d),\n", s.PID)
	}
	fmt.Printf("then run '%s up %s'. Until then llm-helper leaves it alone.\n", cmdName, name)
	if !s.APIKeySet {
		fmt.Println()
		fmt.Println("NOTE: the running server has NO --api-key. Once llm-helper runs it, it will")
		fmt.Printf("require the key in %s -- update your clients.\n", cfg.Dirs.SecretsFile())
	}
	return nil
}

// appleBackend says which vLLM backend an Apple Silicon Mac serves with, since
// the same machine runs on its GPU or its CPU depending on what is installed.
// Empty on anything else, and when new profiles use another provider: that
// server is not vLLM's backend, and telling its user to install vllm-metal
// would send them after something they do not need.
func appleBackend(hw detect.Hardware, newProfiles string) string {
	if hw.GPUVendor != "apple" || newProfiles != engine.VLLM {
		return ""
	}
	if !hw.Metal {
		return "vLLM CPU backend (install vllm-metal to serve on the GPU)"
	}
	if hw.MetalWorkingSetGB > 0 {
		return fmt.Sprintf("vllm-metal on the GPU, %d GB Metal working set", hw.MetalWorkingSetGB)
	}
	return "vllm-metal on the GPU (Metal working set unknown)"
}
