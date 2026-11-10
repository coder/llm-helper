// Package cli implements the llm-helper commands.
package cli

import (
	"bufio"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hardware"
	"github.com/coder/llm-helper/internal/modeltype"
	"github.com/coder/llm-helper/internal/plugin"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/recipe"
	"github.com/coder/llm-helper/internal/service"
)

// Version is stamped by goreleaser via ldflags.
var Version = "dev"

// Commit is the git commit this binary was built from, stamped via ldflags.
// It is what identifies a dev build: those all share one moving release tag,
// so the version string alone can't tell two of them apart.
var Commit = ""

// cmdName is what the CLI calls itself. The binary also ships, and keeps
// working, under its old name, vllm-helper: service units, scripts and
// muscle memory on existing installs all invoke that one.
const cmdName = "llm-helper"

// Command groups, so the help stays sorted by what you're trying to do
// rather than alphabetically.
const (
	groupModels  = "models"
	groupServer  = "server"
	groupMachine = "machine"
	// Node, pool and shard drive several machines; they don't belong under a
	// heading about the one this binary runs on.
	groupMultiNode = "multi-node"
)

func newRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   cmdName,
		Short: "Manage an inference server on one GPU machine or several",
		Long: cmdName + " (formerly vllm-helper) -- serve large models with vLLM, mlx-lm or llama.cpp\n" +
			"on one GPU machine or several (models, several at once if memory allows,\n" +
			"sharding, replica pools, recipes, optional Coder Agents sync).",
		SilenceUsage:  true,
		SilenceErrors: true,
		// describeBuild, not Version: on the dev channel every build shares
		// one release tag, so the commit is the only thing that identifies
		// which one this is. --version and 'version' must not disagree.
		Version: describeBuild(),
		// The generated shell completions are not wired into packaging, so
		// advertising them would only produce a command that goes nowhere.
		CompletionOptions: cobra.CompletionOptions{DisableDefaultCmd: true},
	}
	// Whether a Mac serves on its GPU depends on the configured vLLM, so
	// detect has to be told where that is before anything sizes memory. A
	// missing or broken config is left for the command itself to report.
	// cobra runs only the nearest PersistentPreRun, so a subcommand that
	// defines its own skips this one and must register the probe itself.
	root.PersistentPreRun = func(*cobra.Command, []string) {
		// Local hardware definitions describe the machine, not the config,
		// so they load whether or not config.env does. A bad file is
		// skipped here and reported by doctor.
		hardware.UseLocal(filepath.Join(config.ResolveDirs().Config, hardware.LocalDir))
		// Engines load after hardware, because a local engine may name a
		// vendor only a local hardware definition adds.
		engine.UseLocal(filepath.Join(config.ResolveDirs().Config, engine.LocalDir))
		modeltype.UseLocal(filepath.Join(config.ResolveDirs().Config, modeltype.LocalDir))
		// Recipes load last, because a local recipe may run on a local
		// engine.
		recipe.UseLocal(filepath.Join(config.ResolveDirs().Config, recipe.LocalDir))
		if cfg, err := loadConfig(); err == nil {
			useMetalProbe(cfg.VLLMBin())
			// config.env can set CUDA_VISIBLE_DEVICES, and sizing every GPU
			// for a server allowed one of them overstates the budget.
			control.UseServerGPUs(cfg)
		}
	}
	root.SetVersionTemplate(cmdName + " {{.Version}}\n")
	root.SetFlagErrorFunc(flagError)

	root.AddGroup(
		&cobra.Group{ID: groupModels, Title: "Model profiles:"},
		&cobra.Group{ID: groupServer, Title: "Running server:"},
		&cobra.Group{ID: groupMachine, Title: "This machine:"},
		&cobra.Group{ID: groupMultiNode, Title: "More than one machine:"},
	)

	root.AddCommand(
		newListCmd(),
		newUpCmd(),
		newDownCmd(),
		newDefaultCmd(),
		newSwitchCmd(),
		newNewCmd(),
		newRecipesCmd(),
		newEnginesCmd(),
		newModelTypesCmd(),
		newSearchCmd(),
		newInspectCmd(),
		newInstallGGUFCmd(),
		newEditCmd(),
		newSetCmd(),
		newRemoveCmd(),

		newStatusCmd(),
		newRestartCmd(),
		newLogsCmd(),
		newServeCmd(),
		newRouterCmd(),

		newSetupCmd(),
		newDoctorCmd(),
		newHardwareCmd(),
		newMetalLimitCmd(),
		newAdoptCmd(),
		newMigrateCmd(),
		newUpgradeCmd(),

		newStartCmd(),
		newVersionCmd(),

		// Multi-machine commands, so they don't hide in "This machine:".
		newNodeCmd(),
		newPoolCmd(),
		newShardCmd(),
	)
	// Integrations contribute their own commands, so the set of them is a
	// property of what's registered rather than a list maintained here.
	// Core assigns the help group: where a command is listed is a decision
	// about this CLI's shape, not something a plugin should have to know.
	for _, c := range plugin.Commands() {
		c.GroupID = groupMachine
		root.AddCommand(c)
	}
	return root
}

// Run dispatches a command line. Returns a process exit code.
func Run(args []string) int {
	root := newRootCmd()
	if len(args) == 0 {
		// Bare invocation is a usage error, not a success, so scripts that
		// forget the command notice.
		_ = root.Help()
		return 1
	}
	root.SetArgs(args)
	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, cmdName+":", err)
		return exitCode(err)
	}
	return 0
}

func exitCode(err error) int {
	var ee *exitError
	if errors.As(err, &ee) {
		return ee.code
	}
	return 1
}

// pflag reads a single-dash long flag as a cluster of shorthands, so the
// "-json" that stdlib flag used to accept now fails naming a letter nobody
// typed. Say what to write instead, but only when the letters actually spell
// a flag this command has -- otherwise pflag's own message is the honest one.
var shorthandCluster = regexp.MustCompile(`^unknown shorthand flag: '.' in -(\w[\w-]+)$`)

func flagError(c *cobra.Command, err error) error {
	m := shorthandCluster.FindStringSubmatch(err.Error())
	if m == nil || c.Flags().Lookup(m[1]) == nil {
		return err
	}
	return fmt.Errorf("%s takes --%s, not -%s", c.Name(), m[1], m[1])
}

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the version",
		Args:  cobra.NoArgs,
		Run: func(*cobra.Command, []string) {
			fmt.Println(cmdName, describeBuild())
		},
	}
}

func loadConfig() (*config.Config, error) {
	return config.Load()
}

// runner is how the CLI drives whatever is actually serving an instance:
// the init-system service, or a recipe's container cluster. Label names which
// one, so a report about a two-node cluster does not say "systemd".
//
// Installed/Install are excluded as setup-only, and Start because
// "vllm-helper start" is the exec entrypoint the service manager runs.
type runner interface {
	IsActive() string
	NRestarts() int
	Health() service.Health
	Restart() error
	Stop() error
	StartTail(n int) (*exec.Cmd, error)
	PrintLogs(n int) error
	LastLogs(n int) string
	LogHint() string
	Label() string
}

// errNoInstances is what a command that acts on a running model says when
// nothing runs.
var errNoInstances = errors.New("no model is running; start one with '" + cmdName + " up <profile>'")

// instanceService is the init-system unit that serves one instance.
func instanceService(cfg *config.Config, name string) service.Service {
	return service.Service{
		Name:  service.InstanceName(name),
		Scope: cfg.ServiceScope(),
		// Without a terminal, sudo must fail fast instead of prompting
		// for a password nothing can type.
		NonInteractive: !stdinIsTTY(),
	}
}

// legacyService is the single unit used before instances existed.
func legacyService(cfg *config.Config) service.Service {
	return service.Service{Name: cfg.ServiceName(), Scope: cfg.ServiceScope(), NonInteractive: !stdinIsTTY()}
}

// instanceRunner drives one instance: its recipe's cluster when a recipe owns
// the profile, otherwise its unit.
func instanceRunner(cfg *config.Config, name string) runner {
	if c := clusterRunnerForName(cfg, name); c != nil {
		return c
	}
	return instanceService(cfg, name)
}

// clusterRunnerForName is nil when the profile is not a recipe's.
func clusterRunnerForName(cfg *config.Config, name string) *clusterRunner {
	p, err := profile.Load(cfg.Dirs, name)
	if err != nil || !p.Generated() {
		return nil
	}
	c, err := clusterRunnerFor(cfg, p)
	if err != nil {
		return nil
	}
	return c
}

// pickInstance resolves the instance a command acts on: the one named, or the
// only one running.
func pickInstance(cfg *config.Config, args []string) (*control.Instance, error) {
	if len(args) > 0 && args[0] != "" {
		inst, err := control.LoadInstance(cfg, args[0])
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("'%s' is not running; start it with '%s up %s'", args[0], cmdName, args[0])
		}
		return inst, err
	}
	list, err := control.ListInstances(cfg)
	if err != nil {
		return nil, err
	}
	switch len(list) {
	case 0:
		return nil, errNoInstances
	case 1:
		return list[0], nil
	}
	names := make([]string, len(list))
	for i, inst := range list {
		names[i] = inst.Name
	}
	return nil, fmt.Errorf("%d models are running (%s); name one", len(list), strings.Join(names, ", "))
}

// ask prompts for a line of input with a default. Without a terminal it
// returns the default immediately.
func ask(prompt, def string) string {
	if !stdinIsTTY() {
		return def
	}
	if def != "" {
		fmt.Printf("%s [%s]: ", prompt, def)
	} else {
		fmt.Printf("%s: ", prompt)
	}
	sc := bufio.NewScanner(os.Stdin)
	if !sc.Scan() {
		return def
	}
	v := strings.TrimSpace(sc.Text())
	if v == "" {
		return def
	}
	return v
}

// confirm asks a yes/no question; def is the answer for a bare Enter or a
// non-interactive session.
func confirm(prompt string, def bool) bool {
	hint := "Y/n"
	if !def {
		hint = "y/N"
	}
	v := strings.ToLower(ask(fmt.Sprintf("%s [%s]", prompt, hint), ""))
	if v == "" {
		return def
	}
	return v == "y" || v == "yes"
}
