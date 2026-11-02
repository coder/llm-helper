// Package cli implements the llm-helper commands.
package cli

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/plugin"
	"github.com/coder/vllm-helper/internal/profile"
	"github.com/coder/vllm-helper/internal/service"
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
			"on one GPU machine or several (models, switching, sharding, replica pools,\n" +
			"recipes, optional Coder Agents sync). Per machine, only one native model is\n" +
			"loaded at a time.",
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
		newSwitchCmd(),
		newNewCmd(),
		newRecipesCmd(),
		newSearchCmd(),
		newInspectCmd(),
		newInstallGGUFCmd(),
		newEditCmd(),
		newSetCmd(),
		newRemoveCmd(),

		newStatusCmd(),
		newRestartCmd(),
		newStopCmd(),
		newLogsCmd(),
		newServeCmd(),

		newSetupCmd(),
		newDoctorCmd(),
		newAdoptCmd(),
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
		return 1
	}
	return 0
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

// runner is how the CLI drives whatever is actually serving the active model:
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

// svc is the runner for the active model: the cluster when the active profile
// is owned by a recipe, otherwise the init system. Picking on the profile is
// the whole point -- status and logs used to report systemd regardless.
func svc(cfg *config.Config) runner {
	if c := activeClusterRunner(cfg); c != nil {
		return c
	}
	return systemdService(cfg)
}

// runnerIdentity names what a runner holds the GPUs and port with, so a switch
// can tell when it is about to leave one runner for another.
func runnerIdentity(r runner) string {
	if c, ok := r.(*clusterRunner); ok {
		return c.identity()
	}
	if s, ok := r.(service.Service); ok {
		return "native:" + s.Name
	}
	return r.Label()
}

// stopOutgoing stops the runner a switch is leaving when the one taking over
// is a different runner. The incoming runner only replaces its own
// containers or unit, so without this a recipe kept the GPUs and the API port
// after a switch to a native profile, and the reverse left vLLM running.
//
// A cluster is always told to stop, even with its head down, because
// headless workers can outlive the head and still hold GPU memory. A native
// unit that is not running is left alone, which also keeps platforms without
// a service manager working.
func stopOutgoing(outgoing, incoming runner) error {
	if runnerIdentity(outgoing) == runnerIdentity(incoming) {
		return nil
	}
	c, cluster := outgoing.(*clusterRunner)
	if cluster && c.needNodes() != nil {
		// A profile from before the topology was recorded: there is no node
		// list to act on, and refusing would trap the user on it.
		fmt.Fprintf(os.Stderr, "warning: the previous profile does not record its nodes, so any %s containers are still running; stop them with 'vllm-helper recipes stop %s --nodes <n1,n2>'\n", c.recipe.ID, c.recipe.ID)
		return nil
	}
	state := ""
	if !cluster {
		state = outgoing.IsActive()
	}
	if !outgoingNeedsStop(runnerIdentity(outgoing), runnerIdentity(incoming), cluster, state) {
		return nil
	}
	fmt.Printf("Stopping the %s the previous profile ran on...\n", outgoing.Label())
	if err := outgoing.Stop(); err != nil {
		return fmt.Errorf("stopping the outgoing %s, which would keep the GPUs and port: %w", outgoing.Label(), err)
	}
	return nil
}

// outgoingNeedsStop is stopOutgoing's decision, kept pure for testing.
func outgoingNeedsStop(outgoingID, incomingID string, outgoingIsCluster bool, outgoingState string) bool {
	if outgoingID == incomingID {
		return false
	}
	return outgoingIsCluster || outgoingState == "active" || outgoingState == "activating"
}

// activeClusterRunner returns nil when the active model is not a recipe, which
// is the signal to fall back rather than an error worth surfacing here.
func activeClusterRunner(cfg *config.Config) *clusterRunner {
	name := cfg.ActiveProfile()
	if name == "" {
		return nil
	}
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

// systemdService is the init-system runner. Setup needs the concrete type
// because installing a unit is not part of the runner contract.
func systemdService(cfg *config.Config) service.Service {
	return service.Service{
		Name:  cfg.ServiceName(),
		Scope: cfg.ServiceScope(),
		// Without a terminal, sudo must fail fast instead of prompting
		// for a password nothing can type.
		NonInteractive: !stdinIsTTY(),
	}
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
