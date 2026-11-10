package cli

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/service"
)

// exitError makes Run exit with code instead of 1.
type exitError struct {
	code int
	err  error
}

func (e *exitError) Error() string { return e.err.Error() }
func (e *exitError) Unwrap() error { return e.err }

func newStartCmd() *cobra.Command {
	var name string
	c := &cobra.Command{
		Use:   "start",
		Short: "Run one instance's server in the foreground",
		// The service manager's ExecStart: it execs the server over this
		// process and never returns.
		Hidden: true,
		Args:   cobra.NoArgs,
		RunE:   func(*cobra.Command, []string) error { return runStart(name) },
	}
	c.Flags().StringVar(&name, "instance", "", "the instance (profile) to serve; without it, the pre-instance unit's model")
	return c
}

// startOrderTimeout bounds how long an instance waits for older ones at boot.
const startOrderTimeout = 20 * time.Minute

func runStart(name string) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	if name == "" {
		return runLegacyStart(cfg)
	}
	inst, err := control.LoadInstance(cfg, name)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		// A clean exit: a failure would send the service manager into a
		// restart loop with nothing to serve.
		fmt.Printf("'%s' is not set up to run -- start it with '%s up %s'.\n", name, cmdName, name)
		return nil
	case err != nil:
		return fmt.Errorf("reading %s: %w", cfg.Dirs.InstanceEnvFile(name), err)
	}

	if id := strings.TrimSpace(inst.Env.Get(profile.RecipeKey)); id != "" {
		return fmt.Errorf("'%s' is generated from recipe %q and runs in a container; start it with '%s recipes launch %s'", name, id, cmdName, id)
	}

	waitForOlder(cfg, inst)

	if inst.Env.Get("INSTANCE_KV_CACHE") == "" {
		if err := control.CheckFitFor(cfg, name, "", inst.Env); err != nil {
			return &exitError{code: service.ExitNoFit, err: err}
		}
	}
	return execServer(cfg, inst.Env)
}

// runLegacyStart keeps a unit from before instances serving after an upgrade,
// until 'migrate' replaces it.
func runLegacyStart(cfg *config.Config) error {
	env, err := config.ParseEnvFile(cfg.Dirs.LegacyActiveEnvFile())
	switch {
	case errors.Is(err, fs.ErrNotExist):
		fmt.Printf("Nothing to serve -- run '%s migrate', then '%s up <profile>'.\n", cmdName, cmdName)
		return nil
	case err != nil:
		return fmt.Errorf("reading %s: %w", cfg.Dirs.LegacyActiveEnvFile(), err)
	}
	if id := strings.TrimSpace(env.Get(profile.RecipeKey)); id != "" {
		return fmt.Errorf("the profile is generated from recipe %q and runs in a container; start it with '%s recipes launch %s'", id, cmdName, id)
	}
	fmt.Fprintf(os.Stderr, "This unit predates instances; run '%s migrate' to move its model to one.\n", cmdName)
	return execServer(cfg, env)
}

func execServer(cfg *config.Config, env *config.EnvFile) error {
	spec, err := serveSpecFromEnv(env)
	if err != nil {
		return err
	}
	// A profile can override VLLM_BIN, and the binary decides whether the
	// CPU backend's free-RAM clamp applies.
	if spec.provider().Capabilities().VLLMArgs {
		useMetalProbe(spec.Bin)
	}
	useServerGPUs(env)
	args, err := spec.args()
	if err != nil {
		return err
	}
	awaitWiredLimit(cfg)
	return syscall.Exec(spec.Bin, args, withProviderEnv(serveEnvFromInstance(env), spec))
}

// waitForOlder lets instances started earlier load first after a reboot, so
// they claim the memory they had before.
func waitForOlder(cfg *config.Config, inst *control.Instance) {
	others, err := control.OtherInstances(cfg, inst.Name)
	if err != nil {
		return
	}
	deadline := time.Now().Add(startOrderTimeout)
	for _, o := range others {
		if !o.Since.Before(inst.Since) || clusterRunnerForName(cfg, o.Name) != nil {
			continue
		}
		for time.Now().Before(deadline) {
			if _, ok := o.Probe(); ok {
				break
			}
			if s := instanceService(cfg, o.Name).IsActive(); s != "active" && s != "activating" {
				break
			}
			time.Sleep(5 * time.Second)
		}
	}
}
