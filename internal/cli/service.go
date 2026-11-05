package cli

import (
	"errors"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/profile"
)

func newStopCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "stop",
		Short:   "Stop the server, keeping profiles and config",
		GroupID: groupServer,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runStop() },
	}
}

// runStop stops the vLLM service without touching profiles or config.
func runStop() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	// Name the runner, not a unit: a container recipe has no systemd unit, and
	// saying "stopped vllm.service" about a two-node cluster is a lie.
	label := svc(cfg).Label()
	// A cluster is stopped whatever its head reports: headless workers can
	// outlive a crashed head and keep holding GPU memory.
	_, cluster := svc(cfg).(*clusterRunner)
	if state := svc(cfg).IsActive(); !cluster && state != "active" && state != "activating" {
		fmt.Printf("the %s is not running (state: %s) -- nothing to stop.\n", label, state)
		return nil
	}
	if err := svc(cfg).Stop(); err != nil {
		return fmt.Errorf("stopping the %s: %w", label, err)
	}
	fmt.Printf("Stopped the %s. Start it again with 'llm-helper restart' or by\n", label)
	fmt.Println("switching to a profile.")
	if cfg.CoderConfigured() {
		fmt.Println("Note: the active model stays enabled in Coder's picker until the next switch.")
	}
	return nil
}

func newRestartCmd() *cobra.Command {
	var w waitFlags
	c := &cobra.Command{
		Use:     "restart",
		Short:   "Re-apply the active profile and restart the server",
		GroupID: groupServer,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runRestart(w.opts()) },
	}
	w.register(c)
	return c
}

// runRestart re-applies the active profile (picking up any edits to it or to
// config.env) and restarts it through whichever runner owns it. Unlike switch
// it never changes which profile is active and never touches Coder, which is
// what makes it the right verb for a supervisor: a recovery should bring the
// same model back, not re-publish it.
func runRestart(o watchOpts) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	name := cfg.ActiveProfile()
	if name == "" {
		return errors.New("no active profile -- use 'llm-helper switch <profile>' first")
	}
	p, err := profile.Load(cfg.Dirs, name)
	if err != nil {
		return fmt.Errorf("the active profile can't be loaded (%w) -- switch to another one", err)
	}
	// Regenerate active.env so 'edit'/'set' changes take effect.
	if err := control.WriteActiveEnv(cfg, p); err != nil {
		return err
	}

	fmt.Printf("Reloading profile '%s' (%s)...\n", name, svc(cfg).Label())
	if err := svc(cfg).Restart(); err != nil {
		return fmt.Errorf("restarting service: %w", err)
	}
	if !o.follow && !o.wait {
		fmt.Println("Not waiting for the model to load. Check progress with 'llm-helper status'.")
		return nil
	}
	fmt.Println()
	return watchBoot(cfg, name, o)
}

func newLogsCmd() *cobra.Command {
	var (
		lines    int
		follow   bool
		noFollow bool
	)
	c := &cobra.Command{
		Use:     "logs",
		Short:   "Show the server log, following by default on a terminal",
		GroupID: groupServer,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runLogs(lines, follow, noFollow) },
	}
	c.Flags().IntVarP(&lines, "lines", "n", 50, "how many past lines to show")
	c.Flags().BoolVar(&follow, "follow", false, "keep following even when output is piped")
	c.Flags().BoolVar(&noFollow, "no-follow", false, "print the lines and exit instead of following")
	return c
}

// runLogs shows the service journal: last -n lines, following by default
// on a terminal. Piped output prints and exits so scripts never hang.
func runLogs(lines int, follow, noFollow bool) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	s := svc(cfg)
	doFollow := stdoutIsTTY()
	if follow {
		doFollow = true
	}
	if noFollow {
		doFollow = false
	}
	if !doFollow {
		return s.PrintLogs(lines)
	}
	fmt.Println("Following the service log -- Ctrl+C to stop (the service keeps running):")
	cmd, err := s.StartTail(lines)
	if err != nil {
		return err
	}
	// Ctrl+C interrupts the log reader too -- that's the normal way to leave,
	// not an error worth reporting.
	_ = cmd.Wait()
	return nil
}
