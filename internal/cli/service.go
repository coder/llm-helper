package cli

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/profile"
)

func newRestartCmd() *cobra.Command {
	var w waitFlags
	c := &cobra.Command{
		Use:     "restart [profile]",
		Short:   "Re-apply a running profile and restart its server",
		GroupID: groupServer,
		Args:    cobra.MaximumNArgs(1),
		RunE:    func(_ *cobra.Command, args []string) error { return runRestart(args, w.opts()) },
	}
	w.register(c)
	return c
}

// runRestart re-applies a running profile, picking up edits to it or to
// config.env, and restarts it. It never touches Coder, which makes it the
// right verb for a supervisor: a recovery brings the same model back.
func runRestart(args []string, o watchOpts) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	inst, err := pickInstance(cfg, args)
	if err != nil {
		return err
	}
	p, err := profile.Load(cfg.Dirs, inst.Name)
	if err != nil {
		return fmt.Errorf("profile '%s' can't be loaded (%w); stop it with '%s down %s'", inst.Name, err, cmdName, inst.Name)
	}
	if err := control.WriteInstanceEnv(cfg, p, inst.Opts()); err != nil {
		return err
	}
	r := instanceRunner(cfg, inst.Name)
	fmt.Printf("Reloading profile '%s' (%s)...\n", inst.Name, r.Label())
	if err := r.Restart(); err != nil {
		return fmt.Errorf("restarting service: %w", err)
	}
	if !o.follow && !o.wait {
		fmt.Println("Not waiting for the model to load. Check progress with 'llm-helper status'.")
		return nil
	}
	fmt.Println()
	return watchBoot(cfg, inst.Name, o)
}

func newLogsCmd() *cobra.Command {
	var (
		lines    int
		follow   bool
		noFollow bool
	)
	c := &cobra.Command{
		Use:     "logs [profile]",
		Short:   "Show a running model's server log, following by default on a terminal",
		GroupID: groupServer,
		Args:    cobra.MaximumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			return runLogs(args, lines, follow, noFollow)
		},
	}
	c.Flags().IntVarP(&lines, "lines", "n", 50, "how many past lines to show")
	c.Flags().BoolVar(&follow, "follow", false, "keep following even when output is piped")
	c.Flags().BoolVar(&noFollow, "no-follow", false, "print the lines and exit instead of following")
	return c
}

// runLogs shows an instance's log: the last -n lines, following by default
// on a terminal. Piped output prints and exits so scripts never hang.
func runLogs(args []string, lines int, follow, noFollow bool) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	inst, err := pickInstance(cfg, args)
	if err != nil {
		return err
	}
	s := instanceRunner(cfg, inst.Name)
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
	// Ctrl+C is the normal way to leave, not an error.
	_ = cmd.Wait()
	return nil
}
