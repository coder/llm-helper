package cli

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/profile"
)

func newStartCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "start",
		Short: "Run the active model in the foreground",
		// The service manager's ExecStart, not something to type: it execs
		// vLLM over this process and never returns.
		Hidden: true,
		Args:   cobra.NoArgs,
		RunE:   func(*cobra.Command, []string) error { return runStart() },
	}
}

// runStart is what the service manager runs: source the generated active.env
// and exec the active profile's server, replacing this process so signals flow
// straight to it.
func runStart() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	active, err := config.ParseEnvFile(cfg.Dirs.ActiveEnvFile())
	switch {
	case errors.Is(err, os.ErrNotExist):
		// Exit cleanly instead of failing. The unit gets installed before the
		// first switch, and a non-zero exit here would send systemd's
		// Restart=on-failure -- or launchd's KeepAlive -- into a crash loop
		// with no model to serve. Only a missing file means "not set up yet";
		// anything else is a fault worth reporting.
		fmt.Println("No active model yet -- run 'vllm-helper switch <profile>' to load one.")
		return nil
	case err != nil:
		return fmt.Errorf("reading the active model config at %s: %w", cfg.Dirs.ActiveEnvFile(), err)
	}

	// A recipe-generated profile describes a container launch, so serving it
	// natively here would start vLLM without the image's flags and quietly
	// produce a different server than the one that was verified.
	if id := active.Get(profile.RecipeKey); strings.TrimSpace(id) != "" {
		return fmt.Errorf("the active profile is generated from recipe %q and runs in a container; start it with 'vllm-helper recipes launch %s'", strings.TrimSpace(id), strings.TrimSpace(id))
	}

	spec, err := serveSpecFromActive(active)
	if err != nil {
		return err
	}
	args, err := spec.args()
	if err != nil {
		return err
	}
	return syscall.Exec(spec.Bin, args, withProviderEnv(serveEnvFromActive(active), spec))
}
