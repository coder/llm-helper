package cli

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
)

// ggufMissingMessage explains the gap and how to close it, naming the
// environment rather than a generic pip so the package cannot land in the
// wrong interpreter.
func ggufMissingMessage(cfg *config.Config) string {
	cmd := "<your-vllm-venv>/bin/python -m pip install " + control.GGUFPlugin
	if cfg != nil {
		if c := control.GGUFInstallCommand(cfg); c != "" {
			cmd = c
		}
	}
	return fmt.Sprintf(
		"this vLLM can't load GGUF: support moved out of vLLM core in 0.27.1 and needs %s.\n"+
			"  Install it with 'llm-helper install-gguf', or by hand:\n    %s",
		control.GGUFPlugin, cmd)
}

func newInstallGGUFCmd() *cobra.Command {
	var yes bool
	c := &cobra.Command{
		Use:     "install-gguf",
		Short:   "Install the plugin that lets vLLM load GGUF models",
		GroupID: groupModels,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runInstallGGUF(yes) },
	}
	c.Flags().BoolVarP(&yes, "yes", "y", false, "install without asking")
	return c
}

func runInstallGGUF(yes bool) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	cmd := control.GGUFInstallCommand(cfg)
	if cmd == "" {
		return fmt.Errorf("couldn't find the interpreter next to %s -- install by hand:\n"+
			"  <your-vllm-venv>/bin/python -m pip install %s", cfg.VLLMBin(), control.GGUFPlugin)
	}
	if vllmCaps(cfg).SupportsGGUF() {
		fmt.Println("This vLLM already loads GGUF. Nothing to do.")
		return nil
	}

	fmt.Println("GGUF support left vLLM core in 0.27.1 and now lives in a separate package.")
	fmt.Println()
	fmt.Println("  " + cmd)
	fmt.Println()
	fmt.Println("Worth knowing first:")
	for _, c := range control.GGUFCaveats() {
		fmt.Println("  - " + c)
	}
	fmt.Println()
	if !yes && !confirm("Install it?", false) {
		fmt.Println("Left alone.")
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(),
		control.GGUFInstallTimeout*time.Second)
	defer cancel()
	if err := control.InstallGGUF(ctx, cfg, os.Stdout); err != nil {
		return err
	}

	control.ForgetCaps(cfg.Dirs)
	fmt.Println()
	if vllmCaps(cfg).SupportsGGUF() {
		fmt.Println("Done -- this vLLM can now load GGUF models.")
		fmt.Println("It takes effect when an instance next starts; nothing running was disturbed.")
		return nil
	}
	return fmt.Errorf("installed, but vLLM still doesn't register 'gguf'.\n" +
		"The plugin may need a matching vLLM version, or the build may have failed above")
}
