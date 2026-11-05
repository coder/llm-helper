// Package spark adds DGX Spark commands: a prerequisite check and installer
// built on the platform toolchain checks every machine gets from doctor,
// plus a check that the GPU is the GB10 a Spark is built on.
package spark

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/plugin"
	"github.com/coder/llm-helper/internal/toolchain"
)

func init() { plugin.Register(Plugin{}) }

// Plugin contributes the spark command group.
type Plugin struct{}

func (Plugin) Name() string { return "spark" }

// Configured stays false: Spark support is opt-in command surface, not an
// integration that should receive lifecycle events on every machine.
func (Plugin) Configured(_ *config.Config) bool { return false }
func (Plugin) Events() []plugin.Event           { return nil }
func (Plugin) Handle(context.Context, plugin.Event, plugin.State) error {
	return nil
}
func (Plugin) Check(context.Context, *config.Config) error { return nil }

func (Plugin) Commands() []*cobra.Command {
	c := &cobra.Command{Use: "spark", Short: "DGX Spark helpers"}
	c.AddCommand(newDoctorCmd(), newInstallPrereqsCmd())
	return []*cobra.Command{c}
}

func newDoctorCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "doctor",
		Short: "Check DGX Spark vLLM runtime prerequisites",
		Long: "Check the build tools vLLM compiles kernels with, the same ones 'llm-helper doctor' checks,\n" +
			"and that this machine's GPU is a GB10.",
		Args: cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			anyMissing := false
			for _, c := range gatherChecks() {
				mark := "ok"
				if !c.OK {
					mark = "missing"
					anyMissing = true
				}
				fmt.Printf("%-18s %s", c.Name, mark)
				switch {
				case !c.OK && c.Package != "":
					fmt.Printf(" (install %s)", c.Package)
				case !c.OK && c.Hint != "":
					fmt.Printf(" (%s)", c.Hint)
				}
				fmt.Println()
			}
			if anyMissing {
				return errors.New("missing Spark prerequisites; run 'llm-helper spark install-prereqs'")
			}
			return nil
		},
	}
}

func newInstallPrereqsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "install-prereqs",
		Short: "Install DGX Spark JIT build prerequisites",
		Long:  "Install the missing build tools vLLM compiles kernels with, the same ones 'llm-helper doctor --yes' installs.",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			if runtime.GOOS != "linux" {
				return errors.New("Spark prerequisites are only installable on Linux")
			}
			missing := toolchain.Missing(toolchain.Gather(vllmBin()))
			if len(missing) == 0 {
				fmt.Println("All build prerequisites are installed.")
				return nil
			}
			interactive := term.IsTerminal(int(os.Stdin.Fd()))
			if installer, _ := toolchain.Installer(); installer == "sudo" {
				args := []string{"apt-get", "update"}
				if !interactive {
					args = append([]string{"-n"}, args...)
				}
				if err := run("sudo", args...); err != nil {
					return err
				}
			}
			return toolchain.Install(missing, interactive)
		},
	}
}

// gatherChecks is the platform's build toolchain plus the one thing that
// makes this a Spark: a GB10, as the hardware definitions recognise it.
func gatherChecks() []toolchain.Check {
	checks := toolchain.Gather(vllmBin())
	return append(checks, toolchain.Check{Name: "GB10 GPU", OK: detect.Probe().HardwareID == sparkHardwareID})
}

// sparkHardwareID is the hardware definition every GB10 system matches.
const sparkHardwareID = "dgx_spark_gb10"

// vllmBin is the configured vllm, whose interpreter's headers are the ones
// that matter, or "" before setup has run.
func vllmBin() string {
	cfg, err := config.Load()
	if err != nil {
		return ""
	}
	return cfg.VLLMBin()
}

func run(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}
