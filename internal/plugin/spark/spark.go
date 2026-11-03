package spark

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/plugin"
)

func init() { plugin.Register(Plugin{}) }

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
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			checks := gatherChecks()
			anyMissing := false
			for _, c := range checks {
				mark := "ok"
				if !c.OK {
					mark = "missing"
					anyMissing = true
				}
				fmt.Printf("%-18s %s", c.Name, mark)
				if !c.OK && c.Package != "" {
					fmt.Printf(" (install %s)", c.Package)
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
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			if runtime.GOOS != "linux" {
				return errors.New("Spark prerequisites are only installable on Linux")
			}
			if err := run("sudo", "apt-get", "update"); err != nil {
				return err
			}
			return run("sudo", "env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "build-essential", "python3.12-dev", "ninja-build")
		},
	}
}

type prereqCheck struct {
	Name    string
	OK      bool
	Package string
}

func gatherChecks() []prereqCheck {
	return []prereqCheck{
		{Name: "gcc", OK: commandOK("gcc"), Package: "build-essential"},
		{Name: "Python.h", OK: fileOK("/usr/include/python3.12/Python.h"), Package: "python3.12-dev"},
		{Name: "ninja", OK: commandOK("ninja"), Package: "ninja-build"},
		{Name: "GB10 GPU", OK: strings.Contains(nvidiaSMI(), "GB10")},
	}
}

func commandOK(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

func fileOK(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func nvidiaSMI() string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	if err != nil {
		return ""
	}
	return string(out)
}

func run(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}
