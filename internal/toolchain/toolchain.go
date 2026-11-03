// Package toolchain checks for, and installs, the build tools vLLM compiles
// kernels with on this machine. Which tools those are depends on the
// platform, not on a particular machine: every Linux GPU box JIT-compiles
// Triton and FlashInfer kernels, and a Mac without vllm-metal builds vLLM's
// CPU backend from source. Doctor and the machine-specific helpers share it,
// so they agree on what is missing and how to install it.
package toolchain

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
	"time"
)

// Check is one build prerequisite and whether it is present.
type Check struct {
	Name    string `json:"name"`
	OK      bool   `json:"ok"`
	Package string `json:"package,omitempty"` // apt package on Linux, brew formula on macOS
	Hint    string `json:"hint,omitempty"`    // fix for what a package manager can't install
}

// Missing lists the packages that would install the checks that failed, in
// order and without repeats.
func Missing(checks []Check) []string {
	var out []string
	seen := map[string]bool{}
	for _, c := range checks {
		if !c.OK && c.Package != "" && !seen[c.Package] {
			seen[c.Package] = true
			out = append(out, c.Package)
		}
	}
	return out
}

// Install installs pkgs with this platform's package manager, attached to
// the terminal. Without a terminal to answer a sudo prompt, sudo is told not
// to ask, so a script fails fast instead of hanging.
func Install(pkgs []string, interactive bool) error {
	if len(pkgs) == 0 {
		return nil
	}
	installer, args := Installer()
	if installer == "" {
		return errors.New("no supported package manager found; install these yourself: " + strings.Join(pkgs, " "))
	}
	if installer == "sudo" && !interactive {
		args = append([]string{"-n"}, args...)
	}
	cmd := exec.Command(installer, append(args, pkgs...)...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func hasBins(bins ...string) bool {
	for _, b := range bins {
		if _, err := exec.LookPath(b); err != nil {
			return false
		}
	}
	return true
}

// commandTimeout bounds asking an interpreter about itself, so a wedged or
// misconfigured Python cannot hang doctor.
var commandTimeout = 10 * time.Second

func output(name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), commandTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.WaitDelay = time.Second
	out, err := cmd.Output()
	return strings.TrimSpace(string(out)), err
}
