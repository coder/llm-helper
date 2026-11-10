// Package service manages the long-running model servers through whatever init
// system the host provides: systemd on Linux, launchd on macOS. Every
// implementation honors the configured scope, "system" or "user".
package service

import (
	"os"
	"os/exec"
)

// ExitNoFit is the exit code of an instance whose model does not fit in the
// memory left. The init system does not restart it.
const ExitNoFit = 78

// Service identifies the unit (systemd) or job (launchd) running the server.
type Service struct {
	Name  string // systemd unit name, or launchd label
	Scope string // "system" or "user"

	// Args, when set, is what the unit runs after the binary instead of
	// "start --instance <profile>".
	Args []string

	// NonInteractive makes privileged calls fail fast (sudo -n) instead of
	// prompting for a password nothing can answer (scripts, the UI daemon).
	NonInteractive bool
}

// Health is what the init system knows about the unit beyond "is it active".
// It exists so callers can tell a model that is still loading (which can take
// many minutes) apart from one that is crash-looping -- both leave the
// endpoint unresponsive, and reporting the second as the first sends people
// off to wait for something that is never going to come up.
type Health struct {
	State     string `json:"state"`     // same vocabulary as IsActive
	SubState  string `json:"sub_state"` // init-system detail, "" when there is none
	Result    string `json:"result"`    // why it last stopped, "" when unknown
	NRestarts int    `json:"nrestarts"` // -1 when it can't be read
	Crashed   bool   `json:"crashed"`   // the unit failed or is restarting after a failure
}

func (s Service) userScope() bool { return s.Scope == "user" }

// sudo builds a privileged invocation. Without a terminal to answer a
// password prompt, fail fast rather than block on one forever.
func (s Service) sudo(args ...string) (string, []string) {
	if s.NonInteractive {
		return "sudo", append([]string{"-n"}, args...)
	}
	return "sudo", args
}

// run executes a command. Stderr always reaches the terminal: that is where
// sudo writes its password prompt and where anything that went wrong shows
// up, and swallowing either turns a failure into an unexplained hang. out
// may be nil to discard ordinary output.
func run(out *os.File, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdin, cmd.Stderr = os.Stdin, os.Stderr
	if out != nil {
		cmd.Stdout = out
	}
	return cmd.Run()
}
