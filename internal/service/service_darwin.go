package service

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
)

// domain is the launchd domain the job lives in: the calling user's GUI
// session for user scope, the system domain for system scope.
func (s Service) domain() string {
	if s.userScope() {
		return "gui/" + strconv.Itoa(os.Getuid())
	}
	return "system"
}

func (s Service) target() string { return s.domain() + "/" + s.Name }

// launchctl builds a launchctl invocation; the system domain needs root.
func (s Service) launchctl(args ...string) (string, []string) {
	if s.userScope() {
		return "launchctl", args
	}
	if s.NonInteractive {
		return "sudo", append([]string{"-n", "launchctl"}, args...)
	}
	return "sudo", append([]string{"launchctl"}, args...)
}

// printJob builds 'launchctl print' for the job. Reading a job's state needs
// no root, even in the system domain. sudo would fail without a terminal and
// report a serving job as inactive.
func (s Service) printJob() (string, []string) {
	return "launchctl", []string{"print", s.target()}
}

// loaded reports whether the job is registered in its domain. launchctl
// only accepts kickstart/kill for jobs it already knows about.
func (s Service) loaded() bool {
	bin, args := s.printJob()
	return exec.Command(bin, args...).Run() == nil
}

// IsActive maps launchd's job state onto the same vocabulary systemd uses, so
// callers don't have to care which init system answered.
func (s Service) IsActive() string {
	bin, args := s.printJob()
	out, err := exec.Command(bin, args...).CombinedOutput()
	if err != nil {
		// print fails when the job isn't loaded at all.
		return "inactive"
	}
	text := string(out)
	switch {
	case strings.Contains(text, "state = running"):
		return "active"
	case strings.Contains(text, "state = waiting"):
		// Loaded but not running: distinguish a clean stop from a crash.
		if code := lastExitStatus(text); code != 0 {
			return "failed"
		}
		return "inactive"
	}
	return "unknown"
}

// lastExitStatus pulls "last exit code = N" out of `launchctl print` output.
func lastExitStatus(text string) int {
	for line := range strings.SplitSeq(text, "\n") {
		_, v, ok := strings.Cut(strings.TrimSpace(line), "last exit code = ")
		if !ok {
			continue
		}
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return 0
		}
		return n
	}
	return 0
}

// NRestarts approximates systemd's restart counter from launchd's run
// count, so crash-loop detection works the same way. Returns -1 when the
// count isn't available, which callers treat as "can't tell".
func (s Service) NRestarts() int {
	bin, args := s.printJob()
	out, err := exec.Command(bin, args...).CombinedOutput()
	if err != nil {
		return -1
	}
	return runCount(string(out))
}

// runCount pulls "runs = N" out of `launchctl print` output, or -1.
func runCount(text string) int {
	for line := range strings.SplitSeq(text, "\n") {
		_, v, ok := strings.Cut(strings.TrimSpace(line), "runs = ")
		if !ok {
			continue
		}
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n
		}
	}
	return -1
}

// Health maps launchd's job state onto the same shape systemd reports, so
// callers can tell a slow boot from a crash loop without caring which init
// system answered.
func (s Service) Health() Health {
	bin, args := s.printJob()
	out, err := exec.Command(bin, args...).CombinedOutput()
	if err != nil {
		// print fails when the job isn't loaded at all.
		return Health{State: "inactive", NRestarts: -1}
	}
	return healthFromPrint(string(out))
}

// healthFromPrint is the pure half of Health.
func healthFromPrint(text string) Health {
	h := Health{NRestarts: -1}
	if n := runCount(text); n >= 0 {
		h.NRestarts = n
	}
	exit := lastExitStatus(text)
	if exit != 0 {
		h.Result = fmt.Sprintf("exit-code %d", exit)
	}
	switch {
	case strings.Contains(text, "state = running"):
		h.State, h.SubState = "active", "running"
	case strings.Contains(text, "state = waiting"):
		// Loaded but not running. launchd throttles a job that keeps
		// dying, so a non-zero last exit here is the crash loop.
		h.SubState = "waiting"
		if exit != 0 {
			h.State, h.Crashed = "failed", true
		} else {
			h.State = "inactive"
		}
	default:
		h.State = "unknown"
	}
	return h
}

// Restart reloads the model. kickstart -k kills the running job and starts
// it again, the launchd equivalent of `systemctl restart`. A job that was
// booted out (by Stop) has to be bootstrapped back in first, which makes
// restart-after-stop work the way it does under systemd.
func (s Service) Restart() error {
	if !s.loaded() {
		return s.Start()
	}
	bin, args := s.launchctl("kickstart", "-k", s.target())
	cmd := exec.Command(bin, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

// Stop unloads the job. Signalling the process instead would not stick:
// the plist's KeepAlive brings it straight back after a non-zero exit.
func (s Service) Stop() error {
	if !s.loaded() {
		return nil
	}
	bin, args := s.launchctl("bootout", s.target())
	cmd := exec.Command(bin, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

// Start loads the job back into its domain; RunAtLoad launches it.
func (s Service) Start() error {
	if s.loaded() {
		return nil
	}
	bin, args := s.launchctl("bootstrap", s.domain(), s.plistPath())
	cmd := exec.Command(bin, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return s.explainBootstrap(err)
	}
	return nil
}

// StartTail follows the job's log file. launchd has no journal, so the
// plist redirects stdout and stderr to a file we own. -F (rather than -f)
// waits for the file when the server hasn't written anything yet.
func (s Service) StartTail(n int) (*exec.Cmd, error) {
	cmd := exec.Command("tail", "-n", strconv.Itoa(n), "-F", s.logPath())
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// PrintLogs prints the last n log lines and returns.
func (s Service) PrintLogs(n int) error {
	if _, err := os.Stat(s.logPath()); err != nil {
		return fmt.Errorf("no log yet at %s -- the server hasn't run", s.logPath())
	}
	cmd := exec.Command("tail", "-n", strconv.Itoa(n), s.logPath())
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd.Run()
}

// LastLogs returns the last n log lines as a string (for failure reports);
// best-effort, "" if the log isn't readable.
func (s Service) LastLogs(n int) string {
	out, err := exec.Command("tail", "-n", strconv.Itoa(n), s.logPath()).Output()
	if err != nil {
		return ""
	}
	return strings.TrimRight(string(out), "\n")
}

// LogHint tells the user how to read the server's own output.
func (s Service) LogHint() string {
	return "tail -n 100 " + s.logPath()
}

func (s Service) logDir() string {
	if s.userScope() {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, "Library", "Logs", "vllm-helper")
	}
	return "/Library/Logs/vllm-helper"
}

func (s Service) logPath() string {
	return filepath.Join(s.logDir(), s.Name+".log")
}

func (s Service) plistPath() string {
	if s.userScope() {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, "Library", "LaunchAgents", s.Name+".plist")
	}
	return filepath.Join("/Library/LaunchDaemons", s.Name+".plist")
}

// Installed reports whether the plist already exists.
func (s Service) Installed() bool {
	_, err := os.Stat(s.plistPath())
	return err == nil
}

// Install writes the launchd plist pointing at "<self> start", bootstraps it,
// and returns the path it wrote.
func (s Service) Install(self string) (string, error) {
	path := s.plistPath()
	plist, err := s.renderPlist(self)
	if err != nil {
		return "", err
	}

	if s.userScope() {
		for _, d := range []string{filepath.Dir(path), s.logDir()} {
			if err := os.MkdirAll(d, 0o755); err != nil {
				return "", err
			}
		}
		if err := os.WriteFile(path, plist, 0o644); err != nil {
			return "", err
		}
	} else {
		tmp, err := os.CreateTemp("", "vllm-*.plist")
		if err != nil {
			return "", err
		}
		defer os.Remove(tmp.Name())
		if _, err := tmp.Write(plist); err != nil {
			return "", err
		}
		tmp.Close()
		// launchd refuses to load a daemon plist that isn't root-owned and
		// non-writable by anyone else, and it can't spawn the job at all if
		// the log directory is missing. Letting any of these fail quietly
		// produces a service that installs "successfully" and then never
		// runs, so treat every one as fatal.
		u, err := user.Current()
		if err != nil {
			return "", fmt.Errorf("finding the user to run the daemon as: %w", err)
		}
		steps := [][]string{
			{"mkdir", "-p", s.logDir()},
			// The daemon runs as this user, who has to be able to write
			// the log launchd opens for it. The file is created and
			// chowned too: one left by an older daemon that ran as root
			// is root-owned, and the job could not open it.
			{"touch", s.logPath()},
			{"chown", u.Username, s.logDir(), s.logPath()},
			{"cp", tmp.Name(), path},
			{"chown", "root:wheel", path},
			{"chmod", "644", path},
		}
		for _, step := range steps {
			bin, args := s.sudo(step...)
			if err := run(os.Stdout, bin, args...); err != nil {
				return "", fmt.Errorf("installing plist -- 'sudo %s' failed (needs root): %w",
					strings.Join(step, " "), err)
			}
		}
	}

	// Replace any previous registration so a re-install picks up edits.
	// Only when there is one: booting out a job that was never loaded is an
	// error, and reporting it would be noise during a first install.
	if s.loaded() {
		bin, args := s.launchctl("bootout", s.target())
		run(nil, bin, args...)
	}
	bin, args := s.launchctl("bootstrap", s.domain(), path)
	if err := run(os.Stdout, bin, args...); err != nil {
		return "", s.explainBootstrap(fmt.Errorf("bootstrapping %s: %w", s.Name, err))
	}
	return path, nil
}

// explainBootstrap names the usual reason a LaunchAgent will not load: the
// user has no GUI login session, which is the normal state of a headless Mac
// reached over SSH. launchd reports it only as error 125.
func (s Service) explainBootstrap(err error) error {
	if !s.userScope() || exec.Command("launchctl", "print", s.domain()).Run() == nil {
		return err
	}
	return fmt.Errorf("%w\n%s has no GUI login session, so there is nowhere to load a LaunchAgent. "+
		"On a Mac that runs headless, set SERVICE_SCOPE=system in config.env and run 'vllm-helper setup' again: "+
		"that installs a LaunchDaemon that starts at boot and runs as you", err, s.domain())
}

func (s Service) renderPlist(self string) ([]byte, error) {
	esc := func(v string) (string, error) {
		var b bytes.Buffer
		if err := xml.EscapeText(&b, []byte(v)); err != nil {
			return "", err
		}
		return b.String(), nil
	}
	label, err := esc(s.Name)
	if err != nil {
		return nil, err
	}
	prog, err := esc(self)
	if err != nil {
		return nil, err
	}
	log, err := esc(s.logPath())
	if err != nil {
		return nil, err
	}

	// A LaunchDaemon runs as root unless told otherwise, and as root the
	// server would read root's config and model cache, not the ones setup
	// wrote. Run it as the installing user, as the systemd unit's User= does.
	// HOME is set too: launchd does not derive it from UserName for daemons
	// on every macOS release, and vllm-helper and Hugging Face both need it.
	daemonUser, daemonHome := "", ""
	if !s.userScope() {
		u, err := user.Current()
		if err != nil {
			return nil, fmt.Errorf("finding the user to run the daemon as: %w", err)
		}
		name, err := esc(u.Username)
		if err != nil {
			return nil, err
		}
		home, err := esc(u.HomeDir)
		if err != nil {
			return nil, err
		}
		daemonUser = fmt.Sprintf("\t<key>UserName</key>\n\t<string>%s</string>\n", name)
		daemonHome = fmt.Sprintf("\t\t<key>HOME</key>\n\t\t<string>%s</string>\n", home)
	}

	// ProcessType Interactive keeps launchd from CPU-throttling the job the
	// way it does background work -- a model server needs the full machine.
	// KeepAlive/SuccessfulExit=false mirrors systemd's Restart=on-failure.
	body := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>%s</string>
	<key>ProgramArguments</key>
	<array>
		<string>%s</string>
		<string>start</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key>
		<false/>
	</dict>
	<key>ProcessType</key>
	<string>Interactive</string>
%s	<key>StandardOutPath</key>
	<string>%s</string>
	<key>StandardErrorPath</key>
	<string>%s</string>
	<key>EnvironmentVariables</key>
	<dict>
%s		<key>PATH</key>
		<string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
	</dict>
</dict>
</plist>
`, label, prog, daemonUser, log, log, daemonHome)
	return []byte(body), nil
}

func (s Service) Label() string { return "launchd" }
