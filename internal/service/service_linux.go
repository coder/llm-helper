package service

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

func (s Service) systemctlArgs(args ...string) (string, []string) {
	if s.userScope() {
		return "systemctl", append([]string{"--user"}, args...)
	}
	if s.NonInteractive {
		return "sudo", append([]string{"-n", "systemctl"}, args...)
	}
	return "sudo", append([]string{"systemctl"}, args...)
}

// IsActive returns systemd's activity string ("active", "inactive",
// "failed", ...) without needing privileges.
func (s Service) IsActive() string {
	args := []string{"is-active", s.Name}
	if s.userScope() {
		args = append([]string{"--user"}, args...)
	}
	out, _ := exec.Command("systemctl", args...).CombinedOutput()
	state := strings.TrimSpace(string(out))
	if state == "" {
		state = "unknown"
	}
	return state
}

// NRestarts returns the unit's restart counter, or -1 if it can't be read.
// A counter that climbs right after a restart means a crash loop.
func (s Service) NRestarts() int {
	args := []string{"show", "-p", "NRestarts", "--value", s.Name}
	if s.userScope() {
		args = append([]string{"--user"}, args...)
	}
	out, err := exec.Command("systemctl", args...).Output()
	if err != nil {
		return -1
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		return -1
	}
	return n
}

// healthProperties are the unit properties that, together, say whether the
// service is coming up or falling over.
var healthProperties = []string{"ActiveState", "SubState", "Result", "NRestarts"}

// Health asks systemd for the properties that distinguish "still loading"
// from "crash-looping".
func (s Service) Health() Health {
	args := []string{"show"}
	for _, p := range healthProperties {
		args = append(args, "-p", p)
	}
	args = append(args, s.Name)
	if s.userScope() {
		args = append([]string{"--user"}, args...)
	}
	out, err := exec.Command("systemctl", args...).Output()
	if err != nil {
		return Health{State: "unknown", NRestarts: -1}
	}
	return healthFromProperties(parseShowProperties(string(out)))
}

// parseShowProperties turns `systemctl show` KEY=VALUE lines into a map.
// --value would be terser but ties us to systemd emitting the properties in
// the order we asked for, which it does not promise.
func parseShowProperties(out string) map[string]string {
	props := make(map[string]string)
	for line := range strings.SplitSeq(out, "\n") {
		k, v, ok := strings.Cut(strings.TrimSpace(line), "=")
		if ok && k != "" {
			props[k] = v
		}
	}
	return props
}

// healthFromProperties is the pure half of Health.
func healthFromProperties(props map[string]string) Health {
	h := Health{
		State:     props["ActiveState"],
		SubState:  props["SubState"],
		Result:    props["Result"],
		NRestarts: -1,
	}
	if h.State == "" {
		h.State = "unknown"
	}
	if n, err := strconv.Atoi(strings.TrimSpace(props["NRestarts"])); err == nil {
		h.NRestarts = n
	}
	// Only unambiguous signals count. Result alone does not: it still holds
	// the previous run's verdict while a freshly restarted unit is
	// activating, which would flag every normal switch as a crash.
	switch {
	case h.State == "failed":
		// Gave up: hit the start limit, or a one-shot failure.
		h.Crashed = true
	case h.SubState == "auto-restart":
		// Sleeping between attempts of a restart loop. ActiveState reads
		// "activating" here, which is exactly what makes a crash loop
		// look like a slow boot.
		h.Crashed = true
	}
	return h
}

// Restart restarts the unit, streaming any sudo prompt to the terminal.
func (s Service) Restart() error { return s.runPrivileged("restart") }

// Stop stops the unit.
func (s Service) Stop() error { return s.runPrivileged("stop") }

// Start starts the unit.
func (s Service) Start() error { return s.runPrivileged("start") }

func (s Service) runPrivileged(verb string) error {
	bin, args := s.systemctlArgs(verb, s.Name)
	cmd := exec.Command(bin, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func (s Service) journalArgs(extra ...string) []string {
	args := []string{"-u", s.Name, "-o", "cat"}
	if s.userScope() {
		args = append([]string{"--user"}, args...)
	}
	return append(args, extra...)
}

// StartTail spawns journalctl following the unit's log (last n lines of
// context first), writing to this process's stdout. The caller stops
// watching by killing the returned process; the service itself is
// unaffected.
func (s Service) StartTail(n int) (*exec.Cmd, error) {
	cmd := exec.Command("journalctl", s.journalArgs("-n", strconv.Itoa(n), "-f")...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// PrintLogs prints the last n journal lines and returns.
func (s Service) PrintLogs(n int) error {
	cmd := exec.Command("journalctl", s.journalArgs("-n", strconv.Itoa(n))...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd.Run()
}

// LastLogs returns the last n journal lines as a string (for failure
// reports); best-effort, "" if the journal isn't readable.
func (s Service) LastLogs(n int) string {
	out, err := exec.Command("journalctl", s.journalArgs("-n", strconv.Itoa(n))...).Output()
	if err != nil {
		return ""
	}
	return strings.TrimRight(string(out), "\n")
}

// LogHint tells the user how to read the server's own output.
func (s Service) LogHint() string {
	if s.userScope() {
		return fmt.Sprintf("journalctl --user -u %s -n 100", s.Name)
	}
	return fmt.Sprintf("journalctl -u %s -n 100", s.Name)
}

func (s Service) unitPath() string {
	if s.userScope() {
		home, _ := os.UserHomeDir()
		base := os.Getenv("XDG_CONFIG_HOME")
		if base == "" {
			base = filepath.Join(home, ".config")
		}
		return filepath.Join(base, "systemd", "user", s.Name)
	}
	return filepath.Join("/etc/systemd/system", s.Name)
}

// Installed reports whether the unit file already exists.
func (s Service) Installed() bool {
	_, err := os.Stat(s.unitPath())
	return err == nil
}

// Install renders and installs the unit, pointing it at "<self> start", and
// returns the path it wrote.
func (s Service) Install(self string) (string, error) {
	userLine, wantedBy := "User="+os.Getenv("USER"), "multi-user.target"
	if s.userScope() {
		userLine, wantedBy = "", "default.target"
	}
	unit := fmt.Sprintf(`[Unit]
Description=vLLM OpenAI-compatible server (active model set via llm-helper)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
%s
ExecStart=%s start
Restart=on-failure
RestartSec=5
TimeoutStartSec=600

[Install]
WantedBy=%s
`, userLine, self, wantedBy)

	path := s.unitPath()
	if s.userScope() {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", err
		}
		if err := os.WriteFile(path, []byte(unit), 0o644); err != nil {
			return "", err
		}
		run(nil, "systemctl", "--user", "daemon-reload")
		run(nil, "systemctl", "--user", "enable", s.Name)
		if err := exec.Command("loginctl", "enable-linger", os.Getenv("USER")).Run(); err != nil {
			fmt.Printf("note: couldn't enable lingering -- the server stops at logout (fix: sudo loginctl enable-linger %s)\n", os.Getenv("USER"))
		}
		return path, nil
	}

	tmp, err := os.CreateTemp("", "vllm-unit-*.service")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.WriteString(unit); err != nil {
		return "", err
	}
	tmp.Close()
	if err := run(os.Stdout, "sudo", "cp", tmp.Name(), path); err != nil {
		return "", fmt.Errorf("installing unit (needs sudo): %w", err)
	}
	run(nil, "sudo", "chmod", "644", path)
	run(nil, "sudo", "systemctl", "daemon-reload")
	run(nil, "sudo", "systemctl", "enable", s.Name)
	return path, nil
}

func (s Service) Label() string { return "systemd" }
