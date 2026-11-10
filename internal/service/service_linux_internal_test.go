//go:build linux

package service

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These paths belong to units that already exist, so they are pinned.
func TestUnitPath(t *testing.T) {
	home := t.TempDir()

	t.Run("user scope follows XDG_CONFIG_HOME", func(t *testing.T) {
		t.Setenv("HOME", home)
		t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, "xdg"))
		got := (Service{Name: "vllm.service", Scope: "user"}).unitPath()
		want := filepath.Join(home, "xdg", "systemd", "user", "vllm.service")
		if got != want {
			t.Errorf("unitPath() = %q, want %q", got, want)
		}
	})

	t.Run("user scope falls back to ~/.config", func(t *testing.T) {
		t.Setenv("HOME", home)
		t.Setenv("XDG_CONFIG_HOME", "")
		got := (Service{Name: "vllm.service", Scope: "user"}).unitPath()
		want := filepath.Join(home, ".config", "systemd", "user", "vllm.service")
		if got != want {
			t.Errorf("unitPath() = %q, want %q", got, want)
		}
	})

	t.Run("an instance unit lives in its template", func(t *testing.T) {
		got := (Service{Name: InstanceName("qwen"), Scope: "system"}).unitPath()
		if want := "/etc/systemd/system/llm-helper@.service"; got != want {
			t.Errorf("unitPath() = %q, want %q", got, want)
		}
	})

	t.Run("system scope", func(t *testing.T) {
		got := (Service{Name: "vllm.service", Scope: "system"}).unitPath()
		if want := "/etc/systemd/system/vllm.service"; got != want {
			t.Errorf("unitPath() = %q, want %q", got, want)
		}
	})
}

func TestSystemctlArgs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		svc     Service
		wantBin string
		want    string
	}{{
		name:    "user scope needs no privileges",
		svc:     Service{Name: "vllm.service", Scope: "user"},
		wantBin: "systemctl",
		want:    "--user restart vllm.service",
	}, {
		name:    "system scope goes through sudo",
		svc:     Service{Name: "vllm.service", Scope: "system"},
		wantBin: "sudo",
		want:    "systemctl restart vllm.service",
	}, {
		name:    "system scope without a terminal fails fast",
		svc:     Service{Name: "vllm.service", Scope: "system", NonInteractive: true},
		wantBin: "sudo",
		want:    "-n systemctl restart vllm.service",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			bin, args := tt.svc.systemctlArgs("restart", tt.svc.Name)
			if bin != tt.wantBin {
				t.Errorf("bin = %q, want %q", bin, tt.wantBin)
			}
			if got := strings.Join(args, " "); got != tt.want {
				t.Errorf("args = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLogHint(t *testing.T) {
	t.Parallel()

	tests := []struct {
		scope string
		want  string
	}{
		{"user", "journalctl --user -u vllm.service -n 100"},
		{"system", "journalctl -u vllm.service -n 100"},
	}
	for _, tt := range tests {
		t.Run(tt.scope, func(t *testing.T) {
			t.Parallel()
			if got := (Service{Name: "vllm.service", Scope: tt.scope}).LogHint(); got != tt.want {
				t.Errorf("LogHint() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestJournalArgs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		scope string
		want  string
	}{
		{"user", "--user -u vllm.service -o cat -n 50"},
		{"system", "-u vllm.service -o cat -n 50"},
	}
	for _, tt := range tests {
		t.Run(tt.scope, func(t *testing.T) {
			t.Parallel()
			got := (Service{Name: "vllm.service", Scope: tt.scope}).journalArgs("-n", "50")
			if strings.Join(got, " ") != tt.want {
				t.Errorf("journalArgs = %v, want %q", got, tt.want)
			}
		})
	}
}

// A user-scope install needs no root, so the unit it writes is checkable
// for real. The system-scope path needs sudo and is left to the machine.
func TestInstallUserScope(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, "xdg"))
	// Keep the install off the host: no systemctl or loginctl to find.
	t.Setenv("PATH", t.TempDir())
	t.Setenv("USER", "")

	s := Service{Name: InstanceName("qwen"), Scope: "user"}
	if s.Installed() {
		t.Fatal("Installed() should be false before Install")
	}

	path, err := s.Install("/opt/bin/llm-helper")
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if want := s.unitPath(); path != want {
		t.Errorf("Install returned %q, want %q", path, want)
	}
	if !s.Installed() {
		t.Error("Installed() should be true after Install")
	}
	if !(Service{Name: InstanceName("other"), Scope: "user"}).Installed() {
		t.Error("every instance shares the template unit")
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	unit := string(raw)
	for _, want := range []string{
		"ExecStart=/opt/bin/llm-helper start --instance %i",
		"Restart=on-failure",
		"RestartPreventExitStatus=78",
		"WantedBy=default.target",
		"Type=simple",
	} {
		if !strings.Contains(unit, want) {
			t.Errorf("unit is missing %q:\n%s", want, unit)
		}
	}
	if strings.Contains(unit, "User=") {
		t.Errorf("a user-scope unit already runs as the user:\n%s", unit)
	}
}

func TestRenderUnitWithArgs(t *testing.T) {
	t.Setenv("USER", "dev")
	unit := (Service{Name: RouterName, Scope: "system", Args: []string{"router", "serve", "--port", "8000"}}).renderUnit("/opt/bin/llm-helper")
	for _, want := range []string{
		"ExecStart=/opt/bin/llm-helper router serve --port 8000",
		"Restart=on-failure",
		"User=dev",
		"WantedBy=multi-user.target",
	} {
		if !strings.Contains(unit, want) {
			t.Errorf("unit is missing %q:\n%s", want, unit)
		}
	}
	if strings.Contains(unit, "%i") || strings.Contains(unit, "RestartPreventExitStatus") {
		t.Errorf("router unit should not be an instance template:\n%s", unit)
	}
	if (Service{Name: RouterName}).unitPath() != "/etc/systemd/system/"+RouterName {
		t.Error("router unit is not a template")
	}
}
