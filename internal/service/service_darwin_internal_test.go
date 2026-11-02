//go:build darwin

package service

import (
	"encoding/xml"
	"os"
	"os/user"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// plistFile is enough of the plist to prove the interesting parts survived
// XML encoding.
type plistFile struct {
	XMLName xml.Name `xml:"plist"`
	Dict    struct {
		Keys    []string `xml:"key"`
		Strings []string `xml:"string"`
		Array   struct {
			Strings []string `xml:"string"`
		} `xml:"array"`
	} `xml:"dict"`
}

func parsePlist(t *testing.T, raw []byte) plistFile {
	t.Helper()
	var p plistFile
	if err := xml.Unmarshal(raw, &p); err != nil {
		t.Fatalf("launchd would reject this plist: %v\n%s", err, raw)
	}
	return p
}

func TestRenderPlist(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	s := Service{Name: "com.coder.vllm", Scope: "user"}
	raw, err := s.renderPlist("/opt/homebrew/bin/vllm-helper")
	if err != nil {
		t.Fatalf("renderPlist: %v", err)
	}
	p := parsePlist(t, raw)

	if got := p.Dict.Strings[0]; got != "com.coder.vllm" {
		t.Errorf("Label = %q, want the service name", got)
	}
	wantArgs := []string{"/opt/homebrew/bin/vllm-helper", "start"}
	if strings.Join(p.Dict.Array.Strings, " ") != strings.Join(wantArgs, " ") {
		t.Errorf("ProgramArguments = %v, want %v", p.Dict.Array.Strings, wantArgs)
	}
	for _, key := range []string{"Label", "ProgramArguments", "RunAtLoad", "KeepAlive", "ProcessType", "StandardOutPath", "StandardErrorPath"} {
		if !slices.Contains(p.Dict.Keys, key) {
			t.Errorf("plist is missing the %s key: %v", key, p.Dict.Keys)
		}
	}
	log := filepath.Join(home, "Library", "Logs", "vllm-helper", "com.coder.vllm.log")
	if !strings.Contains(string(raw), log) {
		t.Errorf("plist should redirect output to %s:\n%s", log, raw)
	}
}

// A LaunchDaemon runs as root unless the plist names a user, and as root the
// server would read root's config and model cache rather than the ones setup
// wrote for the installing user.
func TestRenderPlistRunsDaemonAsInstallingUser(t *testing.T) {
	me, err := user.Current()
	if err != nil {
		t.Skipf("no current user: %v", err)
	}
	raw, err := (Service{Name: "com.coder.vllm", Scope: "system"}).renderPlist("/usr/local/bin/vllm-helper")
	if err != nil {
		t.Fatalf("renderPlist: %v", err)
	}
	p := parsePlist(t, raw)
	if !slices.Contains(p.Dict.Keys, "UserName") {
		t.Fatalf("system-scope plist has no UserName, so it would run as root:\n%s", raw)
	}
	if !slices.Contains(p.Dict.Strings, me.Username) {
		t.Errorf("UserName should be %q:\n%s", me.Username, raw)
	}
	if !strings.Contains(string(raw), "<key>HOME</key>\n\t\t<string>"+me.HomeDir+"</string>") {
		t.Errorf("system-scope plist should set HOME to %s:\n%s", me.HomeDir, raw)
	}

	// A LaunchAgent already runs as its user, so it names none.
	t.Setenv("HOME", t.TempDir())
	raw, err = (Service{Name: "com.coder.vllm", Scope: "user"}).renderPlist("/usr/local/bin/vllm-helper")
	if err != nil {
		t.Fatalf("renderPlist: %v", err)
	}
	if slices.Contains(parsePlist(t, raw).Dict.Keys, "UserName") {
		t.Errorf("user-scope plist should not set UserName:\n%s", raw)
	}
}

// A label or path with XML metacharacters must not break the file launchd
// parses.
func TestRenderPlistEscapesXML(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	const (
		label = `a&b<c"d`
		self  = `/tmp/a & b/<vllm>-"helper"`
	)
	raw, err := (Service{Name: label, Scope: "user"}).renderPlist(self)
	if err != nil {
		t.Fatalf("renderPlist: %v", err)
	}
	p := parsePlist(t, raw)

	if got := p.Dict.Strings[0]; got != label {
		t.Errorf("Label = %q, want %q back after escaping", got, label)
	}
	if got := p.Dict.Array.Strings[0]; got != self {
		t.Errorf("ProgramArguments[0] = %q, want %q back after escaping", got, self)
	}
}

func TestLastExitStatus(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		text string
		want int
	}{{
		name: "clean stop",
		text: "com.coder.vllm = {\n\tstate = waiting\n\truns = 3\n\tlast exit code = 0\n}\n",
		want: 0,
	}, {
		name: "crashed",
		text: "com.coder.vllm = {\n\tstate = waiting\n\truns = 3\n\tlast exit code = 2\n}\n",
		want: 2,
	}, {
		name: "killed by a signal",
		text: "com.coder.vllm = {\n\tlast exit code = 137\n}\n",
		want: 137,
	}, {
		name: "never exited",
		text: "com.coder.vllm = {\n\tstate = running\n\tlast exit code = (never exited)\n}\n",
		want: 0,
	}, {
		name: "no such line",
		text: "com.coder.vllm = {\n\tstate = running\n\truns = 1\n}\n",
		want: 0,
	}, {
		name: "empty",
		text: "",
		want: 0,
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := lastExitStatus(tt.text); got != tt.want {
				t.Errorf("lastExitStatus = %d, want %d", got, tt.want)
			}
		})
	}
}

// A user-scope install needs no root, so the plist it writes is checkable for
// real. With no launchctl on PATH the bootstrap at the end fails, which is
// where this stops: registering the job with the running launchd session is
// left to the machine. The file itself, and the directories launchd needs to
// exist before it will spawn anything, are asserted here.
func TestInstallUserScope(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("PATH", t.TempDir())

	s := Service{Name: "com.coder.vllm-test", Scope: "user"}
	if s.Installed() {
		t.Fatal("Installed() should be false before Install")
	}

	_, err := s.Install("/opt/homebrew/bin/vllm-helper")
	if err == nil {
		t.Fatal("Install should report the failure to bootstrap when launchctl is absent")
	}
	if !strings.Contains(err.Error(), "bootstrapping") {
		t.Errorf("error should name the step that failed, got %v", err)
	}

	// Install writes the plist before it bootstraps, so a failed bootstrap
	// still leaves the file behind and Installed() reads true. Anything that
	// treats Installed() as "the job is running" is wrong.
	if !s.Installed() {
		t.Error("the plist should be on disk even though bootstrap failed")
	}
	raw, err := os.ReadFile(s.plistPath())
	if err != nil {
		t.Fatalf("Install should have written %s: %v", s.plistPath(), err)
	}
	p := parsePlist(t, raw)
	wantArgs := []string{"/opt/homebrew/bin/vllm-helper", "start"}
	if strings.Join(p.Dict.Array.Strings, " ") != strings.Join(wantArgs, " ") {
		t.Errorf("ProgramArguments = %v, want %v", p.Dict.Array.Strings, wantArgs)
	}

	// launchd will not spawn a job whose StandardOutPath directory is
	// missing, and it fails silently when it happens.
	if fi, err := os.Stat(s.logDir()); err != nil {
		t.Errorf("Install should create the log directory: %v", err)
	} else if !fi.IsDir() {
		t.Errorf("%s should be a directory", s.logDir())
	}
}

// These paths belong to installs that already exist, so they are pinned.
func TestPathLayout(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	user := Service{Name: "com.coder.vllm", Scope: "user"}
	system := Service{Name: "com.coder.vllm", Scope: "system"}

	tests := []struct {
		name string
		got  string
		want string
	}{
		{"user plist", user.plistPath(), filepath.Join(home, "Library", "LaunchAgents", "com.coder.vllm.plist")},
		{"user log dir", user.logDir(), filepath.Join(home, "Library", "Logs", "vllm-helper")},
		{"user log", user.logPath(), filepath.Join(home, "Library", "Logs", "vllm-helper", "com.coder.vllm.log")},
		{"system plist", system.plistPath(), "/Library/LaunchDaemons/com.coder.vllm.plist"},
		{"system log dir", system.logDir(), "/Library/Logs/vllm-helper"},
		{"system log", system.logPath(), "/Library/Logs/vllm-helper/com.coder.vllm.log"},
		{"user log hint", user.LogHint(), "tail -n 100 " + user.logPath()},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %q, want %q", tt.name, tt.got, tt.want)
		}
	}
}

func TestLaunchctlArgs(t *testing.T) {
	t.Parallel()

	uid := strconv.Itoa(os.Getuid())
	tests := []struct {
		name    string
		svc     Service
		wantBin string
		want    string
	}{{
		name:    "user scope needs no privileges",
		svc:     Service{Name: "com.coder.vllm", Scope: "user"},
		wantBin: "launchctl",
		want:    "print gui/" + uid + "/com.coder.vllm",
	}, {
		name:    "system scope goes through sudo",
		svc:     Service{Name: "com.coder.vllm", Scope: "system"},
		wantBin: "sudo",
		want:    "launchctl print system/com.coder.vllm",
	}, {
		name:    "system scope without a terminal fails fast",
		svc:     Service{Name: "com.coder.vllm", Scope: "system", NonInteractive: true},
		wantBin: "sudo",
		want:    "-n launchctl print system/com.coder.vllm",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			bin, args := tt.svc.launchctl("print", tt.svc.target())
			if bin != tt.wantBin {
				t.Errorf("bin = %q, want %q", bin, tt.wantBin)
			}
			if got := strings.Join(args, " "); got != tt.want {
				t.Errorf("args = %q, want %q", got, tt.want)
			}
		})
	}
}
