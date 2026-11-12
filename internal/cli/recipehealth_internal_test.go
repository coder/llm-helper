package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/llm-helper/internal/config"
)

// The watchdog's whole job is deciding when a silent endpoint means trouble.
// A load legitimately takes minutes, so "activating" must never be treated as
// a miss: that would restart the launch it is waiting for, forever.
func TestHealthDecision(t *testing.T) {
	cases := []struct {
		name      string
		state     string
		launching bool
		stopped   bool
		healthy   bool
		misses    int
		want      healthAction
		wantCount int
	}{
		{"loading is not a failure", "activating", false, false, false, 1, healthSkip, 1},
		{"healthy clears the count", "active", false, false, true, 1, healthClear, 0},
		{"first miss only counts", "active", false, false, false, 0, healthCount, 1},
		{"second miss relaunches", "active", false, false, false, 1, healthRestart, 0},
		{"a dead unit still relaunches", "failed", false, false, false, 1, healthRestart, 0},
		{"healthy while inactive is still healthy", "inactive", false, false, true, 1, healthClear, 0},
		// The case that cost a launch: a deliberate launch looks exactly like
		// an outage from the outside, and the unit is not "activating"
		// because the launch was started by hand rather than by systemd.
		{"a launch in progress is left alone", "active", true, false, false, 1, healthSkip, 1},
		{"even a failed-looking unit is left alone mid-launch", "failed", true, false, false, 1, healthSkip, 1},
		// A healthy probe must be heard while the marker exists: it is the
		// only thing that clears the marker, so skipping it left every
		// launch unwatched for the whole stand-off.
		{"a healthy probe mid-launch still clears", "active", true, false, true, 1, healthClear, 0},
		// Deliberate stops: 'vllm-helper stop' leaves the oneshot unit
		// active, and 'systemctl stop' leaves it inactive. Neither is an
		// outage, and relaunching either undoes what the operator just did.
		{"a stopped cluster stays stopped", "active", false, true, false, 1, healthSkip, 1},
		{"a stopped unit stays stopped", "inactive", false, false, false, 1, healthSkip, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, n := healthDecision(c.state, c.launching, c.stopped, c.healthy, c.misses, 2)
			if got != c.want || n != c.wantCount {
				t.Errorf("healthDecision(%q, launching=%v, stopped=%v, %v, %d) = %v/%d, want %v/%d",
					c.state, c.launching, c.stopped, c.healthy, c.misses, got, n, c.want, c.wantCount)
			}
		})
	}
}

// The stop marker has no expiry, unlike the launch marker: a cluster someone
// stopped stays stopped until someone launches it.
func TestStoppedMarkerLastsUntilALaunch(t *testing.T) {
	cfg := &config.Config{Dirs: config.Dirs{State: t.TempDir()}}
	if stoppedOnPurpose(cfg, "a") {
		t.Error("no marker, but a deliberate stop was reported")
	}
	markStopped(cfg, "a")
	if !stoppedOnPurpose(cfg, "a") {
		t.Error("marker written, but no stop was reported")
	}
	if stoppedOnPurpose(cfg, "b") {
		t.Error("stopping one instance silenced another instance's watchdog")
	}
	clearStopped(cfg, "a")
	if stoppedOnPurpose(cfg, "a") {
		t.Error("marker cleared, but a stop was still reported")
	}
}

// A launcher that dies must not disable the watchdog forever, so the marker
// carries a deadline rather than a pid.
func TestLaunchMarkerExpires(t *testing.T) {
	cfg := &config.Config{Dirs: config.Dirs{State: t.TempDir()}}
	if launchInProgress(cfg, "a") {
		t.Error("no marker, but a launch was reported")
	}
	markLaunching(cfg, "a", time.Minute)
	if !launchInProgress(cfg, "a") {
		t.Error("marker written, but no launch was reported")
	}
	if launchInProgress(cfg, "b") {
		t.Error("launching one instance held off another instance's watchdog")
	}
	// A check that lands seconds into a launch is still seeing the OLD
	// server, so it must not clear the marker. This is the bug that left a
	// real load unguarded: the marker was gone 20s in.
	clearLaunching(cfg, "a", time.Minute)
	if !launchInProgress(cfg, "a") {
		t.Error("a fresh marker was cleared by a healthy check, which is the old server answering")
	}
	clearLaunching(cfg, "a", 0)
	if launchInProgress(cfg, "a") {
		t.Error("marker cleared, but a launch was still reported")
	}
	markLaunching(cfg, "a", -time.Minute)
	if launchInProgress(cfg, "a") {
		t.Error("an expired marker must not hold the watchdog off")
	}
}

// The counter lives in /run so a reboot clears it, and the names follow the
// unit so a renamed service cannot leave an orphaned timer behind.
func TestHealthNamesFollowTheUnit(t *testing.T) {
	svc, timer := healthUnitNames("vllm-cluster.service")
	if svc != "vllm-cluster-health.service" || timer != "vllm-cluster-health.timer" {
		t.Errorf("names = %q, %q", svc, timer)
	}
	if got := healthStatePath("vllm-cluster.service"); got != "/run/vllm-cluster-misses" {
		t.Errorf("state path = %q", got)
	}
}

func TestHealthUnitsRunTheCheckerAgainstTheirUnit(t *testing.T) {
	svc, timer := renderHealthUnits(
		recipeSystemdUnit{Home: "/home/serve", Self: "/usr/local/bin/vllm-helper", Profile: "x"},
		"vllm-cluster.service", 2)
	for _, want := range []string{
		"ExecStart=/usr/local/bin/vllm-helper recipes health-check --unit vllm-cluster.service --profile x --interval 2",
		"Environment=HOME=/home/serve",
		"Type=oneshot",
	} {
		if !strings.Contains(svc, want) {
			t.Errorf("health service missing %q:\n%s", want, svc)
		}
	}
	for _, want := range []string{"OnUnitActiveSec=2min", "WantedBy=timers.target", "OnBootSec=5min"} {
		if !strings.Contains(timer, want) {
			t.Errorf("timer missing %q:\n%s", want, timer)
		}
	}
	// No User= line: the checker restarts the unit, which needs root.
	if strings.Contains(svc, "User=") {
		t.Errorf("health service pins a user, so it could not restart the unit:\n%s", svc)
	}
}

func TestHealthUnitsInUserScope(t *testing.T) {
	svc, _ := renderHealthUnits(
		recipeSystemdUnit{Home: "/home/serve", Self: "/usr/local/bin/vllm-helper", Profile: "x", UserScope: true},
		"vllm-cluster.service", 2)
	want := "ExecStart=/usr/local/bin/vllm-helper recipes health-check --unit vllm-cluster.service --profile x --interval 2 --user-scope\n"
	if !strings.Contains(svc, want) {
		t.Errorf("health service missing %q:\n%s", want, svc)
	}
	if got := healthSystemctl(true, "restart", "u"); strings.Join(got, " ") != "--user restart u" {
		t.Errorf("user args = %v", got)
	}
	if got := healthSystemctl(false, "restart", "u"); strings.Join(got, " ") != "restart u" {
		t.Errorf("system args = %v", got)
	}
}

// A user cannot write /run, so the count goes to the user's runtime dir.
func TestUserHealthStatePath(t *testing.T) {
	t.Setenv("XDG_RUNTIME_DIR", "/run/user/1000")
	if got := userHealthStatePath("vllm-cluster.service"); got != "/run/user/1000/vllm-cluster-misses" {
		t.Errorf("state path = %q", got)
	}
	t.Setenv("XDG_RUNTIME_DIR", "")
	if got, want := userHealthStatePath("vllm-cluster.service"), filepath.Join(os.TempDir(), "vllm-cluster-misses"); got != want {
		t.Errorf("fallback state path = %q, want %q", got, want)
	}
}
