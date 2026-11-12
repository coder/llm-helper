package cli

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/detect"
)

// Two consecutive misses, two minutes apart, before relaunching. One miss is
// not evidence: a single probe can lose to a long request or a restart that
// has not opened the socket yet, and relaunching a working cluster costs
// minutes of loading.
const (
	defaultHealthMisses   = 2
	defaultHealthInterval = 2
)

func newRecipesHealthCheckCmd() *cobra.Command {
	var (
		unit      string
		prof      string
		threshold int
		statePath string
		interval  int
		userScope bool
	)
	c := &cobra.Command{
		Use:    "health-check",
		Short:  "Relaunch the cluster when the endpoint stops answering",
		Long:   "Plumbing for the watchdog timer that 'recipes install-service' installs. Safe to run by hand: it only acts after repeated failures.",
		Hidden: true,
		Args:   cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			if strings.TrimSpace(unit) == "" {
				return fmt.Errorf("--unit is required")
			}
			if statePath == "" {
				statePath = healthStatePath(unit)
				if userScope {
					statePath = userHealthStatePath(unit)
				}
			}
			return runHealthCheck(unit, prof, threshold, statePath, time.Duration(interval)*time.Minute, userScope)
		},
	}
	c.Flags().StringVar(&unit, "unit", defaultRecipeServiceName, "unit to relaunch")
	c.Flags().StringVar(&prof, "profile", "", "recipe profile whose endpoint to check (default: any running)")
	c.Flags().IntVar(&threshold, "misses", defaultHealthMisses, "consecutive failures before relaunching")
	c.Flags().StringVar(&statePath, "state", "", "where the miss count is kept")
	c.Flags().IntVar(&interval, "interval", defaultHealthInterval, "minutes between checks, which is also how long a fresh launch marker is trusted")
	c.Flags().BoolVar(&userScope, "user-scope", false, "the unit runs in the user's systemd manager")
	return c
}

// healthStatePath keeps the counter in /run: it is root-writable, and it is
// cleared by a reboot, which is exactly the lifetime a miss count should have.
func healthStatePath(unit string) string {
	return "/run/" + strings.TrimSuffix(unit, ".service") + "-misses"
}

// userHealthStatePath is the user-scope counterpart: the user's runtime dir is
// also cleared by a reboot.
func userHealthStatePath(unit string) string {
	dir := os.Getenv("XDG_RUNTIME_DIR")
	if dir == "" {
		dir = os.TempDir()
	}
	return filepath.Join(dir, strings.TrimSuffix(unit, ".service")+"-misses")
}

// healthSystemctl returns the args for unit's manager.
func healthSystemctl(userScope bool, args ...string) []string {
	if userScope {
		return append([]string{"--user"}, args...)
	}
	return args
}

// launchMarkerName is the file a launch leaves behind while it is working.
// It lives in the state dir so the launcher (the serving user) writes it and
// the watchdog (root, with HOME set to that user) reads it.
const launchMarkerName = "launch-in-progress"

// markerPath keeps each instance's markers apart, so stopping or launching
// one recipe does not silence the watchdog of another.
func markerPath(cfg *config.Config, base, prof string) string {
	return filepath.Join(cfg.Dirs.State, base+"-"+prof)
}

// launchStandOff bounds how long a launch can hold the watchdog off. Loads
// here take about five minutes, so this leaves room for a slow one without
// leaving the cluster unwatched for long if the launch never comes up.
const launchStandOff = 15 * time.Minute

// markLaunching announces that a launch is under way. Without it the watchdog
// reads the silence of a loading model as an outage and restarts the unit,
// which relaunches the old launch over the top of the new one.
//
// The marker outlives the launcher on purpose: "recipes launch" returns
// minutes before the endpoint answers. It is cleared by the first health
// check that sees the endpoint up, and otherwise expires.
func markLaunching(cfg *config.Config, prof string, ttl time.Duration) {
	if cfg == nil {
		return
	}
	if err := os.MkdirAll(cfg.Dirs.State, 0o755); err != nil {
		return
	}
	path := markerPath(cfg, launchMarkerName, prof)
	deadline := time.Now().Add(ttl).UTC().Format(time.RFC3339)
	if err := os.WriteFile(path, []byte(deadline), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "could not tell the watchdog a launch is starting:", err)
	}
}

// clearLaunching ends the stand-off once the endpoint answers, but refuses
// while the marker is younger than minAge: for the first seconds of a launch
// the PREVIOUS server is still answering, and a check landing there would
// clear a marker whose load has not started. One interval is enough, because
// by the next tick the old server is down.
func clearLaunching(cfg *config.Config, prof string, minAge time.Duration) {
	if cfg == nil {
		return
	}
	path := markerPath(cfg, launchMarkerName, prof)
	fi, err := os.Stat(path)
	if err != nil || time.Since(fi.ModTime()) < minAge {
		return
	}
	_ = os.Remove(path)
}

// launchInProgress reports whether a launch is working right now. A marker
// past its deadline is ignored and cleared: a crashed launcher must not leave
// the cluster unwatched.
func launchInProgress(cfg *config.Config, prof string) bool {
	if cfg == nil {
		return false
	}
	path := markerPath(cfg, launchMarkerName, prof)
	b, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	deadline, err := time.Parse(time.RFC3339, strings.TrimSpace(string(b)))
	if err != nil || time.Now().After(deadline) {
		_ = os.Remove(path)
		return false
	}
	return true
}

// stoppedMarkerName records that the cluster was stopped on purpose. The unit
// is oneshot with RemainAfterExit, so after 'vllm-helper stop' systemd still
// calls it active, and nothing else distinguishes "someone stopped it" from
// "it died". Without this the watchdog relaunched every deliberate stop.
const stoppedMarkerName = "stopped-on-purpose"

// markStopped is written by every deliberate stop, including the unit's own
// ExecStop, and lifted by the next launch. It has no expiry: a stopped
// cluster stays stopped until someone starts it.
func markStopped(cfg *config.Config, prof string) {
	if cfg == nil {
		return
	}
	if err := os.MkdirAll(cfg.Dirs.State, 0o755); err != nil {
		return
	}
	path := markerPath(cfg, stoppedMarkerName, prof)
	stamp := time.Now().UTC().Format(time.RFC3339)
	if err := os.WriteFile(path, []byte(stamp), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "could not tell the watchdog this stop is deliberate:", err)
	}
}

func clearStopped(cfg *config.Config, prof string) {
	if cfg == nil {
		return
	}
	_ = os.Remove(markerPath(cfg, stoppedMarkerName, prof))
}

func stoppedOnPurpose(cfg *config.Config, prof string) bool {
	if cfg == nil {
		return false
	}
	_, err := os.Stat(markerPath(cfg, stoppedMarkerName, prof))
	return err == nil
}

type healthAction string

const (
	healthSkip    healthAction = "skip"
	healthClear   healthAction = "clear"
	healthCount   healthAction = "count"
	healthRestart healthAction = "restart"
)

// healthDecision is the whole watchdog policy, kept pure so the interesting
// cases can be tested without a systemd host or a model.
//
// A healthy endpoint is judged first, before anything that would skip the
// check: it is what clears a launch marker, so skipping it while the marker
// existed left every launch unwatched for the whole stand-off.
//
// After that the watchdog stays out of the way of anything deliberate. A
// stopped or inactive unit is somebody's decision, not an outage -- "failed"
// is the state that means the unit gave up. "activating" is left alone because
// a load takes minutes and the endpoint is legitimately silent for all of it;
// restarting then would loop forever.
func healthDecision(unitState string, launching, stopped, healthy bool, misses, threshold int) (healthAction, int) {
	if healthy {
		return healthClear, 0
	}
	switch {
	case stopped, launching,
		unitState == "activating", unitState == "inactive", unitState == "deactivating":
		return healthSkip, misses
	}
	misses++
	if misses >= threshold {
		return healthRestart, 0
	}
	return healthCount, misses
}

func runHealthCheck(unit, prof string, threshold int, statePath string, interval time.Duration, userScope bool) error {
	cfg, _ := loadConfig()
	state := strings.TrimSpace(systemctlOutput(healthSystemctl(userScope, "is-active", unit)...))
	healthy := endpointHealthy(cfg, prof)
	misses := readMisses(statePath)

	action, next := healthDecision(state, launchInProgress(cfg, prof), stoppedOnPurpose(cfg, prof), healthy, misses, threshold)
	switch action {
	case healthSkip:
		return nil
	case healthClear:
		_ = os.Remove(statePath)
		// The endpoint is up, so whatever launch was in flight is done --
		// unless the marker is so fresh that this is the old server answering.
		clearLaunching(cfg, prof, interval)
		return nil
	case healthCount:
		writeMisses(statePath, next)
		fmt.Fprintf(os.Stderr, "%s: endpoint did not answer (%d/%d)\n", unit, misses+1, threshold)
		return nil
	default:
		_ = os.Remove(statePath)
		fmt.Fprintf(os.Stderr, "%s: endpoint unhealthy %d times; relaunching\n", unit, threshold)
		return runPoolCommand(os.Stdout, "systemctl", healthSystemctl(userScope, "restart", unit)...)
	}
}

// endpointHealthy asks the endpoint status reports on, so the watchdog and
// the operator agree on what healthy means. For a recipe that is the head
// node on the recipe's port, which need not be this machine.
func endpointHealthy(cfg *config.Config, prof string) bool {
	if cfg == nil {
		return false
	}
	if prof != "" {
		if c := clusterRunnerForName(cfg, prof); c != nil {
			_, ok := c.probeHealth()
			return ok
		}
	}
	list, _ := control.ListInstances(cfg)
	for _, inst := range list {
		if c := clusterRunnerForName(cfg, inst.Name); c != nil {
			if _, ok := c.probeHealth(); ok {
				return true
			}
		} else if _, ok := inst.Probe(); ok {
			return true
		}
	}
	if len(list) > 0 {
		return false
	}
	for _, ep := range detect.ProbePorts([]string{cfg.Port()}, cfg.APIKey()) {
		if ep.Healthy {
			return true
		}
	}
	return false
}

func systemctlOutput(args ...string) string {
	// is-active exits non-zero for anything but active, so the output is the
	// answer and the status is not.
	out, _ := exec.Command("systemctl", args...).Output()
	return string(out)
}

func readMisses(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

func writeMisses(path string, n int) {
	if err := os.WriteFile(path, []byte(strconv.Itoa(n)), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "could not record the miss count:", err)
	}
}

// renderHealthUnits builds the watchdog pair.
//
// A system-scope service runs as root so it can restart the unit without a
// sudo rule, and carries HOME so it reads the same config as the user who
// installed it. A user-scope one runs in that user's manager.
func renderHealthUnits(u recipeSystemdUnit, unitName string, intervalMin int) (service, timer string) {
	scope := ""
	if u.UserScope {
		scope = " --user-scope"
	}
	service = fmt.Sprintf("[Unit]\n"+
		"Description=Health check for %s\n\n"+
		"[Service]\n"+
		"Type=oneshot\n"+
		"Environment=HOME=%s\n"+
		"ExecStart=%s recipes health-check --unit %s --profile %s --interval %d%s\n",
		unitName, u.Home, u.Self, unitName, u.Profile, intervalMin, scope)
	timer = fmt.Sprintf("[Unit]\n"+
		"Description=Run the health check for %s every %d minutes\n\n"+
		"[Timer]\n"+
		"OnBootSec=5min\n"+
		"OnUnitActiveSec=%dmin\n"+
		"AccuracySec=15s\n\n"+
		"[Install]\n"+
		"WantedBy=timers.target\n",
		unitName, intervalMin, intervalMin)
	return service, timer
}

// healthUnitNames derives the watchdog names from the unit they watch, so a
// renamed service does not leave an orphaned timer behind.
func healthUnitNames(unit string) (service, timer string) {
	base := strings.TrimSuffix(unit, ".service") + "-health"
	return base + ".service", base + ".timer"
}
