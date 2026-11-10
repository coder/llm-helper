package cli

import (
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
)

// watchOpts controls how a restart is observed.
type watchOpts struct {
	follow  bool          // stream the boot log while waiting
	wait    bool          // block until healthy even without following
	timeout time.Duration // 0 = wait forever (first downloads are slow)
}

func crashReason(now, baseline int) string {
	if baseline >= 0 && now > baseline {
		return fmt.Sprintf("the service is crash-looping (%d restarts since it started)", now-baseline)
	}
	return "the service is crash-looping"
}

// watchBoot waits for the restarted server to come up: optionally streams
// the journal, polls /health until it answers, and detects failed units
// and crash loops instead of letting the user stare at a stalled log.
// Returns nil once the endpoint is healthy.
func watchBoot(cfg *config.Config, name string, opts watchOpts) error {
	inst, err := control.LoadInstance(cfg, name)
	if err != nil {
		return err
	}
	t, err := bootTargetFor(instanceRunner(cfg, name), inst)
	if err != nil {
		return err
	}
	return watchTarget(t, name, opts)
}

// bootTarget is what a boot is watched through: the runner that reports
// crashes and logs, and the URL whose health path says the model is up.
type bootTarget struct {
	runner runner
	base   string
	where  string
	// health is the path the engine answers on once it serves; empty is
	// /health.
	health string
	// probe, when set, is asked instead of base and health.
	probe func() (int, bool)
}

func (t bootTarget) healthy() bool {
	if t.probe != nil {
		_, ok := t.probe()
		return ok
	}
	_, ok := control.ProbeHealthPath(t.base, t.health)
	return ok
}

// bootTargetFor points the health probe where the runner actually serves. A
// cluster serves from its head node on the recipe's port; probing localhost on
// the configured port instead never succeeds when the head is another machine,
// so the wait hung with the model already up.
func bootTargetFor(r runner, inst *control.Instance) (bootTarget, error) {
	if c, ok := r.(*clusterRunner); ok {
		base, err := c.endpointURL()
		if err != nil {
			return bootTarget{}, err
		}
		return bootTarget{runner: r, base: base, where: base, health: c.recipe.HealthPath(), probe: c.probeHealth}, nil
	}
	return bootTarget{runner: r, base: "http://localhost:" + inst.Port, where: "port " + inst.Port, health: inst.HealthPath()}, nil
}

func watchTarget(t bootTarget, label string, opts watchOpts) error {
	s := t.runner
	start := time.Now()

	var tail interface{ Kill() error }
	if opts.follow {
		// Spell out that Ctrl+C only detaches. People assume it kills the
		// server, especially on a first run where the model download makes
		// this sit here for a long time.
		fmt.Println("Watching boot log. Ctrl+C stops watching only -- the server keeps loading in the background.")
		fmt.Println("Check on it later with 'llm-helper status', or skip this wait next time with --no-follow.")
		cmd, err := s.StartTail(10)
		if err != nil {
			fmt.Fprintln(os.Stderr, "couldn't follow the service log (continuing to wait):", err)
		} else {
			tail = cmd.Process
			defer func() {
				_ = cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			}()
		}
	}

	baseline := s.NRestarts()
	fail := func(reason string) error {
		if tail != nil {
			_ = tail.Kill()
		}
		if !opts.follow {
			if logs := s.LastLogs(30); logs != "" {
				fmt.Fprintln(os.Stderr, logs)
			}
		}
		if hint := s.LogHint(); hint != "" {
			return fmt.Errorf("%s -- full log: %s", reason, hint)
		}
		return errors.New(reason)
	}

	for {
		if t.healthy() {
			elapsed := time.Since(start).Round(time.Second)
			fmt.Printf("\n%s serving on %s (took %s) -- '%s' is live.\n",
				checkmark(), t.where, elapsed, label)
			return nil
		}
		h := s.Health()
		switch {
		case h.State == "failed":
			return fail("the service entered a failed state")
		case h.State == "inactive", h.State == "deactivating":
			// The unit is started before we get here, so a stopped unit
			// means the server exited on its own (or something else
			// stopped it). Waiting for a health check it will never answer
			// would hang forever with the default timeout.
			return fail("the service stopped instead of coming up")
		case h.Crashed:
			// Caught while the init system sleeps between restarts, which
			// otherwise reads as "activating" and looks like a slow boot.
			return fail(crashReason(h.NRestarts, baseline))
		}
		if baseline >= 0 && h.NRestarts > baseline+1 {
			return fail(crashReason(h.NRestarts, baseline))
		}
		if opts.timeout > 0 && time.Since(start) > opts.timeout {
			return fail(fmt.Sprintf("gave up after %s (the model may still be downloading)", opts.timeout))
		}
		time.Sleep(2 * time.Second)
	}
}

// checkmark degrades on dumb terminals.
func checkmark() string {
	if os.Getenv("TERM") == "dumb" {
		return "OK:"
	}
	return "\u2713"
}
