package cli

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/detect"
	"github.com/coder/vllm-helper/internal/vllm"
)

func runMetalLimit(o metalLimitOpts) error {
	cfg, err := loadConfig()
	// Showing or resetting the cap works without a config; --set records
	// its value there, so it needs one.
	if err != nil && (o.set != "" || !errors.Is(err, config.ErrNotConfigured)) {
		return err
	}
	ram, err := readSysctlInt("hw.memsize")
	if err != nil {
		return fmt.Errorf("reading installed RAM: %w", err)
	}
	live, err := readSysctlInt(wiredLimitSysctl)
	if err != nil {
		return fmt.Errorf("reading %s (metal-limit needs Apple Silicon and macOS 15 or later, as vllm-metal does): %w", wiredLimitSysctl, err)
	}

	switch {
	case o.reset:
		return resetWiredLimit(cfg, live, o.yes)
	case o.set != "":
		return setWiredLimit(cfg, ram, live, o.set, o.yes)
	}

	var m vllm.Metal
	if cfg != nil {
		m = vllm.ProbeMetal(cfg.VLLMBin())
	}
	renderWiredLimit(os.Stdout, newWiredLimit(cfg, ram, live, m.WorkingSetBytes, wiredLimitDaemonInstalled()), m.Installed)
	return nil
}

func setWiredLimit(cfg *config.Config, ram, live int64, arg string, yes bool) error {
	var mib int64
	if strings.EqualFold(strings.TrimSpace(arg), "auto") {
		ws := vllm.ProbeMetal(cfg.VLLMBin()).WorkingSetBytes
		// Without either figure the floor would be a guess, and a guess
		// could lower the cap below what macOS already allows.
		if ws == 0 && live == 0 {
			return fmt.Errorf("could not read the Metal working set from %s (is vllm-metal installed?); pass a size in MiB instead of 'auto'", orUnknown(cfg.VLLMBin()))
		}
		// An unreadable METAL_WIRED_LIMIT_MB counts as unset, so it is replaced.
		configured, _ := configuredWiredLimit(cfg)
		mib = autoWiredLimit(ram, ws, live, configured, wiredLimitDaemonInstalled())
		if mib == 0 {
			s, _ := suggestWiredLimit(ram, ws, live)
			fmt.Printf("No raise needed: the GPU may already use %s of the %s installed, which leaves macOS %d GiB or less.\n",
				formatMiB(s), formatMiB(ram>>20), wiredLimitHeadroomMiB(ram>>20)/1024)
			return nil
		}
		if mib == live {
			fmt.Printf("No raise needed. Keeping the current %s and reapplying it at every boot.\n", formatMiB(live))
		}
	} else {
		n, err := parseWiredLimitArg(arg)
		if err != nil {
			return err
		}
		mib = n
	}
	if err := validateWiredLimit(mib, ram); err != nil {
		return err
	}

	fmt.Print(describeWiredLimitPlan([]string{
		fmt.Sprintf("install %s, which sets %s=%d at every boot", wiredLimitPlist, wiredLimitSysctl, mib),
		fmt.Sprintf("run 'sudo sysctl %s=%d' now", wiredLimitSysctl, mib),
		fmt.Sprintf("record %s=%d in %s", metalWiredLimitKey, mib, cfg.Dirs.ConfigFile()),
	}))
	fmt.Printf("That leaves macOS %s of the %s installed.\n", formatMiB(ram>>20-mib), formatMiB(ram>>20))
	if !yes && !confirm("Continue?", false) {
		fmt.Println("Left the wired limit as it is.")
		return nil
	}

	// The daemon and the live value come first, so config.env only ever
	// records a limit that was actually applied. Recording one that was not
	// would make every 'start' wait for a daemon that does not exist.
	if err := installWiredLimitDaemon(mib); err != nil {
		return err
	}
	if err := applyWiredLimit(mib); err != nil {
		return err
	}
	cfg.File.Set(metalWiredLimitKey, strconv.FormatInt(mib, 10))
	if err := cfg.File.Write(cfg.Dirs.ConfigFile(), 0o644); err != nil {
		// The limit is in force and survives reboots; only 'start' and
		// doctor lose track of it. Say so, rather than suggest it failed.
		return fmt.Errorf("the wired limit is %d MiB now and %s reapplies it at every boot, but recording it in %s failed: %w; "+
			"add %s=%d to that file by hand so 'start' and 'doctor' know about it",
			mib, wiredLimitPlist, cfg.Dirs.ConfigFile(), err, metalWiredLimitKey, mib)
	}
	fmt.Println()
	fmt.Printf("The GPU may now wire up to %s, and will again after every reboot.\n", formatMiB(mib))
	fmt.Println("The server read the old limit when it started; run 'vllm-helper restart' so it sizes against the new one.")
	return nil
}

func resetWiredLimit(cfg *config.Config, live int64, yes bool) error {
	installed := wiredLimitDaemonInstalled()
	configured := cfg != nil && cfg.File.Has(metalWiredLimitKey)
	if !installed && live == 0 && !configured {
		fmt.Println("Nothing to reset: the wired limit is the macOS default and no boot-time daemon is installed.")
		return nil
	}

	var steps []string
	if installed {
		steps = append(steps, "remove "+wiredLimitPlist)
	}
	if live != 0 {
		steps = append(steps, fmt.Sprintf("run 'sudo sysctl %s=0' now, the macOS default", wiredLimitSysctl))
	}
	if configured {
		steps = append(steps, fmt.Sprintf("remove %s from %s", metalWiredLimitKey, cfg.Dirs.ConfigFile()))
	}
	fmt.Print(describeWiredLimitPlan(steps))
	if !yes && !confirm("Continue?", false) {
		fmt.Println("Left the wired limit as it is.")
		return nil
	}

	if installed {
		if err := removeWiredLimitDaemon(); err != nil {
			return err
		}
	}
	if live != 0 {
		if err := applyWiredLimit(0); err != nil {
			return err
		}
	}
	if configured {
		cfg.File.Delete(metalWiredLimitKey)
		if err := cfg.File.Write(cfg.Dirs.ConfigFile(), 0o644); err != nil {
			return fmt.Errorf("removing %s from %s: %w", metalWiredLimitKey, cfg.Dirs.ConfigFile(), err)
		}
	}
	fmt.Println()
	fmt.Println("The wired limit is back to the macOS default.")
	if live != 0 {
		fmt.Println("Run 'vllm-helper restart' so the server sizes against it.")
	}
	return nil
}

// readSysctlInt reads one integer sysctl. Neither key this reads needs root.
func readSysctlInt(key string) (int64, error) {
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return 0, fmt.Errorf("sysctl -n %s: %w", key, err)
	}
	return parseSysctlInt(out)
}

// runSudo runs a privileged step. Stderr always reaches the terminal, since
// that is where sudo prompts and where a failure explains itself. Without a
// terminal, sudo fails fast instead of waiting on a prompt nobody can answer.
func runSudo(out *os.File, args ...string) error {
	if !stdinIsTTY() {
		args = append([]string{"-n"}, args...)
	}
	cmd := exec.Command("sudo", args...)
	cmd.Stdin, cmd.Stderr = os.Stdin, os.Stderr
	if out != nil {
		cmd.Stdout = out
	}
	return cmd.Run()
}

// applyWiredLimit sets the live cap. It takes effect at once for processes
// that start afterwards; a running vllm-metal keeps the cap it read.
func applyWiredLimit(mib int64) error {
	arg := fmt.Sprintf("%s=%d", wiredLimitSysctl, mib)
	if err := runSudo(os.Stdout, "sysctl", arg); err != nil {
		return fmt.Errorf("'sudo sysctl %s' failed (needs root): %w", arg, err)
	}
	return nil
}

func wiredLimitDaemonInstalled() bool {
	_, err := os.Stat(wiredLimitPlist)
	return err == nil
}

// wiredLimitDaemonLoaded reports whether launchd knows the job. Booting out
// one it does not know is an error, and reporting that would be noise.
// Reading a system-domain job needs no root.
func wiredLimitDaemonLoaded() bool {
	return exec.Command("launchctl", "print", "system/"+wiredLimitLabel).Run() == nil
}

// installWiredLimitDaemon writes and loads the boot-time daemon. launchd
// refuses a daemon plist that is not root-owned and closed to other writers,
// and a daemon that installs but never loads would silently lose the limit at
// the next reboot, so every step is fatal.
func installWiredLimitDaemon(mib int64) error {
	tmp, err := os.CreateTemp("", "vllm-helper-wired-limit-*.plist")
	if err != nil {
		return fmt.Errorf("writing the wired-limit daemon: %w", err)
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(renderWiredLimitPlist(mib)); err != nil {
		tmp.Close()
		return fmt.Errorf("writing the wired-limit daemon: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("writing the wired-limit daemon: %w", err)
	}

	// Replace any earlier registration, or launchd keeps running the old
	// value at boot.
	if wiredLimitDaemonLoaded() {
		_ = runSudo(nil, "launchctl", "bootout", "system/"+wiredLimitLabel)
	}
	steps := [][]string{
		{"cp", tmp.Name(), wiredLimitPlist},
		{"chown", "root:wheel", wiredLimitPlist},
		{"chmod", "644", wiredLimitPlist},
		{"launchctl", "bootstrap", "system", wiredLimitPlist},
	}
	for _, step := range steps {
		if err := runSudo(os.Stdout, step...); err != nil {
			return fmt.Errorf("installing the wired-limit daemon: 'sudo %s' failed (needs root): %w", strings.Join(step, " "), err)
		}
	}
	return nil
}

func removeWiredLimitDaemon() error {
	if wiredLimitDaemonLoaded() {
		if err := runSudo(nil, "launchctl", "bootout", "system/"+wiredLimitLabel); err != nil {
			return fmt.Errorf("unloading the wired-limit daemon: 'sudo launchctl bootout system/%s' failed: %w", wiredLimitLabel, err)
		}
	}
	if err := runSudo(nil, "rm", "-f", wiredLimitPlist); err != nil {
		return fmt.Errorf("removing %s: 'sudo rm -f' failed: %w", wiredLimitPlist, err)
	}
	return nil
}

// awaitWiredLimit holds 'start' until the boot-time daemon has applied the
// configured cap, because vllm-metal reads it once at startup. launchd starts
// daemons in no particular order, so at boot the server can win the race.
func awaitWiredLimit(cfg *config.Config) {
	want, err := configuredWiredLimit(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "warning:", err)
		return
	}
	if want == 0 {
		return
	}
	read := func() (int64, error) { return readSysctlInt(wiredLimitSysctl) }
	waitForWiredLimit(want, read, time.Sleep, os.Stdout, os.Stderr)
}

// gatherWiredLimit is doctor's view of the cap, or nil when the Mac does not
// serve on the GPU or the sysctl is missing.
func gatherWiredLimit(cfg *config.Config, hw detect.Hardware) *wiredLimit {
	if !hw.Metal {
		return nil
	}
	live, err := readSysctlInt(wiredLimitSysctl)
	if err != nil {
		return nil
	}
	wl := newWiredLimit(cfg, hw.TotalMemBytes, live, hw.MetalWorkingSetBytes, wiredLimitDaemonInstalled())
	return &wl
}
