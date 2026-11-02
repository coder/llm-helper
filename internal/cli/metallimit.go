package cli

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/config"
)

// metalWiredLimitKey is the config.env key that records the wired limit
// 'metal-limit --set' chose, so 'start' can tell whether the boot-time daemon
// has applied it yet.
const metalWiredLimitKey = "METAL_WIRED_LIMIT_MB"

// wiredLimitSysctl caps how much unified memory macOS lets the GPU wire, in
// MiB. 0 means the macOS default. MLX reads the cap once, at startup, as its
// max_recommended_working_set_size.
const wiredLimitSysctl = "iogpu.wired_limit_mb"

// wiredLimitLabel and wiredLimitPlist name the root LaunchDaemon that sets the
// sysctl at every boot: macOS resets it to the default on each restart.
const (
	wiredLimitLabel = "com.coder.vllm-helper.wired-limit"
	wiredLimitPlist = "/Library/LaunchDaemons/" + wiredLimitLabel + ".plist"
)

// Bounds on a wired limit, in MiB. Below the minimum the GPU has too little to
// serve any useful model; above RAM minus the reserve, macOS itself can stall
// for want of memory it cannot page out.
const (
	minWiredLimitMiB     = 8192
	wiredLimitReserveMiB = 4096
)

// wiredLimitWait bounds how long 'start' waits for the boot-time daemon. The
// daemon runs early in boot, so a minute is generous; blocking longer would
// keep the server down for a problem a warning describes just as well.
const (
	wiredLimitWait     = 60 * time.Second
	wiredLimitInterval = time.Second
)

func newMetalLimitCmd() *cobra.Command {
	var set string
	var reset, yes bool
	c := &cobra.Command{
		Use:     "metal-limit",
		Short:   "Show or raise how much unified memory macOS lets the GPU use",
		GroupID: groupMachine,
		Long: "macOS caps how much unified memory the GPU may wire, and vllm-metal sizes\n" +
			"--gpu-memory-utilization as a share of that cap, not of installed RAM. On a Mac\n" +
			"dedicated to serving models the default leaves memory idle.\n\n" +
			"With no flags, show the current cap and a suggested one. --set raises it now and\n" +
			"installs a LaunchDaemon that applies it again at every boot, since macOS resets\n" +
			"it on restart. The server reads the cap once at startup, so run\n" +
			"'vllm-helper restart' afterwards. macOS only.",
		Example: "  vllm-helper metal-limit\n" +
			"  vllm-helper metal-limit --set auto\n" +
			"  vllm-helper metal-limit --set 245760\n" +
			"  vllm-helper metal-limit --reset",
		Args: cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return runMetalLimit(metalLimitOpts{set: set, reset: reset, yes: yes})
		},
	}
	c.Flags().StringVar(&set, "set", "", "set the cap to this many MiB, or 'auto' for the suggested value")
	c.Flags().BoolVar(&reset, "reset", false, "return to the macOS default and remove the boot-time daemon")
	c.Flags().BoolVar(&yes, "yes", false, "apply without asking")
	c.MarkFlagsMutuallyExclusive("set", "reset")
	return c
}

type metalLimitOpts struct {
	set   string
	reset bool
	yes   bool
}

// wiredLimit is what metal-limit and doctor know about the GPU wired memory
// cap on this Mac.
type wiredLimit struct {
	RAMBytes int64 `json:"ram_bytes"`
	// LimitMiB is iogpu.wired_limit_mb right now; 0 is the macOS default.
	LimitMiB int64 `json:"limit_mib"`
	// WorkingSetBytes is MLX's max_recommended_working_set_size, 0 when it
	// could not be read.
	WorkingSetBytes int64 `json:"working_set_bytes"`
	// ConfiguredMiB is METAL_WIRED_LIMIT_MB from config.env, 0 when unset.
	ConfiguredMiB int64 `json:"configured_mib,omitempty"`
	// ConfiguredInvalid holds a METAL_WIRED_LIMIT_MB that is not a number.
	ConfiguredInvalid string `json:"configured_invalid,omitempty"`
	DaemonInstalled   bool   `json:"daemon_installed"`
	SuggestedMiB      int64  `json:"suggested_mib"`
	// RaiseNeeded is whether SuggestedMiB is above both the working set and
	// the live cap.
	RaiseNeeded bool `json:"raise_needed"`
}

// newWiredLimit assembles a wiredLimit from what the caller measured and what
// config.env records. cfg may be nil when there is no config yet.
func newWiredLimit(cfg *config.Config, ramBytes, limitMiB, workingSetBytes int64, daemonInstalled bool) wiredLimit {
	wl := wiredLimit{
		RAMBytes:        ramBytes,
		LimitMiB:        limitMiB,
		WorkingSetBytes: workingSetBytes,
		DaemonInstalled: daemonInstalled,
	}
	if n, err := configuredWiredLimit(cfg); err != nil {
		wl.ConfiguredInvalid = strings.TrimSpace(cfg.Lookup(metalWiredLimitKey, ""))
	} else {
		wl.ConfiguredMiB = n
	}
	wl.SuggestedMiB, wl.RaiseNeeded = suggestWiredLimit(ramBytes, workingSetBytes, limitMiB)
	return wl
}

// wiredLimitHeadroomMiB is what the suggestion leaves for macOS and whatever
// else runs. Common guidance for a dedicated LLM Mac is 8 to 16 GB; a large
// Mac gets the upper end because its kernel and file cache grow with RAM.
func wiredLimitHeadroomMiB(ramMiB int64) int64 {
	if ramMiB >= 128*1024 {
		return 16 * 1024
	}
	return 8 * 1024
}

// suggestWiredLimit returns the cap to suggest, in MiB, and whether it raises
// the current one. It never suggests lowering: the floor is the larger of the
// working set and a cap already in force. When neither is known, the floor is
// the share of RAM macOS commonly allows by default, so the suggestion does not
// shrink a small Mac's GPU memory below what it had.
func suggestWiredLimit(ramBytes, workingSetBytes, limitMiB int64) (int64, bool) {
	ramMiB := ramBytes >> 20
	floor := max(workingSetBytes>>20, limitMiB)
	if floor == 0 {
		floor = ramMiB * 3 / 4
	}
	s := ramMiB - wiredLimitHeadroomMiB(ramMiB)
	if s < minWiredLimitMiB || s <= floor {
		return floor, false
	}
	return s, true
}

// autoWiredLimit is the cap 'metal-limit --set auto' applies, or 0 when there
// is nothing to do. A live cap at or above the suggestion is kept rather than
// raised, but it is still persisted unless config.env and the boot-time daemon
// already carry it: macOS drops a cap set by hand at the next reboot.
func autoWiredLimit(ramBytes, workingSetBytes, liveMiB, configuredMiB int64, daemonInstalled bool) int64 {
	s, raise := suggestWiredLimit(ramBytes, workingSetBytes, liveMiB)
	switch {
	case raise:
		return s
	case liveMiB == 0, daemonInstalled && configuredMiB == liveMiB:
		return 0
	}
	return liveMiB
}

// validateWiredLimit rejects a cap that would starve either the GPU or macOS.
func validateWiredLimit(mib, ramBytes int64) error {
	ramMiB := ramBytes >> 20
	if ramMiB <= 0 {
		return errors.New("could not read installed RAM, so there is nothing to check the limit against")
	}
	if mib < minWiredLimitMiB {
		return fmt.Errorf("%d MiB is below the %d MiB minimum: the GPU would have too little memory to serve a model", mib, minWiredLimitMiB)
	}
	if most := ramMiB - wiredLimitReserveMiB; mib > most {
		return fmt.Errorf("%d MiB would leave macOS less than %d MiB of the %d MiB installed; the most this accepts is %d MiB",
			mib, wiredLimitReserveMiB, ramMiB, most)
	}
	return nil
}

// parseWiredLimitArg reads a --set value other than "auto".
func parseWiredLimitArg(arg string) (int64, error) {
	n, err := strconv.ParseInt(strings.TrimSpace(arg), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("--set takes a size in MiB or 'auto', not %q", arg)
	}
	return n, nil
}

// parseSysctlInt reads an integer sysctl value. It accepts both 'sysctl -n'
// output and the "key: value" form.
func parseSysctlInt(out []byte) (int64, error) {
	s := strings.TrimSpace(string(out))
	if i := strings.LastIndex(s, ":"); i >= 0 {
		s = strings.TrimSpace(s[i+1:])
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing sysctl output %q: %w", strings.TrimSpace(string(out)), err)
	}
	return n, nil
}

// configuredWiredLimit reads METAL_WIRED_LIMIT_MB. It returns 0 and no error
// when the key is unset.
func configuredWiredLimit(cfg *config.Config) (int64, error) {
	if cfg == nil {
		return 0, nil
	}
	raw := strings.TrimSpace(cfg.Lookup(metalWiredLimitKey, ""))
	if raw == "" {
		return 0, nil
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("%s=%s in config.env is not a size in MiB", metalWiredLimitKey, raw)
	}
	return n, nil
}

// renderWiredLimitPlist is the LaunchDaemon that applies the cap at boot. It
// runs sysctl once and exits, so it has no KeepAlive. Every value is fixed or
// an integer, so nothing needs XML escaping.
func renderWiredLimitPlist(mib int64) []byte {
	return fmt.Appendf(nil, `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>%s</string>
	<key>ProgramArguments</key>
	<array>
		<string>/usr/sbin/sysctl</string>
		<string>%s=%d</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
</dict>
</plist>
`, wiredLimitLabel, wiredLimitSysctl, mib)
}

// formatMiB shows a sysctl-unit size with its GiB equivalent.
func formatMiB(mib int64) string {
	gib := strconv.FormatFloat(float64(mib)/1024, 'f', 1, 64)
	return fmt.Sprintf("%d MiB (%s GiB)", mib, strings.TrimSuffix(gib, ".0"))
}

// renderWiredLimit prints the no-flag 'metal-limit' report. metalInstalled
// says whether the configured vLLM has vllm-metal, which decides how the
// working set line reads when it is 0.
func renderWiredLimit(w io.Writer, wl wiredLimit, metalInstalled bool) {
	ramMiB := wl.RAMBytes >> 20
	fmt.Fprintf(w, "Installed RAM:      %s\n", formatMiB(ramMiB))
	if wl.LimitMiB == 0 {
		fmt.Fprintf(w, "Wired limit now:    macOS default (%s=0)\n", wiredLimitSysctl)
	} else {
		fmt.Fprintf(w, "Wired limit now:    %s\n", formatMiB(wl.LimitMiB))
	}
	switch {
	case wl.WorkingSetBytes > 0:
		fmt.Fprintf(w, "Metal working set:  %s, what vllm-metal sizes --gpu-memory-utilization against\n", formatMiB(wl.WorkingSetBytes>>20))
	case metalInstalled:
		fmt.Fprintln(w, "Metal working set:  unknown (MLX did not report it)")
	default:
		fmt.Fprintln(w, "Metal working set:  unknown (the configured vLLM has no vllm-metal)")
	}
	switch {
	case wl.ConfiguredInvalid != "":
		fmt.Fprintf(w, "In config.env:      %s=%s is not a size in MiB\n", metalWiredLimitKey, wl.ConfiguredInvalid)
	case wl.ConfiguredMiB == 0:
		fmt.Fprintf(w, "In config.env:      %s not set\n", metalWiredLimitKey)
	case wl.DaemonInstalled:
		fmt.Fprintf(w, "In config.env:      %s=%d, applied at boot by %s\n", metalWiredLimitKey, wl.ConfiguredMiB, wiredLimitPlist)
	default:
		fmt.Fprintf(w, "In config.env:      %s=%d, but %s is missing, so it is lost on reboot\n", metalWiredLimitKey, wl.ConfiguredMiB, wiredLimitPlist)
	}
	headroom := wiredLimitHeadroomMiB(ramMiB)
	if wl.RaiseNeeded {
		fmt.Fprintf(w, "Suggested:          %s, RAM minus %d GiB for macOS\n", formatMiB(wl.SuggestedMiB), headroom/1024)
	} else {
		fmt.Fprintf(w, "Suggested:          no raise needed, macOS already keeps %d GiB or less for itself\n", headroom/1024)
	}

	switch {
	case wl.ConfiguredMiB > 0 && wl.LimitMiB != wl.ConfiguredMiB:
		fmt.Fprintf(w, "\nThe live limit does not match config.env. Apply it with 'vllm-helper metal-limit --set %d'.\n", wl.ConfiguredMiB)
	case wl.RaiseNeeded:
		fmt.Fprintln(w, "\nRaise it with 'vllm-helper metal-limit --set auto', then run 'vllm-helper restart'.")
	case wl.ConfiguredMiB > 0 || wl.LimitMiB > 0:
		fmt.Fprintln(w, "\nReturn to the macOS default with 'vllm-helper metal-limit --reset'.")
	}
}

// doctorWiredLimitLines are the HARDWARE lines doctor prints for the cap.
// workingSetGB is the decimal-GB figure doctor already shows elsewhere.
func doctorWiredLimitLines(wl wiredLimit, workingSetGB int) []string {
	ws := ""
	if workingSetGB > 0 {
		ws = fmt.Sprintf(" (%d GB Metal working set)", workingSetGB)
	}
	var lines []string
	switch {
	case wl.LimitMiB == 0 && wl.ConfiguredMiB > 0:
		lines = append(lines,
			fmt.Sprintf("  wired limit: macOS default%s, but %s=%d", ws, metalWiredLimitKey, wl.ConfiguredMiB),
			fmt.Sprintf("               the boot-time daemon did not apply it; run 'vllm-helper metal-limit --set %d'", wl.ConfiguredMiB))
	case wl.LimitMiB == 0:
		lines = append(lines, "  wired limit: macOS default"+ws)
		// A smaller gain is not worth a root daemon and a restart.
		if wl.RaiseNeeded && wl.WorkingSetBytes > 0 && wl.SuggestedMiB-(wl.WorkingSetBytes>>20) >= 8*1024 {
			lines = append(lines, "               raise it with 'vllm-helper metal-limit --set auto'")
		}
	case wl.LimitMiB == wl.ConfiguredMiB:
		lines = append(lines, fmt.Sprintf("  wired limit: %d MiB (set by vllm-helper metal-limit)", wl.LimitMiB))
	case wl.ConfiguredMiB > 0:
		lines = append(lines,
			fmt.Sprintf("  wired limit: %d MiB, but %s=%d", wl.LimitMiB, metalWiredLimitKey, wl.ConfiguredMiB),
			fmt.Sprintf("               run 'vllm-helper metal-limit --set %d' to apply it", wl.ConfiguredMiB))
	default:
		lines = append(lines, fmt.Sprintf("  wired limit: %d MiB (set outside vllm-helper, lost on reboot)", wl.LimitMiB))
	}
	if wl.ConfiguredInvalid != "" {
		lines = append(lines, fmt.Sprintf("               %s=%s in config.env is not a size in MiB", metalWiredLimitKey, wl.ConfiguredInvalid))
	}
	return lines
}

// waitForWiredLimit polls until the live cap equals want or wiredLimitWait
// passes. vllm-metal reads the cap once at startup, so a server started before
// the boot-time daemon ran would keep the smaller working set until its next
// restart. It reports whether the cap matched, and never fails: serving with
// the default cap beats not serving.
func waitForWiredLimit(want int64, read func() (int64, error), sleep func(time.Duration), out, warn io.Writer) bool {
	got, err := read()
	if err != nil {
		fmt.Fprintf(warn, "warning: could not read %s (%v); starting without waiting for %s=%d\n", wiredLimitSysctl, err, metalWiredLimitKey, want)
		return false
	}
	if got == want {
		return true
	}
	fmt.Fprintf(out, "Waiting up to %ds for %s to apply %s=%d (now %d): vllm-metal reads it once at startup.\n",
		int(wiredLimitWait.Seconds()), wiredLimitLabel, wiredLimitSysctl, want, got)
	for range int(wiredLimitWait / wiredLimitInterval) {
		sleep(wiredLimitInterval)
		if got, err = read(); err == nil && got == want {
			return true
		}
	}
	fmt.Fprintf(warn, "warning: %s is still %d, not the %d in config.env; starting with the smaller Metal working set. Run 'vllm-helper metal-limit --set %d', then 'vllm-helper restart'.\n",
		wiredLimitSysctl, got, want, want)
	return false
}

// describeWiredLimitPlan lists what --set or --reset is about to do, so the
// confirmation names every privileged step.
func describeWiredLimitPlan(steps []string) string {
	var b bytes.Buffer
	b.WriteString("This will:\n")
	for _, s := range steps {
		fmt.Fprintf(&b, "  - %s\n", s)
	}
	return b.String()
}
