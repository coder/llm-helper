package cli

import (
	"bytes"
	"encoding/xml"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/vllm-helper/internal/config"
)

const (
	gib = int64(1) << 30
	// m5UltraWorkingSet is what MLX reports on a 256 GiB Mac Studio with
	// the macOS default cap.
	m5UltraWorkingSet = int64(239143780352)
)

func TestSuggestWiredLimit(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		ram, ws   int64
		limitMiB  int64
		want      int64
		wantRaise bool
	}{
		// The verified machine: the default leaves about 34 GiB idle.
		{"256 GiB at the macOS default", 256 * gib, m5UltraWorkingSet, 0, 245760, true},
		{"128 GiB keeps 16 GiB back", 128 * gib, 96 * gib, 0, 112 * 1024, true},
		{"64 GiB keeps 8 GiB back", 64 * gib, 48 * gib, 0, 56 * 1024, true},
		// A small Mac's default already leaves macOS less than the headroom.
		{"16 GiB needs no raise", 16 * gib, 10 * gib, 0, 10 * 1024, false},
		// A cap already above the suggestion is never lowered.
		{"already raised past it", 256 * gib, 250 * gib, 250 * 1024, 250 * 1024, false},
		{"equal to the suggestion", 256 * gib, 240 * gib, 245760, 245760, false},
		// With no working set, the live cap is the floor.
		{"unknown working set, live cap", 256 * gib, 0, 250000, 250000, false},
		// With neither, three quarters of RAM stands in for the default.
		{"unknown everything, small Mac", 16 * gib, 0, 0, 12 * 1024, false},
		{"unknown everything, large Mac", 256 * gib, 0, 0, 245760, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, raise := suggestWiredLimit(tc.ram, tc.ws, tc.limitMiB)
			if got != tc.want || raise != tc.wantRaise {
				t.Errorf("suggestWiredLimit(%d, %d, %d) = %d, %v; want %d, %v", tc.ram, tc.ws, tc.limitMiB, got, raise, tc.want, tc.wantRaise)
			}
		})
	}
}

// A cap already high enough is not raised again, but one set by hand, or left
// by a run that stopped partway, still has to survive a reboot.
func TestAutoWiredLimit(t *testing.T) {
	t.Parallel()

	const raised = 250 * 1024
	cases := []struct {
		name             string
		live, configured int64
		daemon           bool
		want             int64
	}{
		{"the macOS default is raised", 0, 0, false, 245760},
		{"a raised cap already persisted is left alone", raised, raised, true, 0},
		{"a cap set by hand is persisted", raised, 0, false, raised},
		{"a cap with no boot-time daemon is persisted", raised, raised, false, raised},
		{"a cap config.env does not match is persisted", raised, 245760, true, raised},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := autoWiredLimit(256*gib, m5UltraWorkingSet, tc.live, tc.configured, tc.daemon); got != tc.want {
				t.Errorf("autoWiredLimit() = %d, want %d", got, tc.want)
			}
		})
	}

	// A small Mac at the macOS default needs nothing, persisted or not.
	if got := autoWiredLimit(16*gib, 10*gib, 0, 0, false); got != 0 {
		t.Errorf("autoWiredLimit() on a 16 GiB Mac = %d, want 0", got)
	}
}

func TestValidateWiredLimit(t *testing.T) {
	t.Parallel()

	ram := 256 * gib
	for _, mib := range []int64{8192, 245760, 256*1024 - 4096} {
		if err := validateWiredLimit(mib, ram); err != nil {
			t.Errorf("validateWiredLimit(%d) = %v, want nil", mib, err)
		}
	}
	for _, mib := range []int64{0, 8191, 256*1024 - 4095, 300 * 1024} {
		if err := validateWiredLimit(mib, ram); err == nil {
			t.Errorf("validateWiredLimit(%d) = nil, want an error", mib)
		}
	}
	if err := validateWiredLimit(16384, 0); err == nil {
		t.Error("validateWiredLimit with unknown RAM = nil, want an error")
	}
}

func TestParseWiredLimitArg(t *testing.T) {
	t.Parallel()

	if n, err := parseWiredLimitArg(" 245760 "); err != nil || n != 245760 {
		t.Errorf("parseWiredLimitArg(245760) = %d, %v", n, err)
	}
	for _, bad := range []string{"", "240G", "1.5", "lots"} {
		if _, err := parseWiredLimitArg(bad); err == nil {
			t.Errorf("parseWiredLimitArg(%q) = nil error, want one", bad)
		}
	}
}

func TestParseSysctlInt(t *testing.T) {
	t.Parallel()

	for in, want := range map[string]int64{
		"0\n":                            0,
		"274877906944\n":                 274877906944,
		"iogpu.wired_limit_mb: 245760\n": 245760,
	} {
		got, err := parseSysctlInt([]byte(in))
		if err != nil || got != want {
			t.Errorf("parseSysctlInt(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "unknown oid\n", "iogpu.wired_limit_mb:\n"} {
		if _, err := parseSysctlInt([]byte(bad)); err == nil {
			t.Errorf("parseSysctlInt(%q) = nil error, want one", bad)
		}
	}
}

// launchd rejects a plist that is not well-formed XML, and it would do so
// only at boot, where nobody sees it.
func TestRenderWiredLimitPlist(t *testing.T) {
	t.Parallel()

	body := renderWiredLimitPlist(245760)
	dec := xml.NewDecoder(bytes.NewReader(body))
	var keys, strs []string
	var elem string
	hasRunAtLoad := false
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("plist is not valid XML: %v\n%s", err, body)
		}
		switch tok := tok.(type) {
		case xml.StartElement:
			elem = tok.Name.Local
			if elem == "true" && len(keys) > 0 && keys[len(keys)-1] == "RunAtLoad" {
				hasRunAtLoad = true
			}
		case xml.CharData:
			v := strings.TrimSpace(string(tok))
			switch {
			case v == "":
			case elem == "key":
				keys = append(keys, v)
			case elem == "string":
				strs = append(strs, v)
			}
		}
	}
	wantKeys := []string{"Label", "ProgramArguments", "RunAtLoad"}
	if strings.Join(keys, ",") != strings.Join(wantKeys, ",") {
		t.Errorf("keys = %v, want %v (no KeepAlive: the job runs once)", keys, wantKeys)
	}
	wantStrs := []string{wiredLimitLabel, "/usr/sbin/sysctl", "iogpu.wired_limit_mb=245760"}
	if strings.Join(strs, ",") != strings.Join(wantStrs, ",") {
		t.Errorf("strings = %v, want %v", strs, wantStrs)
	}
	if !hasRunAtLoad {
		t.Error("RunAtLoad is not true, so the limit would not be applied at boot")
	}
}

func configWith(t *testing.T, content string) *config.Config {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.env")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := config.ParseEnvFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return &config.Config{File: f}
}

func TestNewWiredLimitReadsConfig(t *testing.T) {
	t.Parallel()

	wl := newWiredLimit(configWith(t, "METAL_WIRED_LIMIT_MB=245760\n"), 256*gib, 245760, 245760<<20, true)
	if wl.ConfiguredMiB != 245760 || wl.ConfiguredInvalid != "" {
		t.Errorf("configured = %d, %q; want 245760", wl.ConfiguredMiB, wl.ConfiguredInvalid)
	}
	if wl.RaiseNeeded {
		t.Error("a cap already at the suggestion should need no raise")
	}

	wl = newWiredLimit(configWith(t, "METAL_WIRED_LIMIT_MB=lots\n"), 256*gib, 0, m5UltraWorkingSet, false)
	if wl.ConfiguredMiB != 0 || wl.ConfiguredInvalid != "lots" {
		t.Errorf("configured = %d, %q; want 0, \"lots\"", wl.ConfiguredMiB, wl.ConfiguredInvalid)
	}

	wl = newWiredLimit(nil, 256*gib, 0, m5UltraWorkingSet, false)
	if wl.ConfiguredMiB != 0 || !wl.RaiseNeeded || wl.SuggestedMiB != 245760 {
		t.Errorf("without config: %+v", wl)
	}
}

func TestRenderWiredLimitAtTheDefault(t *testing.T) {
	t.Parallel()

	var b bytes.Buffer
	renderWiredLimit(&b, newWiredLimit(nil, 256*gib, 0, m5UltraWorkingSet, false), true)
	out := b.String()
	for _, want := range []string{
		"Installed RAM:      262144 MiB (256 GiB)",
		"Wired limit now:    macOS default (iogpu.wired_limit_mb=0)",
		"Metal working set:  228065 MiB (222.7 GiB)",
		"In config.env:      METAL_WIRED_LIMIT_MB not set",
		"Suggested:          245760 MiB (240 GiB), RAM minus 16 GiB for macOS",
		"'vllm-helper metal-limit --set auto'",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
}

func TestRenderWiredLimitWhenNoRaiseIsNeeded(t *testing.T) {
	t.Parallel()

	var b bytes.Buffer
	renderWiredLimit(&b, newWiredLimit(nil, 16*gib, 0, 10*gib, false), true)
	out := b.String()
	if !strings.Contains(out, "no raise needed") {
		t.Errorf("output does not say no raise is needed:\n%s", out)
	}
	if strings.Contains(out, "--set auto") {
		t.Errorf("output suggests a raise it just said is not needed:\n%s", out)
	}
}

func TestDoctorWiredLimitLines(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		wl   wiredLimit
		wsGB int
		want []string
	}{
		{
			"default with room to raise",
			newWiredLimit(nil, 256*gib, 0, m5UltraWorkingSet, false),
			239,
			[]string{
				"  wired limit: macOS default (239 GB Metal working set)",
				"               raise it with 'vllm-helper metal-limit --set auto'",
			},
		},
		{
			// 4 GiB more is not worth a root daemon.
			"default with little to gain",
			newWiredLimit(nil, 64*gib, 0, 52*gib, false),
			55,
			[]string{"  wired limit: macOS default (55 GB Metal working set)"},
		},
		{
			"set by metal-limit",
			wiredLimit{LimitMiB: 245760, ConfiguredMiB: 245760},
			257,
			[]string{"  wired limit: 245760 MiB (set by vllm-helper metal-limit)"},
		},
		{
			"set by hand",
			wiredLimit{LimitMiB: 200000},
			209,
			[]string{"  wired limit: 200000 MiB (set outside vllm-helper, lost on reboot)"},
		},
		{
			"configured but not applied",
			wiredLimit{ConfiguredMiB: 245760},
			239,
			[]string{
				"  wired limit: macOS default (239 GB Metal working set), but METAL_WIRED_LIMIT_MB=245760",
				"               the boot-time daemon did not apply it; run 'vllm-helper metal-limit --set 245760'",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := doctorWiredLimitLines(tc.wl, tc.wsGB)
			if strings.Join(got, "\n") != strings.Join(tc.want, "\n") {
				t.Errorf("got:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(tc.want, "\n"))
			}
		})
	}
}

// fakeSysctl returns the given values in turn, then the last one forever.
func fakeSysctl(values ...int64) func() (int64, error) {
	i := 0
	return func() (int64, error) {
		v := values[min(i, len(values)-1)]
		i++
		return v, nil
	}
}

func TestWaitForWiredLimitReturnsAtOnceWhenApplied(t *testing.T) {
	t.Parallel()

	var out, warn bytes.Buffer
	slept := 0
	ok := waitForWiredLimit(245760, fakeSysctl(245760), func(time.Duration) { slept++ }, &out, &warn)
	if !ok || slept != 0 || out.Len() != 0 || warn.Len() != 0 {
		t.Errorf("ok=%v slept=%d out=%q warn=%q; want an immediate, silent return", ok, slept, out.String(), warn.String())
	}
}

func TestWaitForWiredLimitWaitsForTheDaemon(t *testing.T) {
	t.Parallel()

	var out, warn bytes.Buffer
	slept := 0
	ok := waitForWiredLimit(245760, fakeSysctl(0, 0, 0, 245760), func(time.Duration) { slept++ }, &out, &warn)
	if !ok || slept != 3 {
		t.Errorf("ok=%v slept=%d; want true after 3 polls", ok, slept)
	}
	if lines := strings.Count(out.String(), "\n"); lines != 1 || !strings.Contains(out.String(), "Waiting up to 60s") {
		t.Errorf("want one waiting line, got %q", out.String())
	}
	if warn.Len() != 0 {
		t.Errorf("unexpected warning %q", warn.String())
	}
}

func TestWaitForWiredLimitGivesUpAfterAMinute(t *testing.T) {
	t.Parallel()

	var out, warn bytes.Buffer
	var waited time.Duration
	ok := waitForWiredLimit(245760, fakeSysctl(0), func(d time.Duration) { waited += d }, &out, &warn)
	if ok || waited != wiredLimitWait {
		t.Errorf("ok=%v waited=%s; want false after %s", ok, waited, wiredLimitWait)
	}
	if !strings.Contains(warn.String(), "warning: iogpu.wired_limit_mb is still 0, not the 245760") {
		t.Errorf("warning = %q", warn.String())
	}
}

func TestWaitForWiredLimitDoesNotWaitOnAnUnreadableSysctl(t *testing.T) {
	t.Parallel()

	var out, warn bytes.Buffer
	slept := 0
	read := func() (int64, error) { return 0, errors.New("unknown oid") }
	ok := waitForWiredLimit(245760, read, func(time.Duration) { slept++ }, &out, &warn)
	if ok || slept != 0 || !strings.Contains(warn.String(), "unknown oid") {
		t.Errorf("ok=%v slept=%d warn=%q; want an immediate warning", ok, slept, warn.String())
	}
}
