package hardware_test

import (
	"regexp"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/hardware"
)

// The descriptors replaced one hardcoded pattern. Until a descriptor is added
// on purpose, the shipped set has to call exactly the same GPUs unified that
// the pattern did: a GPU wrongly called unified gets a static cap instead of
// the CPU backend's live clamp (issue #13), and one wrongly called discrete
// leaves a Spark uncapped.
func TestShippedDescriptorsMatchTheUnifiedNamesTheyReplaced(t *testing.T) {
	t.Parallel()
	replaced := regexp.MustCompile(`(?i)GB10|GB200|GB300|GH200|Grace|Orin|Thor|Tegra`)
	names := []string{
		"NVIDIA GB10",
		"NVIDIA GB200",
		"NVIDIA GB300",
		"NVIDIA GH200 480GB",
		"NVIDIA Grace Hopper Superchip",
		"Orin (nvgpu)",
		"NVIDIA Thor",
		"NVIDIA Tegra X1",
		"nvidia gb10",
		"NVIDIA RTX A6000",
		"NVIDIA A100 80GB PCIe",
		"NVIDIA H100 80GB HBM3",
		"NVIDIA B200",
		"NVIDIA GeForce RTX 5090",
		"NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
		"Apple M4 Max",
		"Unknown Integrated",
		"",
	}
	reg := hardware.Default()
	for _, name := range names {
		d, ok := reg.MatchGPU(name)
		got := ok && d.Unified
		want := name != "" && replaced.MatchString(name)
		if got != want {
			t.Errorf("%q: unified = %v (descriptor %q), the replaced pattern said %v", name, got, d.ID, want)
		}
	}
}

// Recipes record dgx_spark_gb10 as the hardware they were verified on, and the
// forbidden-flag check keys on its architecture. A broader entry ahead of it
// in the file would match the name first and silently drop that check.
func TestGB10ResolvesToTheSparkWithItsArchitecture(t *testing.T) {
	t.Parallel()
	reg := hardware.Default()
	d, ok := reg.MatchGPU("NVIDIA GB10")
	if !ok || d.ID != "dgx_spark_gb10" {
		t.Fatalf("NVIDIA GB10 matched %q (ok=%v), want dgx_spark_gb10", d.ID, ok)
	}
	byID, ok := reg.ByID("dgx_spark_gb10")
	if !ok || byID.Arch != "sm121" {
		t.Fatalf("ByID(dgx_spark_gb10) = %+v, %v; want arch sm121", byID, ok)
	}
	a, ok := reg.Architecture("sm121")
	if !ok || a.Generation != "blackwell" || len(a.ForbiddenArgs) == 0 {
		t.Fatalf("Architecture(sm121) = %+v, %v; want blackwell with forbidden args", a, ok)
	}
}

// Every rule in this file is someone's claim about a machine, so a mistake in
// one has to stop the build rather than quietly disable the claim.
func TestParseRejectsInvalidDescriptors(t *testing.T) {
	t.Parallel()
	const arch = `{"arch":"sm121","generation":"blackwell","forbidden_args":[{"flag":"--x","value":"y","why":"aborts"}]}`
	const desc = `{"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"unified":true,"arch":"sm121","why":"shares memory"}`
	tests := []struct {
		name string
		json string
		want string
	}{
		{"unknown key", `{"architectures":[],"descriptors":[{"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"unifed":true,"why":"w"}]}`, "unknown field"},
		{"duplicate id", `{"architectures":[` + arch + `],"descriptors":[` + desc + `,` + desc + `]}`, "defined twice"},
		{"undefined architecture", `{"architectures":[],"descriptors":[` + desc + `]}`, "not defined"},
		{"bad pattern", `{"architectures":[],"descriptors":[{"id":"box","display_name":"Box","match":{"gpu_name":"("},"why":"w"}]}`, "match.gpu_name"},
		{"no match rule", `{"architectures":[],"descriptors":[{"id":"box","display_name":"Box","match":{},"why":"w"}]}`, "no match rule"},
		{"no why", `{"architectures":[],"descriptors":[{"id":"box","display_name":"Box","match":{"gpu_name":"Box"}}]}`, "no why"},
		{"bad id", `{"architectures":[],"descriptors":[{"id":"Big Box","display_name":"Box","match":{"gpu_name":"Box"},"why":"w"}]}`, "lowercase"},
		{"forbidden flag without dashes", `{"architectures":[{"arch":"a","generation":"g","forbidden_args":[{"flag":"x","value":"y","why":"w"}]}],"descriptors":[]}`, "does not start with --"},
		{"forbidden flag without why", `{"architectures":[{"arch":"a","generation":"g","forbidden_args":[{"flag":"--x","value":"y"}]}],"descriptors":[]}`, "has no why"},
		{"duplicate architecture", `{"architectures":[` + arch + `,` + arch + `],"descriptors":[]}`, "defined twice"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := hardware.Parse([]byte(tt.json))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Parse = %v, want an error containing %q", err, tt.want)
			}
		})
	}
}

// A descriptor is authoritative once it matches, in either direction, and the
// first match in file order wins.
func TestMatchGPUUsesFileOrderAndHonoursDiscrete(t *testing.T) {
	t.Parallel()
	reg, err := hardware.Parse([]byte(`{"architectures":[],"descriptors":[
		{"id":"big_card","display_name":"Big card","match":{"gpu_name":"Big Card 96GB"},"unified":false,"why":"discrete, whatever its size"},
		{"id":"any_card","display_name":"Any card","match":{"gpu_name":"Card"},"unified":true,"why":"shares memory"}
	]}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if d, ok := reg.MatchGPU("ACME Big Card 96GB"); !ok || d.ID != "big_card" || d.Unified {
		t.Errorf("big card matched %+v, %v; want big_card, discrete", d, ok)
	}
	if d, ok := reg.MatchGPU("acme small card"); !ok || d.ID != "any_card" {
		t.Errorf("small card matched %+v, %v; want any_card", d, ok)
	}
	if _, ok := reg.MatchGPU(""); ok {
		t.Error("an empty GPU name matched a descriptor")
	}
}
