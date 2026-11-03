package hardware_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/hardware"
)

// The definitions replaced one hardcoded pattern. Until a definition is added
// on purpose, the shipped set has to call exactly the same NVIDIA GPUs unified
// that the pattern did: a GPU wrongly called unified gets a static cap instead
// of the CPU backend's live clamp (issue #13), and one wrongly called discrete
// leaves a Spark uncapped.
//
// One difference is deliberate: the pattern matched anywhere in the name, so
// "Integrated" read as Tegra. The definitions match whole words.
func TestShippedDefinitionsMatchTheUnifiedNamesTheyReplaced(t *testing.T) {
	t.Parallel()
	replaced := regexp.MustCompile(`(?i)\b(GB10|GB200|GB300|GH200|Grace|Orin|Thor|Tegra)\b`)
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
		"NVIDIA GB100",
		"",
	}
	reg := hardware.Shipped()
	for _, name := range names {
		d, ok := reg.Match(hardware.Observed{Vendor: "nvidia", GPUName: name})
		got := ok && d.Unified()
		want := name != "" && replaced.MatchString(name)
		if got != want {
			t.Errorf("%q: unified = %v (definition %q), the replaced pattern said %v", name, got, d.ID, want)
		}
	}
	if d, ok := reg.Match(hardware.Observed{Vendor: "nvidia", GPUName: "Unknown Integrated"}); ok {
		t.Errorf("\"Integrated\" matched %q", d.ID)
	}
}

// Apple Silicon used to be a hardcoded prefix check. It is data now, and only
// for a GPU the probe says Apple made.
func TestAppleSiliconIsUnifiedOnlyWhenAppleMadeIt(t *testing.T) {
	t.Parallel()
	reg := hardware.Shipped()
	if d, ok := reg.Match(hardware.Observed{Vendor: "apple", GPUName: "Apple M4 Max"}); !ok || d.ID != "apple_silicon" || !d.Unified() {
		t.Errorf("Apple M4 Max matched %q (ok=%v)", d.ID, ok)
	}
	if d, ok := reg.Match(hardware.Observed{Vendor: "nvidia", GPUName: "Apple M4 Max"}); ok {
		t.Errorf("a GPU NVIDIA reported matched %q", d.ID)
	}
}

// Recipes record dgx_spark_gb10 as the hardware they were verified on, and the
// forbidden-flag check keys on its architecture. A broader entry ahead of it
// in the file would match the name first and silently drop that check.
func TestGB10ResolvesToTheSparkWithItsArchitecture(t *testing.T) {
	t.Parallel()
	reg := hardware.Shipped()
	d, ok := reg.Match(hardware.Observed{Vendor: "nvidia", GPUName: "NVIDIA GB10"})
	if !ok || d.ID != "dgx_spark_gb10" || d.Memory != hardware.MemoryUnified {
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

// A GPU no definition names still gets its architecture's rules from the
// compute capability it reports. The match is exact: 12.0 is a different chip
// from 12.1, and the GB10 kernels' rules must not reach an RTX 5090.
func TestArchitectureByComputeCap(t *testing.T) {
	t.Parallel()
	reg := hardware.Shipped()
	for cc, want := range map[string]string{"12.1": "sm121", " 12.1 ": "sm121", "12.0": "sm120", "9.0": "sm90", "8.7": "sm87"} {
		if a, ok := reg.ArchitectureByComputeCap(cc); !ok || a.Arch != want {
			t.Errorf("ArchitectureByComputeCap(%q) = %q (ok=%v), want %s", cc, a.Arch, ok, want)
		}
	}
	for _, cc := range []string{"", "[N/A]", "12", "99.9"} {
		if a, ok := reg.ArchitectureByComputeCap(cc); ok {
			t.Errorf("ArchitectureByComputeCap(%q) = %q, want none", cc, a.Arch)
		}
	}
}

func TestVendorByPCI(t *testing.T) {
	t.Parallel()
	reg := hardware.Shipped()
	for id, want := range map[string]string{"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel", "0x10DE": "nvidia"} {
		if v, ok := reg.VendorByPCI(id); !ok || v.ID != want {
			t.Errorf("VendorByPCI(%q) = %q (ok=%v), want %s", id, v.ID, ok, want)
		}
	}
	// A BMC's display controller is not an accelerator anyone serves on.
	if v, ok := reg.VendorByPCI("0x1a03"); ok {
		t.Errorf("ASPEED matched vendor %q", v.ID)
	}
}

// Every rule in this file is someone's claim about a machine, so a mistake in
// one has to stop the build rather than quietly disable the claim.
func TestParseRejectsInvalidDefinitions(t *testing.T) {
	t.Parallel()
	const arch = `{"arch":"sm121","generation":"blackwell","forbidden_args":[{"flag":"--x","value":"y","why":"aborts"}]}`
	const def = `{"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"memory":"unified","arch":"sm121","why":"shares memory"}`
	wrap := func(vendors, archs, defs string) string {
		return `{"schema_version":1,"vendors":[` + vendors + `],"architectures":[` + archs + `],"definitions":[` + defs + `]}`
	}
	tests := []struct {
		name string
		json string
		want string
	}{
		{"unknown key", wrap("", "", `{"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"memory":"unified","unifed":true,"why":"w"}`), "unknown field"},
		{"the old unified key", wrap("", "", `{"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"unified":true,"why":"w"}`), "unknown field"},
		{"no schema version", `{"definitions":[]}`, "schema_version"},
		{"a newer schema", `{"schema_version":2}`, "schema_version is 2"},
		{"duplicate id", wrap("", arch, def+`,`+def), "defined twice"},
		{"undefined architecture", wrap("", "", def), "not defined"},
		{"undefined vendor", wrap("", "", `{"id":"box","display_name":"Box","match":{"vendor":"acme","gpu_name":"Box"},"memory":"unified","why":"w"}`), `vendor "acme"`},
		{"bad memory", wrap("", "", `{"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"memory":"shared","why":"w"}`), "unified, discrete or coherent"},
		{"no memory", wrap("", "", `{"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"why":"w"}`), "unified, discrete or coherent"},
		{"bad pattern", wrap("", "", `{"id":"box","display_name":"Box","match":{"gpu_name":"("},"memory":"unified","why":"w"}`), "match.gpu_name"},
		{"no match rule", wrap("", "", `{"id":"box","display_name":"Box","match":{},"memory":"unified","why":"w"}`), "no match rule"},
		{"no why", wrap("", "", `{"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"memory":"unified"}`), "no why"},
		{"bad id", wrap("", "", `{"id":"Big Box","display_name":"Box","match":{"gpu_name":"Box"},"memory":"unified","why":"w"}`), "lowercase"},
		{"forbidden flag without dashes", wrap("", `{"arch":"a","generation":"g","forbidden_args":[{"flag":"x","value":"y","why":"w"}]}`, ""), "does not start with --"},
		{"forbidden flag without why", wrap("", `{"arch":"a","generation":"g","forbidden_args":[{"flag":"--x","value":"y"}]}`, ""), "has no why"},
		{"duplicate architecture", wrap("", arch+`,`+arch, ""), "defined twice"},
		{"bad compute cap", wrap("", `{"arch":"a","generation":"g","compute_cap":"sm_121"}`, ""), "must look like 12.1"},
		{"shared compute cap", wrap("", `{"arch":"a","generation":"g","compute_cap":"9.0"},{"arch":"b","generation":"g","compute_cap":"9.0"}`, ""), "belongs to both"},
		{"bad PCI id", wrap(`{"id":"acme","name":"Acme","pci_ids":["10de"]}`, "", ""), "must look like 0x10de"},
		{"shared PCI id", wrap(`{"id":"a","name":"A","pci_ids":["0x1234"]},{"id":"b","name":"B","pci_ids":["0x1234"]}`, "", ""), "belongs to both"},
		{"vendor without a name", wrap(`{"id":"acme"}`, "", ""), "has no name"},
		{"a second object", wrap("", "", "") + `{"schema_version":1}`, "after the top-level object"},
		{"trailing garbage", wrap("", "", "") + ` x`, "after the top-level object"},
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

// A definition is authoritative once it matches, in either direction, and the
// first match in file order wins. A vendor rule has to hold as well as the
// name.
func TestMatchUsesFileOrderVendorAndHonoursDiscrete(t *testing.T) {
	t.Parallel()
	reg, err := hardware.Parse([]byte(`{"schema_version":1,
		"vendors":[{"id":"acme","name":"Acme"}],
		"definitions":[
		{"id":"big_card","display_name":"Big card","match":{"gpu_name":"Big Card 96GB"},"memory":"discrete","why":"discrete, whatever its size"},
		{"id":"acme_card","display_name":"Acme card","match":{"vendor":"acme","gpu_name":"Card"},"memory":"unified","why":"shares memory"}
	]}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if d, ok := reg.Match(hardware.Observed{GPUName: "ACME Big Card 96GB"}); !ok || d.ID != "big_card" || d.Unified() {
		t.Errorf("big card matched %+v, %v; want big_card, discrete", d, ok)
	}
	if d, ok := reg.Match(hardware.Observed{Vendor: "acme", GPUName: "acme small card"}); !ok || d.ID != "acme_card" {
		t.Errorf("Acme's small card matched %+v, %v; want acme_card", d, ok)
	}
	if d, ok := reg.Match(hardware.Observed{Vendor: "other", GPUName: "acme small card"}); ok {
		t.Errorf("another vendor's card matched %q", d.ID)
	}
	if _, ok := reg.Match(hardware.Observed{Vendor: "acme"}); ok {
		t.Error("an empty GPU name matched a definition")
	}
}

// A local file teaches llm-helper about a new machine. It must not be able to
// change what is already known, because that would let a stale file turn a
// Spark discrete or drop a flag that aborts it.
func TestExtendOnlyAdds(t *testing.T) {
	t.Parallel()
	ship := hardware.Shipped()
	reg, err := ship.Extend([]byte(`{"schema_version":1,
		"vendors":[{"id":"acme","name":"Acme","pci_ids":["0xacce"]}],
		"architectures":[
			{"arch":"sm121","forbidden_args":[{"flag":"--new-flag","value":"on","why":"aborts too"}]},
			{"arch":"acme1","generation":"acme"}
		],
		"definitions":[
			{"id":"acme_box","display_name":"Acme box","match":{"vendor":"acme","gpu_name":"Acme"},"memory":"unified","arch":"acme1","why":"measured on one"},
			{"id":"late_gb10","display_name":"Shadowed","match":{"vendor":"nvidia","gpu_name":"GB10"},"memory":"discrete","why":"too late to matter"}
		]}`), "/etc/acme.json")
	if err != nil {
		t.Fatalf("Extend: %v", err)
	}
	if d, ok := reg.Match(hardware.Observed{Vendor: "acme", GPUName: "Acme 1"}); !ok || d.ID != "acme_box" || d.Source != "/etc/acme.json" {
		t.Errorf("the local definition: %+v, %v", d, ok)
	}
	// Local definitions come after the shipped ones, so they cannot shadow
	// a machine the binary already knows.
	if d, _ := reg.Match(hardware.Observed{Vendor: "nvidia", GPUName: "NVIDIA GB10"}); d.ID != "dgx_spark_gb10" {
		t.Errorf("GB10 now matches %q", d.ID)
	}
	if a, _ := reg.Architecture("sm121"); len(a.ForbiddenArgs) != 3 {
		t.Errorf("sm121 forbidden args = %+v, want the shipped two and the local one", a.ForbiddenArgs)
	}
	if a, _ := ship.Architecture("sm121"); len(a.ForbiddenArgs) != 2 {
		t.Errorf("Extend changed the registry it extended: %+v", a.ForbiddenArgs)
	}

	for name, body := range map[string]string{
		"a shipped id":          `{"schema_version":1,"definitions":[{"id":"dgx_spark_gb10","display_name":"x","match":{"gpu_name":"GB10"},"memory":"discrete","why":"w"}]}`,
		"a shipped vendor":      `{"schema_version":1,"vendors":[{"id":"nvidia","name":"Not NVIDIA"}]}`,
		"a shipped generation":  `{"schema_version":1,"architectures":[{"arch":"sm121","generation":"hopper"}]}`,
		"a shipped compute cap": `{"schema_version":1,"architectures":[{"arch":"sm121","compute_cap":"12.0"}]}`,
		"a claimed PCI id":      `{"schema_version":1,"vendors":[{"id":"acme","name":"Acme","pci_ids":["0x10de"]}]}`,
		"an invalid entry":      `{"schema_version":1,"definitions":[{"id":"acme","display_name":"Acme","match":{"gpu_name":"Acme"},"memory":"unified"}]}`,
	} {
		if _, err := ship.Extend([]byte(body), "local.json"); err == nil {
			t.Errorf("Extend accepted %s", name)
		}
	}
}

// Not parallel: UseLocal replaces what Default returns for the process.
func TestUseLocalSkipsABadFileAndKeepsTheRest(t *testing.T) {
	dir := t.TempDir()
	t.Cleanup(func() { hardware.UseLocal(filepath.Join(dir, "missing")) })
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("10-bad.json", `{"schema_version":1,"definitions":[{"id":"oops"`)
	write("20-good.json", `{"schema_version":1,"vendors":[{"id":"acme","name":"Acme"}],"definitions":[{"id":"acme_box","display_name":"Acme box","match":{"vendor":"acme","gpu_name":"Acme"},"memory":"unified","why":"w"}]}`)
	write("notes.txt", "not a definition")

	errs := hardware.UseLocal(dir)
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "10-bad.json") {
		t.Errorf("UseLocal = %v, want one error naming 10-bad.json", errs)
	}
	if got := hardware.LocalErrors(); len(got) != 1 {
		t.Errorf("LocalErrors = %v", got)
	}
	if _, ok := hardware.Default().ByID("acme_box"); !ok {
		t.Error("the good file was not loaded")
	}
	if _, ok := hardware.Shipped().ByID("acme_box"); ok {
		t.Error("a local definition leaked into the shipped set")
	}

	if errs := hardware.UseLocal(filepath.Join(dir, "missing")); len(errs) != 0 {
		t.Errorf("a missing directory: %v", errs)
	}
	if _, ok := hardware.Default().ByID("acme_box"); ok {
		t.Error("UseLocal kept a definition from a directory it no longer reads")
	}
}
