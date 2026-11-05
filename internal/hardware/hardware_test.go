package hardware_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/coder/llm-helper/internal/hardware"
)

// The definitions replaced one hardcoded pattern. Until a definition is added
// on purpose, the shipped set has to call exactly the same NVIDIA GPUs unified
// that the pattern did: a GPU wrongly called unified gets a static cap instead
// of the CPU backend's live clamp (issue #13), and one wrongly called discrete
// leaves a Spark uncapped.
//
// Two differences are deliberate. The pattern matched anywhere in the name, so
// "Integrated" read as Tegra; the definitions match whole words. And it took
// "Grace" alone as a superchip; the definitions need Grace Hopper or Grace
// Blackwell, the only GPUs Grace is paired with, so no catch-all overlaps them.
func TestShippedDefinitionsMatchTheUnifiedNamesTheyReplaced(t *testing.T) {
	t.Parallel()
	replaced := regexp.MustCompile(`(?i)\b(GB10|GB200|GB300|GH200|Grace Hopper|Grace Blackwell|Orin|Thor|Tegra)\b`)
	names := []string{
		"NVIDIA GB10",
		"NVIDIA GB200",
		"NVIDIA GB300",
		"NVIDIA GH200 480GB",
		"NVIDIA Grace Hopper Superchip",
		"NVIDIA GH200 Grace Hopper Superchip",
		"NVIDIA GB200 Grace Blackwell Superchip",
		"NVIDIA Grace Blackwell",
		"NVIDIA Grace",
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
		o := hardware.Observed{Vendor: "nvidia", GPUName: name}
		if m := reg.Matches(o); len(m) > 1 {
			t.Errorf("%q matches %d shipped machines; a GPU may match only one", name, len(m))
		}
		d, ok := reg.Match(o)
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
// forbidden-flag check keys on its architecture. A broader machine that also
// matched the name could win and silently drop that check.
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

// A Grace superchip resolves to its own machine whether its name gives the
// chip code, spells out Grace Hopper or Grace Blackwell, or both, so a Grace
// Hopper keeps its sm90 rules when nvidia-smi gives no compute capability.
func TestGraceSuperchipsResolveToTheirOwnMachine(t *testing.T) {
	t.Parallel()
	reg := hardware.Shipped()
	for name, want := range map[string]string{
		"NVIDIA GH200 480GB":                     "nvidia_grace_hopper",
		"NVIDIA Grace Hopper Superchip":          "nvidia_grace_hopper",
		"NVIDIA GH200 Grace Hopper Superchip":    "nvidia_grace_hopper",
		"NVIDIA GB200":                           "nvidia_grace_blackwell",
		"NVIDIA GB300":                           "nvidia_grace_blackwell",
		"NVIDIA GB200 Grace Blackwell Superchip": "nvidia_grace_blackwell",
	} {
		d, ok := reg.Match(hardware.Observed{Vendor: "nvidia", GPUName: name})
		if !ok || d.ID != want {
			t.Errorf("%q matched %q (ok=%v), want %s", name, d.ID, ok, want)
		}
		if want == "nvidia_grace_hopper" && d.Arch != "sm90" {
			t.Errorf("%q: arch %q, want sm90", name, d.Arch)
		}
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

// load reads a catalogue made of files, keyed by their path in it.
func load(files map[string]string) (*hardware.Registry, error) {
	fsys := fstest.MapFS{}
	for name, body := range files {
		fsys[name] = &fstest.MapFile{Data: []byte(body)}
	}
	return hardware.Load(fsys)
}

// Every rule in the catalogue is someone's claim about a machine, so a mistake
// in one has to stop the build rather than quietly disable the claim.
func TestLoadRejectsInvalidEntries(t *testing.T) {
	t.Parallel()
	const arch = `{"schema_version":1,"arch":"sm121","generation":"blackwell","forbidden_args":[{"flag":"--x","value":"y","why":"aborts"}]}`
	box := func(fields string) map[string]string {
		return map[string]string{
			"architectures/sm121.json": arch,
			"machines/box.json":        `{"schema_version":1,"id":"box",` + fields + `}`,
		}
	}
	tests := []struct {
		name  string
		files map[string]string
		want  string
	}{
		{"unknown key", box(`"display_name":"Box","match":{"gpu_name":"Box"},"memory":"unified","unifed":true,"why":"w"`), "unknown field"},
		{"the old unified key", box(`"display_name":"Box","match":{"gpu_name":"Box"},"unified":true,"why":"w"`), "unknown field"},
		{"no schema version", map[string]string{"vendors/acme.json": `{"id":"acme","name":"Acme"}`}, "schema_version is 0"},
		{"a newer schema", map[string]string{"vendors/acme.json": `{"schema_version":2,"id":"acme","name":"Acme"}`}, "schema_version is 2"},
		{"a file not named after its id", map[string]string{"vendors/acme.json": `{"schema_version":1,"id":"other","name":"Other"}`}, "name it after the id"},
		{"an architecture not named after its arch", map[string]string{"architectures/sm120.json": arch}, "name it after the id"},
		{"undefined architecture", map[string]string{"machines/box.json": `{"schema_version":1,"id":"box","display_name":"Box","match":{"gpu_name":"Box"},"memory":"unified","arch":"sm121","why":"w"}`}, "not defined"},
		{"undefined vendor", box(`"display_name":"Box","match":{"vendor":"acme","gpu_name":"Box"},"memory":"unified","why":"w"`), `vendor "acme"`},
		{"bad memory", box(`"display_name":"Box","match":{"gpu_name":"Box"},"memory":"shared","why":"w"`), "unified, discrete or coherent"},
		{"no memory", box(`"display_name":"Box","match":{"gpu_name":"Box"},"why":"w"`), "unified, discrete or coherent"},
		{"bad pattern", box(`"display_name":"Box","match":{"gpu_name":"("},"memory":"unified","why":"w"`), "match.gpu_name"},
		{"no match rule", box(`"display_name":"Box","match":{},"memory":"unified","why":"w"`), "no match rule"},
		{"no why", box(`"display_name":"Box","match":{"gpu_name":"Box"},"memory":"unified"`), "no why"},
		{"bad id", map[string]string{"machines/Big Box.json": `{"schema_version":1,"id":"Big Box","display_name":"Box","match":{"gpu_name":"Box"},"memory":"unified","why":"w"}`}, "lowercase"},
		{"forbidden flag without dashes", map[string]string{"architectures/a.json": `{"schema_version":1,"arch":"a","generation":"g","forbidden_args":[{"flag":"x","value":"y","why":"w"}]}`}, "does not start with --"},
		{"forbidden flag without why", map[string]string{"architectures/a.json": `{"schema_version":1,"arch":"a","generation":"g","forbidden_args":[{"flag":"--x","value":"y"}]}`}, "has no why"},
		{"bad compute cap", map[string]string{"architectures/a.json": `{"schema_version":1,"arch":"a","generation":"g","compute_cap":"sm_121"}`}, "must look like 12.1"},
		{"shared compute cap", map[string]string{
			"architectures/a.json": `{"schema_version":1,"arch":"a","generation":"g","compute_cap":"9.0"}`,
			"architectures/b.json": `{"schema_version":1,"arch":"b","generation":"g","compute_cap":"9.0"}`,
		}, "belongs to both"},
		{"bad PCI id", map[string]string{"vendors/acme.json": `{"schema_version":1,"id":"acme","name":"Acme","pci_ids":["10de"]}`}, "must look like 0x10de"},
		{"shared PCI id", map[string]string{
			"vendors/a.json": `{"schema_version":1,"id":"a","name":"A","pci_ids":["0x1234"]}`,
			"vendors/b.json": `{"schema_version":1,"id":"b","name":"B","pci_ids":["0x1234"]}`,
		}, "belongs to both"},
		{"vendor without a name", map[string]string{"vendors/acme.json": `{"schema_version":1,"id":"acme"}`}, "has no name"},
		{"a second object", map[string]string{"vendors/acme.json": `{"schema_version":1,"id":"acme","name":"Acme"}{"schema_version":1}`}, "after the top-level object"},
		{"trailing garbage", map[string]string{"vendors/acme.json": `{"schema_version":1,"id":"acme","name":"Acme"} x`}, "after the top-level object"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := load(tt.files)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Load = %v, want an error containing %q", err, tt.want)
			}
		})
	}
}

// A definition is authoritative once it matches, in either direction, and a
// vendor rule has to hold as well as the name. When two definitions claim the
// same GPU, Matches returns both, so doctor can name the overlap.
func TestMatchHonoursVendorAndDiscrete(t *testing.T) {
	t.Parallel()
	reg, err := load(map[string]string{
		"vendors/acme.json":       `{"schema_version":1,"id":"acme","name":"Acme"}`,
		"machines/big_card.json":  `{"schema_version":1,"id":"big_card","display_name":"Big card","match":{"gpu_name":"Big Card 96GB"},"memory":"discrete","why":"discrete, whatever its size"}`,
		"machines/acme_card.json": `{"schema_version":1,"id":"acme_card","display_name":"Acme card","match":{"vendor":"acme","gpu_name":"Card"},"memory":"unified","why":"shares memory"}`,
	})
	if err != nil {
		t.Fatalf("Load: %v", err)
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
	if m := reg.Matches(hardware.Observed{Vendor: "acme", GPUName: "ACME Big Card 96GB"}); len(m) != 2 {
		t.Errorf("Matches for a GPU both claim = %+v, want both", m)
	}
}

// A local file teaches llm-helper about a new machine. It must not be able to
// change what is already known, because that would let a stale file turn a
// Spark discrete or drop a flag that aborts it.
func TestExtendOnlyAdds(t *testing.T) {
	t.Parallel()
	ship := hardware.Shipped()
	reg := ship
	for _, f := range []struct {
		kind       hardware.Kind
		path, body string
	}{
		{hardware.KindVendor, "/etc/vendors/acme.json", `{"schema_version":1,"id":"acme","name":"Acme","pci_ids":["0xacce"]}`},
		{hardware.KindArchitecture, "/etc/architectures/sm121.json", `{"schema_version":1,"arch":"sm121","forbidden_args":[{"flag":"--new-flag","value":"on","why":"aborts too"}]}`},
		{hardware.KindArchitecture, "/etc/architectures/acme1.json", `{"schema_version":1,"arch":"acme1","generation":"acme"}`},
		{hardware.KindMachine, "/etc/machines/acme_box.json", `{"schema_version":1,"id":"acme_box","display_name":"Acme box","match":{"vendor":"acme","gpu_name":"Acme"},"memory":"unified","arch":"acme1","why":"measured on one"}`},
		{hardware.KindMachine, "/etc/machines/late_gb10.json", `{"schema_version":1,"id":"late_gb10","display_name":"Shadowed","match":{"vendor":"nvidia","gpu_name":"GB10"},"memory":"discrete","why":"too late to matter"}`},
	} {
		next, err := reg.Extend(f.kind, f.path, []byte(f.body))
		if err != nil {
			t.Fatalf("Extend %s: %v", f.path, err)
		}
		reg = next
	}
	if d, ok := reg.Match(hardware.Observed{Vendor: "acme", GPUName: "Acme 1"}); !ok || d.ID != "acme_box" || d.Source != "/etc/machines/acme_box.json" {
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

	for name, f := range map[string]struct {
		kind       hardware.Kind
		path, body string
	}{
		"a shipped id":          {hardware.KindMachine, "dgx_spark_gb10.json", `{"schema_version":1,"id":"dgx_spark_gb10","display_name":"x","match":{"gpu_name":"GB10"},"memory":"discrete","why":"w"}`},
		"a shipped vendor":      {hardware.KindVendor, "nvidia.json", `{"schema_version":1,"id":"nvidia","name":"Not NVIDIA"}`},
		"a shipped generation":  {hardware.KindArchitecture, "sm121.json", `{"schema_version":1,"arch":"sm121","generation":"hopper"}`},
		"a shipped compute cap": {hardware.KindArchitecture, "sm121.json", `{"schema_version":1,"arch":"sm121","compute_cap":"12.0"}`},
		"a claimed PCI id":      {hardware.KindVendor, "acme.json", `{"schema_version":1,"id":"acme","name":"Acme","pci_ids":["0x10de"]}`},
		"an invalid entry":      {hardware.KindMachine, "acme.json", `{"schema_version":1,"id":"acme","display_name":"Acme","match":{"gpu_name":"Acme"},"memory":"unified"}`},
		"a misnamed file":       {hardware.KindVendor, "acme.json", `{"schema_version":1,"id":"acme_corp","name":"Acme"}`},
		"the wrong folder":      {hardware.KindVendor, "acme.json", `{"schema_version":1,"id":"acme","display_name":"Acme","match":{"gpu_name":"Acme"},"memory":"unified","why":"w"}`},
	} {
		if _, err := ship.Extend(f.kind, f.path, []byte(f.body)); err == nil {
			t.Errorf("Extend accepted %s", name)
		}
	}
}

// Not parallel: UseLocal replaces what Default returns for the process.
func TestUseLocalSkipsABadFileAndKeepsTheRest(t *testing.T) {
	dir := t.TempDir()
	t.Cleanup(func() { hardware.UseLocal(filepath.Join(dir, "missing")) })
	write := func(name, body string) {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("machines/oops.json", `{"schema_version":1,"id":"oops"`)
	write("vendors/acme.json", `{"schema_version":1,"id":"acme","name":"Acme"}`)
	write("machines/acme_box.json", `{"schema_version":1,"id":"acme_box","display_name":"Acme box","match":{"vendor":"acme","gpu_name":"Acme"},"memory":"unified","why":"w"}`)
	write("old.json", `{"schema_version":1,"definitions":[]}`)
	write("machines/notes.txt", "not a definition")

	errs := hardware.UseLocal(dir)
	var got []string
	for _, err := range errs {
		got = append(got, err.Error())
	}
	if len(errs) != 2 || !strings.Contains(got[0], "old.json") || !strings.Contains(got[1], "oops.json") {
		t.Errorf("UseLocal = %q, want old.json named as not loaded, then oops.json", got)
	}
	if got := hardware.LocalErrors(); len(got) != 2 {
		t.Errorf("LocalErrors = %v", got)
	}
	if _, ok := hardware.Default().ByID("acme_box"); !ok {
		t.Error("the good files were not loaded")
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
