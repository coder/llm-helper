package recipe

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/hardware"
)

func TestTestedRecipesAreWellFormed(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	if len(tt.Recipes) == 0 {
		t.Fatal("no tested recipes shipped")
	}
	seen := map[string]bool{}
	for _, r := range tt.Recipes {
		if seen[r.ID] {
			t.Errorf("duplicate id %q", r.ID)
		}
		seen[r.ID] = true

		if r.DisplayName == "" {
			t.Errorf("%s: display_name is what users see, it cannot be empty", r.ID)
		}
		if r.Model == "" || r.Port == 0 || len(r.Args) == 0 {
			t.Errorf("%s: model, port and args are all required", r.ID)
		}
		switch r.Topology {
		case "solo", "shard", "pool":
		default:
			t.Errorf("%s: unknown topology %q", r.ID, r.Topology)
		}
		if r.Topology != "solo" && r.Nodes < 2 {
			t.Errorf("%s: topology %q needs nodes >= 2, got %d", r.ID, r.Topology, r.Nodes)
		}
		// A moving tag is not a pin: eugr's :latest advanced under us mid-session.
		if !strings.Contains(r.Image, "@sha256:") {
			t.Errorf("%s: image must be digest-pinned, got %q", r.ID, r.Image)
		}
		if r.Verified.Date == "" || r.Verified.Hardware == "" {
			t.Errorf("%s: a tested recipe must record when and where it was verified", r.ID)
		}
		if r.Verified.Smoke == "" {
			t.Errorf("%s: missing smoke-test evidence", r.ID)
		}
	}
}

// The tested args are what we actually launch, so they must satisfy the same
// arch invariant the resolver enforces.
func TestTestedRecipesHoldTheArchInvariant(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	for _, r := range tt.Recipes {
		target, ok := r.Target()
		if !ok {
			continue
		}
		if err := checkForbidden(r.Args, target); err != nil {
			t.Errorf("%s: %v", r.ID, err)
		}
	}
}

// The forbidden-flag check finds a recipe's architecture through the hardware
// id it was verified on. An id no descriptor knows would skip that check
// without a word, so a typo in it has to fail here instead.
func TestTestedRecipesNameKnownHardware(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	for _, r := range tt.Recipes {
		if _, ok := hardware.Default().ByID(r.HardwareID()); !ok {
			t.Errorf("%s: verified hardware %q names no hardware descriptor", r.ID, r.Verified.Hardware)
		}
	}
}

// The Spark recipes are what the forbidden-flag check was written for, so they
// must still resolve to sm121 now that the mapping is data.
func TestSparkRecipesTargetSM121(t *testing.T) {
	r := TestedRecipe{Verified: Verification{Hardware: "dgx_spark_gb10 x2"}}
	got, ok := r.Target()
	if !ok || got != sparkTarget {
		t.Errorf("Target() = %+v, %v; want %+v", got, ok, sparkTarget)
	}
	if _, ok := (TestedRecipe{Verified: Verification{Hardware: "some_other_box"}}).Target(); ok {
		t.Error("an unknown hardware id resolved to a target")
	}
}

// Provenance is what makes recipediff possible and satisfies upstream attribution,
// so an entry without a pinned source commit is not shippable.
func TestTestedRecipesCiteAPinnedSource(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	for _, r := range tt.Recipes {
		if r.Source.Repo == "" {
			t.Errorf("%s: no source.repo", r.ID)
		}
		if len(r.Source.Commit) != 40 {
			t.Errorf("%s: source.commit %q is not a full 40-char sha", r.ID, r.Source.Commit)
		}
		if r.Source.License == "" {
			t.Errorf("%s: no source.license", r.ID)
		}
		if !strings.Contains(r.Source.Describe(), r.Source.Repo) {
			t.Errorf("%s: Describe() omits the repo: %q", r.ID, r.Source.Describe())
		}
	}
}

// A recipe that serves a different context than it claims to have verified is
// worse than no recipe, so the two must be edited together.
func TestVerifiedContextMatchesMaxModelLen(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	for _, r := range tt.Recipes {
		if r.Verified.Context == 0 {
			continue
		}
		want := strconv.Itoa(r.Verified.Context)
		var got string
		for i, a := range r.Args {
			if a == "--max-model-len" && i+1 < len(r.Args) {
				got = r.Args[i+1]
			}
			if v, ok := strings.CutPrefix(a, "--max-model-len="); ok {
				got = v
			}
		}
		if got != want {
			t.Errorf("%s: --max-model-len %q but verified.context %q", r.ID, got, want)
		}
	}
}

func TestDeepSeekRecipeSummary(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	r, ok := tt.Get("deepseek-v4-flash-2spark")
	if !ok {
		t.Fatal("deepseek-v4-flash-2spark missing")
	}
	if r.DisplayName != "DeepSeek V4 Flash" {
		t.Errorf("display name = %q", r.DisplayName)
	}
	got := r.Summary()
	// Derive the concurrency from the entry rather than pinning the number: a
	// re-measurement is supposed to change it, and a test that breaks on every
	// honest re-stamp teaches people to edit tests without reading them.
	for _, want := range []string{"2 nodes, sharded", "256K ctx", fmt.Sprintf("%.1fx conc", r.Verified.Concurrency)} {
		if !strings.Contains(got, want) {
			t.Errorf("summary %q missing %q", got, want)
		}
	}
	// The image-scoped backend name that vllm-project gets wrong for this image.
	joined := strings.Join(r.Args, " ")
	if !strings.Contains(joined, "--attention-backend B12X") || strings.Contains(joined, "B12X_MLA_SPARSE") {
		t.Errorf("attention backend must be B12X, got: %s", joined)
	}
}

// Deviations are the entry's claim about how it differs from its source, so
// they have to stay true to the args beside them. A setting that names a flag
// must match that flag's value, and a setting recorded as dropped must not
// appear in the args at all.
func TestDeviationsMatchTheArgsTheyDescribe(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	for _, r := range tt.Recipes {
		seen := map[string]bool{}
		for _, d := range r.Source.Deviations {
			if d.Setting == "" || d.Why == "" {
				t.Errorf("%s: a deviation needs a setting and a reason, got %+v", r.ID, d)
				continue
			}
			if seen[d.Setting] {
				t.Errorf("%s: duplicate deviation for %q", r.ID, d.Setting)
			}
			seen[d.Setting] = true
			if d.Upstream == d.Ours {
				t.Errorf("%s: %q records no change at all", r.ID, d.Setting)
			}

			// Where the setting lives decides where the claim is checked.
			// Guessing it from the name turned an env key into the flag
			// --VLLM-MOE-SKIP-PADDING, which no entry could ever carry.
			switch d.Kind {
			case DeviationFlag:
				// A dropped setting leaves no trace in the entry, so the
				// shape of the name is the only local check on the kind.
				if strings.Contains(d.Setting, "/") || d.Setting != strings.ToLower(d.Setting) {
					t.Errorf("%s: %q is declared as a flag but is not named like one", r.ID, d.Setting)
				}
				flag := "--" + strings.ReplaceAll(d.Setting, "_", "-")
				got, present := deviationArgValue(r.Args, flag)
				switch {
				case present && got != d.Ours:
					t.Errorf("%s: %s says ours is %q but the args carry %q", r.ID, d.Setting, d.Ours, got)
				case !present && d.Ours != "":
					t.Errorf("%s: %s claims ours is %q but %s is not in the args", r.ID, d.Setting, d.Ours, flag)
				}
			case DeviationEnv:
				if d.Setting != strings.ToUpper(d.Setting) {
					t.Errorf("%s: %q is declared as an env key but is not named like one", r.ID, d.Setting)
				}
				got, present := r.Env[d.Setting]
				switch {
				case present && got != d.Ours:
					t.Errorf("%s: env %s says ours is %q but the entry sets %q", r.ID, d.Setting, d.Ours, got)
				case !present && d.Ours != "":
					t.Errorf("%s: env %s claims ours is %q but the entry does not set it", r.ID, d.Setting, d.Ours)
				}
			case DeviationMod:
				if !strings.Contains(d.Setting, "/") {
					t.Errorf("%s: %q is declared as a mod but is not a mod path", r.ID, d.Setting)
				}
				// We consume the published image by digest and apply no mods,
				// so the only honest claim about one is that we dropped it.
				if d.Ours != "" {
					t.Errorf("%s: mod %s claims ours is %q, but we apply no mods", r.ID, d.Setting, d.Ours)
				}
			default:
				t.Errorf("%s: %s has kind %q, want flag, env or mod", r.ID, d.Setting, d.Kind)
			}
		}
	}
}

// deviationArgValue reports a flag's value, and whether the flag is there.
func deviationArgValue(args []string, flag string) (string, bool) {
	for i, a := range args {
		if a != flag {
			continue
		}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
			return args[i+1], true
		}
		return "", true
	}
	return "", false
}

func TestDeviationChangeReadsAsAMove(t *testing.T) {
	cases := []struct {
		dev  Deviation
		want string
	}{
		{Deviation{Upstream: "auto", Ours: "262144"}, "auto -> 262144"},
		{Deviation{Upstream: "6"}, "6 -> not used"},
		{Deviation{Ours: "32"}, "added: 32"},
		{Deviation{}, "not used here"},
	}
	for _, c := range cases {
		if got := c.dev.Change(); got != c.want {
			t.Errorf("Change() = %q, want %q", got, c.want)
		}
	}
}
