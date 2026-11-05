package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/nodes"
	"github.com/coder/llm-helper/internal/recipe"
)

func glmRecipe(t *testing.T) recipe.TestedRecipe {
	t.Helper()
	tt, err := recipe.LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	r, ok := tt.Get("glm-5.3-flash-tf-2spark")
	if !ok {
		t.Fatal("glm-5.3-flash-tf-2spark missing")
	}
	return r
}

// The hardware definitions' forbidden flags are vLLM's, so they gate only an
// engine that takes vLLM's args. What every engine is held to is the GPU
// vendor it runs on.
func TestCheckRecipeMembersGatesOnTheEngine(t *testing.T) {
	r := glmRecipe(t)
	if r.Engine().Capabilities().VLLMArgs {
		t.Fatalf("%s runs on %s, which takes vLLM's args", r.ID, r.Engine().ID)
	}
	spark := recipeMember{node: nodes.Node{Name: "a", Caps: nodes.Capabilities{HardwareID: "dgx_spark_gb10", Arch: "sm121"}}}
	var w strings.Builder
	vllmOnly := append(r.ServeArgs(), "--moe-backend", "deep_gemm_mega_moe")
	if err := checkRecipeMembers(&w, r, []recipeMember{spark}, vllmOnly); err != nil {
		t.Errorf("a vLLM-only refusal was applied to %s: %v", r.Engine().ID, err)
	}

	mac := recipeMember{node: nodes.Node{Name: "mac", Caps: nodes.Capabilities{HardwareID: "apple_silicon"}}}
	err := checkRecipeMembers(&w, r, []recipeMember{spark, mac}, r.ServeArgs())
	if err == nil || !strings.Contains(err.Error(), "node mac is apple") {
		t.Errorf("checkRecipeMembers = %v, want node mac refused for its GPU vendor", err)
	}

	unknown := recipeMember{node: nodes.Node{Name: "u", Caps: nodes.Capabilities{HardwareID: "not_a_definition"}}}
	if err := checkRecipeMembers(&w, r, []recipeMember{unknown}, r.ServeArgs()); err != nil {
		t.Errorf("a node that matched no definition was refused: %v", err)
	}
}

// A pinned revision that is not in a node's cache is named with the command
// that downloads it, before anything is started.
func TestCheckStagedNamesWhatIsMissing(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	r := glmRecipe(t)
	cache := r.Engine().HFCacheHostPath(r.Params)
	refs := r.ModelRefs()
	if cache == "" || len(refs) < 2 {
		t.Fatalf("%s pins %d models into cache %q", r.ID, len(refs), cache)
	}
	weights, drafter := refs["weights"], refs["drafter"]
	stage := func(m engine.ModelRef) {
		t.Helper()
		dir := filepath.Join(home, engine.SnapshotPath(cache, m))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("w"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	stage(weights)
	local := recipeMember{node: nodes.Node{Name: "head", Host: "http://127.0.0.1:8001"}}

	err := checkStaged(r, []recipeMember{local}, "")
	if err == nil {
		t.Fatal("a missing drafter passed the staging check")
	}
	msg := err.Error()
	for _, want := range []string{drafter.ID + "@" + drafter.Revision, "node head", "hf download " + drafter.ID + " --revision " + drafter.Revision} {
		if !strings.Contains(msg, want) {
			t.Errorf("error %q does not mention %q", msg, want)
		}
	}
	if strings.Contains(msg, weights.ID) {
		t.Errorf("error %q names the weights, which are staged", msg)
	}

	// A snapshot directory with nothing in it is a download that did not
	// finish.
	if err := os.MkdirAll(filepath.Join(home, engine.SnapshotPath(cache, drafter)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := checkStaged(r, []recipeMember{local}, ""); err == nil || !strings.Contains(err.Error(), drafter.ID) {
		t.Errorf("an empty drafter snapshot: err = %v", err)
	}
	stage(drafter)
	if err := checkStaged(r, []recipeMember{local}, ""); err != nil {
		t.Errorf("checkStaged with everything staged = %v", err)
	}
}
