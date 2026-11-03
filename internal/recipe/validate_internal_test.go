package recipe

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func glm(t *testing.T) TestedRecipe {
	t.Helper()
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	r, ok := tt.Get("glm-5.3-flash-tf-2spark")
	if !ok {
		t.Fatal("glm-5.3-flash-tf-2spark missing")
	}
	return r
}

// A recipe a launch could not carry out is refused when it loads, not
// halfway through starting a cluster.
func TestValidateRefusesWhatALaunchCannotDo(t *testing.T) {
	t.Parallel()
	if err := glm(t).validate(); err != nil {
		t.Fatalf("the shipped recipe does not validate: %v", err)
	}
	for _, tc := range []struct {
		name string
		edit func(*TestedRecipe)
		want string
	}{
		{"unpinned image", func(r *TestedRecipe) { r.Image = "ghcr.io/x/y:latest" }, "not pinned by digest"},
		{"unknown source format", func(r *TestedRecipe) { r.Source.Format = "yaml" }, "source.format"},
		{"unknown engine", func(r *TestedRecipe) { r.Provider = "ollama" }, "ollama"},
		{"process-only engine", func(r *TestedRecipe) { r.Provider = "mlx"; r.Params = nil; r.Models = nil; r.Args = nil }, "no container runtime"},
		{"undeclared param", func(r *TestedRecipe) { r.Params = map[string]string{"threads": "4"} }, `"threads"`},
		{"unknown placeholder", func(r *TestedRecipe) { r.Args = append(r.Args, "{nope}") }, "{nope}"},
		{"drafter path with no drafter", func(r *TestedRecipe) { r.Models = r.Models[:1] }, "pinned drafter model"},
		{"short revision", func(r *TestedRecipe) { r.Models[0].Revision = "078455ff" }, "full commit sha"},
		{"branch revision", func(r *TestedRecipe) { r.Models[0].Revision = "main" }, "full commit sha"},
		{"bad role", func(r *TestedRecipe) { r.Models[1].Role = "draft" }, "role"},
		{"two models in one role", func(r *TestedRecipe) { r.Models[1].Role = r.Models[0].Role }, "is taken"},
		{"staging", func(r *TestedRecipe) { r.Models[0].Staging = "copy" }, "staging"},
		{"unknown hardware", func(r *TestedRecipe) { r.Requires = "dgx_spark_gb11" }, "no hardware definition"},
	} {
		r := glm(t)
		r.Models = append([]Model(nil), r.Models...)
		r.Args = append([]string(nil), r.Args...)
		tc.edit(&r)
		err := r.validate()
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want it to mention %q", tc.name, err, tc.want)
		}
	}
}

func writeRecipes(t *testing.T, path string, recipes ...TestedRecipe) {
	t.Helper()
	data, err := json.Marshal(Tested{SchemaVersion: SchemaVersion, Recipes: recipes})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// Not parallel: UseLocal changes what every LoadTested returns.
func TestUseLocalAddsRecipesAndReportsBadFiles(t *testing.T) {
	dir := t.TempDir()
	mine := glm(t)
	mine.ID = "glm-mine"
	bad := glm(t)
	bad.ID = "glm-bad"
	bad.Image = "ghcr.io/x/y:latest"
	writeRecipes(t, filepath.Join(dir, "a-mine.json"), mine)
	writeRecipes(t, filepath.Join(dir, "b-redefine.json"), glm(t))
	writeRecipes(t, filepath.Join(dir, "c-bad.json"), bad)
	writeRecipes(t, filepath.Join(dir, "d-again.json"), mine)
	if err := os.WriteFile(filepath.Join(dir, "e-broken.json"), []byte(`{"schema_version": 1, "recipes": [], "extra": 1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { UseLocal(filepath.Join(t.TempDir(), "gone")) })

	errs := UseLocal(dir)
	wantErrs := []string{"b-redefine.json", "c-bad.json", "d-again.json", "e-broken.json"}
	if len(errs) != len(wantErrs) {
		t.Fatalf("errors = %v, want one each for %v", errs, wantErrs)
	}
	for i, name := range wantErrs {
		if !strings.Contains(errs[i].Error(), name) {
			t.Errorf("error %d = %v, want it to name %s", i, errs[i], name)
		}
	}
	if len(LocalErrors()) != len(wantErrs) {
		t.Errorf("LocalErrors = %v", LocalErrors())
	}

	tt, err := LoadTested()
	if err != nil {
		t.Fatal(err)
	}
	got, ok := tt.Get("glm-mine")
	if !ok || got.Local != filepath.Join(dir, "a-mine.json") {
		t.Fatalf("glm-mine = %q, %v", got.Local, ok)
	}
	if shipped, _ := tt.Get("glm-5.3-flash-tf-2spark"); shipped.Local != "" {
		t.Errorf("a local file redefined a shipped recipe: %q", shipped.Local)
	}
	if _, ok := tt.Get("glm-bad"); ok {
		t.Error("a recipe that fails validation was loaded")
	}

	UseLocal(filepath.Join(dir, "gone"))
	tt, _ = LoadTested()
	if _, ok := tt.Get("glm-mine"); ok || len(LocalErrors()) != 0 {
		t.Error("UseLocal kept what a directory it no longer reads added")
	}
}
