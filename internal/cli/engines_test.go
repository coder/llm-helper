package cli_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/provider"
	"github.com/coder/vllm-helper/internal/recipe"
)

const toyEngine = `{"schema_version":1,"engines":[{"id":"toy","display_name":"Toy","why":"a test engine",` +
	`"runtime":["container"],"health":{"path":"/health","mode":"ready"},` +
	`"container":{"api_key":"none","launch_order":"head_first",` +
	`"roles":[{"name":"all","ranks":"all","serves_http":true,"args":["toy","{model}","{args}"]}]},"capabilities":{}}]}`

// An engines.d file is loaded before every command, so 'engines list' shows
// what a machine owner added and where it came from, and a broken file is
// named rather than taking the others with it.
func TestRunEnginesListShowsLocalDefinitions(t *testing.T) {
	configDir, _ := newTree(t)
	dir := filepath.Join(configDir, provider.LocalDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	good := filepath.Join(dir, "20-toy.json")
	files := map[string]string{
		filepath.Join(dir, "10-bad.json"): `{"schema_version":1,"engines":[`,
		good:                              toyEngine,
	}
	for path, body := range files {
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Later tests in this process must not see this directory's engines.
	t.Cleanup(func() {
		provider.UseLocal(filepath.Join(dir, "gone"))
		recipe.UseLocal(filepath.Join(dir, "gone"))
	})

	code, stdout, stderr := runCLI(t, "engines", "list")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, stderr)
	}
	for _, id := range []string{"vllm", "mlx", "llamacpp", "tensorfold-python"} {
		if !strings.Contains(stdout, id) {
			t.Errorf("shipped engine %s is missing:\n%s", id, stdout)
		}
	}
	var toy string
	for _, line := range strings.Split(stdout, "\n") {
		if strings.HasPrefix(line, "toy ") {
			toy = line
		}
	}
	if !strings.HasSuffix(toy, good) {
		t.Errorf("toy row = %q, want it to end with its file %s", toy, good)
	}
	if !strings.Contains(stderr, "10-bad.json") {
		t.Errorf("stderr = %q, want the broken file named", stderr)
	}

	code, stdout, _ = runCLI(t, "engines", "list", "--json")
	if code != 0 {
		t.Fatalf("--json exit %d", code)
	}
	var rows []struct {
		ID     string `json:"id"`
		Source string `json:"source"`
	}
	if err := json.Unmarshal([]byte(stdout), &rows); err != nil {
		t.Fatalf("--json is not JSON: %v\n%s", err, stdout)
	}
	var found bool
	for _, r := range rows {
		found = found || (r.ID == "toy" && r.Source == good)
	}
	if !found {
		t.Errorf("--json rows = %+v, want toy from %s", rows, good)
	}
}

func TestRunEnginesShow(t *testing.T) {
	newTree(t)
	code, stdout, stderr := runCLI(t, "engines", "show", "tensorfold-python")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, stderr)
	}
	for _, want := range []string{"container key", "/v1/models (ready)", "Container role head (ranks first)", "tensor_parallel"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("output lacks %q:\n%s", want, stdout)
		}
	}
	if code, _, stderr := runCLI(t, "engines", "show", "ollama"); code == 0 || !strings.Contains(stderr, "vllm") {
		t.Errorf("an unknown engine: exit %d, stderr %q; want an error naming the choices", code, stderr)
	}
}

// What a node has to have before a launch, and what the recipe departs from,
// is shown before anything runs.
func TestRunRecipesShowNamesTheEngineAndPinnedModels(t *testing.T) {
	newTree(t)
	code, stdout, stderr := runCLI(t, "recipes", "show", "glm-5.3-flash-tf-2spark")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, stderr)
	}
	for _, want := range []string{
		"tensorfold-python", "source format  env-sh", "requires       dgx_spark_gb10",
		"drafter  incoai/GLM-5.3-Flash-DFlash2@", "SPILL_GIB: 0 -> 128", "CC BY-NC-ND", "tensor_parallel=2",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("output lacks %q:\n%s", want, stdout)
		}
	}
}
