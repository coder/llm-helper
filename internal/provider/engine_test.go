package provider_test

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/provider"
)

const secretKey = "sk-never-on-argv"

// Every shipped engine has to launch for every rank a recipe can give it,
// and keep a key that is not a flag off argv, where ps shows it to every
// user on the machine.
func TestShippedEnginesHoldTheContract(t *testing.T) {
	t.Parallel()
	for _, e := range provider.Shipped().Engines() {
		if strings.TrimSpace(e.Why) == "" {
			t.Errorf("%s: no why", e.ID)
		}
		if e.Has(provider.RuntimeProcess) {
			if e.InstallHint("linux") == "" || e.InstallHint("darwin") == "" {
				t.Errorf("%s: a process engine needs an install hint on every OS", e.ID)
			}
			argv, err := e.Command(provider.Spec{Bin: "/x", Model: "org/m", Port: "8001", APIKey: secretKey})
			if err != nil {
				t.Fatalf("%s: %v", e.ID, err)
			}
			onArgv := slices.Contains(argv, secretKey)
			if flag := strings.HasPrefix(e.APIKey, "flag:"); onArgv != flag {
				t.Errorf("%s: api_key %q, but the key on argv is %v: %q", e.ID, e.APIKey, onArgv, argv)
			}
			for _, a := range argv {
				if strings.ContainsAny(a, "{}") {
					t.Errorf("%s: unrendered placeholder %q", e.ID, a)
				}
			}
		}
		if !e.Has(provider.RuntimeContainer) {
			continue
		}
		serving := 0
		for _, r := range e.Container.Roles {
			if r.ServesHTTP {
				serving++
			}
		}
		if serving != 1 {
			t.Errorf("%s: %d roles serve HTTP, want 1", e.ID, serving)
		}
		models := map[string]provider.ModelRef{}
		for _, role := range e.UsesModelPaths() {
			models[role] = provider.ModelRef{ID: "org/" + role, Revision: strings.Repeat("a", 40)}
		}
		for rank := range 4 {
			r, err := e.RoleFor(rank)
			if err != nil {
				t.Fatalf("%s: %v", e.ID, err)
			}
			_, args, env := e.RenderRole(r, provider.RoleValues{
				Model: "m", Port: 8000, Rank: rank, Nodes: 4, MasterAddr: "10.0.0.1", MasterPort: 29500,
				Args: []string{"--x", "1"}, Models: models,
			})
			for _, a := range args {
				if strings.ContainsAny(a, "{}") || a == "" {
					t.Errorf("%s rank %d: argument %q did not render", e.ID, rank, a)
				}
			}
			for k, v := range env {
				if strings.ContainsAny(v, "{}") {
					t.Errorf("%s rank %d: env %s=%q did not render", e.ID, rank, k, v)
				}
			}
			if !slices.Contains(args, "--x") {
				t.Errorf("%s rank %d: the recipe's args are missing from %q", e.ID, rank, args)
			}
		}
		if _, err := e.RenderMounts("/home/u", nil); err != nil {
			t.Errorf("%s: default mounts: %v", e.ID, err)
		}
	}
}

// TensorFold's server checks no key, and its image's own entrypoint sets up
// CUDA, so neither may be overridden.
func TestTensorFoldPythonKeepsTheImageEntrypointAndTakesNoKey(t *testing.T) {
	t.Parallel()
	e, ok := provider.Shipped().Get("tensorfold-python")
	if !ok {
		t.Fatal("tensorfold-python is not shipped")
	}
	if e.Has(provider.RuntimeProcess) {
		t.Error("tensorfold-python runs only in a container")
	}
	if e.Capabilities().APIKey || e.ContainerKeyEnv() != "" {
		t.Error("tensorfold-python claims to check a key")
	}
	for rank := range 2 {
		r, _ := e.RoleFor(rank)
		if ep, _, _ := e.RenderRole(r, provider.RoleValues{}); ep != "" {
			t.Errorf("rank %d overrides the image entrypoint with %q", rank, ep)
		}
	}
	if !slices.Contains(e.GPUVendors, "nvidia") || len(e.GPUVendors) != 1 {
		t.Errorf("gpu_vendors = %v, want nvidia only", e.GPUVendors)
	}
}

const toyContainer = `{
  "schema_version": 1,
  "engines": [
    {
      "id": "toy",
      "display_name": "Toy",
      "why": "a test engine",
      "runtime": ["container"],
      "params": [{ "name": "threads", "default": "4", "why": "test" }],
      "health": { "path": "/health", "mode": "ready" },
      "container": {
        "api_key": "none",
        "launch_order": "head_first",
        "roles": [
          {
            "name": "all",
            "ranks": "all",
            "serves_http": true,
            "args": ["toy", "serve", "{model}", "--port", "{port}", "--threads", "{threads}", "{args}"]
          }
        ]
      },
      "capabilities": {}
    }
  ]
}`

const toyProcess = `{
  "schema_version": 1,
  "engines": [
    {
      "id": "toyp",
      "display_name": "Toy process",
      "why": "a test engine",
      "runtime": ["process"],
      "install": { "any": "pip install toy" },
      "bin": { "key": "TOY_BIN", "names": ["toy"] },
      "serve": ["{bin}", "--model", "{model}", "--port", "{port}"],
      "api_key": "env:TOY_KEY",
      "health": { "path": "/health", "mode": "live" },
      "capabilities": {}
    }
  ]
}`

// A definition that would launch something other than what it says is
// refused whole, not half read.
func TestParseIsStrict(t *testing.T) {
	t.Parallel()
	for _, base := range []string{toyContainer, toyProcess} {
		if _, err := provider.Parse([]byte(base)); err != nil {
			t.Fatalf("the base definition does not parse: %v", err)
		}
	}
	for _, tc := range []struct {
		name, base, old, new, want string
	}{
		{"unknown key", toyContainer, `"why": "a test engine"`, `"why": "a test engine", "wy": "typo"`, "unknown field"},
		{"wrong schema version", toyContainer, `"schema_version": 1`, `"schema_version": 2`, "schema_version"},
		{"unknown placeholder", toyContainer, `"{threads}"`, `"{thread}"`, "{thread}"},
		{"key on argv without a flag", toyProcess, `"{port}"]`, `"{port}", "--key", "{api_key}"]`, "ps shows it"},
		{"container key as a flag", toyContainer, `"api_key": "none"`, `"api_key": "flag:--api-key"`, "docker's command line"},
		{"no serving role", toyContainer, `"serves_http": true,`, ``, "exactly one role"},
		{"no why", toyContainer, `"why": "a test engine"`, `"why": " "`, "no why"},
		{"param without a reason", toyContainer, `"why": "test"`, `"why": ""`, "has no why"},
		{"param shadows a built-in", toyContainer, `"name": "threads"`, `"name": "port"`, "shadows"},
		{"mount outside home", toyContainer, `"launch_order"`, `"mounts": [{"host": "/etc", "container": "/x"}], "launch_order"`, "under the home"},
		{"process engine without install hint", toyProcess, `"install": { "any": "pip install toy" },`, ``, "install hint"},
		{"trailing data", toyContainer, ``, ``, "after the top-level"},
	} {
		data := strings.Replace(tc.base, tc.old, tc.new, 1)
		if tc.name == "trailing data" {
			data += "{}"
		}
		if data == tc.base && tc.name != "trailing data" {
			t.Fatalf("%s: the edit matched nothing", tc.name)
		}
		_, err := provider.Parse([]byte(data))
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want it to mention %q", tc.name, err, tc.want)
		}
	}
}

func TestExtendOnlyAdds(t *testing.T) {
	t.Parallel()
	r, err := provider.Shipped().Extend([]byte(toyContainer), "/etc/toy.json")
	if err != nil {
		t.Fatal(err)
	}
	e, ok := r.Get("toy")
	if !ok || e.Source != "/etc/toy.json" {
		t.Fatalf("toy = %+v, %v", e, ok)
	}
	if _, ok := provider.Shipped().Get("toy"); ok {
		t.Error("Extend changed the registry it extended")
	}

	redefine := strings.Replace(toyContainer, `"id": "toy"`, `"id": "vllm"`, 1)
	if _, err := provider.Shipped().Extend([]byte(redefine), "x.json"); err == nil || !strings.Contains(err.Error(), "already defined") {
		t.Errorf("redefining vllm: err = %v", err)
	}
	invalid := strings.Replace(toyContainer, `"{threads}"`, `"{nope}"`, 1)
	if _, err := provider.Shipped().Extend([]byte(invalid), "x.json"); err == nil {
		t.Error("an invalid local engine was accepted")
	}
}

// Not parallel: UseLocal swaps the registry every Lookup reads.
func TestUseLocalSkipsAndReportsBadFiles(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"a-broken.json":   "{",
		"b-toy.json":      toyContainer,
		"c-redefine.json": strings.Replace(toyProcess, `"id": "toyp"`, `"id": "vllm"`, 1),
		"d-toyp.json":     toyProcess,
		"notes.txt":       "not a definition",
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { provider.UseLocal(filepath.Join(t.TempDir(), "gone")) })

	errs := provider.UseLocal(dir)
	if len(errs) != 2 {
		t.Fatalf("errors = %v, want the broken and the redefining file", errs)
	}
	for i, name := range []string{"a-broken.json", "c-redefine.json"} {
		if !strings.Contains(errs[i].Error(), name) {
			t.Errorf("error %d = %v, want it to name %s", i, errs[i], name)
		}
	}
	if got := provider.LocalErrors(); len(got) != 2 {
		t.Errorf("LocalErrors = %v", got)
	}
	for _, id := range []string{"toy", "toyp"} {
		e, ok := provider.Default().Get(id)
		if !ok || !strings.HasPrefix(e.Source, dir) {
			t.Errorf("%s = %+v, %v; a good file after a bad one must still load", id, e, ok)
		}
	}
	if e, _ := provider.Default().Get("vllm"); e.Source != "" {
		t.Errorf("vllm comes from %q; a local file redefined it", e.Source)
	}
	if _, err := provider.LookupEngine("toyp"); err != nil {
		t.Errorf("a local engine cannot be selected: %v", err)
	}

	if errs := provider.UseLocal(filepath.Join(dir, "missing")); len(errs) != 0 {
		t.Errorf("a missing directory is an error: %v", errs)
	}
	if _, ok := provider.Default().Get("toy"); ok {
		t.Error("UseLocal kept engines from a directory it no longer reads")
	}
	if len(provider.LocalErrors()) != 0 {
		t.Error("UseLocal kept errors from a directory it no longer reads")
	}
}
