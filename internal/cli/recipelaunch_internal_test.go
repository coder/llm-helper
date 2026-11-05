package cli

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/nodes"
	"github.com/coder/llm-helper/internal/recipe"
)

// Locality decides the hop. Before this was a property of the node, every
// call ran locally, so a container with the same name on another box could be
// inspected and reported as the head's.
func TestNodeCommandHopsByNodeLocality(t *testing.T) {
	local := recipeMember{node: nodes.Node{Name: "head", Host: "http://127.0.0.1:8001"}}
	remote := recipeMember{node: nodes.Node{Name: "worker", Host: "http://not-this-machine.invalid:8001"}}

	cmd, err := nodeCommand(local, "", "echo hi", nil)
	if err != nil {
		t.Fatal(err)
	}
	if cmd.Args[0] != "bash" {
		t.Errorf("local node ran %q, want bash", cmd.Args[0])
	}

	cmd, err = nodeCommand(remote, "/tmp/key", "echo hi", nil)
	if err != nil {
		t.Fatal(err)
	}
	if cmd.Args[0] != "ssh" {
		t.Errorf("remote node ran %q, want ssh", cmd.Args[0])
	}
	if joined := strings.Join(cmd.Args, " "); !strings.Contains(joined, "-i /tmp/key") {
		t.Errorf("ssh lost the key flag: %s", joined)
	}
}

func testRecipe(t *testing.T) recipe.TestedRecipe {
	t.Helper()
	tt, err := recipe.LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	r, ok := tt.Get("deepseek-v4-flash-2spark")
	if !ok {
		t.Fatal("deepseek-v4-flash-2spark missing")
	}
	return r
}

func writeProfile(t *testing.T, body string) *config.Config {
	t.Helper()
	dir := t.TempDir()
	profiles := filepath.Join(dir, "models")
	if err := os.MkdirAll(profiles, 0o755); err != nil {
		t.Fatal(err)
	}
	if body != "" {
		path := filepath.Join(profiles, "deepseek-v4-flash-2spark.env")
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return &config.Config{Dirs: config.Dirs{Config: dir, Profiles: profiles}}
}

// A profile that only restates the recipe is not an override: reporting one
// would downgrade the verified claim for every launch.
func TestRecipeOverridesIgnoreRestatements(t *testing.T) {
	r := testRecipe(t)
	cfg := writeProfile(t, "MODEL="+r.Model+"\nMAX_MODEL_LEN=262144\nGPU_MEMORY_UTILIZATION=0.85\nKV_CACHE_MEMORY_BYTES=\nEXTRA_ARGS=\"\"\nRECIPE="+r.ID+"\n")
	local, err := recipeOverrides(cfg, r)
	if err != nil {
		t.Fatalf("recipeOverrides: %v", err)
	}
	if local.profile == nil {
		t.Fatal("the existing profile should have been returned")
	}
	if len(local.args) != 0 {
		t.Errorf("expected no overrides, got %v", local.args)
	}
}

// An EXTRA_ARGS value that matches the recipe still overrides a knob, because
// it comes later in the same argv. Treating it as a restatement on its own
// would launch with the knob's value instead, and writing back without it
// would change the next launch.
func TestRecipeOverridesResolveKnobsAndExtraArgsTogether(t *testing.T) {
	r := testRecipe(t)
	recipeLen, ok := recipe.FlagValue(r.Args, "--max-model-len")
	if !ok {
		t.Skip("test recipe sets no --max-model-len")
	}
	cfg := writeProfile(t, "MODEL="+r.Model+"\nMAX_MODEL_LEN=4096\nEXTRA_ARGS=\"--max-model-len "+recipeLen+"\"\nRECIPE="+r.ID+"\n")
	local, err := recipeOverrides(cfg, r)
	if err != nil {
		t.Fatalf("recipeOverrides: %v", err)
	}
	args, err := r.ServeArgsWith(local.args)
	if err != nil {
		t.Fatalf("ServeArgsWith: %v", err)
	}
	if v, _ := recipe.FlagValue(args, "--max-model-len"); v != recipeLen {
		t.Errorf("--max-model-len = %q, want %s: EXTRA_ARGS comes after the knob", v, recipeLen)
	}
	if v, _ := recipe.FlagValue(local.extra, "--max-model-len"); v != recipeLen {
		t.Errorf("EXTRA_ARGS written back as %v; dropping the token would let MAX_MODEL_LEN win next time", local.extra)
	}
}

// An edited knob has to reach argv. Coder reads MAX_MODEL_LEN from this same
// profile, so an ignored edit advertises one context and serves another.
func TestRecipeOverridesCarryEditsIntoArgv(t *testing.T) {
	r := testRecipe(t)
	cfg := writeProfile(t, "MODEL="+r.Model+"\nMAX_MODEL_LEN=131072\nEXTRA_ARGS=\"--max-num-seqs 16 --enforce-eager\"\nRECIPE="+r.ID+"\n")
	local, err := recipeOverrides(cfg, r)
	if err != nil {
		t.Fatalf("recipeOverrides: %v", err)
	}
	args, err := r.ServeArgsWith(local.args)
	if err != nil {
		t.Fatalf("ServeArgsWith: %v", err)
	}
	if v, _ := recipe.FlagValue(args, "--max-model-len"); v != "131072" {
		t.Errorf("--max-model-len = %q, want the profile value", v)
	}
	if v, _ := recipe.FlagValue(args, "--max-num-seqs"); v != "16" {
		t.Errorf("--max-num-seqs = %q, want the EXTRA_ARGS value", v)
	}
	if _, ok := recipe.FlagValue(args, "--enforce-eager"); !ok {
		t.Error("a flag only EXTRA_ARGS sets must be added")
	}
	if flags := recipe.OverrideFlags(local.args); len(flags) != 3 {
		t.Errorf("expected 3 overridden flags, got %v", flags)
	}
	// The EXTRA_ARGS written back keeps the real overrides and nothing else.
	if got := strings.Join(local.extra, " "); got != "--max-num-seqs 16 --enforce-eager" {
		t.Errorf("EXTRA_ARGS to write back = %q", got)
	}
}

// The first launcher wrote the recipe's whole flag list into EXTRA_ARGS. That
// profile is byte-for-byte the recipe, so it must not be reported as drift.
func TestRecipeOverridesIgnoreAVerbatimExtraArgsCopy(t *testing.T) {
	r := testRecipe(t)
	body := "MODEL=" + r.Model + "\nMAX_MODEL_LEN=262144\nGPU_MEMORY_UTILIZATION=0.85\nEXTRA_ARGS=" + strconv.Quote(strings.Join(r.Args, " ")) + "\nRECIPE=" + r.ID + "\n"
	local, err := recipeOverrides(writeProfile(t, body), r)
	if err != nil {
		t.Fatalf("recipeOverrides: %v", err)
	}
	if local.profile == nil {
		t.Fatal("profile not loaded")
	}
	if len(local.args) != 0 {
		t.Errorf("a verbatim copy of the recipe is not an override: %v", local.args)
	}
	// It also has to be pruned on write: left in place, those tokens would
	// override the recipe's own value the day the recipe changes.
	if len(local.extra) != 0 || !local.pruned {
		t.Errorf("expected the copy to be pruned, got extra=%v pruned=%v", local.extra, local.pruned)
	}
}

// No profile yet is the first launch, not an error.
func TestRecipeOverridesToleratesNoProfile(t *testing.T) {
	r := testRecipe(t)
	local, err := recipeOverrides(writeProfile(t, ""), r)
	if err != nil || local.profile != nil || local.args != nil {
		t.Fatalf("first launch should be clean: %+v err=%v", local, err)
	}
}

// Swapping MODEL means a different model, which the recipe's verified flags do
// not describe; that belongs in its own profile.
func TestRecipeOverridesRefuseADifferentModel(t *testing.T) {
	r := testRecipe(t)
	cfg := writeProfile(t, "MODEL=other/model\nRECIPE="+r.ID+"\n")
	_, err := recipeOverrides(cfg, r)
	if err == nil || !strings.Contains(err.Error(), "other/model") {
		t.Fatalf("expected a refusal naming the model, got %v", err)
	}
}

// A stop used to swallow every docker failure and print "stopped" regardless,
// so stopping twice, or stopping a node whose docker was unreachable, claimed
// to have freed GPUs it never touched.
func TestRecipeStopReportsWhatItFound(t *testing.T) {
	bin := t.TempDir()
	fake := "#!/bin/sh\ncase \"$FAKE_DOCKER\" in\n" +
		"present) echo vllm-node; exit 0 ;;\n" +
		"absent) echo 'Error response from daemon: No such container: vllm-node' >&2; exit 1 ;;\n" +
		"absent-ok) echo 'Error response from daemon: No such container: vllm-node' >&2; exit 0 ;;\n" +
		"*) echo 'Cannot connect to the Docker daemon' >&2; exit 1 ;;\nesac\n"
	if err := os.WriteFile(filepath.Join(bin, "docker"), []byte(fake), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))

	for _, tc := range []struct {
		docker, want, wantErr string
	}{
		{"present", "stopped vllm-node on spark2", ""},
		{"absent", "no vllm-node container on spark2; nothing to stop", ""},
		// Newer docker CLIs report the missing container but exit 0 under -f.
		{"absent-ok", "no vllm-node container on spark2; nothing to stop", ""},
		{"down", "", "Cannot connect to the Docker daemon"},
	} {
		t.Run(tc.docker, func(t *testing.T) {
			cmd := exec.Command("bash", "-c", stopScript("vllm-node"))
			cmd.Env = append(os.Environ(), "FAKE_DOCKER="+tc.docker)
			out, err := cmd.Output()
			if err != nil {
				t.Fatalf("script failed: %v", err)
			}
			got, err := stopReport("vllm-node", "spark2", string(out))
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want it to carry %q", err, tc.wantErr)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Errorf("report = %q, %v; want %q", got, err, tc.want)
			}
		})
	}
}

// launch takes the same wait flags as switch and restart; it used to return
// as soon as the containers started while restart on the same recipe waited.
func TestLaunchWaitsLikeRestart(t *testing.T) {
	c := newRecipesLaunchCmd()
	for _, name := range []string{"no-follow", "wait", "timeout"} {
		if c.Flags().Lookup(name) == nil {
			t.Errorf("recipes launch has no --%s", name)
		}
	}
	// status reads the active profile, which --no-activate leaves alone.
	if note := notWaitingNote(recipeLaunchOptions{skipCoder: true, container: "vllm-node"}); !strings.Contains(note, "docker logs -f vllm-node") {
		t.Errorf("--no-activate --no-follow points at %q", note)
	}
}

// A recipe's profile names its engine unless that is vLLM, including an
// engine that takes vLLM's args and so keeps the sizing keys. Without the
// name, switch and the watchdog would treat it as plain vLLM.
//
// Not parallel: UseLocal swaps the engines every Lookup reads.
func TestNewRecipeProfileNamesTheEngine(t *testing.T) {
	dir := t.TempDir()
	toy := `{"schema_version":1,"id":"toyv","display_name":"Toy vLLM","why":"a test engine",` +
		`"runtime":["container"],"health":{"path":"/ready","mode":"ready"},` +
		`"container":{"api_key":"none","launch_order":"head_first",` +
		`"roles":[{"name":"all","ranks":"all","serves_http":true,"args":["toyv","{model}","{args}"]}]},` +
		`"capabilities":{"vllm_args":true}}`
	if err := os.WriteFile(filepath.Join(dir, "toyv.json"), []byte(toy), 0o644); err != nil {
		t.Fatal(err)
	}
	if errs := engine.UseLocal(dir); len(errs) > 0 {
		t.Fatal(errs)
	}
	t.Cleanup(func() { engine.UseLocal(filepath.Join(dir, "gone")) })

	vllmLike := testRecipe(t)
	vllmLike.ID = "toyv-recipe"
	vllmLike.Provider = "toyv"
	for _, tc := range []struct {
		r            recipe.TestedRecipe
		provider     string
		sizingKeyset bool
	}{
		{testRecipe(t), "", true},
		{vllmLike, "toyv", true},
		{glmRecipe(t), "tensorfold-python", false},
	} {
		cfg := writeProfile(t, "")
		p, err := newRecipeProfile(cfg.Dirs, tc.r, "65536")
		if err != nil {
			t.Fatalf("%s: %v", tc.r.ID, err)
		}
		if got := p.Provider(); got != tc.provider {
			t.Errorf("%s: PROVIDER = %q, want %q", tc.r.ID, got, tc.provider)
		}
		if got := p.File.Get("MAX_MODEL_LEN"); got != "65536" {
			t.Errorf("%s: MAX_MODEL_LEN = %q, want the recipe's limit", tc.r.ID, got)
		}
		if has := p.File.Has("GPU_MEMORY_UTILIZATION"); has != tc.sizingKeyset {
			t.Errorf("%s: has GPU_MEMORY_UTILIZATION = %v, want %v", tc.r.ID, has, tc.sizingKeyset)
		}
	}
}
