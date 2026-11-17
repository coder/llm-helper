package cli_test

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/recipe"
)

const testRev = "0123456789abcdef0123456789abcdef01234567"

// seedHub writes a repo into a hub cache the way huggingface_hub does: blobs
// named by hash and snapshot files linked to them.
func seedHub(t *testing.T, hub, repo, rev string, files map[string]string) {
	t.Helper()
	root := filepath.Join(hub, "models--"+strings.ReplaceAll(repo, "/", "--"))
	mustWrite := func(p, body string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite(filepath.Join(root, "refs", "main"), rev)
	for name, body := range files {
		sum := sha256.Sum256([]byte(body))
		blob := filepath.Join(root, "blobs", hex.EncodeToString(sum[:]))
		mustWrite(blob, body)
		link := filepath.Join(root, "snapshots", rev, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			t.Fatal(err)
		}
		rel, err := filepath.Rel(filepath.Dir(link), blob)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(rel, link); err != nil {
			t.Fatal(err)
		}
	}
}

var qwenFiles = map[string]string{
	"config.json":       `{"model_type":"qwen3"}`,
	"model.safetensors": strings.Repeat("weights", 100),
}

func snapshotFile(hub, repo, rev, name string) string {
	return filepath.Join(hub, "models--"+strings.ReplaceAll(repo, "/", "--"), "snapshots", rev, name)
}

func TestImportCopiesANamedModelIntoTheCache(t *testing.T) {
	newTree(t)
	drive, hub := t.TempDir(), filepath.Join(t.TempDir(), "hub")
	seedHub(t, filepath.Join(drive, "hub"), "Qwen/Qwen3-8B", testRev, qwenFiles)

	code, out, errOut := runCLI(t, "import", drive, "Qwen/Qwen3-8B", "--cache", hub)
	if code != 0 {
		t.Fatalf("exit %d\n%s\n%s", code, out, errOut)
	}
	got, err := os.ReadFile(snapshotFile(hub, "Qwen/Qwen3-8B", testRev, "model.safetensors"))
	if err != nil || string(got) != qwenFiles["model.safetensors"] {
		t.Fatalf("imported weights: %v", err)
	}
	if ref, _ := os.ReadFile(filepath.Join(hub, "models--Qwen--Qwen3-8B", "refs", "main")); string(ref) != testRev {
		t.Errorf("refs/main = %q", ref)
	}
	if !strings.Contains(out, "llm-helper new Qwen/Qwen3-8B") {
		t.Errorf("output does not say how to serve it:\n%s", out)
	}
}

func TestImportNeedsNamesOrAllWithoutATerminal(t *testing.T) {
	newTree(t)
	drive, hub := t.TempDir(), filepath.Join(t.TempDir(), "hub")
	seedHub(t, drive, "Qwen/Qwen3-8B", testRev, qwenFiles)
	seedHub(t, drive, "org/other", testRev, qwenFiles)

	code, out, errOut := runCLI(t, "import", drive, "--cache", hub)
	if code == 0 || !strings.Contains(errOut, "--all") {
		t.Fatalf("exit %d, stderr %q", code, errOut)
	}
	if !strings.Contains(out, "Qwen/Qwen3-8B") || !strings.Contains(out, "org/other") {
		t.Errorf("the models on the drive were not listed:\n%s", out)
	}
	if _, err := os.Stat(hub); err == nil {
		t.Error("the cache was written without a choice")
	}

	if code, out, errOut := runCLI(t, "import", drive, "--all", "--cache", hub); code != 0 {
		t.Fatalf("--all: exit %d\n%s\n%s", code, out, errOut)
	}
	for _, repo := range []string{"Qwen/Qwen3-8B", "org/other"} {
		if _, err := os.Stat(snapshotFile(hub, repo, testRev, "config.json")); err != nil {
			t.Errorf("%s was not imported: %v", repo, err)
		}
	}
}

func TestImportDryRunCopiesNothing(t *testing.T) {
	newTree(t)
	drive, hub := t.TempDir(), filepath.Join(t.TempDir(), "hub")
	seedHub(t, drive, "Qwen/Qwen3-8B", testRev, qwenFiles)

	code, out, _ := runCLI(t, "import", drive, "Qwen/Qwen3-8B", "--cache", hub, "--dry-run")
	if code != 0 || !strings.Contains(out, "would import Qwen/Qwen3-8B") {
		t.Fatalf("exit %d\n%s", code, out)
	}
	if _, err := os.Stat(hub); err == nil {
		t.Error("a dry run wrote the cache")
	}
}

func TestImportRefusesAnIncompleteModel(t *testing.T) {
	newTree(t)
	drive, hub := t.TempDir(), filepath.Join(t.TempDir(), "hub")
	seedHub(t, drive, "Qwen/Qwen3-8B", testRev, qwenFiles)
	repoDir := filepath.Join(drive, "models--Qwen--Qwen3-8B")
	entries, _ := os.ReadDir(filepath.Join(repoDir, "blobs"))
	for _, e := range entries {
		if err := os.Remove(filepath.Join(repoDir, "blobs", e.Name())); err != nil {
			t.Fatal(err)
		}
	}

	code, _, errOut := runCLI(t, "import", drive, "Qwen/Qwen3-8B", "--cache", hub)
	if code == 0 || !strings.Contains(errOut, "incomplete") {
		t.Fatalf("exit %d, stderr %q", code, errOut)
	}
}

func TestImportFailsOnADamagedFile(t *testing.T) {
	newTree(t)
	drive, hub := t.TempDir(), filepath.Join(t.TempDir(), "hub")
	seedHub(t, drive, "Qwen/Qwen3-8B", testRev, qwenFiles)
	snap := snapshotFile(drive, "Qwen/Qwen3-8B", testRev, "model.safetensors")
	if err := os.WriteFile(snap, []byte(strings.Repeat("WEIGHTS", 100)), 0o644); err != nil {
		t.Fatal(err)
	}

	code, _, errOut := runCLI(t, "import", drive, "Qwen/Qwen3-8B", "--cache", hub)
	if code == 0 {
		t.Fatal("a damaged file was imported")
	}
	if _, err := os.Stat(snapshotFile(hub, "Qwen/Qwen3-8B", testRev, "model.safetensors")); err == nil {
		t.Errorf("the damaged file reached the snapshot (stderr %q)", errOut)
	}
}

func TestExportThenImportRoundTrips(t *testing.T) {
	newTree(t)
	src, drive, dst := t.TempDir(), t.TempDir(), filepath.Join(t.TempDir(), "hub")
	seedHub(t, src, "Qwen/Qwen3-8B", testRev, qwenFiles)

	code, out, errOut := runCLI(t, "export", "Qwen/Qwen3-8B", drive, "--cache", src)
	if code != 0 {
		t.Fatalf("export: exit %d\n%s\n%s", code, out, errOut)
	}
	_ = filepath.WalkDir(drive, func(p string, d os.DirEntry, _ error) error {
		if d != nil && d.Type()&os.ModeSymlink != 0 {
			t.Errorf("the drive holds a link: %s", p)
		}
		return nil
	})

	code, out, errOut = runCLI(t, "import", drive, "--all", "--cache", dst)
	if code != 0 {
		t.Fatalf("import: exit %d\n%s\n%s", code, out, errOut)
	}
	got, err := os.ReadFile(snapshotFile(dst, "Qwen/Qwen3-8B", testRev, "config.json"))
	if err != nil || string(got) != qwenFiles["config.json"] {
		t.Fatalf("round trip lost config.json: %v", err)
	}
}

func TestExportRefusesWhatIsNotComplete(t *testing.T) {
	newTree(t)
	src, drive := t.TempDir(), t.TempDir()
	seedHub(t, src, "Qwen/Qwen3-8B", testRev, map[string]string{"config.json": "{}"})

	code, _, errOut := runCLI(t, "export", "Qwen/Qwen3-8B", drive, "--cache", src)
	if code == 0 || !strings.Contains(errOut, "incomplete") {
		t.Fatalf("exit %d, stderr %q", code, errOut)
	}
	code, _, errOut = runCLI(t, "export", "Qwen/Missing", drive, "--cache", src)
	if code == 0 || !strings.Contains(errOut, "not in the cache") {
		t.Fatalf("exit %d, stderr %q", code, errOut)
	}
	code, _, errOut = runCLI(t, "export", "Qwen/Qwen3-8B", filepath.Join(drive, "nope"), "--cache", src)
	if code == 0 || !strings.Contains(errOut, "mount the drive") {
		t.Fatalf("exit %d, stderr %q", code, errOut)
	}
}

func TestImportTakesAFolderWithARepoName(t *testing.T) {
	newTree(t)
	folder, hub := t.TempDir(), filepath.Join(t.TempDir(), "hub")
	for name, body := range qwenFiles {
		if err := os.WriteFile(filepath.Join(folder, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	code, _, errOut := runCLI(t, "import", folder, "--cache", hub, "--all")
	if code == 0 || !strings.Contains(errOut, "--repo") {
		t.Fatalf("exit %d, stderr %q", code, errOut)
	}
	code, _, errOut = runCLI(t, "import", folder, "--cache", hub, "--all", "--repo", "Qwen/Qwen3-8B")
	if code == 0 || !strings.Contains(errOut, "revision") {
		t.Fatalf("a folder with no commit was imported: exit %d, stderr %q", code, errOut)
	}
	code, out, errOut := runCLI(t, "import", folder, "--cache", hub, "--all", "--repo", "Qwen/Qwen3-8B", "--revision", testRev)
	if code != 0 {
		t.Fatalf("exit %d\n%s\n%s", code, out, errOut)
	}
	if _, err := os.Stat(snapshotFile(hub, "Qwen/Qwen3-8B", testRev, "model.safetensors")); err != nil {
		t.Error(err)
	}
}

func TestImportForARecipeStagesWhereItsEngineReads(t *testing.T) {
	newTree(t)
	tested, err := recipe.LoadTested()
	if err != nil {
		t.Fatal(err)
	}
	var r recipe.TestedRecipe
	for _, c := range tested.Recipes {
		if c.Engine().HFCacheHostPath(c.Params) != "" && len(c.Models) > 0 {
			r = c
			break
		}
	}
	if r.ID == "" {
		t.Skip("no recipe pins models in a host cache")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	drive := t.TempDir()
	for _, m := range r.Models {
		seedHub(t, filepath.Join(drive, "hub"), m.ID, m.Revision, qwenFiles)
	}

	code, out, errOut := runCLI(t, "import", drive, "--recipe", r.ID)
	if code != 0 {
		t.Fatalf("exit %d\n%s\n%s", code, out, errOut)
	}
	for _, m := range r.Models {
		dir := filepath.Join(home, engine.SnapshotPath(r.Engine().HFCacheHostPath(r.Params), engine.ModelRef{ID: m.ID, Revision: m.Revision}))
		if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
			t.Errorf("%s is not where the recipe reads it: %v", m.ID, err)
		}
	}
	if !strings.Contains(out, "staged for recipe "+r.ID) {
		t.Errorf("output:\n%s", out)
	}

	other := t.TempDir()
	seedHub(t, filepath.Join(other, "hub"), r.Models[0].ID, testRev, qwenFiles)
	if code, _, errOut := runCLI(t, "import", other, "--recipe", r.ID); code == 0 || !strings.Contains(errOut, "pinned snapshot") {
		t.Errorf("a revision the recipe does not pin was imported: exit %d, stderr %q", code, errOut)
	}
}

const newerRev = "fedcba9876543210fedcba9876543210fedcba98"

func recipeWithModels(t *testing.T) recipe.TestedRecipe {
	t.Helper()
	tested, err := recipe.LoadTested()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range tested.Recipes {
		if c.Engine().HFCacheHostPath(c.Params) != "" && len(c.Models) > 0 {
			return c
		}
	}
	t.Skip("no recipe pins models in a host cache")
	return recipe.TestedRecipe{}
}

func TestImportForARecipeReadsThePinnedSnapshotNotRefsMain(t *testing.T) {
	newTree(t)
	r := recipeWithModels(t)
	home := t.TempDir()
	t.Setenv("HOME", home)
	drive := t.TempDir()
	for _, m := range r.Models {
		seedHub(t, filepath.Join(drive, "hub"), m.ID, m.Revision, qwenFiles)
		// A later seed moves refs/main off the pinned revision.
		seedHub(t, filepath.Join(drive, "hub"), m.ID, newerRev, qwenFiles)
	}
	code, out, errOut := runCLI(t, "import", drive, "--recipe", r.ID)
	if code != 0 {
		t.Fatalf("exit %d\n%s\n%s", code, out, errOut)
	}
	for _, m := range r.Models {
		dir := filepath.Join(home, engine.SnapshotPath(r.Engine().HFCacheHostPath(r.Params), engine.ModelRef{ID: m.ID, Revision: m.Revision}))
		if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
			t.Errorf("%s: %v", m.ID, err)
		}
	}
}

func TestImportForARecipeNeedsNoRefsMain(t *testing.T) {
	newTree(t)
	r := recipeWithModels(t)
	t.Setenv("HOME", t.TempDir())
	drive := t.TempDir()
	for _, m := range r.Models {
		seedHub(t, filepath.Join(drive, "hub"), m.ID, m.Revision, qwenFiles)
		root := filepath.Join(drive, "hub", "models--"+strings.ReplaceAll(m.ID, "/", "--"))
		if err := os.Remove(filepath.Join(root, "refs", "main")); err != nil {
			t.Fatal(err)
		}
		// A second snapshot, so there is no single one to pick by default.
		seedHub(t, filepath.Join(drive, "hub"), m.ID, newerRev, qwenFiles)
		if err := os.Remove(filepath.Join(root, "refs", "main")); err != nil {
			t.Fatal(err)
		}
	}
	code, out, errOut := runCLI(t, "import", drive, "--recipe", r.ID)
	if code != 0 {
		t.Fatalf("exit %d\n%s\n%s", code, out, errOut)
	}
}

func TestImportRefusesACacheTheServerReadsLiterally(t *testing.T) {
	configDir, _ := newTree(t)
	cfg, err := os.ReadFile(filepath.Join(configDir, "config.env"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(cfg) + "PROVIDER=llamacpp\nLLAMA_CACHE=~/models\n"
	if err := os.WriteFile(filepath.Join(configDir, "config.env"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_HUB_CACHE", "")
	t.Setenv("HF_HOME", "")
	drive := t.TempDir()
	seedHub(t, filepath.Join(drive, "hub"), "Qwen/Qwen3-8B", testRev, qwenFiles)
	code, _, errOut := runCLI(t, "import", drive, "Qwen/Qwen3-8B")
	if code == 0 || !strings.Contains(errOut, "absolute") {
		t.Fatalf("exit %d, stderr %q", code, errOut)
	}
}

func TestSearchOfflineListsTheCachedModels(t *testing.T) {
	configDir, _ := newTree(t)
	hub := filepath.Join(t.TempDir(), "hub")
	seedHub(t, hub, "Qwen/Qwen3-8B", testRev, qwenFiles)
	cfg, err := os.ReadFile(filepath.Join(configDir, "config.env"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(cfg) + "HF_HUB_CACHE=" + hub + "\nHF_HUB_OFFLINE=1\n"
	if err := os.WriteFile(filepath.Join(configDir, "config.env"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}

	code, out, errOut := runCLI(t, "search", "qwen3", "--json")
	if code != 0 {
		t.Fatalf("exit %d\n%s\n%s", code, out, errOut)
	}
	if !strings.Contains(out, "Qwen/Qwen3-8B") || !strings.Contains(errOut, "local cache") {
		t.Errorf("stdout:\n%s\nstderr:\n%s", out, errOut)
	}
}
