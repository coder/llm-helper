package hfcache

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const rev = "0123456789abcdef0123456789abcdef01234567"

func gitID(content string) string {
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00%s", len(content), content)
	return hex.EncodeToString(h.Sum(nil))
}

func sha(content string) string {
	s := sha256.Sum256([]byte(content))
	return hex.EncodeToString(s[:])
}

func write(t *testing.T, p, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func read(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// cacheOpts shape a cache fixture.
type cacheOpts struct {
	// noLinks writes real files in the snapshot and no blobs, as a cache on
	// exFAT does.
	noLinks bool
	// tree writes trees/<commit>.json.
	tree bool
	// lfs names the files stored by sha256 rather than git blob id.
	lfs []string
	// shared puts blob content in the cache-wide store, as newer caches do.
	shared bool
}

var modelFiles = map[string]string{
	"config.json":                  `{"model_type":"toy"}`,
	"model.safetensors.index.json": `{"weight_map":{"a":"model-1.safetensors","b":"sub/model-2.safetensors"}}`,
	"model-1.safetensors":          "weights one",
	"sub/model-2.safetensors":      "weights two",
}

// makeCache writes a hub cache holding repo, and returns the hub directory.
func makeCache(t *testing.T, repo string, files map[string]string, o cacheOpts) string {
	t.Helper()
	hub := filepath.Join(t.TempDir(), "hub")
	repoDir := RepoDir(hub, repo)
	tree := map[string]treeFile{}
	for _, name := range sortedKeys(files) {
		content := files[name]
		etag, tf := gitID(content), treeFile{Size: int64(len(content)), BlobID: gitID(content)}
		if slices.Contains(o.lfs, name) {
			etag = sha(content)
			tf.LFSSha256 = etag
			tf.BlobID = gitID("pointer " + name)
		}
		tree[name] = tf
		dst := filepath.Join(repoDir, "snapshots", rev, filepath.FromSlash(name))
		if o.noLinks {
			write(t, dst, content)
			continue
		}
		blob := filepath.Join(repoDir, "blobs", etag)
		if o.shared {
			store := filepath.Join(hub, "blobs", etag[:2], etag)
			write(t, store, content)
			if err := os.MkdirAll(filepath.Dir(blob), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(filepath.Join("..", "..", "blobs", etag[:2], etag), blob); err != nil {
				t.Fatal(err)
			}
		} else {
			write(t, blob, content)
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			t.Fatal(err)
		}
		up := strings.Repeat("../", 2+strings.Count(name, "/"))
		if err := os.Symlink(up+"blobs/"+etag, dst); err != nil {
			t.Fatal(err)
		}
	}
	write(t, filepath.Join(repoDir, "refs", "main"), rev)
	if o.tree {
		raw, _ := json.Marshal(map[string]any{"format_version": 1, "files": tree})
		write(t, filepath.Join(repoDir, "trees", rev+".json"), string(raw))
	}
	return hub
}

func scanOne(t *testing.T, hub, repo string) Model {
	t.Helper()
	m, err := ScanRepo(hub, repo)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestDirPrecedence(t *testing.T) {
	t.Parallel()
	home := "/home/u"
	for _, tc := range []struct {
		name  string
		env   map[string]string
		extra []string
		want  string
	}{
		{"default", nil, nil, "/home/u/.cache/huggingface/hub"},
		{"xdg", map[string]string{"XDG_CACHE_HOME": "/xdg"}, nil, "/xdg/huggingface/hub"},
		{"hf home beats xdg", map[string]string{"XDG_CACHE_HOME": "/xdg", "HF_HOME": "/hf"}, nil, "/hf/hub"},
		{"old hub cache beats hf home", map[string]string{"HF_HOME": "/hf", "HUGGINGFACE_HUB_CACHE": "/old"}, nil, "/old"},
		{"hub cache beats old", map[string]string{"HUGGINGFACE_HUB_CACHE": "/old", "HF_HUB_CACHE": "/new"}, nil, "/new"},
		{"server variable beats all", map[string]string{"HF_HUB_CACHE": "/new", "LLAMA_CACHE": "/llama"}, []string{"LLAMA_CACHE"}, "/llama"},
		{"server variable unset", map[string]string{"HF_HUB_CACHE": "/new"}, []string{"LLAMA_CACHE"}, "/new"},
		{"tilde", map[string]string{"HF_HOME": "~/models"}, nil, "/home/u/models/hub"},
		{"$HOME", map[string]string{"HF_HUB_CACHE": "$HOME/hub"}, nil, "/home/u/hub"},
		{"blank is unset", map[string]string{"HF_HUB_CACHE": "  ", "HF_HOME": "/hf"}, nil, "/hf/hub"},
	} {
		got := Dir(func(k string) string { return tc.env[k] }, home, tc.extra...)
		if got != tc.want {
			t.Errorf("%s: Dir = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestOffline(t *testing.T) {
	t.Parallel()
	for v, want := range map[string]bool{"1": true, "true": true, "TRUE": true, "on": true, "yes": true, "0": false, "": false, "false": false, "no": false} {
		for _, key := range []string{"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"} {
			if got := Offline(func(k string) string {
				if k == key {
					return v
				}
				return ""
			}); got != want {
				t.Errorf("%s=%q: Offline = %v, want %v", key, v, got, want)
			}
		}
	}
}

func TestScanReadsALinkedCache(t *testing.T) {
	t.Parallel()
	hub := makeCache(t, "org/toy", modelFiles, cacheOpts{tree: true, lfs: []string{"model-1.safetensors"}})
	m := scanOne(t, hub, "org/toy")
	if !m.Complete() || m.Revision != rev || len(m.Files) != 4 {
		t.Fatalf("model = %+v", m)
	}
	for _, f := range m.Files {
		want := gitID(modelFiles[f.Path])
		if f.Path == "model-1.safetensors" {
			want = sha(modelFiles[f.Path])
		}
		if f.Etag != want {
			t.Errorf("%s etag = %s, want %s", f.Path, f.Etag, want)
		}
	}
	if m.Size() != int64(len(modelFiles["config.json"])+len(modelFiles["model.safetensors.index.json"])+len("weights one")+len("weights two")) {
		t.Errorf("Size = %d", m.Size())
	}
}

func TestScanReadsACacheWithoutLinks(t *testing.T) {
	t.Parallel()
	hub := makeCache(t, "org/toy", modelFiles, cacheOpts{noLinks: true, tree: true, lfs: []string{"model-1.safetensors"}})
	m := scanOne(t, hub, "org/toy")
	if !m.Complete() {
		t.Fatalf("problems: %v", m.Problems)
	}
	// With no blob names to read, the listing says what each file is called.
	for _, f := range m.Files {
		if f.Path == "model-1.safetensors" && f.Etag != sha("weights one") {
			t.Errorf("etag = %q, want the listing's lfs_sha256", f.Etag)
		}
		if f.Etag == "" {
			t.Errorf("%s has no etag", f.Path)
		}
	}
	hub = makeCache(t, "org/toy", modelFiles, cacheOpts{noLinks: true})
	for _, f := range scanOne(t, hub, "org/toy").Files {
		if f.Etag != "" {
			t.Errorf("%s: etag %q with nothing to say it", f.Path, f.Etag)
		}
	}
}

func TestScanFindsTheCacheWhereverItSits(t *testing.T) {
	t.Parallel()
	for _, sub := range []string{"", "hub", "huggingface/hub", ".cache/huggingface/hub"} {
		root := makeCache(t, "org/toy", modelFiles, cacheOpts{})
		if sub != "" {
			root = t.TempDir()
			dst := filepath.Join(root, filepath.FromSlash(sub))
			if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.Rename(makeCache(t, "org/toy", modelFiles, cacheOpts{}), dst); err != nil {
				t.Fatal(err)
			}
		}
		found, err := Scan(root, ScanOptions{})
		if err != nil || found.Kind != KindCache || len(found.Models) != 1 || found.Models[0].Repo != "org/toy" {
			t.Errorf("%q: Scan = %+v, %v", sub, found, err)
		}
	}
	// One copied models--org--name folder is enough.
	hub := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	found, err := Scan(RepoDir(hub, "org/toy"), ScanOptions{})
	if err != nil || len(found.Models) != 1 {
		t.Errorf("single repo folder: %+v, %v", found, err)
	}
}

func TestScanSkipsWhatIsNotAModel(t *testing.T) {
	t.Parallel()
	hub := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	for _, junk := range []string{"datasets--org--data", ".locks", "models--bad", "CACHEDIR.TAG"} {
		write(t, filepath.Join(hub, junk, "x"), "x")
	}
	write(t, filepath.Join(RepoDir(hub, "org/toy"), "blobs", "abc.incomplete"), "partial")
	models, err := ScanCache(hub)
	if err != nil || len(models) != 1 || !models[0].Complete() {
		t.Fatalf("ScanCache = %+v, %v", models, err)
	}
}

func TestScanFollowsTheSharedBlobStore(t *testing.T) {
	t.Parallel()
	hub := makeCache(t, "org/toy", modelFiles, cacheOpts{shared: true})
	m := scanOne(t, hub, "org/toy")
	if !m.Complete() {
		t.Fatalf("problems: %v", m.Problems)
	}
	// The name is the repo's blob link, not the store's path.
	for _, f := range m.Files {
		if f.Etag != gitID(modelFiles[f.Path]) {
			t.Errorf("%s etag = %s", f.Path, f.Etag)
		}
	}
	drive := t.TempDir()
	if _, err := Export(m, filepath.Join(drive, "hub"), nil); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(RepoDir(filepath.Join(drive, "hub"), "org/toy"), "snapshots", rev, "model-1.safetensors")
	if st, err := os.Lstat(p); err != nil || !st.Mode().IsRegular() || read(t, p) != "weights one" {
		t.Errorf("exported file: %v, %v", st, err)
	}
}

func TestScanFlagsIncompleteModels(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		mutate func(t *testing.T, hub string)
		want   string
	}{
		{"dangling link", func(t *testing.T, hub string) {
			os.Remove(filepath.Join(RepoDir(hub, "org/toy"), "blobs", gitID("weights two")))
		}, "dangling link: sub/model-2.safetensors"},
		{"a file the listing names is missing", func(t *testing.T, hub string) {
			os.Remove(filepath.Join(RepoDir(hub, "org/toy"), "snapshots", rev, "config.json"))
		}, "missing config.json"},
		{"size differs from the listing", func(t *testing.T, hub string) {
			write(t, filepath.Join(RepoDir(hub, "org/toy"), "blobs", gitID("weights one")), "short")
		}, "the listing says"},
		{"a shard the index names is missing", func(t *testing.T, hub string) {
			os.Remove(filepath.Join(RepoDir(hub, "org/toy"), "snapshots", rev, "model-1.safetensors"))
		}, "missing model-1.safetensors"},
	} {
		hub := makeCache(t, "org/toy", modelFiles, cacheOpts{tree: tc.name != "a shard the index names is missing"})
		tc.mutate(t, hub)
		m := scanOne(t, hub, "org/toy")
		if m.Complete() || !strings.Contains(strings.Join(m.Problems, "; "), tc.want) {
			t.Errorf("%s: problems = %v, want %q", tc.name, m.Problems, tc.want)
		}
	}
	// Config only is what an interrupted first download leaves.
	hub := makeCache(t, "org/toy", map[string]string{"config.json": "{}"}, cacheOpts{})
	if m := scanOne(t, hub, "org/toy"); m.Complete() || m.Problems[0] != "no weight files" {
		t.Errorf("config only: %v", m.Problems)
	}
	// Two snapshots and no ref leave nothing to choose.
	hub = makeCache(t, "org/toy", modelFiles, cacheOpts{})
	os.Remove(filepath.Join(RepoDir(hub, "org/toy"), "refs", "main"))
	if err := os.MkdirAll(filepath.Join(RepoDir(hub, "org/toy"), "snapshots", strings.Repeat("b", 40)), 0o755); err != nil {
		t.Fatal(err)
	}
	if m := scanOne(t, hub, "org/toy"); m.Complete() {
		t.Error("a repo with two snapshots and no refs/main was complete")
	}
}

func TestImportWritesTheCacheHuggingFaceReads(t *testing.T) {
	t.Parallel()
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{noLinks: true, tree: true, lfs: []string{"model-1.safetensors"}})
	dst := filepath.Join(t.TempDir(), "hub")
	var seen []string
	res, err := Import(scanOne(t, src, "org/toy"), dst, func(p string, _ int64, skipped bool) {
		seen = append(seen, fmt.Sprint(p, skipped))
	})
	if err != nil || !res.Links || res.Copied != 4 || res.Skipped != 0 || len(seen) != 4 {
		t.Fatalf("Import = %+v, %v, %v", res, err, seen)
	}
	repoDir := RepoDir(dst, "org/toy")
	for name, content := range modelFiles {
		p := filepath.Join(repoDir, "snapshots", rev, filepath.FromSlash(name))
		target, err := os.Readlink(p)
		if err != nil {
			t.Fatalf("%s is not a link: %v", name, err)
		}
		want := gitID(content)
		if name == "model-1.safetensors" {
			want = sha(content)
		}
		wantTarget := strings.Repeat("../", 2+strings.Count(name, "/")) + "blobs/" + want
		if target != wantTarget || filepath.IsAbs(target) {
			t.Errorf("%s -> %s, want %s", name, target, wantTarget)
		}
		if read(t, p) != content {
			t.Errorf("%s content differs", name)
		}
	}
	if got := read(t, filepath.Join(repoDir, "refs", "main")); got != rev {
		t.Errorf("refs/main = %q, want the bare commit", got)
	}
	if read(t, filepath.Join(repoDir, "trees", rev+".json")) != read(t, filepath.Join(RepoDir(src, "org/toy"), "trees", rev+".json")) {
		t.Error("the tree listing was not carried over")
	}
	if m := scanOne(t, dst, "org/toy"); !m.Complete() {
		t.Errorf("what was imported does not scan complete: %v", m.Problems)
	}
	entries, _ := os.ReadDir(filepath.Join(repoDir, "blobs"))
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".incomplete") {
			t.Errorf("left %s behind", e.Name())
		}
	}
}

func TestImportResumes(t *testing.T) {
	t.Parallel()
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	m := scanOne(t, src, "org/toy")
	dst := filepath.Join(t.TempDir(), "hub")
	if _, err := Import(m, dst, nil); err != nil {
		t.Fatal(err)
	}
	res, err := Import(m, dst, nil)
	if err != nil || res.Copied != 0 || res.Skipped != 4 {
		t.Errorf("second import = %+v, %v; want everything skipped", res, err)
	}

	// An interrupted import: one blob and its link gone, a partial left.
	repoDir := RepoDir(dst, "org/toy")
	os.Remove(filepath.Join(repoDir, "snapshots", rev, "model-1.safetensors"))
	os.Remove(filepath.Join(repoDir, "blobs", gitID("weights one")))
	write(t, partialPath(repoDir, "model-1.safetensors"), "half")
	res, err = Import(m, dst, nil)
	if err != nil || res.Copied != 1 || res.Skipped != 3 {
		t.Errorf("resumed import = %+v, %v; want one file copied", res, err)
	}
	if read(t, filepath.Join(repoDir, "snapshots", rev, "model-1.safetensors")) != "weights one" {
		t.Error("the resumed file is wrong")
	}
}

func TestImportCatchesADamagedSource(t *testing.T) {
	t.Parallel()
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	// Same length, same name, different bytes.
	write(t, filepath.Join(RepoDir(src, "org/toy"), "blobs", gitID("weights two")), "weights twX")
	dst := filepath.Join(t.TempDir(), "hub")
	_, err := Import(scanOne(t, src, "org/toy"), dst, nil)
	var corrupt *CorruptError
	if !errors.As(err, &corrupt) {
		t.Fatalf("Import = %v, want a CorruptError", err)
	}
	repoDir := RepoDir(dst, "org/toy")
	if _, err := os.Stat(filepath.Join(repoDir, "blobs", gitID("weights twX"))); err == nil {
		t.Error("the damaged file was kept")
	}
	if _, err := os.Lstat(filepath.Join(repoDir, "snapshots", rev, "sub", "model-2.safetensors")); err == nil {
		t.Error("a link to the damaged file was made")
	}
	if _, err := os.Stat(filepath.Join(repoDir, "refs", "main")); err == nil {
		t.Error("refs/main was written for a model that did not finish")
	}
}

func TestImportFallsBackToFilesWhenLinksFail(t *testing.T) {
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{tree: true})
	symlink = func(string, string) error { return errors.New("operation not permitted") }
	t.Cleanup(func() { symlink = os.Symlink })
	dst := filepath.Join(t.TempDir(), "hub")
	res, err := Import(scanOne(t, src, "org/toy"), dst, nil)
	if err != nil || res.Links {
		t.Fatalf("Import = %+v, %v", res, err)
	}
	p := filepath.Join(RepoDir(dst, "org/toy"), "snapshots", rev, "sub", "model-2.safetensors")
	if st, err := os.Lstat(p); err != nil || !st.Mode().IsRegular() || read(t, p) != "weights two" {
		t.Errorf("snapshot file: %v, %v", st, err)
	}
	if m := scanOne(t, dst, "org/toy"); !m.Complete() {
		t.Errorf("problems: %v", m.Problems)
	}
}

func TestImportRefusesIncompleteModels(t *testing.T) {
	t.Parallel()
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	os.Remove(filepath.Join(RepoDir(src, "org/toy"), "snapshots", rev, "model-1.safetensors"))
	dst := filepath.Join(t.TempDir(), "hub")
	if _, err := Import(scanOne(t, src, "org/toy"), dst, nil); err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Errorf("Import = %v", err)
	}
	if _, err := os.Stat(RepoDir(dst, "org/toy")); err == nil {
		t.Error("an incomplete model left files behind")
	}
}

func TestImportKeepsWhatIsThere(t *testing.T) {
	t.Parallel()
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{tree: true})
	dst := filepath.Join(t.TempDir(), "hub")
	repoDir := RepoDir(dst, "org/toy")
	other := strings.Repeat("c", 40)
	write(t, filepath.Join(repoDir, "refs", "main"), other)
	write(t, filepath.Join(repoDir, "trees", rev+".json"), `{"format_version":1,"files":{}}`)
	res, err := Import(scanOne(t, src, "org/toy"), dst, nil)
	if err != nil || res.RefKept != other {
		t.Fatalf("Import = %+v, %v; want refs/main kept at %s", res, err, other)
	}
	if read(t, filepath.Join(repoDir, "refs", "main")) != other {
		t.Error("refs/main was overwritten")
	}
	if !strings.Contains(read(t, filepath.Join(repoDir, "trees", rev+".json")), `"files":{}`) {
		t.Error("an existing tree listing was overwritten")
	}

	// A blob of the right name and the wrong size is not ours to replace.
	dst2 := filepath.Join(t.TempDir(), "hub")
	write(t, filepath.Join(RepoDir(dst2, "org/toy"), "blobs", gitID("weights two")), "not what its name says")
	if _, err := Import(scanOne(t, src, "org/toy"), dst2, nil); err == nil || !strings.Contains(err.Error(), "different size") {
		t.Errorf("Import over a bad blob = %v", err)
	}
}

func TestExportWritesAReadableCacheWithoutLinks(t *testing.T) {
	t.Parallel()
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{tree: true})
	drive := filepath.Join(t.TempDir(), "hub")
	if _, err := Export(scanOne(t, src, "org/toy"), drive, nil); err != nil {
		t.Fatal(err)
	}
	filepath.WalkDir(drive, func(p string, d os.DirEntry, err error) error {
		if err == nil && d.Type()&os.ModeSymlink != 0 {
			t.Errorf("%s is a link", p)
		}
		return nil
	})
	m := scanOne(t, drive, "org/toy")
	if !m.Complete() || len(m.Files) != 4 || m.Revision != rev {
		t.Fatalf("exported model = %+v", m)
	}
	if read(t, filepath.Join(RepoDir(drive, "org/toy"), "refs", "main")) != rev {
		t.Error("refs/main")
	}

	// And it imports into a linked cache, files verified against the tree.
	dst := filepath.Join(t.TempDir(), "hub")
	res, err := Import(m, dst, nil)
	if err != nil || !res.Links || res.Copied != 4 {
		t.Fatalf("Import = %+v, %v", res, err)
	}
	if _, err := os.Stat(filepath.Join(RepoDir(dst, "org/toy"), "blobs", gitID("weights two"))); err != nil {
		t.Error("blobs are not named by the etag the listing gave")
	}
}

func TestExportRefusesIncompleteModels(t *testing.T) {
	t.Parallel()
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	os.Remove(filepath.Join(RepoDir(src, "org/toy"), "blobs", gitID("weights one")))
	drive := filepath.Join(t.TempDir(), "hub")
	if _, err := Export(scanOne(t, src, "org/toy"), drive, nil); err == nil {
		t.Error("an incomplete model was exported")
	}
}

func TestExportCatchesADamagedCache(t *testing.T) {
	t.Parallel()
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	write(t, filepath.Join(RepoDir(src, "org/toy"), "blobs", gitID("weights one")), "weights onX")
	_, err := Export(scanOne(t, src, "org/toy"), filepath.Join(t.TempDir(), "hub"), nil)
	var corrupt *CorruptError
	if !errors.As(err, &corrupt) {
		t.Errorf("Export = %v, want a CorruptError", err)
	}
}

func TestCheckSpaceRefusesWhatWillNotFit(t *testing.T) {
	prev := freeBytes
	freeBytes = func(string) (int64, error) { return 100, nil }
	t.Cleanup(func() { freeBytes = prev })
	src := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	m := scanOne(t, src, "org/toy")
	dst := filepath.Join(t.TempDir(), "not", "yet", "hub")
	need := Need([]Model{m}, dst)
	if need != m.Size() {
		t.Fatalf("Need = %d, want %d for an empty cache", need, m.Size())
	}
	if err := CheckSpace(dst, 101); !errors.Is(err, ErrNoSpace) {
		t.Errorf("CheckSpace(101 of 100) = %v", err)
	}
	if err := CheckSpace(dst, 100); err != nil {
		t.Errorf("CheckSpace(100 of 100) = %v", err)
	}
	// What an import already holds is not counted again.
	if _, err := Import(m, dst, nil); err != nil {
		t.Fatal(err)
	}
	if n := Need([]Model{m}, dst); n != 0 {
		t.Errorf("Need after importing = %d, want 0", n)
	}
	freeBytes = func(string) (int64, error) { return 0, errors.New("unknown") }
	if err := CheckSpace(dst, 1<<40); err != nil {
		t.Errorf("an unreadable filesystem was refused: %v", err)
	}
}

func TestCheckDriveRefusesBigFilesOnFAT(t *testing.T) {
	prevFAT, prevFree := isFAT, freeBytes
	isFAT = func(string) bool { return true }
	freeBytes = func(string) (int64, error) { return 1 << 50, nil }
	t.Cleanup(func() { isFAT, freeBytes = prevFAT, prevFree })
	big := Model{Repo: "org/toy", Files: []File{{Path: "model.safetensors", Size: 5 << 30}}}
	err := CheckDrive([]Model{big}, t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "FAT32") || !strings.Contains(err.Error(), "model.safetensors") {
		t.Errorf("CheckDrive = %v", err)
	}
	small := Model{Repo: "org/toy", Files: []File{{Path: "model.safetensors", Size: 3 << 30}}}
	if err := CheckDrive([]Model{small}, t.TempDir()); err != nil {
		t.Errorf("files under 4 GiB on FAT: %v", err)
	}
	isFAT = func(string) bool { return false }
	if err := CheckDrive([]Model{big}, t.TempDir()); err != nil {
		t.Errorf("a big file on exFAT: %v", err)
	}
}

func writeLocalDir(t *testing.T, commits ...string) string {
	t.Helper()
	dir := t.TempDir()
	for i, name := range []string{"config.json", "model.safetensors"} {
		content := modelFiles[name]
		if name == "model.safetensors" {
			content = "local weights"
		}
		write(t, filepath.Join(dir, name), content)
		etag := gitID(content)
		commit := commits[0]
		if i < len(commits) {
			commit = commits[i]
		}
		write(t, filepath.Join(dir, ".cache", "huggingface", "download", name+".metadata"), commit+"\n"+etag+"\n1700000000.0\n")
	}
	return dir
}

func TestScanReadsAnHFLocalDir(t *testing.T) {
	t.Parallel()
	dir := writeLocalDir(t, rev)
	if _, err := Scan(dir, ScanOptions{}); err == nil || !strings.Contains(err.Error(), "--repo") {
		t.Errorf("without --repo: %v", err)
	}
	found, err := Scan(dir, ScanOptions{Repo: "org/toy"})
	if err != nil || found.Kind != KindLocalDir || len(found.Models) != 1 {
		t.Fatalf("Scan = %+v, %v", found, err)
	}
	m := found.Models[0]
	if m.Revision != rev || !m.Complete() || len(m.Files) != 2 {
		t.Fatalf("model = %+v", m)
	}
	for _, f := range m.Files {
		if f.Path == "model.safetensors" && f.Etag != gitID("local weights") {
			t.Errorf("etag = %q, want the metadata's", f.Etag)
		}
	}

	other := strings.Repeat("d", 40)
	if _, err := Scan(writeLocalDir(t, rev, other), ScanOptions{Repo: "org/toy"}); err == nil || !strings.Contains(err.Error(), "different commits") {
		t.Errorf("mixed commits: %v", err)
	}
	if _, err := Scan(dir, ScanOptions{Repo: "org/toy", Revision: other}); err == nil || !strings.Contains(err.Error(), "--revision") {
		t.Errorf("a --revision that disagrees: %v", err)
	}
	write(t, filepath.Join(dir, ".cache", "huggingface", "download", "model.safetensors.abc.incomplete"), "x")
	found, _ = Scan(dir, ScanOptions{Repo: "org/toy"})
	if found.Models[0].Complete() {
		t.Error("a partial download was complete")
	}
}

func TestScanReadsAPlainFolder(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	write(t, filepath.Join(dir, "config.json"), "{}")
	write(t, filepath.Join(dir, "model.safetensors"), "w")
	if _, err := Scan(dir, ScanOptions{}); err == nil || !strings.Contains(err.Error(), "--repo") {
		t.Errorf("without --repo: %v", err)
	}
	if _, err := Scan(dir, ScanOptions{Repo: "org/toy"}); err == nil || !strings.Contains(err.Error(), "--revision") {
		t.Errorf("without a commit it must not invent one: %v", err)
	}
	if _, err := Scan(dir, ScanOptions{Repo: "org/toy", Revision: "main"}); err == nil {
		t.Error("a branch name was taken for a commit")
	}
	if _, err := Scan(dir, ScanOptions{Repo: "toy"}); err == nil {
		t.Error("a repo with no org was accepted")
	}
	found, err := Scan(dir, ScanOptions{Repo: "org/toy", Revision: rev})
	if err != nil || found.Kind != KindFolder || found.Models[0].Revision != rev || len(found.Models[0].Files) != 2 {
		t.Fatalf("Scan = %+v, %v", found, err)
	}
	// Blobs of an unrecorded etag are named as the Hub names them.
	dst := filepath.Join(t.TempDir(), "hub")
	if _, err := Import(found.Models[0], dst, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(RepoDir(dst, "org/toy"), "blobs", gitID("w"))); err != nil {
		t.Errorf("a small file is not named by its git blob id: %v", err)
	}
}

func TestScanTakesAGitCloneRevisionFromHEAD(t *testing.T) {
	t.Parallel()
	for name, setup := range map[string]func(t *testing.T, dir string){
		"detached": func(t *testing.T, dir string) { write(t, filepath.Join(dir, ".git", "HEAD"), rev+"\n") },
		"branch": func(t *testing.T, dir string) {
			write(t, filepath.Join(dir, ".git", "HEAD"), "ref: refs/heads/main\n")
			write(t, filepath.Join(dir, ".git", "refs", "heads", "main"), rev+"\n")
		},
		"packed": func(t *testing.T, dir string) {
			write(t, filepath.Join(dir, ".git", "HEAD"), "ref: refs/heads/main\n")
			write(t, filepath.Join(dir, ".git", "packed-refs"), "# pack-refs\n"+rev+" refs/heads/main\n")
		},
	} {
		dir := t.TempDir()
		write(t, filepath.Join(dir, "config.json"), "{}")
		write(t, filepath.Join(dir, "model.safetensors"), "w")
		write(t, filepath.Join(dir, ".git", "objects", "x"), "git internals")
		setup(t, dir)
		found, err := Scan(dir, ScanOptions{Repo: "org/toy"})
		if err != nil || found.Models[0].Revision != rev || len(found.Models[0].Files) != 2 {
			t.Errorf("%s: %+v, %v", name, found, err)
		}
	}
}

func TestScanFlagsGitLFSPointers(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	write(t, filepath.Join(dir, "config.json"), "{}")
	write(t, filepath.Join(dir, "model.safetensors"), "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
	found, err := Scan(dir, ScanOptions{Repo: "org/toy", Revision: rev})
	if err != nil {
		t.Fatal(err)
	}
	if m := found.Models[0]; m.Complete() || !strings.Contains(m.Problems[0], "git lfs pull") {
		t.Errorf("problems = %v", m.Problems)
	}
}

func TestScanFindsNothingInAnEmptyFolder(t *testing.T) {
	t.Parallel()
	if _, err := Scan(t.TempDir(), ScanOptions{Repo: "org/toy"}); err == nil || !strings.Contains(err.Error(), "no models") {
		t.Errorf("Scan = %v", err)
	}
	if _, err := Scan(filepath.Join(t.TempDir(), "missing"), ScanOptions{}); err == nil {
		t.Error("a missing path was accepted")
	}
}

func TestSnapshotFollowsRefsMain(t *testing.T) {
	t.Parallel()
	hub := makeCache(t, "org/toy", modelFiles, cacheOpts{})
	dir, got, ok := Snapshot(hub, "org/toy")
	if !ok || got != rev || !strings.HasSuffix(dir, filepath.Join("snapshots", rev)) {
		t.Fatalf("Snapshot = %q, %q, %v", dir, got, ok)
	}
	if _, _, ok := Snapshot(hub, "org/other"); ok {
		t.Error("a repo that is not cached has a snapshot")
	}
}

func TestFormatBytes(t *testing.T) {
	t.Parallel()
	for n, want := range map[int64]string{0: "0 B", 999: "999 B", 1500: "1.5 kB", 14_500_000_000: "14.5 GB"} {
		if got := FormatBytes(n); got != want {
			t.Errorf("FormatBytes(%d) = %q, want %q", n, got, want)
		}
	}
}
