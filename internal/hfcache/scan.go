package hfcache

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Kind is the layout a source was read from.
type Kind string

// The layouts Scan reads.
const (
	// KindCache is a hub cache: models--org--name folders.
	KindCache Kind = "Hugging Face cache"
	// KindLocalDir is an `hf download --local-dir` folder.
	KindLocalDir Kind = "hf download --local-dir folder"
	// KindFolder is a plain model folder, such as a git clone.
	KindFolder Kind = "model folder"
)

// File is one file of a model.
type File struct {
	// Path is the file's path in the repo, slash-separated.
	Path string
	Size int64
	// Etag is the blob's name in a cache: the sha256 of an LFS file or the git
	// blob hash of any other. It is "" when the source does not record it.
	Etag string
	// Abs is where the content is read from.
	Abs string
}

// Model is one revision of one repo found in a source.
type Model struct {
	Repo     string
	Revision string
	Files    []File
	// Problems say why the model cannot be copied, and are empty when it is
	// complete.
	Problems []string
	Kind     Kind
	// Root is the source the model was read from.
	Root string

	// tree is the source's trees/<commit>.json, copied alongside the files it
	// lists.
	tree []byte
}

// Size is the total bytes of the model's files.
func (m Model) Size() int64 {
	var n int64
	for _, f := range m.Files {
		n += f.Size
	}
	return n
}

// Complete reports whether the model has every file it needs.
func (m Model) Complete() bool { return len(m.Problems) == 0 }

// ScanOptions name what a source cannot say about itself.
type ScanOptions struct {
	// Repo and Revision identify a folder that is not a cache.
	Repo     string
	Revision string
	// Pins, when set, limits a cache scan to these repos, each read at the
	// revision given instead of at refs/main.
	Pins map[string]string
}

// Found is what Scan read.
type Found struct {
	Kind   Kind
	Models []Model
}

// Scan reads the models at path. A Hugging Face cache is read at path itself,
// or at hub/, huggingface/hub/ or .cache/huggingface/hub/ under it, or from a
// single models--org--name folder. Otherwise path is an `hf download
// --local-dir` folder or a plain model folder, and o names its repo.
func Scan(path string, o ScanOptions) (Found, error) {
	st, err := os.Stat(path)
	if err != nil {
		return Found{}, fmt.Errorf("read %s: %w", path, err)
	}
	if !st.IsDir() {
		return Found{}, fmt.Errorf("%s is not a directory", path)
	}
	path = filepath.Clean(path)
	if _, ok := repoFromDir(filepath.Base(path)); ok {
		models, err := scanCache(filepath.Dir(path), filepath.Base(path), o.Pins)
		return Found{Kind: KindCache, Models: models}, err
	}
	var models []Model
	seen := map[string]bool{}
	for _, hub := range []string{path, filepath.Join(path, "hub"), filepath.Join(path, "huggingface", "hub"), filepath.Join(path, ".cache", "huggingface", "hub")} {
		if !isHub(hub) {
			continue
		}
		found, err := scanCache(hub, "", o.Pins)
		if err != nil {
			return Found{}, err
		}
		for _, m := range found {
			if key := m.Repo + "@" + m.Revision; !seen[key] {
				seen[key] = true
				models = append(models, m)
			}
		}
	}
	if len(models) > 0 {
		return Found{Kind: KindCache, Models: models}, nil
	}

	if _, err := os.Stat(filepath.Join(path, ".cache", "huggingface", "download")); err == nil {
		if o.Repo == "" {
			return Found{}, fmt.Errorf("%s was written by `hf download --local-dir`, which does not record the repo: pass --repo org/model", path)
		}
		m, err := scanLocalDir(path, o)
		return Found{Kind: KindLocalDir, Models: []Model{m}}, err
	}
	if hasFiles(path) {
		if o.Repo == "" {
			return Found{}, fmt.Errorf("no Hugging Face cache found at %s; to import it as a model folder, pass --repo org/model (and --revision <commit> unless it is a git clone)", path)
		}
		m, err := scanFolder(path, o)
		return Found{Kind: KindFolder, Models: []Model{m}}, err
	}
	return Found{}, fmt.Errorf("found no models at %s", path)
}

// ScanCache lists every model in a hub cache.
func ScanCache(hub string) ([]Model, error) { return scanCache(hub, "", nil) }

// ScanRepo reads one repo's refs/main revision from a hub cache.
func ScanRepo(hub, repo string) (Model, error) {
	models, err := scanCache(hub, repoPrefix+strings.ReplaceAll(repo, "/", "--"), nil)
	if err != nil {
		return Model{}, err
	}
	if len(models) == 0 {
		return Model{}, fmt.Errorf("%s is not in the cache at %s", repo, hub)
	}
	return models[0], nil
}

// Has reports whether the hub cache at hub holds any repo.
func Has(hub string) bool { return isHub(hub) }

func isHub(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if _, ok := repoFromDir(e.Name()); ok && e.IsDir() {
			return true
		}
	}
	return false
}

func hasFiles(dir string) bool {
	found := false
	_ = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err == nil && d.IsDir() && d.Name() == ".git" {
			return filepath.SkipDir
		}
		if err == nil && !d.IsDir() {
			found = true
			return filepath.SkipAll
		}
		return nil
	})
	return found
}

// scanCache reads the repos under hub, or only the folder named only. With
// pins it reads only those repos, at the revisions they name.
func scanCache(hub, only string, pins map[string]string) ([]Model, error) {
	entries, err := os.ReadDir(hub)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", hub, err)
	}
	var out []Model
	for _, e := range entries {
		repo, ok := repoFromDir(e.Name())
		if !ok || !e.IsDir() || (only != "" && e.Name() != only) {
			continue
		}
		pin, pinned := pins[repo]
		if pins != nil && !pinned {
			continue
		}
		out = append(out, scanSnapshot(hub, repo, pin))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Repo < out[j].Repo })
	return out, nil
}

// weightSuffixes are the files a model is useless without.
var weightSuffixes = []string{".safetensors", ".bin", ".gguf"}

// scanSnapshot reads repo at pin, or at refs/main when pin is empty.
func scanSnapshot(hub, repo, pin string) Model {
	repoDir := RepoDir(hub, repo)
	m := Model{Repo: repo, Kind: KindCache, Root: hub}
	rev := pin
	if rev == "" {
		var err error
		if rev, err = mainRevision(repoDir); err != nil {
			m.Problems = append(m.Problems, "no refs/main, and not exactly one snapshot to choose")
			return m
		}
	}
	m.Revision = rev
	snap := filepath.Join(repoDir, "snapshots", rev)
	if _, err := os.Stat(snap); err != nil {
		what := "snapshot "
		if pin != "" {
			what = "pinned snapshot "
		}
		m.Problems = append(m.Problems, what+rev+" is missing")
		return m
	}
	err := filepath.WalkDir(snap, func(p string, d fs.DirEntry, err error) error {
		rel, _ := filepath.Rel(snap, p)
		rel = filepath.ToSlash(rel)
		if err != nil {
			m.Problems = append(m.Problems, "cannot read "+rel+": "+err.Error())
			return nil
		}
		if d.IsDir() {
			return nil
		}
		st, err := os.Stat(p)
		if err != nil {
			m.Problems = append(m.Problems, "dangling link: "+rel)
			return nil
		}
		f := File{Path: rel, Size: st.Size(), Abs: p}
		if d.Type()&fs.ModeSymlink != 0 {
			if target, err := os.Readlink(p); err == nil {
				if name := path.Base(filepath.ToSlash(target)); hexPattern.MatchString(name) {
					f.Etag = name
				}
			}
		}
		m.Files = append(m.Files, f)
		return nil
	})
	if err != nil {
		m.Problems = append(m.Problems, "cannot read snapshot: "+err.Error())
	}
	if raw, err := os.ReadFile(filepath.Join(repoDir, "trees", rev+".json")); err == nil {
		m.tree = raw
		m.applyTree(raw)
	}
	m.applyHashes(readHashes(repoDir, rev))
	m.checkComplete()
	return m
}

// applyHashes takes the hash an export recorded for each file the cache does
// not name by hash, when the size still matches.
func (m *Model) applyHashes(recorded map[string]hashEntry) {
	for i, f := range m.Files {
		if e, ok := recorded[f.Path]; ok && f.Etag == "" && e.Size == f.Size {
			m.Files[i].Etag = e.Hash
		}
	}
}

// treeFile is one entry of trees/<commit>.json.
type treeFile struct {
	Size      int64  `json:"size"`
	BlobID    string `json:"blob_id"`
	LFSSha256 string `json:"lfs_sha256"`
}

func (t treeFile) etag() string {
	if hexPattern.MatchString(t.LFSSha256) {
		return t.LFSSha256
	}
	if hexPattern.MatchString(t.BlobID) {
		return t.BlobID
	}
	return ""
}

// applyTree checks the files against the listing the cache recorded for the
// revision: huggingface_hub refuses an offline load that misses any of them.
func (m *Model) applyTree(raw []byte) {
	var t struct {
		Files map[string]treeFile `json:"files"`
	}
	if json.Unmarshal(raw, &t) != nil {
		m.Problems = append(m.Problems, "trees/"+m.Revision+".json is unreadable")
		return
	}
	have := map[string]int{}
	for i, f := range m.Files {
		have[f.Path] = i
	}
	for _, p := range sortedKeys(t.Files) {
		want := t.Files[p]
		i, ok := have[p]
		switch {
		case !ok:
			m.Problems = append(m.Problems, "missing "+p)
		case want.Size > 0 && m.Files[i].Size != want.Size:
			m.Problems = append(m.Problems, fmt.Sprintf("%s is %d bytes, the listing says %d", p, m.Files[i].Size, want.Size))
		case m.Files[i].Etag == "":
			m.Files[i].Etag = want.etag()
		}
	}
}

// checkComplete adds a problem for each way the files cannot serve a model:
// none at all, no weights, or a shard an index names that is not there.
func (m *Model) checkComplete() {
	if len(m.Files) == 0 {
		m.Problems = append(m.Problems, "no files")
		return
	}
	have := map[string]bool{}
	weights := false
	for _, f := range m.Files {
		have[f.Path] = true
		for _, s := range weightSuffixes {
			weights = weights || strings.HasSuffix(strings.ToLower(f.Path), s)
		}
	}
	if !weights {
		m.Problems = append(m.Problems, "no weight files")
	}
	m.Problems = append(m.Problems, missingSplits(have)...)
	for _, f := range m.Files {
		if !strings.HasSuffix(f.Path, ".index.json") {
			continue
		}
		raw, err := os.ReadFile(f.Abs)
		if err != nil {
			continue
		}
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if json.Unmarshal(raw, &idx) != nil {
			continue
		}
		dir := path.Dir(f.Path)
		for _, name := range uniqueSorted(idx.WeightMap) {
			if !have[path.Join(dir, name)] {
				m.Problems = append(m.Problems, "missing "+path.Join(dir, name)+", which "+f.Path+" names")
			}
		}
	}
}

var splitGGUF = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

// missingSplits names the pieces absent from each numbered GGUF split.
func missingSplits(have map[string]bool) []string {
	type group struct {
		total int
		seen  map[int]bool
	}
	groups := map[string]*group{}
	for p := range have {
		sm := splitGGUF.FindStringSubmatch(p)
		if sm == nil {
			continue
		}
		total, _ := strconv.Atoi(sm[3])
		key := sm[1] + "|" + sm[3]
		if groups[key] == nil {
			groups[key] = &group{total: total, seen: map[int]bool{}}
		}
		n, _ := strconv.Atoi(sm[2])
		groups[key].seen[n] = true
	}
	var out []string
	for _, key := range sortedKeys(groups) {
		g := groups[key]
		prefix, of, _ := strings.Cut(key, "|")
		for i := 1; i <= g.total; i++ {
			if !g.seen[i] {
				out = append(out, fmt.Sprintf("missing %s-%05d-of-%s.gguf, a piece of a split model", prefix, i, of))
			}
		}
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func uniqueSorted(m map[string]string) []string {
	set := map[string]bool{}
	for _, v := range m {
		set[v] = true
	}
	return sortedKeys(set)
}

// scanLocalDir reads an `hf download --local-dir` folder, whose
// .cache/huggingface/download/<file>.metadata files record the commit and etag
// of each download: three lines, commit, etag, timestamp.
func scanLocalDir(dir string, o ScanOptions) (Model, error) {
	m := Model{Repo: o.Repo, Kind: KindLocalDir, Root: dir}
	if !ValidRepo(o.Repo) {
		return m, fmt.Errorf("--repo %q is not an org/name Hub id", o.Repo)
	}
	meta := filepath.Join(dir, ".cache", "huggingface", "download")
	commits := map[string]bool{}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p == filepath.Join(dir, ".cache") {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(dir, p)
		rel = filepath.ToSlash(rel)
		st, err := os.Stat(p)
		if err != nil {
			m.Problems = append(m.Problems, "dangling link: "+rel)
			return nil
		}
		f := File{Path: rel, Size: st.Size(), Abs: p}
		if commit, etag, ok := readMetadata(filepath.Join(meta, filepath.FromSlash(rel)+".metadata")); ok {
			if revisionPattern.MatchString(commit) {
				commits[commit] = true
			}
			if hexPattern.MatchString(etag) {
				f.Etag = etag
			}
		}
		m.Files = append(m.Files, f)
		return nil
	})
	if err != nil {
		return m, fmt.Errorf("read %s: %w", dir, err)
	}
	_ = filepath.WalkDir(meta, func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(d.Name(), ".incomplete") {
			m.Problems = append(m.Problems, "partial download: "+d.Name())
		}
		return nil
	})
	if len(commits) > 1 {
		return m, fmt.Errorf("%s holds files from %d different commits; download one revision at a time", dir, len(commits))
	}
	for c := range commits {
		m.Revision = c
	}
	if err := m.useRevision(o.Revision); err != nil {
		return m, err
	}
	m.checkComplete()
	return m, nil
}

// useRevision settles a folder's commit, which the caller's flag may state
// and the source may also record. It never makes one up.
func (m *Model) useRevision(given string) error {
	if given != "" && !revisionPattern.MatchString(given) {
		return fmt.Errorf("--revision %q is not a 40-character commit hash", given)
	}
	switch {
	case m.Revision != "" && given != "" && m.Revision != given:
		return fmt.Errorf("the folder records commit %s, but --revision says %s", m.Revision, given)
	case m.Revision == "":
		m.Revision = given
	}
	if m.Revision == "" {
		return errors.New("the folder does not record its commit: pass --revision <40-character commit hash>")
	}
	return nil
}

func readMetadata(file string) (commit, etag string, ok bool) {
	f, err := os.Open(file)
	if err != nil {
		return "", "", false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	var lines []string
	for sc.Scan() && len(lines) < 2 {
		lines = append(lines, strings.TrimSpace(sc.Text()))
	}
	if len(lines) < 2 {
		return "", "", false
	}
	return lines[0], strings.Trim(lines[1], `"`), true
}

// scanFolder reads a plain model folder. The commit comes from --revision, or
// from the folder's git HEAD.
func scanFolder(dir string, o ScanOptions) (Model, error) {
	m := Model{Repo: o.Repo, Kind: KindFolder, Root: dir}
	if !ValidRepo(o.Repo) {
		return m, fmt.Errorf("--repo %q is not an org/name Hub id", o.Repo)
	}
	m.Revision = gitHead(dir)
	if err := m.useRevision(o.Revision); err != nil {
		return m, err
	}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(dir, p)
		rel = filepath.ToSlash(rel)
		st, err := os.Stat(p)
		if err != nil {
			m.Problems = append(m.Problems, "dangling link: "+rel)
			return nil
		}
		if isLFSPointer(p, st.Size()) {
			m.Problems = append(m.Problems, rel+" is a git-lfs pointer, not the file: run git lfs pull")
		}
		m.Files = append(m.Files, File{Path: rel, Size: st.Size(), Abs: p})
		return nil
	})
	if err != nil {
		return m, fmt.Errorf("read %s: %w", dir, err)
	}
	m.checkComplete()
	return m, nil
}

const lfsPointerHead = "version https://git-lfs.github.com/spec/v1"

func isLFSPointer(file string, size int64) bool {
	if size > 1024 {
		return false
	}
	f, err := os.Open(file)
	if err != nil {
		return false
	}
	defer f.Close()
	head := make([]byte, len(lfsPointerHead))
	n, _ := io.ReadFull(f, head)
	return string(head[:n]) == lfsPointerHead
}

// gitHead is the commit a clone has checked out, "" when dir is not one.
func gitHead(dir string) string {
	read := func(p string) string {
		b, err := os.ReadFile(filepath.Join(dir, ".git", filepath.FromSlash(p)))
		if err != nil {
			return ""
		}
		return strings.TrimSpace(string(b))
	}
	head := read("HEAD")
	ref, ok := strings.CutPrefix(head, "ref: ")
	if !ok {
		if revisionPattern.MatchString(head) {
			return head
		}
		return ""
	}
	if c := read(ref); revisionPattern.MatchString(c) {
		return c
	}
	for line := range strings.SplitSeq(read("packed-refs"), "\n") {
		if c, name, ok := strings.Cut(line, " "); ok && name == ref && revisionPattern.MatchString(c) {
			return c
		}
	}
	return ""
}
