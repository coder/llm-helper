package hfcache

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// partialSuffix ends a file that is still being copied. Caches ignore
// *.incomplete, so an interrupted copy leaves nothing a loader would read.
const partialSuffix = ".llm-helper.incomplete"

// lfsThreshold is the size above which the Hub stores a file with LFS, so its
// blob is named by sha256 rather than by git blob id.
const lfsThreshold = 10 << 20

// Progress is told of each file as it is copied or found already there.
type Progress func(path string, size int64, skipped bool)

// Result is what copying one model did.
type Result struct {
	Copied  int
	Skipped int
	Bytes   int64
	// Links is whether snapshot files are links to blobs, rather than files.
	Links bool
	// RefKept is the revision refs/main already named, when it differs from
	// the one copied. It is left alone, so the new revision is not the one an
	// offline load reads.
	RefKept string
}

// newSum hashes a file the way its blob is named: the sha256 of an LFS file,
// and the git blob id of any other. etag says which when the source records
// it; without one, the size does.
func newSum(etag string, size int64) hash.Hash {
	if len(etag) == 64 || (etag == "" && size > lfsThreshold) {
		return sha256.New()
	}
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", size)
	return h
}

// copyFile copies src to tmp, hashing as it goes, and returns the blob name.
// A recorded etag that the content does not match removes tmp and fails.
func copyFile(src, tmp, etag string, size int64) (string, error) {
	in, err := os.Open(src)
	if err != nil {
		return "", err
	}
	defer in.Close()
	if st, err := in.Stat(); err != nil || st.Size() != size {
		return "", fmt.Errorf("%s changed size while being read", src)
	}
	out, err := createNew(tmp)
	if err != nil {
		return "", err
	}
	h := newSum(etag, size)
	n, err := io.CopyBuffer(io.MultiWriter(out, h), in, make([]byte, 1<<20))
	if err == nil && n != size {
		err = fmt.Errorf("read %d of %d bytes from %s", n, size, src)
	}
	if err == nil {
		err = out.Sync()
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return "", err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if etag != "" && got != etag {
		os.Remove(tmp)
		return "", &CorruptError{Path: src, Want: etag, Got: got}
	}
	return got, nil
}

// createNew creates path empty. Whatever is there, a link included, is removed
// first and never written through.
func createNew(path string) (*os.File, error) {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	return os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
}

// CorruptError is a file whose content does not match the hash recorded for
// it.
type CorruptError struct {
	Path, Want, Got string
}

func (e *CorruptError) Error() string {
	return fmt.Sprintf("%s does not match its recorded hash (want %s, got %s): the copy it came from may be damaged", e.Path, e.Want, e.Got)
}

// partialPath is where the file at rel is copied before it is renamed into
// place.
func partialPath(repoDir, rel string) string {
	sum := sha256.Sum256([]byte(rel))
	return filepath.Join(repoDir, "blobs", hex.EncodeToString(sum[:8])+partialSuffix)
}

// sizeIs reports whether p is a file of the given size.
func sizeIs(p string, size int64) bool {
	st, err := os.Stat(p)
	return err == nil && st.Mode().IsRegular() && st.Size() == size
}

// symlink is os.Symlink, so tests can take links away.
var symlink = os.Symlink

// canLink reports whether dir takes symbolic links.
func canLink(dir string) bool {
	p := filepath.Join(dir, ".llm-helper-link-test")
	os.Remove(p)
	if symlink("x", p) != nil {
		return false
	}
	os.Remove(p)
	return true
}

// place copies a complete model into hub. With links, files become blobs named
// by their hash and snapshot entries relative links to them, as
// huggingface_hub writes them; without, the snapshot holds the files.
// Whatever is already there is kept and never overwritten.
func place(m Model, hub string, links bool, progress Progress) (Result, error) {
	res := Result{Links: links}
	if !ValidRepo(m.Repo) {
		return res, fmt.Errorf("%q is not an org/name Hub id", m.Repo)
	}
	if !m.Complete() {
		return res, fmt.Errorf("%s is incomplete: %s", m.Repo, m.Problems[0])
	}
	repoDir := RepoDir(hub, m.Repo)
	snap := filepath.Join(repoDir, "snapshots", m.Revision)
	for _, d := range []string{filepath.Join(repoDir, "blobs"), snap} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return res, err
		}
	}
	hashes := readHashes(repoDir, m.Revision)
	for _, f := range m.Files {
		dst := filepath.Join(snap, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return res, err
		}
		var copied bool
		var err error
		if links {
			copied, err = placeBlobLink(repoDir, dst, f)
		} else {
			var sum string
			copied, sum, err = placeFile(repoDir, dst, f)
			if sum != "" {
				hashes[f.Path] = hashEntry{Size: f.Size, Hash: sum}
			}
		}
		if err != nil {
			return res, fmt.Errorf("%s: %w", f.Path, err)
		}
		if copied {
			res.Copied++
			res.Bytes += f.Size
		} else {
			res.Skipped++
		}
		if progress != nil {
			progress(f.Path, f.Size, !copied)
		}
	}
	kept, err := writeRef(repoDir, m.Revision)
	if err != nil {
		return res, err
	}
	res.RefKept = kept
	if !links {
		if err := writeHashes(repoDir, m.Revision, hashes); err != nil {
			return res, err
		}
	}
	return res, writeTree(repoDir, m.Revision, m.tree)
}

// placeFile puts f at dst as a regular file, and returns its hash when it is
// known.
func placeFile(repoDir, dst string, f File) (copied bool, sum string, err error) {
	if _, err := os.Lstat(dst); err == nil {
		if sizeIs(dst, f.Size) {
			return false, f.Etag, nil
		}
		return false, "", fmt.Errorf("%s exists with different content", dst)
	}
	tmp := partialPath(repoDir, f.Path)
	sum, err = copyFile(f.Abs, tmp, f.Etag, f.Size)
	if err != nil {
		return false, "", err
	}
	return true, sum, os.Rename(tmp, dst)
}

// placeBlobLink puts f's content in blobs/ and links dst to it.
func placeBlobLink(repoDir, dst string, f File) (copied bool, err error) {
	if _, err := os.Lstat(dst); err == nil {
		if sizeIs(dst, f.Size) {
			return false, nil
		}
		return false, fmt.Errorf("%s exists with different content", dst)
	}
	name := f.Etag
	if name == "" || !sizeIs(filepath.Join(repoDir, "blobs", name), f.Size) {
		tmp := partialPath(repoDir, f.Path)
		got, err := copyFile(f.Abs, tmp, f.Etag, f.Size)
		if err != nil {
			return false, err
		}
		name = got
		blob := filepath.Join(repoDir, "blobs", name)
		switch _, err := os.Lstat(blob); {
		case err != nil:
			if err := os.Rename(tmp, blob); err != nil {
				return false, err
			}
			copied = true
		case sizeIs(blob, f.Size):
			os.Remove(tmp)
		default:
			os.Remove(tmp)
			return false, fmt.Errorf("blob %s exists with a different size", name)
		}
	}
	up := strings.Repeat("../", 2+strings.Count(f.Path, "/"))
	return copied, symlink(up+"blobs/"+name, dst)
}

// writeRef makes refs/main name rev when the cache has none. A ref that names
// another revision is left as it is, and returned.
func writeRef(repoDir, rev string) (kept string, err error) {
	p := filepath.Join(repoDir, "refs", "main")
	if b, err := os.ReadFile(p); err == nil {
		if have := strings.TrimSpace(string(b)); have != rev {
			return have, nil
		}
		return "", nil
	}
	return "", writeAtomic(p, []byte(rev))
}

// writeTree records the source's listing for rev, which names exactly the
// files that were copied. An existing listing is kept.
func writeTree(repoDir, rev string, raw []byte) error {
	if len(raw) == 0 {
		return nil
	}
	p := filepath.Join(repoDir, "trees", rev+".json")
	if _, err := os.Stat(p); err == nil {
		return nil
	}
	return writeAtomic(p, raw)
}

func writeAtomic(p string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmp := p + partialSuffix
	f, err := createNew(tmp)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, p)
}

// Import copies a complete model into the hub cache at hub, so a server finds
// it as if it had been downloaded. Files already there are skipped, which
// lets an interrupted import resume. Every file is hashed as it is copied and
// checked against its recorded hash.
func Import(m Model, hub string, progress Progress) (Result, error) {
	if err := os.MkdirAll(hub, 0o755); err != nil {
		return Result{}, err
	}
	return place(m, hub, canLink(hub), progress)
}

// Export copies a complete model onto a drive as a cache with no links, which
// every filesystem a drive is likely to have can hold, and which is itself a
// cache that can be read offline.
func Export(m Model, hub string, progress Progress) (Result, error) {
	if err := os.MkdirAll(hub, 0o755); err != nil {
		return Result{}, err
	}
	return place(m, hub, false, progress)
}

// ErrNoSpace is wrapped by the error a copy that would not fit returns.
var ErrNoSpace = errors.New("not enough free space")

// Need is the bytes copying the models into hub would write. It counts a file
// as already there when it is, either as a blob or in the snapshot.
func Need(models []Model, hub string) int64 {
	var n int64
	for _, m := range models {
		snap := filepath.Join(RepoDir(hub, m.Repo), "snapshots", m.Revision)
		for _, f := range m.Files {
			if sizeIs(filepath.Join(snap, filepath.FromSlash(f.Path)), f.Size) {
				continue
			}
			if f.Etag != "" && sizeIs(filepath.Join(RepoDir(hub, m.Repo), "blobs", f.Etag), f.Size) {
				continue
			}
			n += f.Size
		}
	}
	return n
}

// Free is the bytes free where dir is, or would be created. ok is false when
// that cannot be told.
func Free(dir string) (free int64, ok bool) {
	probe := dir
	for {
		if _, err := os.Stat(probe); err == nil {
			break
		}
		next := filepath.Dir(probe)
		if next == probe {
			return 0, false
		}
		probe = next
	}
	free, err := freeBytes(probe)
	return free, err == nil
}

// CheckSpace fails when need bytes do not fit where dir is, or will be
// created.
func CheckSpace(dir string, need int64) error {
	free, ok := Free(dir)
	if !ok || need <= free {
		return nil
	}
	return fmt.Errorf("%w on %s: need %s, %s free", ErrNoSpace, dir, FormatBytes(need), FormatBytes(free))
}

// isFAT is onFAT, so tests can name a drive FAT32.
var isFAT = onFAT

// fatFileLimit is the largest file FAT32 holds.
const fatFileLimit = 4<<30 - 1

// CheckDrive fails before an export starts when the models cannot be written
// to the drive at dir: FAT32 cannot hold a file over 4 GiB, and the rest must
// fit in the free space.
func CheckDrive(models []Model, dir string) error {
	if isFAT(dir) {
		for _, m := range models {
			for _, f := range m.Files {
				if f.Size > fatFileLimit {
					return fmt.Errorf("%s is %s, and %s is formatted as FAT32, which cannot hold a file over 4 GiB: reformat the drive as exFAT (or use ext4 or APFS)", f.Path, FormatBytes(f.Size), dir)
				}
			}
		}
	}
	return CheckSpace(dir, Need(models, filepath.Join(dir, "hub")))
}

// FormatBytes writes n the way a person reads a disk: 14.5 GB.
func FormatBytes(n int64) string {
	const unit = 1000
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "kMGTPE"[exp])
}
