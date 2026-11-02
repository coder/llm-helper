package update

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseChannel(t *testing.T) {
	t.Parallel()
	for _, in := range []string{"stable", "dev"} {
		if _, err := ParseChannel(in); err != nil {
			t.Errorf("ParseChannel(%q): %v", in, err)
		}
	}
	for _, in := range []string{"", "Stable", "nightly", "main"} {
		if _, err := ParseChannel(in); err == nil {
			t.Errorf("ParseChannel(%q) was accepted, want an error naming the valid channels", in)
		}
	}
}

// The dev release tag never changes, so the commit in the body is the only
// thing that says which build it holds.
func TestReleaseCommit(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		body string
		want string
	}{
		{"typical notes", "Built from main.\n\ncommit: abc123def456\n\nsome subject", "abc123def456"},
		{"leading whitespace", "  commit:   abc123  \n", "abc123"},
		{"absent", "no marker here\n", ""},
		{"empty", "", ""},
		{"not a prefix match", "the commit: abc123\n", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := (&Release{Body: tt.body}).Commit(); got != tt.want {
				t.Errorf("Commit() = %q, want %q", got, tt.want)
			}
		})
	}
}

// Asset names carry a version that differs per channel, so only the platform
// suffix is predictable enough to match on.
func TestAssetFor(t *testing.T) {
	t.Parallel()
	rel := &Release{TagName: "v0.3.1", Assets: []Asset{
		{Name: "vllm-helper_0.3.1_linux_amd64.tar.gz"},
		{Name: "vllm-helper_0.3.1_linux_arm64.tar.gz"},
		{Name: "vllm-helper_0.3.1_darwin_arm64.tar.gz"},
		{Name: "checksums.txt"},
	}}
	for _, tt := range []struct{ goos, goarch, want string }{
		{"linux", "amd64", "vllm-helper_0.3.1_linux_amd64.tar.gz"},
		{"linux", "arm64", "vllm-helper_0.3.1_linux_arm64.tar.gz"},
		{"darwin", "arm64", "vllm-helper_0.3.1_darwin_arm64.tar.gz"},
	} {
		got, err := rel.AssetFor(tt.goos, tt.goarch)
		if err != nil || got.Name != tt.want {
			t.Errorf("AssetFor(%s,%s) = %q, %v; want %q", tt.goos, tt.goarch, got.Name, err, tt.want)
		}
	}
	// Intel macs aren't built, and saying so beats downloading the wrong one.
	if _, err := rel.AssetFor("darwin", "amd64"); err == nil {
		t.Error("AssetFor(darwin,amd64) succeeded, want an error naming the platform")
	}
	// A dev snapshot's version is generated, not a tag; matching must not
	// depend on its shape.
	dev := &Release{TagName: "dev", Assets: []Asset{{Name: "vllm-helper_0.3.2-dev.0fbc441_linux_amd64.tar.gz"}}}
	if _, err := dev.AssetFor("linux", "amd64"); err != nil {
		t.Errorf("a dev asset didn't match: %v", err)
	}
}

// tarGz builds a release-shaped archive: the binary plus the docs goreleaser
// includes.
func tarGz(t *testing.T, binary string) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, f := range []struct{ name, body string }{
		{"LICENSE", "a license"},
		{"vllm-helper", binary},
		{"README.md", "docs"},
	} {
		if err := tw.WriteHeader(&tar.Header{
			Name: f.name, Mode: 0o755, Size: int64(len(f.body)), Typeflag: tar.TypeReg,
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(f.body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// serveRelease stands in for GitHub: one release, one platform archive, and a
// checksums.txt in goreleaser's format.
func serveRelease(t *testing.T, archive []byte, corruptSum bool) *Release {
	t.Helper()
	const assetName = "vllm-helper_0.3.1_linux_amd64.tar.gz"

	sum := sha256.Sum256(archive)
	digest := hex.EncodeToString(sum[:])
	if corruptSum {
		digest = strings.Repeat("0", len(digest))
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/assets/"+assetName, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(archive)
	})
	mux.HandleFunc("/assets/checksums.txt", func(w http.ResponseWriter, _ *http.Request) {
		// Two spaces between digest and name, as sha256sum writes it.
		fmt.Fprintf(w, "%s  %s\n", digest, assetName)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	return &Release{TagName: "v0.3.1", Assets: []Asset{
		{Name: assetName, DownloadURL: srv.URL + "/assets/" + assetName},
		{Name: "checksums.txt", DownloadURL: srv.URL + "/assets/checksums.txt"},
	}}
}

func TestDownload(t *testing.T) {
	rel := serveRelease(t, tarGz(t, "I am the binary"), false)
	got, err := Download(rel, "linux", "amd64", "")
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if string(got) != "I am the binary" {
		t.Errorf("got %q, want the binary from inside the archive", got)
	}
}

// A bad checksum has to stop the upgrade before anything is unpacked, let
// alone written over the running binary.
func TestDownloadRejectsAChecksumMismatch(t *testing.T) {
	rel := serveRelease(t, tarGz(t, "I am the binary"), true)
	_, err := Download(rel, "linux", "amd64", "")
	if err == nil {
		t.Fatal("a mismatched checksum was accepted")
	}
	if !strings.Contains(err.Error(), "checksum") {
		t.Errorf("error = %v, want it to name the checksum as the problem", err)
	}
}

func TestDownloadWithoutTheBinaryInTheArchive(t *testing.T) {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	_ = tw.WriteHeader(&tar.Header{Name: "README.md", Mode: 0o644, Size: 4, Typeflag: tar.TypeReg})
	_, _ = tw.Write([]byte("docs"))
	_ = tw.Close()
	_ = gz.Close()

	rel := serveRelease(t, buf.Bytes(), false)
	if _, err := Download(rel, "linux", "amd64", ""); err == nil {
		t.Error("an archive with no binary was accepted")
	}
}

// Releases ship the binary as llm-helper and, for installs from before the
// rename, as vllm-helper too. An archive with either name has to upgrade.
func TestBinaryFromArchiveTakesEitherName(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"llm-helper", "vllm-helper"} {
		var buf bytes.Buffer
		gz := gzip.NewWriter(&buf)
		tw := tar.NewWriter(gz)
		_ = tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: 3, Typeflag: tar.TypeReg})
		_, _ = tw.Write([]byte("bin"))
		_ = tw.Close()
		_ = gz.Close()
		got, err := binaryFromArchive(buf.Bytes())
		if err != nil || string(got) != "bin" {
			t.Errorf("archive holding %s: got %q, %v", name, got, err)
		}
	}
}

func TestLatestReportsAMissingRelease(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(srv.Close)
	old := apiBase
	apiBase = srv.URL
	t.Cleanup(func() { apiBase = old })

	if _, err := Latest(Dev, ""); err == nil {
		t.Error("a missing dev release was reported as success")
	}
	if _, err := Latest(Stable, ""); err == nil {
		t.Error("a repo with no releases was reported as success")
	}
}

// While the repository is private, assets are only readable through the API
// URL with credentials: the browser URL redirects to storage that 404s. So a
// token has to change both the endpoint and the Accept header.
func TestDownloadWithATokenUsesTheAuthenticatedEndpoint(t *testing.T) {
	const assetName = "vllm-helper_0.3.1_linux_amd64.tar.gz"
	archive := tarGz(t, "private binary")
	sum := sha256.Sum256(archive)

	var gotAuth, gotAccept string
	mux := http.NewServeMux()
	mux.HandleFunc("/api/assets/1", func(w http.ResponseWriter, r *http.Request) {
		gotAuth, gotAccept = r.Header.Get("Authorization"), r.Header.Get("Accept")
		_, _ = w.Write(archive)
	})
	mux.HandleFunc("/api/assets/2", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintf(w, "%s  %s\n", hex.EncodeToString(sum[:]), assetName)
	})
	// The browser URLs point at a handler that fails the test if used.
	mux.HandleFunc("/public/", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	rel := &Release{TagName: "v0.3.1", Assets: []Asset{
		{Name: assetName, APIURL: srv.URL + "/api/assets/1", DownloadURL: srv.URL + "/public/" + assetName},
		{Name: "checksums.txt", APIURL: srv.URL + "/api/assets/2", DownloadURL: srv.URL + "/public/checksums.txt"},
	}}

	got, err := Download(rel, "linux", "amd64", "s3cret")
	if err != nil {
		t.Fatalf("Download with a token: %v", err)
	}
	if string(got) != "private binary" {
		t.Errorf("got %q, want the binary from the authenticated endpoint", got)
	}
	if gotAuth != "Bearer s3cret" {
		t.Errorf("Authorization = %q, want the token as a bearer credential", gotAuth)
	}
	// Without this the API answers with JSON describing the asset.
	if gotAccept != "application/octet-stream" {
		t.Errorf("Accept = %q, want application/octet-stream", gotAccept)
	}
}

// A 404 is how GitHub reports "private and you can't see it", so the message
// has to raise credentials rather than insisting nothing was released.
func TestLatestSuggestsATokenWhenNoneIsInUse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(srv.Close)
	old := apiBase
	apiBase = srv.URL
	t.Cleanup(func() { apiBase = old })

	_, err := Latest(Stable, "")
	if err == nil {
		t.Fatal("a 404 was reported as success")
	}
	if !strings.Contains(err.Error(), "GITHUB_TOKEN") {
		t.Errorf("error = %v, want it to mention how to supply a token", err)
	}

	// With a token in play, telling the user to set one is unhelpful; the
	// problem is that this one lacks access.
	_, err = Latest(Stable, "s3cret")
	if err == nil {
		t.Fatal("a 404 was reported as success")
	}
	if strings.Contains(err.Error(), "Set GITHUB_TOKEN") {
		t.Errorf("error = %v, want it to say the token in use lacks access", err)
	}
}

// Replace has to work while the old binary is running, which is why it
// renames rather than writing in place. It also has to keep the mode: a
// binary that comes back non-executable is worse than no upgrade.
func TestReplaceKeepsTheModeAndSwapsContent(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "vllm-helper")
	if err := os.WriteFile(path, []byte("old"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := Replace(path, []byte("new")); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "new" {
		t.Errorf("content = %q, want the new binary", got)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0o111 == 0 {
		t.Errorf("mode = %v, want it to stay executable", info.Mode().Perm())
	}
	// The temporary file must not be left behind next to the binary.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("directory holds %d files, want just the binary", len(entries))
	}
}

// A symlinked install (say a name on PATH pointing at a versioned path)
// should have its target replaced, not be turned into a regular file.
func TestReplaceFollowsASymlink(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	target := filepath.Join(dir, "vllm-helper-real")
	link := filepath.Join(dir, "vllm-helper")
	if err := os.WriteFile(target, []byte("old"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := Replace(link, []byte("new")); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	info, err := os.Lstat(link)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Errorf("the symlink was replaced instead of its target (mode %v)", info.Mode())
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "new" {
		t.Errorf("target content = %q, want the new binary", got)
	}
}
