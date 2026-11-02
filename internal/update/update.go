// Package update finds and installs newer vllm-helper binaries from GitHub
// releases.
//
// There are two channels. Stable is the published releases, cut by tagging.
// Dev is a single prerelease named "dev" that CI rebuilds on every push to
// main, so it moves without a version number of its own -- which is why a
// dev build is identified by the commit it was built from rather than by
// comparing versions. A dev version string sorts *below* the release it
// follows under semver ("0.3.1-dev.2" < "0.3.1"), so ordering them would
// report an upgrade as a downgrade.
package update

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Channel is where upgrades come from.
type Channel string

const (
	Stable Channel = "stable"
	Dev    Channel = "dev"
)

// DevTag is the moving tag CI republishes for every commit on main.
const DevTag = "dev"

// ParseChannel validates a channel name.
func ParseChannel(s string) (Channel, error) {
	switch Channel(s) {
	case Stable:
		return Stable, nil
	case Dev:
		return Dev, nil
	}
	return "", fmt.Errorf("unknown channel %q: use %q or %q", s, Stable, Dev)
}

// apiBase is the GitHub API root, swapped by tests.
var apiBase = "https://api.github.com"

// repo is the project these releases come from.
const repo = "coder/vllm-helper"

// maxDownload caps what we'll pull over the network. The real archives are a
// few megabytes; this is only here so a wrong URL can't fill a disk.
const maxDownload = 200 << 20

var httpClient = &http.Client{Timeout: 60 * time.Second}

// Release is the part of a GitHub release we act on.
type Release struct {
	TagName    string  `json:"tag_name"`
	Name       string  `json:"name"`
	Body       string  `json:"body"`
	Prerelease bool    `json:"prerelease"`
	Assets     []Asset `json:"assets"`
}

// Asset is one published file.
//
// Both URLs are kept because they authenticate differently: the browser URL
// is a redirect to public object storage and takes no credentials, while the
// API URL is the only one that serves an asset from a private repository.
type Asset struct {
	Name        string `json:"name"`
	APIURL      string `json:"url"`
	DownloadURL string `json:"browser_download_url"`
	Size        int64  `json:"size"`
}

// url picks the endpoint to fetch this asset from.
func (a Asset) url(token string) (string, string) {
	if token != "" && a.APIURL != "" {
		// The API returns JSON about the asset unless asked for bytes.
		return a.APIURL, "application/octet-stream"
	}
	return a.DownloadURL, ""
}

// Latest fetches the release a channel currently points at.
func Latest(ch Channel, token string) (*Release, error) {
	path := "/repos/" + repo + "/releases/latest"
	if ch == Dev {
		// The dev release is a prerelease, and /releases/latest ignores
		// prereleases, so it has to be fetched by its tag.
		path = "/repos/" + repo + "/releases/tags/" + DevTag
	}
	resp, err := request(apiBase+path, "application/vnd.github+json", token)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		// GitHub answers 404 both for "there is no such release" and for
		// "you can't see this repository", so say both: while the repo is
		// private, an upgrade needs a token that can read it.
		what := "no releases published yet"
		if ch == Dev {
			what = fmt.Sprintf("no %q release published yet", DevTag)
		}
		return nil, fmt.Errorf("%s -- or %s isn't readable with the credentials in use.%s",
			what, repo, tokenHint(token))
	}
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, fmt.Errorf("github API: HTTP %d.%s", resp.StatusCode, tokenHint(token))
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("github API: HTTP %d", resp.StatusCode)
	}
	var r Release
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&r); err != nil {
		return nil, err
	}
	return &r, nil
}

// Commit reports the commit a dev release was built from.
//
// CI puts it in the release body because a moving tag can't carry it: the
// "dev" tag is repointed on every build, so the tag name says nothing about
// which commit is inside. Comparing this against the running binary's stamp
// is how a dev build knows whether it is current.
func (r *Release) Commit() string {
	const marker = "commit:"
	for line := range strings.SplitSeq(r.Body, "\n") {
		line = strings.TrimSpace(line)
		if rest, ok := strings.CutPrefix(line, marker); ok {
			return strings.TrimSpace(rest)
		}
	}
	return ""
}

// AssetFor picks the archive for a platform.
//
// It matches the tail of the name rather than rebuilding it, because the
// version in the middle differs between channels (a tag for stable, a
// generated string for dev) and only the platform part is predictable.
func (r *Release) AssetFor(goos, goarch string) (Asset, error) {
	suffix := fmt.Sprintf("_%s_%s.tar.gz", goos, goarch)
	for _, a := range r.Assets {
		if strings.HasSuffix(a.Name, suffix) {
			return a, nil
		}
	}
	return Asset{}, fmt.Errorf("release %s has no build for %s/%s", r.TagName, goos, goarch)
}

// checksums maps filename to hex digest, from a goreleaser checksums.txt.
func (r *Release) checksums(token string) (map[string]string, error) {
	var asset Asset
	for _, a := range r.Assets {
		if a.Name == "checksums.txt" {
			asset = a
			break
		}
	}
	if asset.Name == "" {
		return nil, fmt.Errorf("release %s publishes no checksums.txt", r.TagName)
	}
	url, accept := asset.url(token)
	body, err := get(url, accept, token)
	if err != nil {
		return nil, err
	}
	sums := make(map[string]string)
	for line := range strings.SplitSeq(string(body), "\n") {
		digest, name, ok := strings.Cut(strings.TrimSpace(line), "  ")
		if ok {
			sums[strings.TrimSpace(name)] = digest
		}
	}
	return sums, nil
}

// request issues a GET, authenticating when a token is supplied.
func request(url, accept, token string) (*http.Response, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return httpClient.Do(req)
}

// tokenHint explains how to supply credentials, tailored to whether any were
// found in the first place.
func tokenHint(token string) string {
	if token != "" {
		return " The token in use can't read it: check that it grants access to " + repo + "."
	}
	return " Set GITHUB_TOKEN in the environment, or in config.env, to read a private repository."
}

func get(url, accept, token string) ([]byte, error) {
	resp, err := request(url, accept, token)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("downloading %s: HTTP %d.%s", url, resp.StatusCode, tokenHint(token))
	}
	return io.ReadAll(io.LimitReader(resp.Body, maxDownload))
}

// Download fetches the platform's archive and returns the binary inside it,
// having checked it against the release's published checksum.
func Download(r *Release, goos, goarch, token string) ([]byte, error) {
	asset, err := r.AssetFor(goos, goarch)
	if err != nil {
		return nil, err
	}
	sums, err := r.checksums(token)
	if err != nil {
		return nil, err
	}
	want, ok := sums[asset.Name]
	if !ok {
		return nil, fmt.Errorf("checksums.txt has no entry for %s", asset.Name)
	}

	url, accept := asset.url(token)
	archive, err := get(url, accept, token)
	if err != nil {
		return nil, err
	}
	// Verify before unpacking, so a corrupted or substituted archive is
	// never parsed, let alone written over the running binary.
	if got := hex.EncodeToString(sha256Sum(archive)); got != want {
		return nil, fmt.Errorf("%s failed its checksum: got %s, want %s", asset.Name, got, want)
	}
	return binaryFromArchive(archive)
}

func sha256Sum(b []byte) []byte {
	sum := sha256.Sum256(b)
	return sum[:]
}

// binaryNames are the names a release ships the executable under. Releases
// carry it as both while installs from before the rename are still upgrading,
// and the two files are the same build, so either one will do.
var binaryNames = map[string]bool{"llm-helper": true, "vllm-helper": true}

// binaryFromArchive pulls the executable out of a release tarball, which also
// carries LICENSE and README.md.
func binaryFromArchive(archive []byte) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewReader(archive))
	if err != nil {
		return nil, fmt.Errorf("reading archive: %w", err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading archive: %w", err)
		}
		if h.Typeflag != tar.TypeReg || !binaryNames[filepath.Base(h.Name)] {
			continue
		}
		return io.ReadAll(io.LimitReader(tr, maxDownload))
	}
	return nil, errors.New("the archive contains no llm-helper binary")
}

// Replace swaps the binary at path for new content.
//
// The replacement is written beside the target and renamed over it: a rename
// is atomic and, unlike writing in place, is allowed while the old binary is
// executing. An interrupted upgrade therefore leaves the old binary intact
// rather than a half-written one.
func Replace(path string, binary []byte) error {
	// Follow symlinks so we replace the real file rather than turning a
	// link into a binary.
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".vllm-helper-upgrade-*")
	if err != nil {
		return fmt.Errorf("%w: can't write to %s", err, dir)
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.Write(binary); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	// Keep whatever mode the existing binary had, defaulting to the usual
	// executable bits for a fresh install.
	mode := os.FileMode(0o755)
	if info, err := os.Stat(path); err == nil {
		mode = info.Mode().Perm()
	}
	if err := os.Chmod(tmp.Name(), mode); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
