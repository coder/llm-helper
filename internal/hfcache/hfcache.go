// Package hfcache locates, reads and writes Hugging Face hub caches, so models
// can move between machines on a drive without a download.
//
// It speaks the layout huggingface_hub, vLLM, mlx-lm and llama.cpp share:
// models--org--name/{blobs, snapshots/<commit>, refs/main, trees/<commit>.json}.
package hfcache

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Env looks up one environment variable, "" when unset.
type Env func(key string) string

// OSEnv reads the process environment.
func OSEnv(key string) string { return os.Getenv(key) }

// Dir is the hub cache directory a server reads, given the environment it
// starts with. The first of extra that is set wins, then HF_HUB_CACHE,
// HUGGINGFACE_HUB_CACHE, $HF_HOME/hub and $XDG_CACHE_HOME/huggingface/hub, and
// last ~/.cache/huggingface/hub under home.
//
// huggingface_hub expands ~ and $HOME in its own variables, so they are
// expanded here. A variable in extra is the engine's, and is read as written.
func Dir(env Env, home string, extra ...string) string {
	get := func(k string) string { return expandHome(strings.TrimSpace(env(k)), home) }
	for _, k := range extra {
		if v := strings.TrimSpace(env(k)); v != "" {
			return v
		}
	}
	for _, k := range []string{"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"} {
		if v := get(k); v != "" {
			return v
		}
	}
	if v := get("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	if v := get("XDG_CACHE_HOME"); v != "" {
		return filepath.Join(v, "huggingface", "hub")
	}
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// Absolute reports whether dir names one place wherever it is read from: a
// relative path, or one with ~ or a variable the server will not expand, does
// not.
func Absolute(dir string) bool {
	return filepath.IsAbs(dir) && !strings.HasPrefix(dir, "~") && !strings.Contains(dir, "$")
}

// expandHome resolves a leading ~ and $HOME, as huggingface_hub does.
func expandHome(v, home string) string {
	switch {
	case v == "~":
		return home
	case strings.HasPrefix(v, "~/"):
		return filepath.Join(home, v[2:])
	}
	v = strings.ReplaceAll(v, "${HOME}", home)
	return strings.ReplaceAll(v, "$HOME", home)
}

// Offline reports whether HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set to a
// true value, as huggingface_hub reads them.
func Offline(env Env) bool {
	for _, k := range []string{"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"} {
		switch strings.ToUpper(strings.TrimSpace(env(k))) {
		case "1", "ON", "YES", "TRUE":
			return true
		}
	}
	return false
}

var (
	repoPattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)
	revisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
	hexPattern      = regexp.MustCompile(`^(?:[0-9a-f]{40}|[0-9a-f]{64})$`)
)

const repoPrefix = "models--"

// ValidRepo reports whether repo is an org/name Hub id.
func ValidRepo(repo string) bool {
	return repoPattern.MatchString(repo) && !strings.Contains(repo, "--") && !strings.Contains(repo, "..") && strings.Trim(repo[strings.IndexByte(repo, '/')+1:], ".") != ""
}

// RepoDir is where a repo lives under a hub cache.
func RepoDir(hub, repo string) string {
	name := repoPrefix + strings.ReplaceAll(repo, "/", "--")
	if strings.Contains(name, "..") || !filepath.IsLocal(name) {
		name = repoPrefix + "invalid"
	}
	return filepath.Join(hub, name)
}

// repoFromDir is the repo a models--org--name folder holds.
func repoFromDir(name string) (string, bool) {
	rest, ok := strings.CutPrefix(name, repoPrefix)
	if !ok {
		return "", false
	}
	org, model, ok := strings.Cut(rest, "--")
	if !ok {
		return "", false
	}
	repo := org + "/" + model
	return repo, ValidRepo(repo)
}

// Snapshot is the directory holding repo's refs/main revision in a hub cache,
// the one an offline load reads.
func Snapshot(hub, repo string) (dir, revision string, ok bool) {
	if !ValidRepo(repo) {
		return "", "", false
	}
	rev, err := mainRevision(RepoDir(hub, repo))
	if err != nil {
		return "", "", false
	}
	dir = filepath.Join(RepoDir(hub, repo), "snapshots", rev)
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		return "", "", false
	}
	return dir, rev, true
}

// mainRevision reads refs/main, or names the one snapshot a repo has when it
// has no ref.
func mainRevision(repoDir string) (string, error) {
	if b, err := os.ReadFile(filepath.Join(repoDir, "refs", "main")); err == nil {
		if rev := strings.TrimSpace(string(b)); revisionPattern.MatchString(rev) {
			return rev, nil
		}
	}
	entries, err := os.ReadDir(filepath.Join(repoDir, "snapshots"))
	if err != nil {
		return "", err
	}
	var revs []string
	for _, e := range entries {
		if e.IsDir() && revisionPattern.MatchString(e.Name()) {
			revs = append(revs, e.Name())
		}
	}
	if len(revs) != 1 {
		return "", os.ErrNotExist
	}
	return revs[0], nil
}
