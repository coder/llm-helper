package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// repoIDPattern is a Hugging Face repo id as a profile's MODEL names one.
var repoIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)

var commitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)

// lastEnv is the last value env gives name, as exec would pass it.
func lastEnv(env []string, name string) (string, bool) {
	value, set := "", false
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == name {
			value, set = v, true
		}
	}
	return value, set
}

func truthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// hubCacheDir is the directory huggingface_hub reads repos from for env:
// HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub.
func hubCacheDir(env []string) string {
	if v, ok := lastEnv(env, "HF_HUB_CACHE"); ok && v != "" {
		return v
	}
	if v, ok := lastEnv(env, "HF_HOME"); ok && v != "" {
		return filepath.Join(v, "hub")
	}
	if v, ok := lastEnv(env, "XDG_CACHE_HOME"); ok && v != "" {
		return filepath.Join(v, "huggingface", "hub")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// argValue reads a "--name value" or "--name=value" flag from a profile's
// EXTRA_ARGS, "" when it is not there.
func argValue(extra, name string) string {
	fields := strings.Fields(extra)
	for i, f := range fields {
		if f == name && i+1 < len(fields) {
			return strings.Trim(fields[i+1], `"'`)
		}
		if v, ok := strings.CutPrefix(f, name+"="); ok {
			return strings.Trim(v, `"'`)
		}
	}
	return ""
}

// offlineCacheProblem explains why a server started with env would not find
// model in the Hugging Face cache, or returns "". It only speaks when the
// server cannot fetch (HF_HUB_OFFLINE): vLLM then resolves the repo's
// revision -- "main" unless EXTRA_ARGS names one -- through refs/<revision>
// and loads snapshots/<commit>. 'hf download --revision <commit>' and a copy
// of a snapshot directory both leave refs/main out, and the server then
// fails minutes into its start with LocalEntryNotFoundError, again on every
// restart.
func offlineCacheProblem(env []string, model, extraArgs string) string {
	if v, _ := lastEnv(env, "HF_HUB_OFFLINE"); !truthy(v) {
		return ""
	}
	if !repoIDPattern.MatchString(model) {
		return "" // a local path, or a GGUF selector: not a cache lookup
	}
	if _, err := os.Stat(model); err == nil {
		return ""
	}
	hub := hubCacheDir(env)
	repoDir := filepath.Join(hub, "models--"+strings.ReplaceAll(model, "/", "--"))
	rev := argValue(extraArgs, "--revision")
	if rev == "" {
		rev = "main"
	}
	snapshots, _ := filepath.Glob(filepath.Join(repoDir, "snapshots", "*"))
	var have []string
	for _, s := range snapshots {
		have = append(have, filepath.Base(s))
	}
	if len(have) == 0 {
		return fmt.Sprintf("HF_HUB_OFFLINE is set and %s is not in the Hugging Face cache at %s, so the server cannot load it. "+
			"Download it first (hf download %s), or unset HF_HUB_OFFLINE in config.env.", model, hub, model)
	}
	commit := rev
	if !commitPattern.MatchString(rev) {
		b, err := os.ReadFile(filepath.Join(repoDir, "refs", rev))
		if err != nil {
			fix := ""
			if len(have) == 1 {
				fix = fmt.Sprintf(" If that snapshot is the one to serve: printf %%s %s > %s", have[0], filepath.Join(repoDir, "refs", rev))
			}
			return fmt.Sprintf("HF_HUB_OFFLINE is set and the cache has %s at snapshot(s) %s but no refs/%s, so the server would fail to "+
				"resolve %q offline ('hf download --revision <commit>' does not write refs/main).%s Or add --revision <commit> to EXTRA_ARGS.",
				model, strings.Join(have, ", "), rev, rev, fix)
		}
		commit = strings.TrimSpace(string(b))
	}
	if fi, err := os.Stat(filepath.Join(repoDir, "snapshots", commit)); err != nil || !fi.IsDir() {
		return fmt.Sprintf("HF_HUB_OFFLINE is set and refs/%s of %s names %s, which is not among the cached snapshots (%s).",
			rev, model, commit, strings.Join(have, ", "))
	}
	return ""
}
