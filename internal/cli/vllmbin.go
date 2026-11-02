package cli

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
)

// findVLLM returns the vllm executable setup should record, or "" when there
// is none. One on PATH wins; otherwise the places this platform's installers
// put one are tried, since those are often not on PATH.
func findVLLM() string {
	if p, err := exec.LookPath("vllm"); err == nil {
		return stableVLLMPath(p)
	}
	for _, p := range vllmCandidates() {
		if isExecutable(p) {
			return p
		}
	}
	return ""
}

// cellarPath matches a file inside a versioned Homebrew keg:
// <prefix>/Cellar/<formula>/<version>/<rest>.
var cellarPath = regexp.MustCompile(`^(.*)/Cellar/([^/]+)/[^/]+/(.+)$`)

// stableVLLMPath resolves symlinks so VLLM_BIN names the real environment, but
// keeps Homebrew's opt link in place of a versioned Cellar path. The Cellar
// path is deleted by the first 'brew upgrade' and 'brew cleanup', which would
// leave the service pointing at nothing; <prefix>/opt/<formula> always tracks
// the installed version, and the formula's interpreter sits beside it.
func stableVLLMPath(p string) string {
	resolved, err := filepath.EvalSymlinks(p)
	if err != nil {
		return p
	}
	if m := cellarPath.FindStringSubmatch(resolved); m != nil {
		opt := filepath.Join(m[1], "opt", m[2], m[3])
		if isExecutable(opt) {
			return opt
		}
	}
	return resolved
}

func isExecutable(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir() && info.Mode()&0o111 != 0
}
