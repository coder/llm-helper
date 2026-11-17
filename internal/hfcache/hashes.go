package hfcache

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// hashesDir is the folder in a repo's cache dir where an export records the
// hash of each file it wrote, since a cache without links has no blob names
// to carry them.
const hashesDir = "llm-helper-hashes"

type hashEntry struct {
	Size int64  `json:"size"`
	Hash string `json:"hash"`
}

type hashList struct {
	FormatVersion int                  `json:"format_version"`
	Files         map[string]hashEntry `json:"files"`
}

func hashesPath(repoDir, rev string) string {
	return filepath.Join(repoDir, hashesDir, rev+".json")
}

// readHashes is the recorded hashes for rev, empty when there are none.
func readHashes(repoDir, rev string) map[string]hashEntry {
	out := map[string]hashEntry{}
	raw, err := os.ReadFile(hashesPath(repoDir, rev))
	if err != nil {
		return out
	}
	var l hashList
	if json.Unmarshal(raw, &l) == nil {
		for p, e := range l.Files {
			if hexPattern.MatchString(e.Hash) {
				out[p] = e
			}
		}
	}
	return out
}

func writeHashes(repoDir, rev string, files map[string]hashEntry) error {
	if len(files) == 0 {
		return nil
	}
	raw, err := json.MarshalIndent(hashList{FormatVersion: 1, Files: files}, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(hashesPath(repoDir, rev), append(raw, '\n'))
}
