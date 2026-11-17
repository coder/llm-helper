package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testCommit = "3acb09951cff50cda7be0200ebbad2c15a3e63ba"

// cacheWith lays out one repo the way 'hf download --revision <commit>'
// leaves it: a snapshot, and refs/main only when withRef is set.
func cacheWith(t *testing.T, withRef bool) string {
	t.Helper()
	home := t.TempDir()
	repo := filepath.Join(home, "hub", "models--nvidia--Nemotron")
	if err := os.MkdirAll(filepath.Join(repo, "snapshots", testCommit), 0o755); err != nil {
		t.Fatal(err)
	}
	if withRef {
		if err := os.MkdirAll(filepath.Join(repo, "refs"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(repo, "refs", "main"), []byte(testCommit), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return home
}

func TestOfflineCacheProblem(t *testing.T) {
	t.Parallel()
	noRef, withRef := cacheWith(t, false), cacheWith(t, true)
	tests := []struct {
		name  string
		env   []string
		model string
		extra string
		want  string // substring; "" means no problem
	}{
		{"online: never checked", []string{"HF_HOME=" + noRef}, "nvidia/Nemotron", "", ""},
		{"offline, refs/main present", []string{"HF_HOME=" + withRef, "HF_HUB_OFFLINE=1"}, "nvidia/Nemotron", "", ""},
		{"offline, refs/main missing (the GB300 failure)", []string{"HF_HOME=" + noRef, "HF_HUB_OFFLINE=1"}, "nvidia/Nemotron", "", "no refs/main"},
		{"offline, refs/main missing, pinned by --revision", []string{"HF_HOME=" + noRef, "HF_HUB_OFFLINE=1"}, "nvidia/Nemotron", "--trust-remote-code --revision " + testCommit, ""},
		{"offline, not downloaded", []string{"HF_HOME=" + noRef, "HF_HUB_OFFLINE=true"}, "nvidia/Other", "", "not in the Hugging Face cache"},
		{"offline, a local path", []string{"HF_HOME=" + noRef, "HF_HUB_OFFLINE=1"}, "/models/weights", "", ""},
		{"last assignment wins", []string{"HF_HUB_OFFLINE=1", "HF_HOME=" + noRef, "HF_HUB_OFFLINE=0"}, "nvidia/Nemotron", "", ""},
	}
	for _, tt := range tests {
		got := offlineCacheProblem(tt.env, tt.model, tt.extra)
		if tt.want == "" && got != "" || tt.want != "" && !strings.Contains(got, tt.want) {
			t.Errorf("%s: got %q, want %q", tt.name, got, tt.want)
		}
	}
	if got := offlineCacheProblem([]string{"HF_HOME=" + noRef, "HF_HUB_OFFLINE=1"}, "nvidia/Nemotron", ""); !strings.Contains(got, "printf %s "+testCommit) {
		t.Errorf("one snapshot should come with the refs/main fix, got %q", got)
	}
}
