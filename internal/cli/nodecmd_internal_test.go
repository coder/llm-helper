package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/control"
	"github.com/coder/vllm-helper/internal/nodes"
)

// 'node ls' showed only the HTTP host, so a node whose SSH target was wrong,
// or silently derived from --host, looked fine until a launch failed on it.
func TestNodeListShowsTheSSHTargetUsed(t *testing.T) {
	t.Setenv("USER", "coder")
	var out bytes.Buffer
	printNodes(&out, []nodes.Node{
		{Name: "spark1", Host: "http://192.168.1.125:8001"},
		{Name: "spark2", Host: "http://192.168.1.234:8001", SSH: "coder@10.44.0.2",
			Caps: nodes.Capabilities{Accelerator: "cuda", GPU: "NVIDIA GB10", GPUs: 1, PoolBytes: 130e9}},
	})
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want a header and two nodes:\n%s", len(lines), out.String())
	}
	for i, want := range [][]string{
		{"NAME", "HOST", "SSH", "HARDWARE"},
		{"spark1", "http://192.168.1.125:8001", "coder@192.168.1.125", "(from", "--host)", "-"},
		{"spark2", "http://192.168.1.234:8001", "coder@10.44.0.2", "cuda,", "NVIDIA", "GB10,", "130", "GB"},
	} {
		if got := strings.Fields(lines[i]); strings.Join(got, " ") != strings.Join(want, " ") {
			t.Errorf("line %d = %q, want fields %q", i, lines[i], want)
		}
	}
}

// The server's CUDA_VISIBLE_DEVICES is assembled from layers, and the probe
// has to read the same one: the last file that sets it, else this process's.
// Set-but-empty hides every GPU, so it must not fall through to the next layer.
func TestServerCUDAVisible(t *testing.T) {
	dir := t.TempDir()
	envFile := func(name, body string) *config.EnvFile {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		f, err := config.ParseEnvFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	cfgFile := envFile("config.env", "PORT=8001\nCUDA_VISIBLE_DEVICES=0,1\n")
	profile := envFile("profile.env", "MODEL=x\nCUDA_VISIBLE_DEVICES=\n")
	bare := envFile("bare.env", "MODEL=x\n")

	t.Setenv("CUDA_VISIBLE_DEVICES", "3")
	for _, tt := range []struct {
		name    string
		files   []*config.EnvFile
		want    string
		wantSet bool
	}{
		{name: "process env when no file sets it", files: []*config.EnvFile{bare, nil}, want: "3", wantSet: true},
		{name: "config.env over the process", files: []*config.EnvFile{cfgFile, bare}, want: "0,1", wantSet: true},
		{name: "a later empty value wins", files: []*config.EnvFile{cfgFile, profile}, want: "", wantSet: true},
	} {
		got, set := control.CUDAVisible(tt.files...)
		if got != tt.want || set != tt.wantSet {
			t.Errorf("%s: got (%q, %v), want (%q, %v)", tt.name, got, set, tt.want, tt.wantSet)
		}
	}
}
