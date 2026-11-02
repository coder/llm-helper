package vllm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseMetal(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		out  string
		want Metal
	}{{
		name: "vllm-metal with a working set",
		out:  "some log line\nVLLM_HELPER_METAL {\"installed\": true, \"working_set_bytes\": 239143780352}\n",
		want: Metal{Installed: true, WorkingSetBytes: 239143780352},
	}, {
		name: "vllm-metal that honors --kv-cache-memory-bytes",
		out:  "VLLM_HELPER_METAL {\"installed\": true, \"working_set_bytes\": 239143780352, \"fixed_kv_cache\": true}\n",
		want: Metal{Installed: true, WorkingSetBytes: 239143780352, FixedKVCache: true},
	}, {
		name: "plain vLLM",
		out:  "VLLM_HELPER_METAL {\"installed\": false, \"working_set_bytes\": 0}\n",
		want: Metal{},
	}, {
		name: "MLX could not be asked",
		out:  "VLLM_HELPER_METAL {\"installed\": true, \"working_set_bytes\": 0}\n",
		want: Metal{Installed: true},
	}, {
		name: "no marker",
		out:  "Traceback (most recent call last):\n",
		want: Metal{},
	}, {
		name: "garbage after the marker",
		out:  "VLLM_HELPER_METAL {not json\n",
		want: Metal{},
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := parseMetal([]byte(tt.out)); got != tt.want {
				t.Errorf("parseMetal() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestParseMetalInstalled(t *testing.T) {
	t.Parallel()

	for out, want := range map[string]bool{
		"VLLM_HELPER_METAL_INSTALLED true\n":            true,
		"a warning\nVLLM_HELPER_METAL_INSTALLED true\n": true,
		"VLLM_HELPER_METAL_INSTALLED false\n":           false,
		"VLLM_HELPER_METAL_INSTALLED maybe\n":           false,
		"Traceback (most recent call last):\n":          false,
		"VLLM_HELPER_METAL {\"installed\": true}\n":     false,
	} {
		if got := parseMetalInstalled([]byte(out)); got != want {
			t.Errorf("parseMetalInstalled(%q) = %v, want %v", out, got, want)
		}
	}
}

// Homebrew links bin/vllm into the formula's private environment, next to
// Homebrew's own python3. The interpreter that owns vLLM is the one beside the
// link's target.
func TestPythonBesideFollowsASymlinkedVLLM(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	envBin := filepath.Join(root, "libexec", "bin")
	linkBin := filepath.Join(root, "bin")
	for _, d := range []string{envBin, linkBin} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, f := range []string{filepath.Join(envBin, "vllm"), filepath.Join(envBin, "python3"), filepath.Join(linkBin, "python3")} {
		if err := os.WriteFile(f, nil, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	link := filepath.Join(linkBin, "vllm")
	if err := os.Symlink(filepath.Join("..", "libexec", "bin", "vllm"), link); err != nil {
		t.Fatal(err)
	}

	want, err := filepath.EvalSymlinks(filepath.Join(envBin, "python3"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := filepath.EvalSymlinks(PythonBeside(link))
	if err != nil {
		t.Fatalf("PythonBeside(%s) = %q: %v", link, PythonBeside(link), err)
	}
	if got != want {
		t.Errorf("PythonBeside(%s) = %s, want the environment's %s", link, got, want)
	}

	// A venv's vllm is a plain file, and its python sits beside it.
	if got := PythonBeside(filepath.Join(envBin, "vllm")); got != filepath.Join(envBin, "python3") {
		t.Errorf("PythonBeside(venv vllm) = %s, want %s", got, filepath.Join(envBin, "python3"))
	}
}
