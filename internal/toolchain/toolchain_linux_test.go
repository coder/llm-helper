package toolchain

import (
	"os"
	"path/filepath"
	"testing"
)

// fakeVenv writes a venv whose python reports include as its header
// directory and version as its version, beside a vllm binary.
func fakeVenv(t *testing.T, include, version string) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\necho '" + include + "'\necho 'python" + version + "-dev'\n"
	if err := os.WriteFile(filepath.Join(bin, "python"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	vllm := filepath.Join(bin, "vllm")
	if err := os.WriteFile(vllm, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return vllm
}

// The headers that matter are the ones the server's own interpreter compiles
// against. The Spark check looked for /usr/include/python3.12 whatever Python
// the venv ran, and recommended python3.12-dev on a 3.11 or a uv-managed one.
func TestPythonHeadersFollowTheVenvInterpreter(t *testing.T) {
	include := t.TempDir()
	vllm := fakeVenv(t, include, "3.11")
	c := pythonHeaders(vllm)
	if c.OK || c.Package != "python3.11-dev" {
		t.Errorf("without Python.h: %+v, want missing with python3.11-dev", c)
	}
	if err := os.WriteFile(filepath.Join(include, "Python.h"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if c := pythonHeaders(vllm); !c.OK {
		t.Errorf("with Python.h in the interpreter's include dir: %+v, want ok", c)
	}
}
