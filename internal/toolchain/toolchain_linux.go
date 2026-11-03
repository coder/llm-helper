package toolchain

import (
	"os"
	"path/filepath"
	"strings"
)

// Gather checks the build tools vLLM's JIT kernel compilation needs (Triton,
// FlashInfer). Which one bites depends on the first model served, so checking
// up front beats discovering them one crash at a time.
//
// vllmBin is the vllm the server runs, or "" when none is configured. Python
// headers are looked for where that venv's own interpreter expects them, since
// that is the interpreter Triton compiles against: a uv-managed Python carries
// its own headers, and a system Python's come from its -dev package for its
// own version, not a version guessed from the distribution.
func Gather(vllmBin string) []Check {
	return []Check{
		{Name: "ninja", Package: "ninja-build", OK: hasBins("ninja")},
		{Name: "gcc/g++", Package: "build-essential", OK: hasBins("gcc", "g++")},
		pythonHeaders(vllmBin),
	}
}

// Installer is the command and leading args that install packages here.
func Installer() (string, []string) {
	if !hasBins("apt-get") {
		return "", nil
	}
	return "sudo", []string{"apt-get", "install", "-y"}
}

// pythonQuery prints an interpreter's header directory and the Debian
// package that provides headers for its version.
const pythonQuery = `import sys, sysconfig; print(sysconfig.get_paths()["include"]); print(f"python{sys.version_info.major}.{sys.version_info.minor}-dev")`

func pythonHeaders(vllmBin string) Check {
	c := Check{Name: "Python.h", Package: "python3-dev"}
	python := "python3"
	if vllmBin != "" {
		if p := filepath.Join(filepath.Dir(vllmBin), "python"); isExecutable(p) {
			python = p
		}
	}
	out, err := output(python, "-c", pythonQuery)
	if err != nil {
		// No interpreter to ask: any installed headers will do.
		found, _ := filepath.Glob("/usr/include/python3*/Python.h")
		c.OK = len(found) > 0
		return c
	}
	lines := strings.Split(out, "\n")
	if len(lines) >= 2 {
		c.Package = strings.TrimSpace(lines[1])
	}
	_, err = os.Stat(filepath.Join(strings.TrimSpace(lines[0]), "Python.h"))
	c.OK = err == nil
	return c
}

func isExecutable(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir() && st.Mode()&0o111 != 0
}
