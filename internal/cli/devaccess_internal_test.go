package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/detect"
)

// A user outside the render group gets a server that finds no GPU, with no
// permission error to say why, so doctor names the nodes and the fix. Only
// nodes that exist count, and only on a machine with an amdgpu GPU.
func TestAMDDeviceWarnings(t *testing.T) {
	dev := t.TempDir()
	for _, n := range []string{"kfd", "dri/renderD128", "dri/card0"} {
		p := filepath.Join(dev, n)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, nil, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	amd := []detect.Accelerator{{Vendor: "amd", VendorName: "AMD", PCIID: "0x1002:0x1586", Kind: "gpu", Driver: "amdgpu"}}
	none := func(string) bool { return false }
	all := func(string) bool { return true }

	w := amdDeviceWarnings(amd, dev, none)
	if len(w) != 1 {
		t.Fatalf("no access: %q, want one warning", w)
	}
	for _, want := range []string{filepath.Join(dev, "kfd"), filepath.Join(dev, "dri", "renderD128"), "sudo usermod -aG render,video $USER"} {
		if !strings.Contains(w[0], want) {
			t.Errorf("warning %q does not name %q", w[0], want)
		}
	}
	if strings.Contains(w[0], "card0") {
		t.Errorf("warning %q names a node servers do not open", w[0])
	}

	if w := amdDeviceWarnings(amd, dev, all); len(w) != 0 {
		t.Errorf("full access: %q", w)
	}
	onlyRender := func(p string) bool { return strings.Contains(p, "renderD") }
	if w := amdDeviceWarnings(amd, dev, onlyRender); len(w) != 1 || strings.Contains(w[0], "renderD128") || !strings.Contains(w[0], "kfd") {
		t.Errorf("render node only: %q, want /dev/kfd alone", w)
	}
	if w := amdDeviceWarnings(amd, t.TempDir(), none); len(w) != 0 {
		t.Errorf("no device nodes at all: %q", w)
	}
	nvidia := []detect.Accelerator{{Vendor: "nvidia", VendorName: "NVIDIA", PCIID: "0x10de:0x2e12", Kind: "gpu", Driver: "nvidia"}}
	if w := amdDeviceWarnings(nvidia, dev, none); len(w) != 0 {
		t.Errorf("no amdgpu GPU: %q", w)
	}
}
