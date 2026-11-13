package cli

import (
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/detect"
)

// The figures are a 128 GB Ryzen AI Max at the stock ttm pages_limit and at
// the one its 103 GB model needed, whose 29360128 pages are what doctor
// suggests.
func TestDoctorGTTLines(t *testing.T) {
	halo := func(gtt int64) detect.Hardware {
		return detect.FromReadings(detect.Readings{SchemaVersion: 1, OS: "linux", MemTotalBytes: 134314070016, PCI: []detect.PCIDevice{
			{Vendor: "0x1002", Device: "0x1586", Class: "0x038000", Driver: "amdgpu", VRAMBytes: 536870912, GTTBytes: gtt},
		}})
	}

	stock := doctorGTTLines(halo(101139496960), 4096)
	if len(stock) != 3 {
		t.Fatalf("stock limit: %q, want the limit and how to raise it", stock)
	}
	if !strings.Contains(stock[0], "101 GB the GPU can map") || !strings.Contains(stock[0], "536 MB carve-out") {
		t.Errorf("limit line = %q", stock[0])
	}
	if !strings.Contains(stock[1], "raise it to 112 GiB with 'options ttm pages_limit=29360128' in /etc/modprobe.d/ttm.conf") ||
		!strings.Contains(stock[2], "update-initramfs -u") {
		t.Errorf("raise lines = %q", stock[1:])
	}

	if raised := doctorGTTLines(halo(120259084288), 4096); len(raised) != 1 || !strings.Contains(raised[0], "120 GB the GPU can map") {
		t.Errorf("raised limit: %q, want only the limit", raised)
	}
	if lines := doctorGTTLines(detect.Hardware{TotalMemBytes: 128e9, GPUName: "NVIDIA GB10"}, 4096); lines != nil {
		t.Errorf("a GPU without GTT: %q", lines)
	}
}
