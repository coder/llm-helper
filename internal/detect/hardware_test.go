//go:build linux || darwin

package detect_test

import (
	"testing"

	"github.com/coder/llm-helper/internal/detect"
)

// The budget drives whether `search` calls a model too big, so a zero here
// would silently rule out every model on the machine.
//
// Asserted on the byte count, which is what the fit check divides: BudgetGB is
// the display rounding and is legitimately 0 on a machine whose real budget is
// under a gigabyte, which describes a small CI runner.
func TestProbeReportsThisMachine(t *testing.T) {
	t.Parallel()

	hw := detect.Probe()
	if hw.TotalMemGB <= 0 {
		t.Errorf("TotalMemGB = %d, want the real system memory", hw.TotalMemGB)
	}
	if hw.BudgetBytes <= 0 {
		t.Errorf("BudgetBytes = %d on a %d GB machine, want a usable weights budget",
			hw.BudgetBytes, hw.TotalMemGB)
	}
	if hw.BudgetBytes >= hw.TotalMemBytes {
		t.Errorf("BudgetBytes = %d, which is the whole machine (%d)", hw.BudgetBytes, hw.TotalMemBytes)
	}
	if hw.BudgetReason == "" {
		t.Error("BudgetReason should explain the budget to the user")
	}
	if hw.GPUMemBytes > 0 && !hw.Unified && hw.BudgetBytes > hw.GPUMemBytes {
		t.Errorf("BudgetBytes = %d exceeds the %d bytes of discrete VRAM it has to fit in",
			hw.BudgetBytes, hw.GPUMemBytes)
	}
}

// fitMemUtilization divides by this, so zero would disable the clamp that
// keeps vLLM's CPU backend from refusing to start.
func TestAvailableMemMB(t *testing.T) {
	t.Parallel()

	avail := detect.AvailableMemMB()
	if avail <= 0 {
		t.Errorf("AvailableMemMB = %d, want the memory that's actually free", avail)
	}
	// TotalMemGB is whole gigabytes, so allow one for the truncation.
	if ceiling := (detect.Probe().TotalMemGB + 1) * 1024; avail > ceiling {
		t.Errorf("AvailableMemMB = %d, more than the %d MB the machine has", avail, ceiling)
	}
}
