package cli

import (
	"strconv"
	"testing"

	"github.com/coder/llm-helper/internal/detect"
)

func TestClampMemUtilization(t *testing.T) {
	t.Parallel()

	const gb = 1024
	tests := []struct {
		name             string
		want             string
		totalMB, availMB int
		value            string
		msg              bool
	}{
		{"fits already", "0.30", 16 * gb, 8 * gb, "0.30", false},
		{"exactly at the limit", "0.85", 16 * gb, 16 * gb, "0.85", false},
		// The issue #13 case: the Linux default on a 15 GiB CPU-only box
		// with about half the RAM in use.
		{"linux default on a busy CPU box", "0.85", 15 * gb, 8 * gb, "0.45", true},
		{"almost nothing free warns but keeps the value", "0.85", 16 * gb, 512, "0.85", true},
		{"unknown total passes through", "0.85", 0, 8 * gb, "0.85", false},
		{"unknown available passes through", "0.85", 16 * gb, 0, "0.85", false},
		{"unparseable passes through", "lots", 16 * gb, 1 * gb, "lots", false},
		{"zero passes through", "0", 16 * gb, 1 * gb, "0", false},
		{"negative passes through", "-1", 16 * gb, 1 * gb, "-1", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			value, msg := clampMemUtilization(tt.want, tt.totalMB, tt.availMB)
			if value != tt.value {
				t.Errorf("value = %q, want %q", value, tt.value)
			}
			if (msg != "") != tt.msg {
				t.Errorf("msg = %q, want a message: %v", msg, tt.msg)
			}
		})
	}
}

func TestCapMemUtilization(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		want  string
		value string
		msg   bool
	}{
		// The DGX Spark default, and what vLLM's own guidance recommends.
		{"the recommended value is left alone", "0.85", "0.85", false},
		{"below the cap passes through", "0.70", "0.70", false},
		// A number copied from a discrete-GPU recipe.
		{"above the cap comes down", "0.95", "0.85", true},
		{"unparseable passes through", "lots", "lots", false},
		{"zero passes through", "0", "0", false},
		{"negative passes through", "-1", "-1", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			value, msg := capMemUtilization(tt.want, maxUnifiedMemUtilization)
			if value != tt.value {
				t.Errorf("value = %q, want %q", value, tt.value)
			}
			if (msg != "") != tt.msg {
				t.Errorf("msg = %q, want a message: %v", msg, tt.msg)
			}
		})
	}
}

// Which rule each kind of machine gets. The unified rows are the regression
// that matters: an earlier version clamped them against free memory, and on a
// DGX Spark serving a 68 GB model that turned 0.85 into 0.06 -- a KV cache far
// too small to run. availMB is deliberately tiny in those rows, so anything
// that starts consulting it again fails here.
func TestPickMemFit(t *testing.T) {
	t.Parallel()

	const gb = 1024
	spark := detect.Hardware{TotalMemBytes: 130 << 30, GPUName: "NVIDIA GB10", Unified: true}
	tests := []struct {
		name     string
		hw       detect.Hardware
		availMB  int
		want     string
		value    string
		msg      bool
		kvPinned bool
	}{
		{name: "spark keeps the recommended value", hw: spark, availMB: 10 * gb, want: "0.85", value: "0.85"},
		{name: "spark keeps a lower value", hw: spark, availMB: 500, want: "0.70", value: "0.70"},
		{name: "spark caps a discrete-GPU recipe", hw: spark, availMB: 10 * gb, want: "0.95", value: "0.85", msg: true},
		{
			// With the cache pinned, this flag is only a ceiling on what
			// vLLM may touch, and the measured one is both roomier and
			// better founded than the fixed 0.85 above.
			name: "a pinned KV cache earns the measured ceiling",
			hw:   spark, availMB: 10 * gb, want: "0.99", value: "0.92",
			msg: true, kvPinned: true,
		},
		{
			name: "unified wins over the GPU name",
			// Both branches match; taking the discrete one would leave the
			// request unbounded on the hardware that most needs a bound.
			hw: spark, availMB: 100, want: "0.95", value: "0.85", msg: true,
		},
		{
			name:  "discrete VRAM is left alone",
			hw:    detect.Hardware{TotalMemBytes: 64 << 30, GPUName: "NVIDIA RTX A6000", GPUMemGB: 48},
			want:  "0.95",
			value: "0.95",
		},
		{
			// Issue #13: no GPU means the CPU backend, where the flag really
			// does reserve system RAM and startup aborts without it.
			name:    "cpu backend still clamps to what is free",
			hw:      detect.Hardware{TotalMemBytes: 15 << 30},
			availMB: 8 * gb,
			want:    "0.85",
			value:   "0.45",
			msg:     true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			value, msg := pickMemFit(tt.hw, tt.availMB, tt.want, tt.kvPinned)
			if value != tt.value {
				t.Errorf("value = %q, want %q", value, tt.value)
			}
			if (msg != "") != tt.msg {
				t.Errorf("msg = %q, want a message: %v", msg, tt.msg)
			}
		})
	}
}

// Whatever we hand back has to survive the round trip into vLLM's flag, and
// must never ask for more memory than the caller did.
func TestClampMemUtilizationNeverRaisesTheRequest(t *testing.T) {
	t.Parallel()

	const gb = 1024
	for _, avail := range []int{256, 1 * gb, 4 * gb, 8 * gb, 15 * gb} {
		value, _ := clampMemUtilization("0.85", 16*gb, avail)
		// Compare as a number: "0.9" sorts after "0.85" as text but is
		// the larger value either way, while "0.850" sorts after it and
		// is not, so string ordering can't answer this question.
		got, err := strconv.ParseFloat(value, 64)
		if err != nil {
			t.Errorf("clampMemUtilization(0.85, %d MiB free) = %q, which vLLM can't parse as a float: %v", avail, value, err)
			continue
		}
		if got > 0.85 {
			t.Errorf("clampMemUtilization(0.85, %d MiB free) = %q, which is higher than asked", avail, value)
		}
	}
}
