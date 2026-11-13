package detect

import (
	"strings"
	"testing"
)

// Figures below come from vLLM's memory profiler on a DGX Spark (GB10, 121.69
// GiB visible), read out of the service journal. They are the reason the
// constants are what they are, so they belong in the tests rather than only in
// a commit message.
// unifiedTotal is MemTotal from a 128 GB unified-memory machine: 2 GB is
// carved off before the OS sees it, which is why the figure is not round.
const unifiedTotal = 127600592 * 1024

func gib(f float64) float64 { return f * gibibyte }

func bi(f float64) int64 { return int64(f * gibibyte) }

func TestTheBudgetLeavesRoomForWhatVLLMActuallyUses(t *testing.T) {
	// Every start this machine logged, as (weights loaded, total consumed
	// including runtime, activation and CUDA graphs) in GiB. If the budget
	// admits a model, the machine has to be able to hold it.
	runs := []struct {
		name              string
		weights, consumed float64
	}{
		{"opt-125m", 0.24, 1.24},
		{"Qwen3-0.6B", 1.12, 2.86},
		{"Nemotron-30B NVFP4", 18.66, 22.49},
		{"Nemotron-30B BF16", 58.93, 64.34},
		{"Ornith-35B", 65.53, 71.24},
	}
	budget, _ := Hardware{TotalMemBytes: unifiedTotal, Unified: true, CUDA: true}.Budget(0)
	for _, r := range runs {
		if gib(r.weights) > float64(budget) {
			continue // too big for this machine; nothing to check
		}
		// The overhead we predict for it must cover what it really used.
		predicted := gib(r.weights)*(1+activationFraction) + runtimeReserve + graphReserve
		if predicted < gib(r.consumed) {
			t.Errorf("%s: budgeted %.2f GiB of memory for a model that used %.2f GiB",
				r.name, predicted/gibibyte, r.consumed)
		}
		// And it must still leave a KV cache worth having.
		spare := unifiedTotal - gib(r.consumed) - defaultBaseline(unifiedTotal)
		if spare < kvFloor {
			t.Errorf("%s: only %.1f GiB left for KV cache", r.name, spare/gibibyte)
		}
	}
}

func TestTheBudgetFitsInTheMachine(t *testing.T) {
	// A budget that cannot itself be served is the bug this replaced: at the
	// old 70%, a model exactly at budget plus overhead overflowed the box.
	for _, total := range []int64{
		8 * gibibyte, 16 * gibibyte, 32 * gibibyte, 64 * gibibyte, unifiedTotal, 512 * gibibyte,
	} {
		budget, _ := Hardware{TotalMemBytes: total, Unified: true, CUDA: true}.Budget(0)
		w := float64(budget)
		committed := w*(1+activationFraction) + runtimeReserve + graphReserve + defaultBaseline(float64(total))
		if committed > float64(total) {
			t.Errorf("total %.0f GiB: budget %.1f GiB commits %.1f GiB",
				float64(total)/gibibyte, w/gibibyte, committed/gibibyte)
		}
	}
}

func TestABusyMachineDoesNotGetABudgetOfNothing(t *testing.T) {
	// Probing while our own service is up reads 111 of 121 GiB in use. Taken
	// as an OS baseline that would leave nothing, and every model on the Hub
	// would be reported as too big -- on the machine that was serving one.
	busy, _ := Hardware{TotalMemBytes: unifiedTotal, Unified: true, CUDA: true}.Budget(111 * gibibyte)
	idle, _ := Hardware{TotalMemBytes: unifiedTotal, Unified: true, CUDA: true}.Budget(0)
	if busy != idle {
		t.Errorf("a loaded model changed the budget: busy %.1f GiB, idle %.1f GiB",
			float64(busy)/gibibyte, float64(idle)/gibibyte)
	}
}

func TestAMeasuredBaselineIsUsedWhenItLooksIdle(t *testing.T) {
	// vLLM reported 5.5-6.0 GiB in use before loading anything. A machine
	// that lean should be rewarded for it.
	lean, _ := Hardware{TotalMemBytes: unifiedTotal, Unified: true, CUDA: true}.Budget(5.5 * gibibyte)
	estimated, _ := Hardware{TotalMemBytes: unifiedTotal, Unified: true, CUDA: true}.Budget(0)
	if lean <= estimated {
		t.Errorf("measured-idle budget %.1f GiB should beat the %.1f GiB estimate",
			float64(lean)/gibibyte, float64(estimated)/gibibyte)
	}
}

func TestDiscreteVRAMIsBudgetedAgainstTheCardNotTheHost(t *testing.T) {
	// 24 GiB card in a 128 GiB host. The weights land in VRAM, so the host's
	// memory says nothing about whether they fit.
	got, reason := Hardware{TotalMemBytes: 128 * gibibyte, GPUMemBytes: 24 * gibibyte, CUDA: true}.Budget(0)
	if float64(got) >= 24*gibibyte {
		t.Errorf("budget %.1f GiB does not fit a 24 GiB card", float64(got)/gibibyte)
	}
	if float64(got) < 10*gibibyte {
		t.Errorf("budget %.1f GiB is uselessly small for a 24 GiB card", float64(got)/gibibyte)
	}
	if reason == "" {
		t.Error("no reason given")
	}
}

// The budget has to be measured against the pool Pool() names, because that
// pool is what callers draw against. A caller charging installed RAM under a
// budget derived from VRAM is describing two different machines.
func TestTheBudgetIsMeasuredAgainstThePoolItReports(t *testing.T) {
	tests := []struct {
		name     string
		hw       Hardware
		wantPool int64
		wantKind string
	}{
		{
			// A 24 GiB card in a 256 GiB host. The weights land in VRAM,
			// so the host's memory says nothing about whether they fit.
			name:     "discrete card",
			hw:       Hardware{TotalMemBytes: bi(256), GPUMemBytes: bi(24), CUDA: true},
			wantPool: bi(24),
			wantKind: "vram",
		},
		{
			name:     "unified memory",
			hw:       Hardware{TotalMemBytes: unifiedTotal, Unified: true, CUDA: true},
			wantPool: unifiedTotal,
			wantKind: "ram",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pool, kind := tt.hw.Pool()
			if pool != tt.wantPool || kind != tt.wantKind {
				t.Fatalf("Pool() = %.1f GiB of %s, want %.1f GiB of %s",
					float64(pool)/gibibyte, kind, float64(tt.wantPool)/gibibyte, tt.wantKind)
			}
			budget, _ := tt.hw.Budget(0)
			if budget <= 0 || budget >= pool {
				t.Errorf("budget %.1f GiB is not bounded by the %.1f GiB of %s it will be loaded into",
					float64(budget)/gibibyte, float64(pool)/gibibyte, kind)
			}
		})
	}
}

func TestASmallMachineIsNotPromisedMoreThanItHas(t *testing.T) {
	// 70% of 8 GiB was 5.6 GB, which no CUDA context and Python runtime will
	// ever leave room for. Refusing outright is the honest answer.
	got, _ := Hardware{TotalMemBytes: 8 * gibibyte, Unified: true, CUDA: true}.Budget(0)
	if float64(got) > 2*gibibyte {
		t.Errorf("8 GiB machine offered a %.1f GiB budget", float64(got)/gibibyte)
	}
}

func TestAnUnmeasurableMachineSaysSoRatherThanGuessing(t *testing.T) {
	got, reason := Hardware{TotalMemBytes: 0, Unified: true, CUDA: true}.Budget(0)
	if got != 0 || reason == "" {
		t.Errorf("got %d, %q; want 0 and an explanation", got, reason)
	}
	if (Hardware{TotalMemBytes: 0, Unified: true, CUDA: true}).BudgetKnown() {
		t.Error("a machine whose memory could not be read reported its zero budget as known")
	}
}

// A zero budget from a machine that was measured is an answer, and callers
// that read 0 as "no budget configured" told the user to run setup instead of
// saying nothing fits.
func TestAZeroBudgetFromAMeasuredMachineIsKnown(t *testing.T) {
	hidden := Hardware{TotalMemBytes: 128e9, CUDA: true, GPUsHidden: 2}
	if b, _ := hidden.Budget(0); b != 0 || !hidden.BudgetKnown() {
		t.Errorf("every GPU hidden: budget %d, known %v; want a known 0", b, hidden.BudgetKnown())
	}
	tiny := Hardware{TotalMemBytes: 4 * gibibyte, Unified: true, CUDA: true}
	if b, _ := tiny.Budget(0); b != 0 || !tiny.BudgetKnown() {
		t.Errorf("too little memory: budget %d, known %v; want a known 0", b, tiny.BudgetKnown())
	}
}

func TestTheOverrideWins(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want float64
	}{{"42", 42}, {" 42 ", 42}} {
		if got := ModelBudget(tc.in); got.GB != tc.want || !got.Known {
			t.Errorf("ModelBudget(%q) = %+v, want %v and known", tc.in, got, tc.want)
		}
	}
	// Garbage and zero fall back to the machine rather than rejecting
	// everything on the Hub. Asked of the parsing rather than by comparing two
	// ModelBudget calls: each one probes, the probe reads memory in use right now,
	// and on a busy machine the two readings differ enough to cross the
	// baseline ceiling and return different budgets. That comparison failed on
	// a CI runner while passing everywhere quiet.
	for _, in := range []string{"", "  ", "abc", "0", "-5"} {
		if v, ok := overrideGB(in); ok {
			t.Errorf("overrideGB(%q) = %v, accepted; want a fall back to the machine", in, v)
		}
	}
}

// A machine with no CUDA runs vLLM's CPU backend, which holds no context and
// captures no graphs. Charging it for both refused small models on machines
// that can serve them: a 7 GB box, which is a standard CI runner, came out at
// a budget of zero.
func TestACPUOnlyMachineIsNotChargedForCUDA(t *testing.T) {
	for _, total := range []int64{bi(7), bi(8), bi(16)} {
		cpu, reason := Hardware{TotalMemBytes: total}.Budget(0)
		gpu, _ := Hardware{TotalMemBytes: total, Unified: true, CUDA: true}.Budget(0)

		if cpu <= 0 {
			t.Errorf("%.0f GiB CPU-only machine got no budget at all", float64(total)/gibibyte)
		}
		if cpu <= gpu {
			t.Errorf("%.0f GiB: CPU-only budget %.2f GiB is no better than the CUDA one at %.2f",
				float64(total)/gibibyte, float64(cpu)/gibibyte, float64(gpu)/gibibyte)
		}
		if reason == "" {
			t.Error("no reason given")
		}
	}
}

// Refusing is a real answer, and has to read as one rather than as a budget
// that happens to be zero.
func TestAMachineThatCannotServeSaysSo(t *testing.T) {
	got, reason := Hardware{TotalMemBytes: bi(3), Unified: true, CUDA: true}.Budget(0)
	if got != 0 {
		t.Errorf("budget %.2f GiB on a 3 GiB machine", float64(got)/gibibyte)
	}
	if !strings.Contains(reason, "too little memory") {
		t.Errorf("reason = %q, want it to say the machine cannot serve a model", reason)
	}
}

// An APU that maps system memory through GTT, as a Strix Halo does, is sized
// against that limit and its carve-out, not against all of RAM and not as the
// CPU backend. Raising the limit raises the budget, until the OS's share of
// RAM is what bounds it. The figures are a Ryzen AI Max with 128 GB at the
// stock ttm pages_limit and at the one its 103 GB model needed.
func TestAGTTLimitBoundsTheBudget(t *testing.T) {
	const ram, carve = 134314070016, 536870912
	stock := FromReadings(strixHaloReadings(ram, carve, 101139496960))
	raised := FromReadings(strixHaloReadings(ram, carve, 120259084288))
	if pool, kind := stock.Pool(); kind != "gtt" || pool != carve+101139496960 {
		t.Errorf("stock pool = %d %s, want the GTT limit plus the carve-out", pool, kind)
	}
	if !strings.Contains(stock.BudgetReason, "GTT limit") || strings.Contains(stock.BudgetReason, "CPU-backend") {
		t.Errorf("reason = %q, want it to name the GTT limit", stock.BudgetReason)
	}
	const model = 103e9
	if stock.BudgetBytes >= model || raised.BudgetBytes <= model {
		t.Errorf("budgets %d (stock) and %d (raised); want the 103 GB model to fit only after the raise", stock.BudgetBytes, raised.BudgetBytes)
	}
	if raised.BudgetBytes > raised.GTTBytes {
		t.Errorf("budget %d exceeds the GTT limit %d", raised.BudgetBytes, raised.GTTBytes)
	}

	// A limit at all of RAM would leave the OS nothing, so RAM less the
	// OS bounds the weights instead.
	all := FromReadings(strixHaloReadings(ram, carve, ram))
	if all.BudgetBytes >= int64(ram-defaultBaseline(ram)) || !strings.Contains(all.BudgetReason, "minus the OS") {
		t.Errorf("GTT at all of RAM: budget %d (%s)", all.BudgetBytes, all.BudgetReason)
	}

	// A large firmware carve-out is memory the OS does not see, and the GPU
	// still uses it.
	carved := FromReadings(strixHaloReadings(32e9, 96e9, 16e9))
	if carved.BudgetBytes <= 96e9 {
		t.Errorf("96 GB carve-out with a 16 GB GTT: budget %d, want more than the carve-out", carved.BudgetBytes)
	}
}

// Without a GTT reading, as from an llm-helper that did not read it, the
// machine is still recognised and sized from system memory as before.
func TestAUnifiedAPUWithoutAGTTReadingFallsBackToRAM(t *testing.T) {
	hw := FromReadings(strixHaloReadings(134314070016, 536870912, 0))
	if hw.HardwareID != "amd_strix_halo" || !hw.Unified {
		t.Fatalf("hardware = %+v", hw)
	}
	if _, kind := hw.Pool(); kind != "ram" {
		t.Errorf("pool kind = %s, want ram", kind)
	}
}

func strixHaloReadings(ram, carve, gtt int64) Readings {
	return Readings{SchemaVersion: 1, OS: "linux", Arch: "amd64", MemTotalBytes: ram, PCI: []PCIDevice{
		{Vendor: "0x1002", Device: "0x1586", Class: "0x038000", Driver: "amdgpu", VRAMBytes: carve, GTTBytes: gtt},
	}}
}
