package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hardware"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/resolve"
)

// station is a DGX Station GB300 whose display GPU the server can also see.
var station = detect.Hardware{
	GPUName: "NVIDIA GB300", Memory: string(hardware.MemoryCoherent), OffloadGB: 478,
	GPUs:   []detect.GPU{{Name: "NVIDIA GB300", UUID: "GPU-aaaa"}, {Name: "NVIDIA RTX PRO 2000 Blackwell", UUID: "GPU-bbbb"}},
	GPUPin: "GPU-aaaa",
}

// A new config on a machine with different GPUs names the biggest, and a
// kept one gets it only when it names no GPUs itself.
func TestSetupPinsTheBiggestGPUWhenTheyDiffer(t *testing.T) {
	prov, err := engine.Lookup(engine.VLLM)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "config.env")
	if err := os.WriteFile(path, []byte(newConfigEnv(prov, "/opt/vllm/bin/vllm", "user", station)), 0o600); err != nil {
		t.Fatal(err)
	}
	env, err := config.ParseEnvFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := env.Get("CUDA_VISIBLE_DEVICES"); got != "GPU-aaaa" {
		t.Errorf("new config CUDA_VISIBLE_DEVICES = %q, want the GB300's UUID", got)
	}
	if strings.Contains(newConfigEnv(prov, "/opt/vllm/bin/vllm", "user", detect.Hardware{}), "CUDA_VISIBLE_DEVICES") {
		t.Error("a machine with one kind of GPU got a CUDA_VISIBLE_DEVICES line")
	}

	kept := filepath.Join(t.TempDir(), "config.env")
	if err := os.WriteFile(kept, []byte("PORT=8001\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if filled, err := fillMissingGPUPin(kept, "GPU-aaaa"); err != nil || !filled {
		t.Fatalf("fillMissingGPUPin = %v, %v; want filled", filled, err)
	}
	if filled, err := fillMissingGPUPin(kept, "GPU-cccc"); err != nil || filled {
		t.Errorf("a second fill replaced the first: %v, %v", filled, err)
	}
	env, err = config.ParseEnvFile(kept)
	if err != nil {
		t.Fatal(err)
	}
	if got := env.Get("CUDA_VISIBLE_DEVICES"); got != "GPU-aaaa" || env.Get("PORT") != "8001" {
		t.Errorf("kept config: CUDA_VISIBLE_DEVICES=%q PORT=%q", got, env.Get("PORT"))
	}
}

func TestDoctorWarnsWhenTheServerCouldPickTheWrongGPU(t *testing.T) {
	t.Parallel()
	if w := gpuPinWarnings(station, false, "GPU-aaaa", true); len(w) != 1 ||
		!strings.Contains(w[0], "CUDA_VISIBLE_DEVICES=GPU-aaaa in config.env") || !strings.Contains(w[0], "only if systemd sets it") {
		t.Errorf("unpinned, with the shell's setting: %q", w)
	}
	if w := gpuPinWarnings(station, true, "", false); len(w) != 1 || !strings.Contains(w[0], "different GPUs") {
		t.Errorf("config.env names both GPUs: %q", w)
	}
	pinned := station
	pinned.GPUs = pinned.GPUs[:1]
	if w := gpuPinWarnings(pinned, true, "", false); len(w) != 0 {
		t.Errorf("config.env names the GB300 alone: %q", w)
	}
	if w := gpuPinWarnings(detect.Hardware{GPUName: "NVIDIA GB10"}, false, "", false); len(w) != 0 {
		t.Errorf("one GPU: %q", w)
	}
}

func TestACoherentGPUIsDescribedWithBothPools(t *testing.T) {
	t.Parallel()
	if got := memoryKind(station); !strings.Contains(got, "NVLink-C2C") {
		t.Errorf("memoryKind = %q", got)
	}
	if got := offloadLine(station); !strings.Contains(got, "478 GB") {
		t.Errorf("offloadLine = %q", got)
	}
	if got := offloadLine(detect.Hardware{Unified: true}); got != "" {
		t.Errorf("unified offloadLine = %q", got)
	}
}

// A mixture-of-experts model too big for a Grace GPU is shown as fitting with
// offload, and its copies say which fit how.
func TestInspectShowsAModelThatFitsWithOffload(t *testing.T) {
	d := resolve.Detail{
		SizeGB: 395, BudgetGB: 244, BudgetKnown: true, FitsBudget: new(false),
		MoE: true, OffloadGB: 478, FitsWithOffload: new(true), OffloadNeededGB: 151,
		All: []hf.Variant{{Name: "big", Label: "BIG", SizeGB: 900}, {Name: "nvfp4", Label: "NVFP4", SizeGB: 395}},
	}
	d.Best = &d.All[1]
	d.Variants = 2
	out, _ := captureStdout(t, func() error { printModelDetail(d, nil); return nil })
	for _, want := range []string{"fits with ~151 GB of its experts in CPU memory", "fits with experts in CPU memory, slower", "too big for this machine"} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
}

// A smaller copy that fits the GPU but cannot load does not hide that the copy
// recommended needs offload.
func TestInspectPrefersTheOffloadVerdictOverAnUnloadableFit(t *testing.T) {
	d := resolve.Detail{
		SizeGB: 149, BudgetGB: 244, BudgetKnown: true, FitsBudget: new(true),
		MoE: true, OffloadGB: 478, FitsWithOffload: new(true), OffloadNeededGB: 151,
		All: []hf.Variant{{Name: "nvfp4", Label: "NVFP4", SizeGB: 395}, {Name: "q2", Label: "Q2", SizeGB: 149, GGUF: true, Blocked: "split"}},
	}
	d.Best = &d.All[0]
	d.Variants = 2
	out, _ := captureStdout(t, func() error { printModelDetail(d, nil); return nil })
	if !strings.Contains(out, "fits with ~151 GB of its experts in CPU memory") || strings.Contains(out, "fits the ~244GB budget") {
		t.Errorf("size line hides the offload:\n%s", out)
	}
}
