package control

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
)

func TestCheckFit(t *testing.T) {
	unified := detect.Hardware{Unified: true}
	discrete := detect.Hardware{GPUName: "NVIDIA RTX 6000", GPUMemGB: 96}
	metal := detect.Hardware{Unified: true, Metal: true}
	cases := []struct {
		name    string
		hw      detect.Hardware
		running []Share
		want    float64
		refuse  bool
	}{
		{"first model", unified, nil, 0.85, false},
		{"two halves", unified, []Share{{Name: "a", Frac: 0.4}}, 0.4, false},
		{"over the unified cap", unified, []Share{{Name: "a", Frac: 0.5}}, 0.4, true},
		{"discrete fits to 1", discrete, []Share{{Name: "a", Frac: 0.5}}, 0.5, false},
		{"discrete over", discrete, []Share{{Name: "a", Frac: 0.6}}, 0.5, true},
		{"unknown share ignored", unified, []Share{{Name: "a"}}, 0.85, false},
		{"unknown request", unified, []Share{{Name: "a", Frac: 0.85}}, 0, false},
		{"cpu only", detect.Hardware{}, []Share{{Name: "a", Frac: 0.9}}, 0.9, false},
		{"metal first model at its default", metal, nil, detect.MetalMemUtilization, false},
		{"metal over", metal, []Share{{Name: "a", Frac: 0.5}}, 0.5, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := CheckFit(tc.hw, tc.running, Share{Name: "new", Frac: tc.want})
			if (err != nil) != tc.refuse {
				t.Fatalf("refuse=%v, got %v", tc.refuse, err)
			}
			if err != nil && (!strings.Contains(err.Error(), "a 0.") || !strings.Contains(err.Error(), "stop one")) {
				t.Errorf("message should name the running model and suggest stopping one: %v", err)
			}
		})
	}
}

func TestCheckFitCountsOnlySharedGPUs(t *testing.T) {
	hw := detect.Hardware{GPUName: "NVIDIA H100", GPUMemGB: 80}
	cases := []struct {
		name       string
		held, want []string
		refuse     bool
	}{
		{"other gpu", []string{"0"}, []string{"1"}, false},
		{"same gpu", []string{"0"}, []string{"0"}, true},
		{"overlap", []string{"0", "1"}, []string{"1", "2"}, true},
		{"held sees all", nil, []string{"1"}, true},
		{"new sees all", []string{"0"}, nil, true},
		{"no gpus", []string{}, []string{"0"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := CheckFit(hw, []Share{{Name: "a", Frac: 0.7, GPUs: tc.held}}, Share{Name: "new", Frac: 0.5, GPUs: tc.want})
			if (err != nil) != tc.refuse {
				t.Fatalf("refuse=%v, got %v", tc.refuse, err)
			}
		})
	}
}

func TestGPUList(t *testing.T) {
	if gpuList("", false) != nil {
		t.Error("unset should be nil")
	}
	if got := gpuList("", true); got == nil || len(got) != 0 {
		t.Errorf("empty: %v", got)
	}
	if got := gpuList(" 0, 2 ", true); strings.Join(got, ",") != "0,2" {
		t.Errorf("list: %v", got)
	}
}

func TestSharesSkipsRecipesOnOtherMachines(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("VLLM_HELPER_CONFIG_DIR", dir)
	nodesJSON := `[{"name":"here","host":"http://127.0.0.1:8888"},{"name":"there","host":"http://192.0.2.10:8888"}]`
	if err := os.WriteFile(filepath.Join(dir, "nodes.json"), []byte(nodesJSON), 0o600); err != nil {
		t.Fatal(err)
	}
	env := func(body string) *config.EnvFile {
		p := filepath.Join(t.TempDir(), "x.env")
		if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		f, err := config.ParseEnvFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	list := []*Instance{
		{Name: "native", Env: env("GPU_MEMORY_UTILIZATION=0.3\n")},
		{Name: "local-recipe", Env: env("RECIPE=r\nNODES=here\nGPU_MEMORY_UTILIZATION=0.3\n")},
		{Name: "remote-recipe", Env: env("RECIPE=r\nNODES=there\nGPU_MEMORY_UTILIZATION=0.3\n")},
	}
	var got []string
	for _, s := range Shares(list) {
		got = append(got, s.Name)
	}
	if strings.Join(got, ",") != "native,local-recipe" {
		t.Errorf("Shares = %v", got)
	}
}
