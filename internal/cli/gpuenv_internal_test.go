package cli

import (
	"slices"
	"testing"
)

func TestWithStableGPUsRewritesEveryCopy(t *testing.T) {
	t.Parallel()
	stable := func(v string) string {
		if v == "1" {
			return "GPU-16afb1f2"
		}
		return v
	}
	in := []string{"PATH=/bin", "CUDA_VISIBLE_DEVICES=1", "HF_HOME=/models/hf", "CUDA_VISIBLE_DEVICES=1"}
	got := withStableGPUs(in, stable)
	want := []string{"PATH=/bin", "CUDA_VISIBLE_DEVICES=GPU-16afb1f2", "HF_HOME=/models/hf", "CUDA_VISIBLE_DEVICES=GPU-16afb1f2"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
	if in[1] != "CUDA_VISIBLE_DEVICES=1" {
		t.Error("the input slice was modified")
	}
	if got := withStableGPUs([]string{"A=1"}, stable); !slices.Equal(got, []string{"A=1"}) {
		t.Errorf("no CUDA_VISIBLE_DEVICES: got %q", got)
	}
}
