package recipe

import (
	"slices"
	"testing"
)

// sparkTarget is the DGX Spark GB10 profile: an exact hardware id, its GPU
// generation, and the arch the forbidden-flag check keys on.
var sparkTarget = Target{Hardware: "dgx_spark_gb10", Generation: "blackwell", Arch: "sm121"}

// Both entries are SM100-only kernels. An entry that carried one would abort at
// startup on GB10, so the check has to fail rather than warn.
func TestForbiddenArgsFailClosed(t *testing.T) {
	args := []string{"--model", "acme/model", "--moe-backend", "deep_gemm_mega_moe"}
	if err := checkForbidden(args, sparkTarget); err == nil {
		t.Error("expected a forbidden sm121 flag to fail the check")
	}
	if err := checkForbidden(args, Target{Arch: "sm100"}); err != nil {
		t.Errorf("sm100 has no forbidden list, so this should pass: %v", err)
	}
	ok := []string{"--model", "acme/model", "--moe-backend", "b12x"}
	if err := checkForbidden(ok, sparkTarget); err != nil {
		t.Errorf("a supported backend should pass: %v", err)
	}
	if err := CheckForbidden([]string{"--moe-backend=deep_gemm_mega_moe"}, "sm121"); err == nil {
		t.Error("the inline --flag=value spelling slipped past the check")
	}
	for _, arch := range []string{"", "sm999"} {
		if err := CheckForbidden(args, arch); err != nil {
			t.Errorf("arch %q has no rules, so this should pass: %v", arch, err)
		}
	}
}

func TestDedupeArgsLastWinsAndKeepsInlineForm(t *testing.T) {
	got := dedupeArgs([]string{
		"--moe-backend", "deep_gemm_mega_moe",
		"--flag=one",
		"--enable-prefix-caching",
		"--moe-backend", "b12x",
		"--flag=two",
	})
	want := []string{"--enable-prefix-caching", "--moe-backend", "b12x", "--flag=two"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}
