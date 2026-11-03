package toolchain

import (
	"runtime"
	"slices"
	"testing"
)

func TestGather(t *testing.T) {
	checks := Gather("")
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no toolchain checks on %s", runtime.GOOS)
	}
	if runtime.GOOS == "linux" && len(checks) == 0 {
		t.Fatal("expected toolchain checks on Linux")
	}
	for _, c := range checks {
		if c.Name == "" {
			t.Errorf("check %+v has no name to show the user", c)
		}
		if !c.OK && c.Package == "" && c.Hint == "" {
			t.Errorf("check %q is missing but offers no way to fix it", c.Name)
		}
	}
}

func TestMissingListsEachPackageOnce(t *testing.T) {
	got := Missing([]Check{
		{Name: "gcc", Package: "build-essential"},
		{Name: "g++", Package: "build-essential"},
		{Name: "ninja", Package: "ninja-build", OK: true},
		{Name: "clang", Hint: "xcode-select --install"},
		{Name: "Python.h", Package: "python3.12-dev"},
	})
	if want := []string{"build-essential", "python3.12-dev"}; !slices.Equal(got, want) {
		t.Errorf("Missing = %v, want %v", got, want)
	}
}
