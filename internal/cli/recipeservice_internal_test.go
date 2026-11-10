package cli

import (
	"errors"
	"runtime"
	"strings"
	"testing"
)

type fakeRecipeService struct {
	units      map[string]string // unit name -> rendered contents
	enabled    []string
	superseded string
	failStop   bool
}

func (f *fakeRecipeService) installed(name string) bool {
	_, ok := f.units[name]
	return ok
}

func (f *fakeRecipeService) Self() (string, error)       { return "/usr/local/bin/vllm-helper", nil }
func (f *fakeRecipeService) Home(string) (string, error) { return "/home/serve", nil }
func (f *fakeRecipeService) EnableStart(n string) error  { f.enabled = append(f.enabled, n); return nil }
func (f *fakeRecipeService) InstallSystemd(n, u string) (string, error) {
	if f.units == nil {
		f.units = map[string]string{}
	}
	f.units[n] = u
	return "/etc/systemd/system/" + n, nil
}
func (f *fakeRecipeService) Supersede(n string) error {
	if f.failStop {
		return errors.New("nope")
	}
	f.superseded = n
	return nil
}

// The unit names the profile it serves, not the recipe behind it.
func TestRecipeUnitNamesNoRecipe(t *testing.T) {
	unit := renderRecipeSystemdUnit(recipeSystemdUnit{
		User: "serve", Home: "/home/serve", Self: "/usr/local/bin/vllm-helper", Timeout: 1800, Profile: "x",
	})
	for _, want := range []string{
		"ExecStart=/usr/local/bin/vllm-helper recipes service-start x",
		"ExecStop=/usr/local/bin/vllm-helper recipes service-stop x",
		"Type=oneshot",
		"RemainAfterExit=yes",
		"TimeoutStartSec=1800",
		"User=serve",
	} {
		if !strings.Contains(unit, want) {
			t.Errorf("unit is missing %q:\n%s", want, unit)
		}
	}
	if strings.Contains(unit, "recipes launch") {
		t.Errorf("the unit pins a recipe instead of a profile:\n%s", unit)
	}
}

// A load takes minutes, so the default systemd start timeout would kill a
// healthy launch.
func TestRecipeUnitAllowsALongLoad(t *testing.T) {
	if defaultRecipeServiceTimeout < 600 {
		t.Errorf("timeout default is %ds, too short for a large model", defaultRecipeServiceTimeout)
	}
}

func TestInstallRecipeServiceStandsDownTheNativeUnit(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("install is implemented for systemd hosts")
	}
	cases := []struct {
		name       string
		opts       recipeServiceOptions
		superseded string
		enabled    []string
	}{
		{
			name:       "supersedes the configured unit",
			opts:       recipeServiceOptions{Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60, Supersede: true, Native: "vllm.service"},
			superseded: "vllm.service",
		},
		{
			name: "leaves it alone when asked",
			opts: recipeServiceOptions{Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60, Supersede: false, Native: "vllm.service"},
		},
		{
			name: "never stands down the unit it just installed",
			opts: recipeServiceOptions{Name: "vllm.service", Profile: "x", User: "serve", Timeout: 60, Supersede: true, Native: "vllm.service"},
		},
		{
			name:    "starts only when asked",
			opts:    recipeServiceOptions{Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60, Start: true},
			enabled: []string{"vllm-cluster.service"},
		},
		{
			name:    "enables the watchdog timer alongside the unit",
			opts:    recipeServiceOptions{Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60, Start: true, Watchdog: true, Interval: 2},
			enabled: []string{"vllm-cluster.service", "vllm-cluster-health.timer"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fake := &fakeRecipeService{}
			c.opts.Service = fake
			if err := installRecipeService(c.opts); err != nil {
				t.Fatal(err)
			}
			if fake.superseded != c.superseded {
				t.Errorf("superseded %q, want %q", fake.superseded, c.superseded)
			}
			if strings.Join(fake.enabled, ",") != strings.Join(c.enabled, ",") {
				t.Errorf("enabled %v, want %v", fake.enabled, c.enabled)
			}
			if !fake.installed(c.opts.Name) {
				t.Errorf("did not install %q, only %v", c.opts.Name, fake.units)
			}
			if c.opts.Watchdog && !fake.installed("vllm-cluster-health.timer") {
				t.Errorf("watchdog asked for but not installed: %v", fake.units)
			}
		})
	}
}

// Standing the old unit down is a courtesy, not the job: failing at it must
// not lose the unit that was just installed.
func TestInstallRecipeServiceSurvivesAFailedSupersede(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("install is implemented for systemd hosts")
	}
	fake := &fakeRecipeService{failStop: true}
	err := installRecipeService(recipeServiceOptions{
		Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60,
		Supersede: true, Native: "vllm.service", Service: fake,
	})
	if err != nil {
		t.Fatalf("install failed because supersede did: %v", err)
	}
	if !fake.installed("vllm-cluster.service") {
		t.Error("the unit was not installed")
	}
}

// The default install starts nothing and prints how to start it later. The
// timer is its own unit, so a hint naming only the service left the watchdog
// installed but never running.
func TestInstallRecipeServiceHintStartsTheWatchdogToo(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("install is implemented for systemd hosts")
	}
	for _, c := range []struct {
		watchdog bool
		want     string
	}{
		{true, "sudo systemctl enable --now vllm-cluster.service vllm-cluster-health.timer\n"},
		{false, "sudo systemctl enable --now vllm-cluster.service\n"},
	} {
		out, err := captureStdout(t, func() error {
			return installRecipeService(recipeServiceOptions{
				Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60,
				Watchdog: c.watchdog, Interval: 2, Service: &fakeRecipeService{},
			})
		})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasSuffix(out, c.want) {
			t.Errorf("watchdog=%v: hint does not end with %q:\n%s", c.watchdog, c.want, out)
		}
	}
}
