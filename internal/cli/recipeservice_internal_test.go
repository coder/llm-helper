package cli

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type fakeRecipeService struct {
	units      map[string]string // unit name -> rendered contents
	enabled    []string
	superseded string
	failStop   bool
	lingered   string
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
func (f *fakeRecipeService) Linger(u string) error { f.lingered = u; return nil }

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

// A user manager has no docker.service to order against and rejects User=.
func TestRecipeUnitInUserScope(t *testing.T) {
	unit := renderRecipeSystemdUnit(recipeSystemdUnit{
		User: "serve", Home: "/home/serve", Self: "/usr/local/bin/vllm-helper", Timeout: 1800, Profile: "x", UserScope: true,
	})
	for _, want := range []string{"After=network-online.target\n", "WantedBy=default.target", "ExecStart=/usr/local/bin/vllm-helper recipes service-start x"} {
		if !strings.Contains(unit, want) {
			t.Errorf("unit is missing %q:\n%s", want, unit)
		}
	}
	for _, bad := range []string{"User=", "docker.service", "multi-user.target"} {
		if strings.Contains(unit, bad) {
			t.Errorf("user unit contains %q:\n%s", bad, unit)
		}
	}
}

func TestUserUnitDirFollowsXDG(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "/tmp/cfg")
	if got := userUnitDir(); got != "/tmp/cfg/systemd/user" {
		t.Errorf("userUnitDir() = %q", got)
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

func TestInstallRecipeServiceInUserScope(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("install is implemented for systemd hosts")
	}
	t.Setenv("USER", "serve")
	fake := &fakeRecipeService{}
	out, err := captureStdout(t, func() error {
		return installRecipeService(recipeServiceOptions{
			Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60,
			Supersede: true, Native: "vllm.service", Watchdog: true, Interval: 2,
			UserScope: true, Service: fake,
		})
	})
	if err != nil {
		t.Fatal(err)
	}
	if fake.lingered != "serve" {
		t.Errorf("lingered %q, want serve", fake.lingered)
	}
	if fake.superseded != "vllm.service" {
		t.Errorf("superseded %q, want vllm.service", fake.superseded)
	}
	if strings.Contains(fake.units["vllm-cluster.service"], "User=") {
		t.Errorf("user unit sets User=:\n%s", fake.units["vllm-cluster.service"])
	}
	if !strings.Contains(fake.units["vllm-cluster-health.service"], "--interval 2 --user-scope\n") {
		t.Errorf("watchdog does not check the user manager:\n%s", fake.units["vllm-cluster-health.service"])
	}
	want := "systemctl --user enable --now vllm-cluster.service vllm-cluster-health.timer\n"
	if !strings.HasSuffix(out, want) {
		t.Errorf("hint does not end with %q:\n%s", want, out)
	}
}

func TestInstallRecipeServiceLingersOnlyInUserScope(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("install is implemented for systemd hosts")
	}
	fake := &fakeRecipeService{}
	if _, err := captureStdout(t, func() error {
		return installRecipeService(recipeServiceOptions{
			Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60, Service: fake,
		})
	}); err != nil {
		t.Fatal(err)
	}
	if fake.lingered != "" {
		t.Errorf("system scope enabled lingering for %q", fake.lingered)
	}
}

// A user manager can only run units as its own user.
func TestInstallRecipeServiceRejectsAnotherUserInUserScope(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("install is implemented for systemd hosts")
	}
	t.Setenv("USER", "me")
	fake := &fakeRecipeService{}
	err := installRecipeService(recipeServiceOptions{
		Name: "vllm-cluster.service", Profile: "x", User: "serve", Timeout: 60,
		UserScope: true, Service: fake,
	})
	if err == nil || !strings.Contains(err.Error(), "SERVICE_SCOPE=system") {
		t.Fatalf("err = %v", err)
	}
	if len(fake.units) != 0 {
		t.Errorf("installed %v anyway", fake.units)
	}
}

// A unit from before instances ran a bare 'stop', which now forgets the
// instance; migrate rewrites it and its watchdog to name the profile.
func TestUpgradeRecipeUnits(t *testing.T) {
	dir := t.TempDir()
	old := "[Unit]\nDescription=LLM Helper cluster (the active recipe)\n\n[Service]\n" +
		"User=serve\nEnvironment=HOME=/home/serve\nExecStart=/opt/llm-helper restart\n" +
		"ExecStop=/opt/llm-helper stop\nTimeoutStartSec=2400\n"
	health := "[Service]\nExecStart=/opt/llm-helper recipes health-check --unit glm.service --interval 5\n"
	other := "[Service]\nExecStart=/usr/bin/sshd\n"
	for name, body := range map[string]string{
		"glm.service":        old,
		"glm-health.service": health,
		"sshd.service":       other,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	fake := &fakeRecipeService{}
	done, err := upgradeRecipeUnits(dir, "glm", fake)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(done, " ") != "glm.service glm-health.service glm-health.timer" {
		t.Fatalf("upgraded %v", done)
	}
	for name, want := range map[string][]string{
		"glm.service":        {"recipes service-start glm", "recipes service-stop glm", "User=serve", "TimeoutStartSec=2400"},
		"glm-health.service": {"--unit glm.service --profile glm --interval 5"},
		"glm-health.timer":   {"OnUnitActiveSec=5min"},
	} {
		for _, w := range want {
			if !strings.Contains(fake.units[name], w) {
				t.Errorf("%s is missing %q:\n%s", name, w, fake.units[name])
			}
		}
	}

	if err := os.WriteFile(filepath.Join(dir, "glm.service"), []byte(fake.units["glm.service"]), 0o644); err != nil {
		t.Fatal(err)
	}
	if done, err := upgradeRecipeUnits(dir, "glm", &fakeRecipeService{}); err != nil || len(done) != 0 {
		t.Fatalf("a current unit was rewritten: %v %v", done, err)
	}
}
