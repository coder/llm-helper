package control

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/profile"
)

func instanceTestConfig(t *testing.T, conf string) *config.Config {
	t.Helper()
	root := t.TempDir()
	dirs := config.Dirs{
		Config:   filepath.Join(root, "config"),
		State:    filepath.Join(root, "state"),
		Data:     filepath.Join(root, "data"),
		Profiles: filepath.Join(root, "config", "models"),
	}
	if err := os.MkdirAll(dirs.Profiles, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dirs.ConfigFile(), []byte(conf), 0o600); err != nil {
		t.Fatal(err)
	}
	f, err := config.ParseEnvFile(dirs.ConfigFile())
	if err != nil {
		t.Fatal(err)
	}
	return &config.Config{Dirs: dirs, File: f}
}

func writeInstance(t *testing.T, cfg *config.Config, name, port, since string) {
	t.Helper()
	if err := os.WriteFile(cfg.Dirs.ProfileFile(name), []byte("MODEL=org/"+name+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(cfg.Dirs.InstancesDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	body := "MODEL=org/" + name + "\nINSTANCE_SINCE=" + since + "\nPORT=" + port + "\n"
	if err := os.WriteFile(cfg.Dirs.InstanceEnvFile(name), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestListInstancesOldestFirst(t *testing.T) {
	cfg := instanceTestConfig(t, "PORT=8001\n")
	if got, err := ListInstances(cfg); err != nil || got != nil {
		t.Fatalf("without an instances dir: %v, %v", got, err)
	}
	writeInstance(t, cfg, "b", "8002", "200")
	writeInstance(t, cfg, "a", "8001", "100")
	writeInstance(t, cfg, "c", "8003", "200")
	if got := LiveProfiles(cfg); len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Errorf("LiveProfiles = %v, want [a b c]", got)
	}
	if err := RemoveInstance(cfg, "b"); err != nil {
		t.Fatal(err)
	}
	if err := RemoveInstance(cfg, "b"); err != nil {
		t.Errorf("removing a missing instance: %v", err)
	}
	if got := LiveProfiles(cfg); len(got) != 2 {
		t.Errorf("after remove: %v", got)
	}
}

func TestWriteInstanceEnvKeepsStartTimeAndOptions(t *testing.T) {
	cfg := instanceTestConfig(t, "PORT=8001\n")
	writeInstance(t, cfg, "m", "8001", "100")
	p, err := profile.Load(cfg.Dirs, "m")
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteInstanceEnv(cfg, p, InstanceOpts{Port: "8004", KVCache: "8G"}); err != nil {
		t.Fatal(err)
	}
	inst, err := LoadInstance(cfg, "m")
	if err != nil {
		t.Fatal(err)
	}
	if inst.Since.Unix() != 100 {
		t.Errorf("restart reset the start time to %v", inst.Since)
	}
	if o := inst.Opts(); o.Port != "8004" || o.KVCache != "8G" || o.MemUtil != "" {
		t.Errorf("opts = %+v", o)
	}
	if inst.Env.Get("KV_CACHE_MEMORY_BYTES") != "8G" {
		t.Error("the KV cache override does not reach the server")
	}
	if inst.BaseURL() != "http://localhost:8004/v1" {
		t.Errorf("BaseURL = %s", inst.BaseURL())
	}

	bad := &profile.Profile{Name: "Bad Name", File: &config.EnvFile{}}
	if err := WriteInstanceEnv(cfg, bad, InstanceOpts{Port: "8005"}); err == nil {
		t.Error("an invalid instance name was accepted")
	}
	if err := WriteInstanceEnv(cfg, p, InstanceOpts{}); err == nil {
		t.Error("an instance without a port was accepted")
	}
}

func TestDefaultInstance(t *testing.T) {
	cfg := instanceTestConfig(t, "PORT=8001\n")
	if got := DefaultInstance(cfg, nil); got != "" {
		t.Errorf("no instances: %q", got)
	}
	live := []string{"a", "b"}
	if got := DefaultInstance(cfg, live); got != "a" {
		t.Errorf("unset: %q, want the oldest", got)
	}
	cfg.File.Set("DEFAULT_PROFILE", "b")
	if got := DefaultInstance(cfg, live); got != "b" {
		t.Errorf("set and running: %q", got)
	}
	cfg.File.Set("DEFAULT_PROFILE", "gone")
	if got := DefaultInstance(cfg, live); got != "a" {
		t.Errorf("set but not running: %q, want the oldest", got)
	}
}

func TestAllocatePort(t *testing.T) {
	busy := map[string]bool{"8002": true}
	old := portFree
	portFree = func(p string) bool { return !busy[p] }
	t.Cleanup(func() { portFree = old })

	cfg := instanceTestConfig(t, "PORT=8001\nROUTER_PORT=8003\n")
	got, err := AllocatePort(cfg, "a", "")
	if err != nil || got != "8001" {
		t.Fatalf("first instance: %q, %v; want PORT", got, err)
	}
	writeInstance(t, cfg, "a", "8001", "100")
	if got, _ := AllocatePort(cfg, "a", ""); got != "8001" {
		t.Errorf("an instance keeps its port, got %q", got)
	}
	if got, _ := AllocatePort(cfg, "b", ""); got != "8004" {
		t.Errorf("second instance got %q, want 8004 (8001 taken, 8002 listening, 8003 router)", got)
	}
	if _, err := AllocatePort(cfg, "b", "8001"); err == nil {
		t.Error("--port on another instance's port was accepted")
	}
	if _, err := AllocatePort(cfg, "b", "nope"); err == nil {
		t.Error("a non-numeric port was accepted")
	}
	if got, err := AllocatePort(cfg, "b", "9000"); err != nil || got != "9000" {
		t.Errorf("--port 9000: %q, %v", got, err)
	}
}

func TestCheckRouterPort(t *testing.T) {
	busy := map[string]bool{"8000": true, "8003": true}
	old := portFree
	portFree = func(p string) bool { return !busy[p] }
	t.Cleanup(func() { portFree = old })

	cfg := instanceTestConfig(t, "ROUTER_PORT=8003\n")
	writeInstance(t, cfg, "a", "8001", "100")
	for port, ok := range map[string]bool{"8001": false, "8000": false, "x": false, "8003": true, "8010": true} {
		if err := CheckRouterPort(cfg, port); (err == nil) != ok {
			t.Errorf("%s: err %v, want ok=%v", port, err, ok)
		}
	}
}
