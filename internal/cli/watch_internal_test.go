package cli

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/nodes"
	"github.com/coder/vllm-helper/internal/service"
)

// The boot wait has to ask the machine that serves. A cluster's head can be
// another node on the recipe's own port; probing localhost on the configured
// port never answered there, so switch and restart waited forever on a model
// that was already up.
func TestBootTargetFollowsTheServer(t *testing.T) {
	r := testRecipe(t)
	useRegistry(t,
		nodes.Node{Name: "here", Host: "http://127.0.0.1:8001"},
		nodes.Node{Name: "there", Host: "http://10.99.0.7:8001", SSH: "serve@10.99.0.7"})
	cfg := &config.Config{File: &config.EnvFile{}}

	cluster := &clusterRunner{recipe: r, opts: recipeLaunchOptions{nodeNames: []string{"there", "here"}}}
	got, err := bootTargetFor(cfg, cluster)
	if err != nil {
		t.Fatal(err)
	}
	if want := "http://10.99.0.7:" + strconv.Itoa(r.Port); got.base != want {
		t.Errorf("cluster probe = %s, want %s", got.base, want)
	}

	// The head is another machine, so it is asked there: an engine on
	// loopback cannot be reached from here.
	if got.probe == nil {
		t.Error("a cluster's boot wait probes from this machine")
	}

	// A native server is asked on its engine's own health path.
	state := t.TempDir()
	if err := os.WriteFile(filepath.Join(state, "active.env"), []byte("PROVIDER=tensorfold-python\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg.Dirs.State = state
	got, err = bootTargetFor(cfg, systemdService(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if want := "http://localhost:" + cfg.Port(); got.base != want {
		t.Errorf("native probe = %s, want %s", got.base, want)
	}
	if got.health != "/v1/models" {
		t.Errorf("native health path = %q, want the active engine's /v1/models", got.health)
	}
}

// The head probe runs on the head with nothing but bash, tries loopback
// first, and reports the status it got.
func TestHeadProbeScriptAsksEachHost(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	run := func(path string, hosts ...string) string {
		t.Helper()
		cmd := exec.Command("bash", "-s")
		cmd.Stdin = strings.NewReader(headProbeScript(u.Port(), path, hosts...))
		out, err := cmd.Output()
		if err != nil {
			t.Fatal(err)
		}
		return strings.TrimSpace(string(out))
	}
	if got := run("/v1/models", "127.0.0.1"); got != "200" {
		t.Errorf("healthy path = %q, want 200", got)
	}
	if got := run("/health", "127.0.0.1"); got != "404" {
		t.Errorf("missing path = %q, want 404", got)
	}
	// Nothing listens on 127.0.0.2, so the next host is asked.
	if got := run("/v1/models", "127.0.0.2", "", "127.0.0.1"); got != "200" {
		t.Errorf("second host = %q, want 200", got)
	}
	if got := run("/v1/models", "127.0.0.2"); got != "0" {
		t.Errorf("no listener = %q, want 0", got)
	}
}

// A local head is probed on the recipe's own port and health path.
func TestClusterProbeUsesTheRecipeHealthPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	r := glmRecipe(t)
	r.Port, _ = strconv.Atoi(u.Port())
	useRegistry(t, nodes.Node{Name: "here", Host: "http://127.0.0.1:8001"}, nodes.Node{Name: "there", Host: "http://10.99.0.7:8001", SSH: "serve@10.99.0.7"})
	c := &clusterRunner{recipe: r, opts: recipeLaunchOptions{nodeNames: []string{"here", "there"}}}
	if code, ok := c.probeHealth(); !ok || code != http.StatusOK {
		t.Errorf("probeHealth = %d, %v; want the head's /v1/models", code, ok)
	}
}

type loadingRunner struct{}

func (loadingRunner) IsActive() string                 { return "active" }
func (loadingRunner) NRestarts() int                   { return 0 }
func (loadingRunner) Health() service.Health           { return service.Health{State: "active"} }
func (loadingRunner) Restart() error                   { return nil }
func (loadingRunner) Stop() error                      { return nil }
func (loadingRunner) StartTail(int) (*exec.Cmd, error) { return nil, nil }
func (loadingRunner) PrintLogs(int) error              { return nil }
func (loadingRunner) LastLogs(int) string              { return "" }
func (loadingRunner) LogHint() string                  { return "" }
func (loadingRunner) Label() string                    { return "test" }

func TestWatchTargetReturnsOnceTheTargetIsHealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	done := make(chan error, 1)
	go func() {
		done <- watchTarget(bootTarget{runner: loadingRunner{}, base: srv.URL, where: srv.URL}, "x", watchOpts{wait: true, timeout: time.Minute})
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("watch did not see the healthy target")
	}
}
