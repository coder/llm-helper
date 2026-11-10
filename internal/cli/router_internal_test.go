package cli

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/router"
	"github.com/coder/llm-helper/internal/service"
)

type fakeRouterService struct {
	installed, restarted, removed bool
	err                           error
}

func (f *fakeRouterService) Install(string) (string, error) {
	f.installed = true
	return "/unit", f.err
}
func (f *fakeRouterService) Restart() error   { f.restarted = true; return nil }
func (f *fakeRouterService) Uninstall() error { f.removed = true; return nil }

func routerTestConfig(t *testing.T, conf string) *config.Config {
	t.Helper()
	dirs := newTree(t)
	writeConfig(t, dirs, conf)
	cfg, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestRouterPort(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	if got := routerPort(cfg, ""); got != defaultRouterPort {
		t.Errorf("unset: %q", got)
	}
	cfg.File.Set("ROUTER_PORT", "9000")
	if got := routerPort(cfg, ""); got != "9000" {
		t.Errorf("installed: %q", got)
	}
	if got := routerPort(cfg, "9100"); got != "9100" {
		t.Errorf("flag: %q", got)
	}
}

func TestRouterBind(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	if got := routerBind(cfg, "8000"); got != "127.0.0.1:8000" {
		t.Errorf("no key: %q", got)
	}
	cfg.File.Set("VLLM_API_KEY", "k")
	if got := routerBind(cfg, "8000"); got != ":8000" {
		t.Errorf("with key: %q", got)
	}
}

func TestRouterMuxRequiresKey(t *testing.T) {
	mux := routerMux(router.NewModels(func() []router.Backend { return nil }, "k"), "k")
	for _, path := range []string{"/v1/models", "/router/status"} {
		for auth, want := range map[string]int{"": http.StatusUnauthorized, "Bearer bad": http.StatusUnauthorized, "Bearer k": http.StatusOK} {
			req := httptest.NewRequest(http.MethodGet, path, nil)
			if auth != "" {
				req.Header.Set("Authorization", auth)
			}
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, req)
			if rec.Code != want {
				t.Errorf("%s with %q: %d, want %d", path, auth, rec.Code, want)
			}
		}
	}
}

func TestInstallRouterRecordsPort(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	svc := &fakeRouterService{}
	if err := installRouter(cfg, "18765", "/bin/llm-helper", svc); err != nil {
		t.Fatal(err)
	}
	if !svc.installed || !svc.restarted {
		t.Errorf("service = %+v", svc)
	}
	reloaded, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.RouterPort(); got != "18765" {
		t.Errorf("ROUTER_PORT = %q", got)
	}

	if err := uninstallRouter(reloaded, svc); err != nil {
		t.Fatal(err)
	}
	reloaded, _ = config.Load()
	if !svc.removed || reloaded.RouterPort() != "" {
		t.Errorf("after uninstall: removed=%v ROUTER_PORT=%q", svc.removed, reloaded.RouterPort())
	}
}

func TestInstallRouterRefusesBadPortOrFailure(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	svc := &fakeRouterService{}
	if err := installRouter(cfg, "nope", "/bin/llm-helper", svc); err == nil || svc.installed {
		t.Errorf("bad port: err %v, installed %v", err, svc.installed)
	}
	svc = &fakeRouterService{err: errors.New("boom")}
	if err := installRouter(cfg, "18766", "/bin/llm-helper", svc); err == nil {
		t.Fatal("want the install error")
	}
	if cfg.RouterPort() != "" {
		t.Errorf("ROUTER_PORT recorded after a failed install: %q", cfg.RouterPort())
	}
}

func TestExitCode(t *testing.T) {
	if got := exitCode(errors.New("plain")); got != 1 {
		t.Errorf("plain error: %d", got)
	}
	wrapped := fmt.Errorf("start: %w", &exitError{code: service.ExitNoFit, err: errors.New("no fit")})
	if got := exitCode(wrapped); got != service.ExitNoFit {
		t.Errorf("exitError: %d, want %d", got, service.ExitNoFit)
	}
}

func TestRecipeServiceProfileRefusesAPlainProfile(t *testing.T) {
	cfg := writeProfile(t, "MODEL=org/m\n")
	if _, err := recipeServiceProfile(cfg, "deepseek-v4-flash-2spark"); err == nil || !strings.Contains(err.Error(), "not a recipe profile") {
		t.Errorf("plain profile: %v", err)
	}
	cfg.Dirs.State = t.TempDir()
	if _, err := recipeServiceProfile(cfg, ""); err == nil || !strings.Contains(err.Error(), "no recipe profile is running") {
		t.Errorf("nothing running: %v", err)
	}
}
