package cli

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/router"
)

// fakeEngine answers like an OpenAI-compatible server serving "served" and
// keeps the last chat request it got.
type fakeEngine struct {
	*httptest.Server
	mu   sync.Mutex
	last map[string]any
}

func newFakeEngine(t *testing.T) *fakeEngine {
	t.Helper()
	f := &fakeEngine{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
		case "/v1/models":
			_, _ = io.WriteString(w, `{"data":[{"id":"served"}]}`)
		case "/v1/chat/completions":
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			f.mu.Lock()
			f.last = body
			f.mu.Unlock()
			_, _ = io.WriteString(w, `{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":90}}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeEngine) lastRequest() map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.last
}

func writeWarmupFile(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "warmup.json")
	body := `{"model":"x","messages":[{"role":"system","content":"rules"},{"role":"user","content":"Hi"}],"stream":true}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func loadInstance(t *testing.T, cfg *config.Config, name string) *control.Instance {
	t.Helper()
	inst, err := control.LoadInstance(cfg, name)
	if err != nil {
		t.Fatal(err)
	}
	return inst
}

func TestWarmupPathPrefersProfile(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	writeInstance(t, cfg.Dirs, "x", "PORT=9123\nMODEL=org/m\n"+profile.WarmupKey+"=/started.json\n")
	if got := warmupPath(cfg, loadInstance(t, cfg, "x")); got != "/started.json" {
		t.Errorf("without a profile: %q", got)
	}
	if err := os.WriteFile(cfg.Dirs.ProfileFile("x"), []byte("MODEL=org/m\n"+profile.WarmupKey+"=/edited.json\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := warmupPath(cfg, loadInstance(t, cfg, "x")); got != "/edited.json" {
		t.Errorf("with a profile: %q", got)
	}
}

func TestWarmupTargetUsesServedModel(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	engine := newFakeEngine(t)
	writeInstance(t, cfg.Dirs, "x", "PORT="+portOf(t, engine.URL)+"\nMODEL=org/m\n")
	base, model := warmupTarget(cfg, loadInstance(t, cfg, "x"))
	if base != "http://localhost:"+portOf(t, engine.URL)+"/v1" || model != "served" {
		t.Errorf("target: %q %q", base, model)
	}

	writeInstance(t, cfg.Dirs, "y", "PORT="+closedPort(t)+"\nMODEL=org/m\n")
	if _, model := warmupTarget(cfg, loadInstance(t, cfg, "y")); model != "org/m" {
		t.Errorf("down server: %q", model)
	}
}

func TestWarmAfterBootSendsTheRequest(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	engine := newFakeEngine(t)
	path := writeWarmupFile(t, t.TempDir())
	writeInstance(t, cfg.Dirs, "x", "PORT="+portOf(t, engine.URL)+"\nMODEL=org/m\n"+profile.WarmupKey+"="+path+"\n")
	out, _ := captureStdout(t, func() error {
		warmAfterBoot(cfg, loadInstance(t, cfg, "x"))
		return nil
	})
	if !strings.Contains(out, "Warmed up: 100 prompt tokens, 90 from cache") {
		t.Errorf("output: %q", out)
	}
	got := engine.lastRequest()
	if got["model"] != "served" || got["max_tokens"] != float64(1) || got["stream"] != false {
		t.Errorf("request: %v", got)
	}
}

func TestWarmAfterBootSkipsWithoutFile(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	engine := newFakeEngine(t)
	writeInstance(t, cfg.Dirs, "x", "PORT="+portOf(t, engine.URL)+"\nMODEL=org/m\n")
	out, _ := captureStdout(t, func() error {
		warmAfterBoot(cfg, loadInstance(t, cfg, "x"))
		return nil
	})
	if out != "" || engine.lastRequest() != nil {
		t.Errorf("warmed without a file: %q", out)
	}
}

func TestSaveWarmupSetsTheProfileKey(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	if err := os.WriteFile(cfg.Dirs.ProfileFile("x"), []byte("MODEL=org/m\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	path, err := saveWarmup(cfg, "x", []byte(`{"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(cfg.Dirs.Data, "warmup", "x.json"); path != want {
		t.Errorf("path %q, want %q", path, want)
	}
	if data, err := os.ReadFile(path); err != nil || string(data) != "{\"messages\":[]}\n" {
		t.Errorf("file: %q %v", data, err)
	}
	p, err := profile.Load(cfg.Dirs, "x")
	if err != nil {
		t.Fatal(err)
	}
	if p.Warmup() != path || p.File.Get("MODEL") != "org/m" {
		t.Errorf("profile: %q %q", p.Warmup(), p.File.Get("MODEL"))
	}
}

func TestInstanceBackendsCarryWarmup(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\n")
	writeInstance(t, cfg.Dirs, "x", "PORT=9123\nMODEL=org/m\n"+profile.WarmupKey+"=/w.json\n")
	got := instanceBackends(cfg)()
	if len(got) != 1 || got[0].Warmup != "/w.json" {
		t.Errorf("backends: %+v", got)
	}
}

func fakeRouter(t *testing.T, h http.HandlerFunc) *config.Config {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return routerTestConfig(t, "PORT=8001\nVLLM_API_KEY=k\nROUTER_PORT="+portOf(t, srv.URL)+"\n")
}

func TestCaptureRequestReadsTheRouter(t *testing.T) {
	cfg := fakeRouter(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" {
			http.Error(w, "no", http.StatusUnauthorized)
			return
		}
		if r.URL.Path != "/router/capture" || r.URL.Query().Get("model") != "org/m" {
			http.Error(w, "the router does not serve it", http.StatusNotFound)
			return
		}
		_, _ = io.WriteString(w, `{"messages":[]}`)
	})
	got, err := captureRequest(context.Background(), cfg, "org/m")
	if err != nil || string(got) != `{"messages":[]}` {
		t.Errorf("capture: %q %v", got, err)
	}
	if _, err := captureRequest(context.Background(), cfg, "org/other"); err == nil || !strings.Contains(err.Error(), "does not serve") {
		t.Errorf("unrouted model: %v", err)
	}
}

func TestCaptureRequestWithoutRouter(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\nROUTER_PORT="+closedPort(t)+"\n")
	if _, err := captureRequest(context.Background(), cfg, "org/m"); err == nil || !strings.Contains(err.Error(), "not answering") {
		t.Errorf("err: %v", err)
	}
}

func TestRouterCacheMatchesInstances(t *testing.T) {
	cfg := fakeRouter(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/router/stats" || r.Header.Get("Authorization") != "Bearer k" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]router.ModelStats{
			"org/m":   {CacheStats: router.CacheStats{Reported: 2, PromptTokens: 200, CachedTokens: 150}, URL: "http://localhost:9123"},
			"org/old": {CacheStats: router.CacheStats{Reported: 1, PromptTokens: 50}},
		})
	})
	writeInstance(t, cfg.Dirs, "x", "PORT=9123\nMODEL=org/m\n")
	writeInstance(t, cfg.Dirs, "y", "PORT=9124\nMODEL=org/n\n")
	got := routerCache(cfg, []*control.Instance{loadInstance(t, cfg, "x"), loadInstance(t, cfg, "y")})
	if len(got) != 1 || got["x"].Percent() != 75 {
		t.Errorf("cache: %+v", got)
	}
}

func TestRouterCacheWithoutRouter(t *testing.T) {
	cfg := routerTestConfig(t, "PORT=8001\nROUTER_PORT="+closedPort(t)+"\n")
	writeInstance(t, cfg.Dirs, "x", "PORT=9123\nMODEL=org/m\n")
	if got := routerCache(cfg, []*control.Instance{loadInstance(t, cfg, "x")}); len(got) != 0 {
		t.Errorf("cache: %+v", got)
	}
}

func TestRouterMuxStatsAndCapture(t *testing.T) {
	mux := routerMux(router.NewModels(func() []router.Backend { return nil }, "k"), "k")
	cases := []struct {
		path, remote string
		want         int
	}{
		{"/router/stats", "192.0.2.1:1234", http.StatusOK},
		{"/router/capture?model=org/m", "192.0.2.1:1234", http.StatusForbidden},
		{"/router/capture?model=org/m", "127.0.0.1:1234", http.StatusNotFound},
		{"/router/capture?model=org/m", "[::1]:1234", http.StatusNotFound},
	}
	for _, c := range cases {
		for auth, code := range map[string]int{"": http.StatusUnauthorized, "Bearer k": c.want} {
			req := httptest.NewRequest(http.MethodGet, c.path, nil)
			req.RemoteAddr = c.remote
			if auth != "" {
				req.Header.Set("Authorization", auth)
			}
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, req)
			if rec.Code != code {
				t.Errorf("%s from %s with %q: %d, want %d", c.path, c.remote, auth, rec.Code, code)
			}
		}
	}
}

func TestRouterMuxCaptureReturnsSharedPrefix(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"data":[{"id":"org/m"}]}`)
			return
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	defer backend.Close()
	models := router.NewModels(func() []router.Backend { return []router.Backend{{URL: backend.URL}} }, "")
	models.Refresh(context.Background())
	mux := routerMux(models, "")

	first := make(chan *httptest.ResponseRecorder)
	go func() {
		req := httptest.NewRequest(http.MethodGet, "/router/capture?model=org/m", nil)
		req.RemoteAddr = "127.0.0.1:1234"
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		first <- rec
	}()
	chat := `{"model":"org/m","messages":[{"role":"system","content":"rules"},{"role":"user","content":"secret task"}]}`
	var got *httptest.ResponseRecorder
	for deadline := time.Now().Add(5 * time.Second); got == nil && time.Now().Before(deadline); {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(chat))
		mux.ServeHTTP(httptest.NewRecorder(), req)
		select {
		case got = <-first:
		case <-time.After(10 * time.Millisecond):
		}
	}
	if got == nil {
		t.Fatal("nothing captured")
	}
	if got.Code != http.StatusOK || strings.Contains(got.Body.String(), "secret task") || !strings.Contains(got.Body.String(), "rules") {
		t.Fatalf("capture: %d %s", got.Code, got.Body)
	}
}

func TestStatusShowsPromptCache(t *testing.T) {
	r := statusReport{Profile: "x", Healthy: true, HTTPStatus: 200, Port: "9123", Cache: &router.CacheStats{Reported: 1, PromptTokens: 200, CachedTokens: 50}}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"prompt_cache":{"reported":1,"prompt_tokens":200,"cached_tokens":50,"unknown":0}`) {
		t.Errorf("json: %s", data)
	}
	cfg := routerTestConfig(t, "PORT=8001\n")
	out, _ := captureStdout(t, func() error {
		printStatusDetail(cfg, r)
		return nil
	})
	if !strings.Contains(out, "cache:    25% of 200 prompt tokens came from the cache (router traffic only)") {
		t.Errorf("output: %q", out)
	}
	if got := describeCache(router.CacheStats{Unknown: 3}); got != "the server does not report cached tokens" {
		t.Errorf("unknown: %q", got)
	}
}
