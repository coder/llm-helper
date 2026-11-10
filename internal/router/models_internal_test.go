package router

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func fakeInstance(t *testing.T, name string, models ...string) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			var data []map[string]string
			for _, m := range models {
				data = append(data, map[string]string{"id": m, "object": "model"})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
			return
		}
		body, _ := io.ReadAll(r.Body)
		_, _ = io.WriteString(w, name+" "+r.URL.Path+" "+string(body))
	}))
	t.Cleanup(s.Close)
	return s
}

func urls(us ...string) func() []Backend {
	return func() []Backend {
		out := make([]Backend, len(us))
		for i, u := range us {
			out[i] = Backend{URL: u}
		}
		return out
	}
}

func do(t *testing.T, h http.Handler, method, path, body string) (int, string) {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, path, strings.NewReader(body)))
	return rec.Code, rec.Body.String()
}

func TestModelsDispatchesByModel(t *testing.T) {
	a := fakeInstance(t, "a", "qwen")
	b := fakeInstance(t, "b", "glm", "qwen")
	m := NewModels(urls(a.URL, b.URL), "")
	m.Refresh(context.Background())

	body := `{"model":"glm","messages":[]}`
	if code, got := do(t, m, http.MethodPost, "/v1/chat/completions", body); code != 200 || got != "b /v1/chat/completions "+body {
		t.Errorf("glm: %d %q", code, got)
	}
	if _, got := do(t, m, http.MethodPost, "/v1/completions", `{"model":"qwen"}`); !strings.HasPrefix(got, "a ") {
		t.Errorf("qwen should stay on the first instance: %q", got)
	}
	if _, got := do(t, m, http.MethodGet, "/v1/models/glm", ""); !strings.HasPrefix(got, "b ") {
		t.Errorf("model path: %q", got)
	}
}

func TestModelsMergesList(t *testing.T) {
	a := fakeInstance(t, "a", "qwen")
	b := fakeInstance(t, "b", "glm", "qwen")
	m := NewModels(urls(a.URL, b.URL), "")

	code, got := do(t, m, http.MethodGet, "/v1/models", "")
	var l struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(got), &l); code != 200 || err != nil {
		t.Fatalf("%d %q %v", code, got, err)
	}
	var ids []string
	for _, d := range l.Data {
		ids = append(ids, d.ID)
	}
	if strings.Join(ids, ",") != "qwen,glm" {
		t.Errorf("ids = %v", ids)
	}
}

func TestModelsUnknownAndMissingModel(t *testing.T) {
	a := fakeInstance(t, "a", "qwen")
	b := fakeInstance(t, "b", "glm")
	m := NewModels(urls(a.URL, b.URL), "")
	m.Refresh(context.Background())

	if code, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"nope"}`); code != 404 || !strings.Contains(got, "model_not_found") {
		t.Errorf("unknown model: %d %q", code, got)
	}
	if code, _ := do(t, m, http.MethodPost, "/v1/chat/completions", `{}`); code != 400 {
		t.Errorf("no model with two instances: %d", code)
	}

	one := NewModels(urls(a.URL), "")
	one.Refresh(context.Background())
	if code, got := do(t, one, http.MethodPost, "/v1/chat/completions", `{}`); code != 200 || !strings.HasPrefix(got, "a ") {
		t.Errorf("no model with one instance: %d %q", code, got)
	}
}

func TestModelsLearnsNewInstanceOnMiss(t *testing.T) {
	a := fakeInstance(t, "a", "qwen")
	backends := []string{a.URL}
	m := NewModels(func() []Backend { return urls(backends...)() }, "")
	m.Refresh(context.Background())
	m.tried = m.tried.Add(-2e9)

	b := fakeInstance(t, "b", "glm")
	backends = append(backends, b.URL)
	if _, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"glm"}`); !strings.HasPrefix(got, "b ") {
		t.Errorf("new instance: %q", got)
	}
}

func TestModelsSendsKeyToBackends(t *testing.T) {
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer sekrit" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"data":[{"id":"qwen"}]}`)
			return
		}
		_, _ = io.WriteString(w, "a")
	}))
	t.Cleanup(a.Close)
	m := NewModels(urls(a.URL), "sekrit")
	if _, got := do(t, m, http.MethodGet, "/v1/models", ""); !strings.Contains(got, "qwen") {
		t.Errorf("list: %q", got)
	}
	if code, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"qwen"}`); code != http.StatusOK || got != "a" {
		t.Errorf("proxy: %d %q", code, got)
	}
}

func assertUnavailable(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("code = %d, body %q", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Retry-After"); got != "10" {
		t.Errorf("Retry-After = %q", got)
	}
	if got := rec.Header().Get("Retry-After-Ms"); got != "10000" {
		t.Errorf("Retry-After-Ms = %q", got)
	}
	var e struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    int    `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
		t.Fatalf("body %q: %v", rec.Body.String(), err)
	}
	if e.Error.Type != "unavailable_error" || e.Error.Code != 503 || e.Error.Message == "" {
		t.Errorf("error = %+v", e.Error)
	}
	// Clients such as Coder give up at once on these phrases.
	for _, p := range []string{"model not found", "invalid model", "unsupported model"} {
		if strings.Contains(strings.ToLower(e.Error.Message), p) {
			t.Errorf("message %q contains %q", e.Error.Message, p)
		}
	}
}

func TestModelsStartingInstance(t *testing.T) {
	var ready atomic.Bool
	a := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"data":[{"id":"glm"}]}`)
			return
		}
		_, _ = io.WriteString(w, "a")
	}))
	t.Cleanup(a.Close)
	b := fakeInstance(t, "b", "qwen")
	m := NewModels(func() []Backend {
		return []Backend{{URL: a.URL, Model: "glm"}, {URL: b.URL, Model: "qwen"}}
	}, "")
	m.Refresh(context.Background())

	rec := httptest.NewRecorder()
	m.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"glm"}`)))
	assertUnavailable(t, rec)
	if code, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"nope"}`); code != 404 || !strings.Contains(got, "model_not_found") {
		t.Errorf("unknown model while another starts: %d %q", code, got)
	}
	if _, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"qwen"}`); !strings.HasPrefix(got, "b ") {
		t.Errorf("ready instance: %q", got)
	}

	ready.Store(true)
	m.Refresh(context.Background())
	if code, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"glm"}`); code != 200 || got != "a" {
		t.Errorf("after loading: %d %q", code, got)
	}
}

func TestModelsStartingNotStuckOnRename(t *testing.T) {
	a := fakeInstance(t, "a", "served-name")
	m := NewModels(func() []Backend { return []Backend{{URL: a.URL, Model: "org/model"}} }, "")
	m.Refresh(context.Background())
	if code, _ := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"org/model"}`); code != 404 {
		t.Errorf("an answering instance under another name is not starting: %d", code)
	}
}

func TestModelsBackendDownBody(t *testing.T) {
	a := fakeInstance(t, "a", "qwen")
	m := NewModels(urls(a.URL), "")
	m.Refresh(context.Background())
	a.Close()
	rec := httptest.NewRecorder()
	m.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"qwen"}`)))
	assertUnavailable(t, rec)
}

func TestRequireKey(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "ok") })
	h := RequireKey("sekrit", ok)
	for _, tc := range []struct {
		header string
		want   int
	}{
		{"", http.StatusUnauthorized},
		{"Bearer wrong", http.StatusUnauthorized},
		{"Bearer sekrit", http.StatusOK},
	} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		if tc.header != "" {
			req.Header.Set("Authorization", tc.header)
		}
		h.ServeHTTP(rec, req)
		if rec.Code != tc.want {
			t.Errorf("%q: got %d, want %d", tc.header, rec.Code, tc.want)
		}
	}
	if code, _ := do(t, RequireKey("", ok), http.MethodGet, "/v1/models", ""); code != http.StatusOK {
		t.Errorf("no key: %d", code)
	}
}
