package router

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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

func do(t *testing.T, h http.Handler, method, path, body string) (int, string) {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, path, strings.NewReader(body)))
	return rec.Code, rec.Body.String()
}

func TestModelsDispatchesByModel(t *testing.T) {
	a := fakeInstance(t, "a", "qwen")
	b := fakeInstance(t, "b", "glm", "qwen")
	m := NewModels(func() []string { return []string{a.URL, b.URL} }, "")
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
	m := NewModels(func() []string { return []string{a.URL, b.URL} }, "")

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
	m := NewModels(func() []string { return []string{a.URL, b.URL} }, "")
	m.Refresh(context.Background())

	if code, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"nope"}`); code != 404 || !strings.Contains(got, "model_not_found") {
		t.Errorf("unknown model: %d %q", code, got)
	}
	if code, _ := do(t, m, http.MethodPost, "/v1/chat/completions", `{}`); code != 400 {
		t.Errorf("no model with two instances: %d", code)
	}

	one := NewModels(func() []string { return []string{a.URL} }, "")
	one.Refresh(context.Background())
	if code, got := do(t, one, http.MethodPost, "/v1/chat/completions", `{}`); code != 200 || !strings.HasPrefix(got, "a ") {
		t.Errorf("no model with one instance: %d %q", code, got)
	}
}

func TestModelsLearnsNewInstanceOnMiss(t *testing.T) {
	a := fakeInstance(t, "a", "qwen")
	backends := []string{a.URL}
	m := NewModels(func() []string { return backends }, "")
	m.Refresh(context.Background())
	m.tried = m.tried.Add(-2e9)

	b := fakeInstance(t, "b", "glm")
	backends = append(backends, b.URL)
	if _, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"glm"}`); !strings.HasPrefix(got, "b ") {
		t.Errorf("new instance: %q", got)
	}
}

func TestModelsBackendDown(t *testing.T) {
	a := fakeInstance(t, "a", "qwen")
	m := NewModels(func() []string { return []string{a.URL} }, "")
	m.Refresh(context.Background())
	a.Close()
	if code, _ := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"qwen"}`); code != http.StatusBadGateway {
		t.Errorf("down instance: %d", code)
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
	m := NewModels(func() []string { return []string{a.URL} }, "sekrit")
	if _, got := do(t, m, http.MethodGet, "/v1/models", ""); !strings.Contains(got, "qwen") {
		t.Errorf("list: %q", got)
	}
	if code, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"qwen"}`); code != http.StatusOK || got != "a" {
		t.Errorf("proxy: %d %q", code, got)
	}
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
