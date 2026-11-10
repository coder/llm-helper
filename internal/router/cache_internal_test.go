package router

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/llm-helper/internal/warmup"
)

func replying(t *testing.T, contentType, reply string) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"data":[{"id":"glm"}]}`)
			return
		}
		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(s.Close)
	return s
}

func TestModelsCountsCachedTokens(t *testing.T) {
	cases := []struct {
		name, contentType, reply string
		want                     CacheStats
	}{
		{"json", "application/json", `{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":80}}}`,
			CacheStats{Reported: 1, PromptTokens: 100, CachedTokens: 80}},
		{"sse", "text/event-stream", "data: {\"choices\":[],\"usage\":null}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"prompt_tokens_details\":{\"cached_tokens\":10}}}\n\ndata: [DONE]\n\n",
			CacheStats{Reported: 1, PromptTokens: 50, CachedTokens: 10}},
		{"responses sse", "text/event-stream", "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":40,\"input_tokens_details\":{\"cached_tokens\":40}}}}\n\n",
			CacheStats{Reported: 1, PromptTokens: 40, CachedTokens: 40}},
		{"no usage", "application/json", `{"choices":[]}`, CacheStats{Unknown: 1}},
		{"no cached field", "application/json", `{"usage":{"prompt_tokens":100}}`, CacheStats{Unknown: 1}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := replying(t, c.contentType, c.reply)
			m := NewModels(urls(s.URL), "")
			m.Refresh(context.Background())
			code, got := do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"glm"}`)
			if code != 200 || got != c.reply {
				t.Fatalf("reply changed: %d %q", code, got)
			}
			if st := m.Stats()["glm"]; st != c.want {
				t.Errorf("stats %+v, want %+v", st, c.want)
			}
		})
	}
}

func TestModelsCountsOnlyCompletions(t *testing.T) {
	s := replying(t, "application/json", `{"usage":{"prompt_tokens":1,"prompt_tokens_details":{"cached_tokens":1}}}`)
	m := NewModels(urls(s.URL), "")
	m.Refresh(context.Background())
	do(t, m, http.MethodPost, "/v1/embeddings", `{"model":"glm"}`)
	if st := m.Stats(); len(st) != 0 {
		t.Errorf("embeddings counted: %+v", st)
	}
	do(t, m, http.MethodPost, "/v1/completions", `{}`)
	if st := m.Stats()["glm"]; st.Reported != 1 {
		t.Errorf("a request naming no model should count for the only one: %+v", m.Stats())
	}
}

func TestCacheStatsPercent(t *testing.T) {
	if p := (CacheStats{}).Percent(); p != -1 {
		t.Errorf("empty: %d", p)
	}
	if p := (CacheStats{Reported: 2, PromptTokens: 200, CachedTokens: 150}).Percent(); p != 75 {
		t.Errorf("75%%: %d", p)
	}
}

func TestModelsCapture(t *testing.T) {
	s := replying(t, "application/json", `{}`)
	m := NewModels(urls(s.URL), "")
	m.Refresh(context.Background())

	got := make(chan []byte)
	go func() {
		body, _ := m.Capture(context.Background(), "glm")
		got <- body
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		m.mu.RLock()
		armed := m.captures["glm"] != nil
		m.mu.RUnlock()
		if armed || time.Now().After(deadline) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	do(t, m, http.MethodPost, "/v1/completions", `{"model":"glm","prompt":"x"}`)
	first := `{"model":"glm","messages":[{"role":"user","content":"one"}]}`
	do(t, m, http.MethodPost, "/v1/chat/completions", first)
	do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"glm","messages":[{"role":"user","content":"two"}]}`)
	select {
	case body := <-got:
		if string(body) != first {
			t.Errorf("captured %q", body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("nothing captured")
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.captures) != 0 {
		t.Error("capture still armed")
	}
}

func TestModelsCaptureRefusesSecond(t *testing.T) {
	m := NewModels(urls(), "")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		_, _ = m.Capture(ctx, "glm")
		close(done)
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		m.mu.RLock()
		armed := m.captures["glm"] != nil
		m.mu.RUnlock()
		if armed || time.Now().After(deadline) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if _, err := m.Capture(context.Background(), "glm"); !errors.Is(err, ErrCaptureBusy) {
		t.Fatalf("second capture: %v", err)
	}
	cancel()
	<-done
	ctx2, cancel2 := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel2()
	if _, err := m.Capture(ctx2, "glm"); errors.Is(err, ErrCaptureBusy) {
		t.Fatal("capture still busy after the first gave up")
	}
}

func TestModelsCountsGzippedReplies(t *testing.T) {
	reply := `{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":80}}}`
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"data":[{"id":"glm"}]}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			w.Header().Set("Content-Encoding", "gzip")
			zw := gzip.NewWriter(w)
			_, _ = io.WriteString(zw, reply)
			_ = zw.Close()
			return
		}
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(s.Close)
	m := NewModels(urls(s.URL), "")
	m.Refresh(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"glm"}`))
	req.Header.Set("Accept-Encoding", "gzip")
	rec := httptest.NewRecorder()
	m.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	body := rec.Body.String()
	if rec.Header().Get("Content-Encoding") == "gzip" {
		zr, err := gzip.NewReader(rec.Body)
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(zr)
		body = string(data)
	}
	if body != reply {
		t.Fatalf("reply changed: %q", body)
	}
	if st := m.Stats()["glm"]; st.Reported != 1 || st.CachedTokens != 80 {
		t.Fatalf("stats %+v", st)
	}
}

func TestModelsCaptureGivesUp(t *testing.T) {
	m := NewModels(urls(), "")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, err := m.Capture(ctx, "glm"); err == nil {
		t.Error("expected the context's error")
	}
}

func TestModelsWarmsBackendThatComesUp(t *testing.T) {
	path := filepath.Join(t.TempDir(), "warmup.json")
	if err := os.WriteFile(path, []byte(`{"messages":[{"role":"system","content":"tools"},{"role":"user","content":"Hi"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var up atomic.Bool
	sent := make(chan map[string]any, 4)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !up.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"data":[{"id":"glm"}]}`)
			return
		}
		if r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("warm-up without the key")
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		sent <- body
		_, _ = io.WriteString(w, `{"usage":{"prompt_tokens":9,"prompt_tokens_details":{"cached_tokens":0}}}`)
	}))
	t.Cleanup(s.Close)

	m := NewModels(func() []Backend { return []Backend{{URL: s.URL, Model: "org/glm", Warmup: path}} }, "k")
	warmed := make(chan warmup.Result, 4)
	m.OnWarm = func(model string, res warmup.Result, err error) {
		if model != "glm" || err != nil {
			t.Errorf("OnWarm(%q, %v)", model, err)
		}
		warmed <- res
	}
	m.Refresh(context.Background())
	up.Store(true)
	m.Refresh(context.Background())
	select {
	case body := <-sent:
		if body["model"] != "glm" || body["max_tokens"] != float64(1) {
			t.Errorf("warm-up body %v", body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no warm-up sent")
	}
	if res := <-warmed; res.PromptTokens != 9 {
		t.Errorf("result %+v", res)
	}
	m.Refresh(context.Background())
	select {
	case <-sent:
		t.Error("warmed again while the backend stayed up")
	case <-time.After(100 * time.Millisecond):
	}
}

func TestModelsNoWarmupForBackendAlreadyUp(t *testing.T) {
	sent := make(chan struct{}, 1)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"data":[{"id":"glm"}]}`)
			return
		}
		sent <- struct{}{}
	}))
	t.Cleanup(s.Close)
	m := NewModels(func() []Backend { return []Backend{{URL: s.URL, Warmup: "/nonexistent"}} }, "")
	m.Refresh(context.Background())
	m.Refresh(context.Background())
	select {
	case <-sent:
		t.Error("warmed a backend that was already up when the router started")
	case <-time.After(100 * time.Millisecond):
	}
}

func TestModelsReportAndFetchStats(t *testing.T) {
	s := replying(t, "application/json", `{"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":60}}}`)
	m := NewModels(urls(s.URL), "")
	m.Refresh(context.Background())
	do(t, m, http.MethodPost, "/v1/chat/completions", `{"model":"glm"}`)

	rep := m.Report()
	if got := rep["glm"]; got.URL != s.URL || got.CachedTokens != 60 {
		t.Fatalf("report %+v", rep)
	}

	stats := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/router/stats" || r.Header.Get("Authorization") != "Bearer k" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(m.Report())
	}))
	t.Cleanup(stats.Close)
	got, err := FetchStats(context.Background(), stats.URL+"/", "k")
	if err != nil {
		t.Fatal(err)
	}
	if got["glm"] != rep["glm"] {
		t.Errorf("fetched %+v, want %+v", got, rep)
	}
	if _, err := FetchStats(context.Background(), stats.URL, "wrong"); err == nil {
		t.Error("a refused request should be an error")
	}
}

func TestStatsAtAddsUpOneBackend(t *testing.T) {
	stats := map[string]ModelStats{
		"a":   {CacheStats: CacheStats{Reported: 1, PromptTokens: 10, CachedTokens: 5}, URL: "http://h:8001"},
		"b":   {CacheStats: CacheStats{Reported: 2, PromptTokens: 30, CachedTokens: 15, Unknown: 1}, URL: "http://h:8001/"},
		"c":   {CacheStats: CacheStats{Reported: 9, PromptTokens: 90}, URL: "http://h:8002"},
		"old": {CacheStats: CacheStats{Reported: 9, PromptTokens: 90}},
	}
	got, ok := StatsAt(stats, "http://h:8001/")
	if want := (CacheStats{Reported: 3, PromptTokens: 40, CachedTokens: 20, Unknown: 1}); !ok || got != want {
		t.Errorf("got %+v %v, want %+v", got, ok, want)
	}
	if _, ok := StatsAt(stats, "http://h:9999"); ok {
		t.Error("no model is served there")
	}
}
