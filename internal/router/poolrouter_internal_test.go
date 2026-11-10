package router

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestPoolProxiesOpenAIPathWithoutDuplicatingV1(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("path = %q, want /v1/models", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"data":[{"id":"m"}]}`)
	}))
	defer backend.Close()

	p, err := NewPool([]string{backend.URL}, func(*http.Request) string { return "" })
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rr := httptest.NewRecorder()
	p.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), `"id":"m"`) {
		t.Fatalf("body = %s", rr.Body.String())
	}
}

func TestPoolSkipsUnhealthyBackendForStickyKey(t *testing.T) {
	p, err := NewPool([]string{"http://first.invalid", "http://second.invalid"}, func(*http.Request) string { return "same" })
	if err != nil {
		t.Fatal(err)
	}
	p.mu.Lock()
	p.backends[0].healthy = false
	p.backends[1].healthy = true
	p.mu.Unlock()

	b, ok := p.pick("same")
	if !ok {
		t.Fatal("expected backend")
	}
	if got := b.url.String(); got != "http://second.invalid" {
		t.Fatalf("picked %s, want second backend", got)
	}
}

func TestStartHealthUpdatesBackendStatus(t *testing.T) {
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Fatalf("health path = %q", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer healthy.Close()
	unhealthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer unhealthy.Close()

	p, err := NewPool([]string{healthy.URL, unhealthy.URL}, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p.StartHealth(ctx, 10*time.Millisecond)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		st := p.Status()
		if len(st.Backends) == 2 && st.Backends[0].Healthy && !st.Backends[1].Healthy {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("health did not update: %+v", p.Status())
}

func TestPoolBackendDownIsUnavailable(t *testing.T) {
	backend := httptest.NewServer(http.NotFoundHandler())
	backend.Close()
	p, err := NewPool([]string{backend.URL}, func(*http.Request) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`)))
	assertUnavailable(t, rec)
}
