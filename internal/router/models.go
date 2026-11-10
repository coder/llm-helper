package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"
)

// Models sends each request to the backend that serves the model it names.
// It learns which backend serves what from each backend's own /v1/models.
type Models struct {
	backends func() []string
	client   *http.Client

	mu      sync.RWMutex
	routes  map[string]*url.URL
	entries []json.RawMessage
	urls    []*url.URL
	tried   time.Time
}

// NewModels returns a router over the base URLs (scheme://host:port) that
// backends returns. backends is called on every refresh.
func NewModels(backends func() []string) *Models {
	return &Models{
		backends: backends,
		client:   &http.Client{Timeout: 5 * time.Second},
		routes:   map[string]*url.URL{},
	}
}

type modelList struct {
	Data []json.RawMessage `json:"data"`
}

type modelID struct {
	ID string `json:"id"`
}

// Refresh asks every backend for its models. When two backends serve the
// same id, the earlier one keeps it.
func (m *Models) Refresh(ctx context.Context) {
	routes := map[string]*url.URL{}
	var entries []json.RawMessage
	var urls []*url.URL
	for _, b := range m.backends() {
		u, err := url.Parse(strings.TrimRight(b, "/"))
		if err != nil {
			continue
		}
		urls = append(urls, u)
		for _, e := range m.fetch(ctx, u) {
			var id modelID
			if json.Unmarshal(e, &id) != nil || id.ID == "" {
				continue
			}
			if _, dup := routes[id.ID]; dup {
				continue
			}
			routes[id.ID] = u
			entries = append(entries, e)
		}
	}
	m.mu.Lock()
	m.routes, m.entries, m.urls, m.tried = routes, entries, urls, time.Now()
	m.mu.Unlock()
}

func (m *Models) fetch(ctx context.Context, u *url.URL) []json.RawMessage {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String()+"/v1/models", nil)
	if err != nil {
		return nil
	}
	resp, err := m.client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var l modelList
	if json.NewDecoder(resp.Body).Decode(&l) != nil {
		return nil
	}
	return l.Data
}

// StartRefresh refreshes now and then every interval until ctx is done.
func (m *Models) StartRefresh(ctx context.Context, every time.Duration) {
	m.Refresh(ctx)
	go func() {
		t := time.NewTicker(every)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				m.Refresh(ctx)
			}
		}
	}()
}

// lookup finds the backend for model, refreshing once on a miss unless a
// refresh ran in the last second.
func (m *Models) lookup(ctx context.Context, model string) *url.URL {
	m.mu.RLock()
	u, recent := m.routes[model], time.Since(m.tried) < time.Second
	only := len(m.urls) == 1
	var single *url.URL
	if only {
		single = m.urls[0]
	}
	m.mu.RUnlock()
	if model == "" {
		return single
	}
	if u != nil || recent {
		return u
	}
	m.Refresh(ctx)
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.routes[model]
}

// Routes reports which backend serves each model.
func (m *Models) Routes() map[string]string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]string, len(m.routes))
	for id, u := range m.routes {
		out[id] = u.String()
	}
	return out
}

func (m *Models) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/v1/models" && r.Method == http.MethodGet {
		m.Refresh(r.Context())
		m.mu.RLock()
		data := m.entries
		m.mu.RUnlock()
		if data == nil {
			data = []json.RawMessage{}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
		return
	}

	model := strings.TrimPrefix(r.URL.Path, "/v1/models/")
	if model == r.URL.Path {
		model = ""
		if r.Body != nil {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				apiError(w, http.StatusBadRequest, "reading the request body: "+err.Error(), "")
				return
			}
			r.Body.Close()
			var req struct {
				Model string `json:"model"`
			}
			_ = json.Unmarshal(body, &req)
			model = req.Model
			r.Body = io.NopCloser(bytes.NewReader(body))
			r.ContentLength = int64(len(body))
		}
	}

	u := m.lookup(r.Context(), model)
	if u == nil {
		if model == "" {
			apiError(w, http.StatusBadRequest, "the request names no model, and more than one is running", "")
			return
		}
		apiError(w, http.StatusNotFound, fmt.Sprintf("no running instance serves the model %q", model), "model_not_found")
		return
	}
	rx := httputil.NewSingleHostReverseProxy(u)
	rx.FlushInterval = -1
	rx.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		apiError(w, http.StatusBadGateway, fmt.Sprintf("the instance serving %q did not answer: %v", model, err), "")
	}
	rx.ServeHTTP(w, r)
}

func apiError(w http.ResponseWriter, status int, msg, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	e := map[string]any{"message": msg, "type": "invalid_request_error"}
	if code != "" {
		e["code"] = code
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"error": e})
}
