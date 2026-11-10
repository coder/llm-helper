package router

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Models sends each request to the backend that serves the model it names.
// It learns which backend serves what from each backend's own /v1/models.
type Models struct {
	backends func() []Backend
	key      string
	client   *http.Client

	mu       sync.RWMutex
	routes   map[string]*url.URL
	starting map[string]bool
	learned  map[string][]string
	entries  []json.RawMessage
	urls     []*url.URL
	tried    time.Time
}

// Backend is one instance: its base URL (scheme://host:port) and the model
// it is configured to serve, "" when unknown.
type Backend struct {
	URL   string
	Model string
}

// RetryAfter is how long a client is told to wait for a starting instance.
const RetryAfter = 10 * time.Second

// NewModels returns a router over the instances backends returns. backends is
// called on every refresh. A non-empty key is sent as the bearer token to
// every backend.
func NewModels(backends func() []Backend, key string) *Models {
	return &Models{
		backends: backends,
		key:      key,
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
// same id, the earlier one keeps it. The model of a backend whose /v1/models
// does not answer yet is starting, as are the ids it last advertised.
func (m *Models) Refresh(ctx context.Context) {
	m.mu.RLock()
	prev := m.learned
	m.mu.RUnlock()
	learned := map[string][]string{}
	routes := map[string]*url.URL{}
	var entries []json.RawMessage
	var urls []*url.URL
	var waiting []string
	for _, b := range m.backends() {
		u, err := url.Parse(strings.TrimRight(b.URL, "/"))
		if err != nil {
			continue
		}
		urls = append(urls, u)
		key := u.String() + "\x00" + b.Model
		list, ok := m.fetch(ctx, u)
		if !ok && b.Model != "" {
			waiting = append(waiting, b.Model)
			waiting = append(waiting, prev[key]...)
			learned[key] = prev[key]
		}
		for _, e := range list {
			var id modelID
			if json.Unmarshal(e, &id) != nil || id.ID == "" {
				continue
			}
			learned[key] = append(learned[key], id.ID)
			if _, dup := routes[id.ID]; dup {
				continue
			}
			routes[id.ID] = u
			entries = append(entries, e)
		}
	}
	starting := map[string]bool{}
	for _, id := range waiting {
		if routes[id] == nil {
			starting[id] = true
		}
	}
	m.mu.Lock()
	m.routes, m.starting, m.learned, m.entries, m.urls, m.tried = routes, starting, learned, entries, urls, time.Now()
	m.mu.Unlock()
}

func (m *Models) fetch(ctx context.Context, u *url.URL) ([]json.RawMessage, bool) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String()+"/v1/models", nil)
	if err != nil {
		return nil, false
	}
	m.authorize(req)
	resp, err := m.client.Do(req)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false
	}
	var l modelList
	if json.NewDecoder(resp.Body).Decode(&l) != nil {
		return nil, false
	}
	return l.Data, len(l.Data) > 0
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
		m.mu.RLock()
		u = m.routes[model]
		starting := m.starting[model]
		m.mu.RUnlock()
		if u == nil && starting {
			unavailable(w, fmt.Sprintf("%s is still loading; try again shortly", model))
			return
		}
		if u == nil {
			apiError(w, http.StatusNotFound, fmt.Sprintf("no running instance serves the model %q", model), "model_not_found")
			return
		}
	}
	rx := httputil.NewSingleHostReverseProxy(u)
	direct := rx.Director
	rx.Director = func(r *http.Request) {
		direct(r)
		m.authorize(r)
	}
	rx.FlushInterval = -1
	rx.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		unavailable(w, fmt.Sprintf("the instance serving %s is not answering; it may be restarting: %v", model, err))
	}
	rx.ServeHTTP(w, r)
}

func (m *Models) authorize(r *http.Request) {
	if m.key != "" {
		r.Header.Set("Authorization", "Bearer "+m.key)
	}
}

// RequireKey refuses requests that do not carry key as a bearer token. An
// empty key lets everything through.
func RequireKey(key string, h http.Handler) http.Handler {
	if key == "" {
		return h
	}
	want := []byte("Bearer " + key)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), want) != 1 {
			apiError(w, http.StatusUnauthorized, "missing or invalid API key", "invalid_api_key")
			return
		}
		h.ServeHTTP(w, r)
	})
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

// unavailable answers 503 the way llama.cpp does while a model loads, with
// Retry-After so OpenAI-compatible clients retry.
func unavailable(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", strconv.Itoa(int(RetryAfter.Seconds())))
	w.Header().Set("Retry-After-Ms", strconv.FormatInt(RetryAfter.Milliseconds(), 10))
	w.WriteHeader(http.StatusServiceUnavailable)
	e := map[string]any{"message": msg, "type": "unavailable_error", "code": http.StatusServiceUnavailable}
	_ = json.NewEncoder(w).Encode(map[string]any{"error": e})
}
