package router

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"sync/atomic"
	"time"
)

type backend struct {
	url     *url.URL
	healthy bool
}

type Pool struct {
	mu       sync.RWMutex
	backends []backend
	sticky   func(*http.Request) string
	rr       uint64
}

// NewPool creates a simple sticky router over backends.
func NewPool(backends []string, sticky func(*http.Request) string) (*Pool, error) {
	var bs []backend
	for _, s := range backends {
		u, err := url.Parse(s)
		if err != nil {
			return nil, err
		}
		bs = append(bs, backend{url: u, healthy: true})
	}
	if sticky == nil {
		sticky = func(*http.Request) string { return "" }
	}
	return &Pool{backends: bs, sticky: sticky}, nil
}

func (p *Pool) pick(key string) (backend, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.backends) == 0 {
		return backend{}, false
	}
	// Start index based on key (0 when empty).
	start := 0
	if key == "" {
		start = int(atomic.AddUint64(&p.rr, 1)-1) % len(p.backends)
	} else {
		var h uint32
		for i := 0; i < len(key); i++ {
			h = h*16777619 ^ uint32(key[i])
		}
		start = int(h) % len(p.backends)
	}
	// Prefer healthy backend at or after start, wrap once.
	for i := 0; i < len(p.backends); i++ {
		b := p.backends[(start+i)%len(p.backends)]
		if b.healthy {
			return b, true
		}
	}
	// No healthy backends; return start anyway.
	return p.backends[start], true
}

// Status returns a snapshot of backend health and URLs.
type Status struct {
	Backends []struct {
		URL     string `json:"url"`
		Healthy bool   `json:"healthy"`
	} `json:"backends"`
}

func (p *Pool) Status() Status {
	p.mu.RLock()
	defer p.mu.RUnlock()
	var s Status
	for _, b := range p.backends {
		s.Backends = append(s.Backends, struct {
			URL     string `json:"url"`
			Healthy bool   `json:"healthy"`
		}{URL: b.url.String(), Healthy: b.healthy})
	}
	return s
}

// StartHealth begins background health checks of backends. It stops when ctx is done.
func (p *Pool) StartHealth(ctx context.Context, every time.Duration) {
	go func() {
		client := &http.Client{Timeout: 2 * time.Second}
		for {
			select {
			case <-ctx.Done():
				return
			case <-time.After(every):
				p.mu.Lock()
				bs := make([]backend, len(p.backends))
				copy(bs, p.backends)
				p.mu.Unlock()
				for i, b := range bs {
					// Probe /health at the host root (vLLM serves it there).
					u := *b.url
					u.Path = "/health"
					ok := false
					if resp, err := client.Get(u.String()); err == nil {
						if resp.Body != nil {
							_ = resp.Body.Close()
						}
						ok = resp.StatusCode == http.StatusOK
					}
					p.mu.Lock()
					if i < len(p.backends) {
						p.backends[i].healthy = ok
					}
					p.mu.Unlock()
				}
			}
		}
	}()
}

func (p *Pool) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := p.sticky(r)
	b, ok := p.pick(key)
	if !ok {
		unavailable(w, "the pool has no backends")
		return
	}
	rx := httputil.NewSingleHostReverseProxy(b.url)
	rx.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		unavailable(w, fmt.Sprintf("the backend is not answering; it may be starting: %v", err))
	}
	rx.ServeHTTP(w, r)
}
