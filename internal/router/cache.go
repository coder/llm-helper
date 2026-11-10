package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/coder/llm-helper/internal/warmup"
)

// CacheStats counts the prompt tokens of the replies the router passed on
// for one model, and how many of them the server took from its cache.
type CacheStats struct {
	// Reported is the number of replies that said how many tokens were cached.
	Reported     int64 `json:"reported"`
	PromptTokens int64 `json:"prompt_tokens"`
	CachedTokens int64 `json:"cached_tokens"`
	// Unknown is the number of replies that did not.
	Unknown int64 `json:"unknown"`
}

// Percent is the share of prompt tokens served from cache, or -1 when no
// reply reported it.
func (s CacheStats) Percent() int {
	if s.Reported == 0 || s.PromptTokens == 0 {
		return -1
	}
	return int(s.CachedTokens * 100 / s.PromptTokens)
}

// Stats reports the cache counts of every model the router has passed a
// reply for since it started.
func (m *Models) Stats() map[string]CacheStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]CacheStats, len(m.stats))
	for id, s := range m.stats {
		out[id] = *s
	}
	return out
}

// ModelStats is one model's line in the router's stats endpoint.
type ModelStats struct {
	CacheStats
	// URL is the backend serving the model, empty when it is not routed now.
	URL string `json:"url,omitempty"`
}

// Report is Stats with the backend each model is routed to.
func (m *Models) Report() map[string]ModelStats {
	stats := m.Stats()
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]ModelStats, len(stats))
	for id, s := range stats {
		e := ModelStats{CacheStats: s}
		if u := m.routes[id]; u != nil {
			e.URL = u.String()
		}
		out[id] = e
	}
	return out
}

// FetchStats reads the stats of the router listening on base.
func FetchStats(ctx context.Context, base, key string) (map[string]ModelStats, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/router/stats", nil)
	if err != nil {
		return nil, err
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("router stats: %s", resp.Status)
	}
	var out map[string]ModelStats
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// StatsAt adds up the stats of every model served from backend.
func StatsAt(stats map[string]ModelStats, backend string) (CacheStats, bool) {
	var sum CacheStats
	found := false
	for _, e := range stats {
		if e.URL == "" || strings.TrimRight(e.URL, "/") != strings.TrimRight(backend, "/") {
			continue
		}
		found = true
		sum.Reported += e.Reported
		sum.PromptTokens += e.PromptTokens
		sum.CachedTokens += e.CachedTokens
		sum.Unknown += e.Unknown
	}
	return sum, found
}

func (m *Models) count(model string, u *warmup.Usage) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := m.stats[model]
	if s == nil {
		s = &CacheStats{}
		m.stats[model] = s
	}
	prompt, cached := -1, -1
	if u != nil {
		prompt, cached = u.Counts()
	}
	if prompt < 0 || cached < 0 {
		s.Unknown++
		return
	}
	s.Reported++
	s.PromptTokens += int64(prompt)
	s.CachedTokens += int64(cached)
}

// modelAt is the first model id served from u.
func (m *Models) modelAt(u *url.URL) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	best := ""
	for id, r := range m.routes {
		if r.String() == u.String() && (best == "" || id < best) {
			best = id
		}
	}
	return best
}

func countsTokens(path string) bool {
	switch path {
	case "/v1/chat/completions", "/v1/completions", "/v1/responses":
		return true
	}
	return false
}

// watchUsage wraps a proxied reply so its usage block is counted as it
// passes, without holding the reply back.
func (m *Models) watchUsage(resp *http.Response, model string) {
	if resp.StatusCode != http.StatusOK || !countsTokens(resp.Request.URL.Path) {
		return
	}
	sse := strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream")
	resp.Body = &usageBody{ReadCloser: resp.Body, sse: sse, done: func(u *warmup.Usage) { m.count(model, u) }}
}

const maxScan = 8 << 20

type usageBody struct {
	io.ReadCloser
	sse   bool
	buf   []byte
	skip  bool
	eof   bool
	usage *warmup.Usage
	done  func(*warmup.Usage)
	once  sync.Once
}

func (b *usageBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.feed(p[:n])
	if err == io.EOF {
		b.eof = true
		b.finish()
	}
	return n, err
}

func (b *usageBody) Close() error {
	b.finish()
	return b.ReadCloser.Close()
}

func (b *usageBody) feed(p []byte) {
	if !b.sse {
		if b.skip || len(b.buf)+len(p) > maxScan {
			b.skip, b.buf = true, nil
			return
		}
		b.buf = append(b.buf, p...)
		return
	}
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			if len(b.buf)+len(p) > maxScan {
				b.skip = true
			}
			if !b.skip {
				b.buf = append(b.buf, p...)
			}
			return
		}
		if !b.skip {
			b.buf = append(b.buf, p[:i]...)
			line := bytes.TrimSpace(bytes.TrimPrefix(bytes.TrimSpace(b.buf), []byte("data:")))
			b.parse(line)
		}
		b.buf, b.skip = b.buf[:0], false
		p = p[i+1:]
	}
}

func (b *usageBody) parse(data []byte) {
	if !bytes.Contains(data, []byte(`"usage"`)) {
		return
	}
	var reply struct {
		Usage    *warmup.Usage `json:"usage"`
		Response *struct {
			Usage *warmup.Usage `json:"usage"`
		} `json:"response"`
	}
	if json.Unmarshal(data, &reply) != nil {
		return
	}
	switch {
	case reply.Usage != nil:
		b.usage = reply.Usage
	case reply.Response != nil && reply.Response.Usage != nil:
		b.usage = reply.Response.Usage
	}
}

func (b *usageBody) finish() {
	b.once.Do(func() {
		if !b.sse && !b.skip && b.eof {
			b.parse(b.buf)
		}
		b.buf = nil
		if b.usage != nil || b.eof {
			b.done(b.usage)
		}
	})
}

// Capture waits for the next chat request for model and returns its body.
func (m *Models) Capture(ctx context.Context, model string) ([]byte, error) {
	ch := make(chan []byte, 1)
	m.mu.Lock()
	m.captures[model] = ch
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		if m.captures[model] == ch {
			delete(m.captures, model)
		}
		m.mu.Unlock()
	}()
	select {
	case body := <-ch:
		return body, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (m *Models) offer(model string, body []byte) {
	m.mu.Lock()
	ch := m.captures[model]
	delete(m.captures, model)
	m.mu.Unlock()
	if ch != nil {
		ch <- body
	}
}

// startWarm sends path's warm-up request to the backend at u, unless one is
// already on its way for model.
func (m *Models) startWarm(u *url.URL, model, path string) {
	m.mu.Lock()
	if m.warming[model] {
		m.mu.Unlock()
		return
	}
	m.warming[model] = true
	m.mu.Unlock()
	go func() {
		res, err := warmup.Run(context.Background(), u.String()+"/v1", m.key, model, path)
		m.mu.Lock()
		delete(m.warming, model)
		m.mu.Unlock()
		if m.OnWarm != nil {
			m.OnWarm(model, res, err)
		}
	}()
}
