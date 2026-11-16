package hf

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/coder/llm-helper/internal/hfcache"
)

// ErrOffline is returned for a model that is not in the local cache while the
// Hub is not being used.
var ErrOffline = errors.New("not in the local cache, and the Hub is not reachable")

// cache is where the Hub readers look when the Hub cannot answer.
var cache struct {
	sync.Mutex
	hub     string
	offline bool
	// downUntil is when the Hub is tried again after a request failed to
	// reach it, so a panel that outlives a network drop recovers.
	downUntil time.Time
}

// retryHubAfter is how long the cache answers after the Hub did not.
const retryHubAfter = time.Minute

// UseCache points the readers at the hub cache a server reads. With offline
// set they never ask the Hub; without it they fall back to the cache once a
// request fails to reach the Hub at all. An empty hub turns the fallback off.
func UseCache(hub string, offline bool) {
	cache.Lock()
	defer cache.Unlock()
	cache.hub, cache.offline, cache.downUntil = hub, offline, time.Time{}
}

// Offline reports whether the readers are answering from the cache: by
// choice, or because the Hub did not answer.
func Offline() bool {
	cache.Lock()
	defer cache.Unlock()
	return cache.hub != "" && (cache.offline || time.Now().Before(cache.downUntil))
}

// local is the cache to read, "" when the Hub is the source.
func local() string {
	cache.Lock()
	defer cache.Unlock()
	if cache.hub != "" && (cache.offline || time.Now().Before(cache.downUntil)) {
		return cache.hub
	}
	return ""
}

// canFallBack is the cache to read after a request failed to reach the Hub,
// and notes that it did not. A cache with no models is no fallback: the
// request's own error says more.
func canFallBack(ctx context.Context, err error) string {
	if err == nil || ctx.Err() != nil {
		return ""
	}
	cache.Lock()
	defer cache.Unlock()
	if cache.hub == "" || !hfcache.Has(cache.hub) {
		return ""
	}
	cache.downUntil = time.Now().Add(retryHubAfter)
	return cache.hub
}

// rawGet reads one file of a repo at main, from the Hub or, when that is not
// being used, from the cached snapshot. A cached repo without the file is a
// 404, as the Hub would say.
func rawGet(ctx context.Context, repo, name string) (*http.Response, error) {
	hub := local()
	if hub == "" {
		resp, err := get(ctx, baseURL+"/"+repo+"/raw/main/"+name)
		if hub = canFallBack(ctx, err); hub == "" {
			return resp, err
		}
	}
	dir, _, ok := hfcache.Snapshot(hub, repo)
	if !ok {
		return nil, fmt.Errorf("%s: %w", repo, ErrOffline)
	}
	f, err := os.Open(filepath.Join(dir, filepath.FromSlash(name)))
	if err != nil {
		return &http.Response{StatusCode: http.StatusNotFound, Body: http.NoBody}, nil
	}
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(f), ContentLength: -1}, nil
}

// localModel is Lookup's answer for a cached repo.
func localModel(hub, repo string) (Model, error) {
	if _, _, ok := hfcache.Snapshot(hub, repo); !ok {
		return Model{}, fmt.Errorf("%s: %w", repo, ErrOffline)
	}
	return Model{ID: repo}, nil
}

// localTree is repoTree for a cached repo: the files its snapshot holds.
func localTree(hub, repo string) ([]treeEntry, error) {
	m, err := hfcache.ScanRepo(hub, repo)
	if err != nil || len(m.Files) == 0 {
		return nil, fmt.Errorf("%s: %w", repo, ErrOffline)
	}
	out := make([]treeEntry, 0, len(m.Files))
	for _, f := range m.Files {
		out = append(out, treeEntry{Type: "file", Path: f.Path, Size: f.Size})
	}
	return out, nil
}

// localSearch lists the cached models whose id holds every word of query.
func localSearch(hub, query string, limit int) []Model {
	models, _ := hfcache.ScanCache(hub)
	words := strings.Fields(strings.ToLower(query))
	var out []Model
	for _, m := range models {
		id := strings.ToLower(m.Repo)
		match := true
		for _, w := range words {
			match = match && strings.Contains(id, w)
		}
		if match {
			out = append(out, Model{ID: m.Repo})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}
