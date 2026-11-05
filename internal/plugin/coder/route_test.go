package coder_test

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/coder/llm-helper/internal/plugin/coder"
)

const (
	orgPath = "/api/v2/organizations/default/chats/models"
	expPath = "/api/experimental/chats/model-configs"
)

// deployment stands in for a Coder that serves exactly one of the two
// collection routes and 404s the other, which is what the real ones do:
// Coder's router reserves the retired path and answers it with a 404 rather
// than letting it fall through.
func deployment(t *testing.T, serves string, seen *[]string) *coder.Client {
	t.Helper()
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		*seen = append(*seen, r.Method+" "+r.URL.Path)
		mu.Unlock()
		if !strings.HasPrefix(r.URL.Path, serves) {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"new-id"}`)
	}))
	t.Cleanup(srv.Close)
	return coder.NewClient(srv.URL, "tok", "provider-uuid", "default")
}

// The bug this fixes: against Coder 2.37 the client called the retired path
// and got a 404 that read like a bad URL or an expired token.
func TestTheClientFindsTheCollectionOnEitherGeneration(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name  string
		route string
	}{
		{"a current deployment", orgPath},
		{"a pre-2.37 deployment", expPath},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var seen []string
			c := deployment(t, tc.route, &seen)
			if err := c.Check(); err != nil {
				t.Fatalf("Check: %v", err)
			}
			id, err := c.Create(coder.ModelConfig{Model: "m", ContextLimit: 4096})
			if err != nil {
				t.Fatalf("Create: %v", err)
			}
			if id != "new-id" {
				t.Errorf("Create returned %q, want the id the API gave", id)
			}
			for _, s := range seen {
				if strings.HasPrefix(s, "POST ") && !strings.Contains(s, tc.route) {
					t.Errorf("posted to %q, but this deployment serves %q", s, tc.route)
				}
			}
		})
	}
}

// Resolution costs one request, not one per call: a sync makes several.
func TestTheRouteIsResolvedOnce(t *testing.T) {
	t.Parallel()
	var seen []string
	c := deployment(t, orgPath, &seen)
	_ = c.Check()
	_, _ = c.Create(coder.ModelConfig{Model: "m", ContextLimit: 4096})
	_ = c.Update("id-1", coder.ModelConfig{Model: "m", ContextLimit: 4096})
	_ = c.Disable("id-2")

	probes := 0
	for _, s := range seen {
		if s == "GET "+orgPath {
			probes++
		}
	}
	// One from resolution, one from Check itself.
	if probes > 2 {
		t.Errorf("probed the route %d times across four calls", probes)
	}
}

// A saved ID outlives its entry when someone deletes the model in Coder's UI,
// or when CODER_URL is pointed at a different deployment. Callers recover by
// re-creating, which they can only do if the error says what happened.
func TestAMissingEntryIsDistinguishable(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == orgPath {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"models":[]}`)
			return
		}
		http.NotFound(w, r) // any specific entry is gone
	}))
	t.Cleanup(srv.Close)
	c := coder.NewClient(srv.URL, "tok", "prov", "default")

	err := c.Update("stale-id", coder.ModelConfig{Model: "m", ContextLimit: 4096})
	if !errors.Is(err, coder.ErrNotFound) {
		t.Errorf("Update on a deleted entry gave %v, want ErrNotFound", err)
	}
	// And the collection being missing must NOT read as a missing entry --
	// that is a broken deployment, not something to recreate around.
	if err := c.Check(); errors.Is(err, coder.ErrNotFound) {
		t.Error("an unusable API reported itself as a missing entry")
	}
}

// The create response echoes the whole entry back. Reading fields we do not
// need is how the client broke once already: a speculative {"model":{"id":..}}
// shape failed to decode the real reply, where "model" is the model's name.
func TestCreateReadsTheIDOutOfTheFullEntry(t *testing.T) {
	t.Parallel()
	const real = `{"id":"abc","organization_id":"org-1","ai_provider_id":"prov",
	               "model":"Qwen/Qwen3-0.6B","display_name":"Qwen3 0.6B","enabled":false,
	               "is_default":false,"context_limit":40960,"compression_threshold":70,
	               "created_at":"2026-09-04T00:00:00Z"}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, real)
	}))
	defer srv.Close()
	id, err := coder.NewClient(srv.URL, "tok", "prov", "default").Create(coder.ModelConfig{Model: "m"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if id != "abc" {
		t.Errorf("got id %q, want abc", id)
	}
}

// An entry we cannot name is one we can never update or disable, and staying
// silent would create a duplicate on every later sync.
func TestCreateWithoutAnIDIsAnError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"model":"m"}`)
	}))
	defer srv.Close()
	if _, err := coder.NewClient(srv.URL, "tok", "prov", "default").Create(coder.ModelConfig{Model: "m"}); err == nil {
		t.Error("a create that returned no id was reported as success")
	}
}

// A deployment with more than one organization has no "default" to fall back
// on, so the organization has to come from configuration rather than being
// assumed from whichever deployment happened to be tested against.
func TestTheOrganizationIsNotAssumed(t *testing.T) {
	t.Parallel()
	var seen []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.URL.Path)
		mu.Unlock()
		if !strings.Contains(r.URL.Path, "/organizations/engineering/") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"new-id"}`)
	}))
	defer srv.Close()

	c := coder.NewClient(srv.URL, "tok", "prov", "engineering")
	if _, err := c.Create(coder.ModelConfig{Model: "m", ContextLimit: 4096}); err != nil {
		t.Fatalf("Create against a named organization: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, p := range seen {
		if strings.Contains(p, "/organizations/default/") {
			t.Errorf("asked for the default organization (%s) despite one being configured", p)
		}
	}
}

// A misconfigured organization and a deployment too old to have the route
// both answer 404, and the fallback then fails too. Blaming only the retired
// path sends the reader looking at CODER_URL and their token, which is the
// diagnosis that already cost real time once.
func TestABadOrganizationIsNamedInTheError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.NotFound(w, nil)
	}))
	defer srv.Close()

	err := coder.NewClient(srv.URL, "tok", "prov", "no-such-org").Check()
	if err == nil {
		t.Fatal("a deployment serving neither route reported success")
	}
	if !strings.Contains(err.Error(), "CODER_ORGANIZATION") ||
		!strings.Contains(err.Error(), "no-such-org") {
		t.Errorf("error does not name the organization as a suspect: %v", err)
	}
}
