package coder_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coder/llm-helper/internal/plugin/coder"
)

// Where a current deployment keeps the collection. The stand-in below answers
// every path, so route resolution settles on this one; the fallback to the
// retired /api/experimental path is covered in route_test.go.
const modelConfigPath = "/api/v2/organizations/default/chats/models"

// capture records what the deployment actually received.
type capture struct {
	method string
	path   string
	token  string
	body   []byte
}

// newClient points a real client at a local deployment stand-in.
func newClient(t *testing.T, status int, respBody string, got *capture) *coder.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		*got = capture{method: r.Method, path: r.URL.Path, token: r.Header.Get("Coder-Session-Token"), body: body}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, respBody)
	}))
	t.Cleanup(srv.Close)
	return coder.NewClient(srv.URL, "session-token", "provider-uuid", "default")
}

func TestRequestShape(t *testing.T) {
	t.Parallel()

	mc := coder.ModelConfig{
		Model:        "Qwen/Qwen3-8B-FP8",
		DisplayName:  "Qwen3 8B FP8",
		ContextLimit: 40960,
		Enabled:      true,
		IsDefault:    true,
	}
	tests := []struct {
		name       string
		call       func(*coder.Client) error
		wantMethod string
		wantPath   string
	}{{
		name:       "check lists the model configs",
		call:       func(c *coder.Client) error { return c.Check() },
		wantMethod: http.MethodGet,
		wantPath:   modelConfigPath,
	}, {
		name:       "create posts to the collection",
		call:       func(c *coder.Client) error { _, err := c.Create(mc); return err },
		wantMethod: http.MethodPost,
		wantPath:   modelConfigPath,
	}, {
		name:       "update patches one entry",
		call:       func(c *coder.Client) error { return c.Update("mc-1", mc) },
		wantMethod: http.MethodPatch,
		wantPath:   modelConfigPath + "/mc-1",
	}, {
		name:       "disable patches one entry",
		call:       func(c *coder.Client) error { return c.Disable("mc-1") },
		wantMethod: http.MethodPatch,
		wantPath:   modelConfigPath + "/mc-1",
	}, {
		name:       "delete removes one entry",
		call:       func(c *coder.Client) error { return c.Delete("mc-1") },
		wantMethod: http.MethodDelete,
		wantPath:   modelConfigPath + "/mc-1",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var got capture
			if err := tt.call(newClient(t, http.StatusOK, `{"id":"mc-1"}`, &got)); err != nil {
				t.Fatalf("call: %v", err)
			}
			if got.method != tt.wantMethod {
				t.Errorf("method = %q, want %q", got.method, tt.wantMethod)
			}
			if got.path != tt.wantPath {
				t.Errorf("path = %q, want %q", got.path, tt.wantPath)
			}
			if got.token != "session-token" {
				t.Errorf("Coder-Session-Token = %q, want the configured token", got.token)
			}
		})
	}
}

// The provider a model belongs to is the client's, not the caller's, so
// Create has to fill it in.
func TestCreateInjectsTheProviderID(t *testing.T) {
	t.Parallel()

	var got capture
	c := newClient(t, http.StatusCreated, `{"id":"mc-42"}`, &got)
	id, err := c.Create(coder.ModelConfig{
		Model:        "Qwen/Qwen3-8B-FP8",
		DisplayName:  "Qwen3 8B FP8",
		ContextLimit: 40960,
		Enabled:      true,
		IsDefault:    true,
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if id != "mc-42" {
		t.Errorf("Create returned %q, want the id the deployment assigned", id)
	}

	var sent map[string]any
	if err := json.Unmarshal(got.body, &sent); err != nil {
		t.Fatalf("request body isn't JSON: %v (%s)", err, got.body)
	}
	want := map[string]any{
		"ai_provider_id": "provider-uuid",
		"model":          "Qwen/Qwen3-8B-FP8",
		"display_name":   "Qwen3 8B FP8",
		"context_limit":  float64(40960),
		"enabled":        true,
		"is_default":     true,
	}
	for k, v := range want {
		if sent[k] != v {
			t.Errorf("body[%q] = %#v, want %#v", k, sent[k], v)
		}
	}
	if _, ok := sent["id"]; ok {
		t.Errorf("body carries an id the deployment hasn't assigned yet: %s", got.body)
	}
}

func TestDisableClearsBothFlags(t *testing.T) {
	t.Parallel()

	var got capture
	if err := newClient(t, http.StatusOK, "", &got).Disable("mc-1"); err != nil {
		t.Fatalf("Disable: %v", err)
	}
	var sent map[string]bool
	if err := json.Unmarshal(got.body, &sent); err != nil {
		t.Fatalf("request body isn't JSON: %v (%s)", err, got.body)
	}
	if sent["enabled"] || sent["is_default"] {
		t.Errorf("body = %s, want both flags false", got.body)
	}
	if len(sent) != 2 {
		t.Errorf("body = %s, want only enabled and is_default so the model isn't touched", got.body)
	}
}

func TestNon2xxIsAnError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		status int
	}{
		{"expired token", http.StatusUnauthorized},
		{"no experimental chats API", http.StatusNotFound},
		{"deployment error", http.StatusInternalServerError},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var got capture
			c := newClient(t, tt.status, "", &got)
			if err := c.Check(); err == nil {
				t.Error("Check should surface the failure")
			}
			if _, err := c.Create(coder.ModelConfig{Model: "a/b"}); err == nil {
				t.Error("Create should surface the failure")
			}
			if err := c.Update("mc-1", coder.ModelConfig{Model: "a/b"}); err == nil {
				t.Error("Update should surface the failure")
			}
			if err := c.Delete("mc-1"); err == nil {
				t.Error("Delete should surface the failure")
			}
		})
	}
}

func TestUnreachableDeployment(t *testing.T) {
	t.Parallel()

	// Port 1 refuses immediately and nothing can bind it by accident. This
	// used to start a server and close it, which left the result depending on
	// whether the freed port stayed free for the length of the test.
	if err := coder.NewClient("http://127.0.0.1:1", "tok", "prov", "default").Check(); err == nil {
		t.Error("Check should fail when nothing is listening")
	}
}
