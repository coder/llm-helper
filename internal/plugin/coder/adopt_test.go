package coder_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/plugin/coder"
)

// listing serves a fixed collection to every path, the way newClient does.
func listing(t *testing.T, body string) *coder.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return coder.NewClient(srv.URL, "session-token", "provider-uuid", "default")
}

// Adoption is what stops a second row appearing for a model that already has
// one. The enabled row wins so adoption settles on the one in use, and rows
// belonging to another provider are never ours to adopt.
func TestFindByModel(t *testing.T) {
	t.Parallel()

	const body = `{"models":[
		{"id":"other-provider","ai_provider_id":"someone-else","model":"deepseek","enabled":true},
		{"id":"disabled-one","ai_provider_id":"provider-uuid","model":"deepseek","enabled":false},
		{"id":"enabled-one","ai_provider_id":"provider-uuid","model":"deepseek","enabled":true},
		{"id":"unrelated","ai_provider_id":"provider-uuid","model":"qwen","enabled":false}
	]}`

	cases := []struct {
		name  string
		model string
		want  string
	}{
		{"prefers the enabled row", "deepseek", "enabled-one"},
		{"falls back to a disabled row", "qwen", "unrelated"},
		{"nothing to adopt", "llama", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := listing(t, body).FindByModel(c.model)
			if err != nil {
				t.Fatal(err)
			}
			if got != c.want {
				t.Errorf("FindByModel(%q) = %q, want %q", c.model, got, c.want)
			}
		})
	}
}

// model_config is omitted unless set, so an entry without options cannot
// clear options an operator configured in Coder.
func TestModelConfigOmitsEmptyOptions(t *testing.T) {
	t.Parallel()

	var got capture
	client := newClient(t, http.StatusOK, `{"id":"new"}`, &got)
	if _, err := client.Create(coder.ModelConfig{Model: "m", ContextLimit: 1}); err != nil {
		t.Fatal(err)
	}
	if body := string(got.body); strings.Contains(body, "model_config") {
		t.Errorf("empty options were still sent: %s", body)
	}

	if _, err := client.Create(coder.ModelConfig{
		Model: "m", ContextLimit: 1,
		Options: map[string]any{"max_output_tokens": 64000},
	}); err != nil {
		t.Fatal(err)
	}
	if body := string(got.body); !strings.Contains(body, `"model_config":{"max_output_tokens":64000}`) {
		t.Errorf("options missing from body: %s", body)
	}
}
