package coder_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/plugin/coder"
)

// The collection is deployment-wide: an Anthropic model sits in the same list
// as ours. Pruning without filtering by provider would delete models this
// machine has no business touching.
func TestListOnlyReturnsThisProvidersEntries(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"models":[
		  {"id":"1","ai_provider_id":"ours","model":"openai/gpt-oss-120b","enabled":true},
		  {"id":"2","ai_provider_id":"anthropic-uuid","model":"claude-opus-4-7","enabled":true},
		  {"id":"3","ai_provider_id":"ours","model":"Qwen/Qwen3-0.6B","enabled":false}
		]}`)
	}))
	defer srv.Close()

	got, err := coder.NewClient(srv.URL, "tok", "ours", "default").List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("List returned %d entries, want the 2 on our provider", len(got))
	}
	for _, e := range got {
		if e.AIProviderID != "ours" {
			t.Errorf("List returned %s, which belongs to provider %q", e.Model, e.AIProviderID)
		}
		if strings.HasPrefix(e.Model, "claude") {
			t.Errorf("another provider's model (%s) is in range of a prune", e.Model)
		}
	}
}

// A deployment with no entries for us is not an error, and must not read as
// "everything is an orphan".
func TestListIsEmptyRatherThanWrongWhenNothingIsOurs(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"models":[{"id":"1","ai_provider_id":"someone-else","model":"claude-opus-4-7"}]}`)
	}))
	defer srv.Close()

	got, err := coder.NewClient(srv.URL, "tok", "ours", "default").List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("List returned %d entries, want none", len(got))
	}
}
