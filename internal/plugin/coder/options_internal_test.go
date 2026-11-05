package coder

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coder/llm-helper/internal/config"
)

func cfgWith(kv map[string]string) *config.Config {
	f := &config.EnvFile{}
	for k, v := range kv {
		f.Set(k, v)
	}
	return &config.Config{File: f}
}

// Unset must send nothing at all: an empty model_config would overwrite
// whatever an operator configured in Coder's UI with nothing.
func TestOutputOptionsOnlySentWhenConfigured(t *testing.T) {
	cases := []struct {
		name string
		cfg  *config.Config
		want any
	}{
		{"nil config", nil, nil},
		{"unset", cfgWith(nil), nil},
		{"blank", cfgWith(map[string]string{keyMaxOutput: "   "}), nil},
		{"not a number", cfgWith(map[string]string{keyMaxOutput: "lots"}), nil},
		{"zero is not a limit", cfgWith(map[string]string{keyMaxOutput: "0"}), nil},
		{"set", cfgWith(map[string]string{keyMaxOutput: "64000"}), int64(64000)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := outputOptions(c.cfg)
			if c.want == nil {
				if got != nil {
					t.Errorf("options = %v, want nothing sent", got)
				}
				return
			}
			if got["max_output_tokens"] != c.want {
				t.Errorf("max_output_tokens = %v, want %v", got["max_output_tokens"], c.want)
			}
		})
	}
}

func TestDuplicateModelsGroupsOnlyRepeats(t *testing.T) {
	entries := []Entry{
		{ID: "a", Model: "one"},
		{ID: "b", Model: "two"},
		{ID: "c", Model: "one", Enabled: true},
		{ID: "d", Model: "three"},
	}
	got := duplicateModels(entries)
	if len(got) != 1 {
		t.Fatalf("groups = %d, want 1: %+v", len(got), got)
	}
	if len(got[0]) != 2 || got[0][0].ID != "a" || got[0][1].ID != "c" {
		t.Errorf("group = %+v, want the two 'one' entries in order", got[0])
	}
}

// A token is always expiring -- Coder caps its lifetime -- so only the last
// couple of days are worth reporting.
func TestTokenExpiringOnlyWarnsNearTheEnd(t *testing.T) {
	cases := []struct {
		name  string
		in    time.Duration
		warns bool
	}{
		{"a week out", 7 * 24 * time.Hour, false},
		{"three days", 72 * time.Hour, false},
		{"tomorrow", 20 * time.Hour, true},
		{"already gone", -time.Hour, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"abc","token_name":"vllm","expires_at":"` +
					time.Now().Add(c.in).UTC().Format(time.RFC3339) + `"}`))
			}))
			defer srv.Close()

			client := NewClient(srv.URL, "abcdefghij-secret", "provider", "default")
			err := tokenExpiring(client)
			if c.warns && err == nil {
				t.Error("expected a warning")
			}
			if !c.warns && err != nil {
				t.Errorf("unexpected warning: %v", err)
			}
		})
	}
}

// An unreadable key must not fail a health check: the sync itself works.
func TestTokenExpiringIsQuietWhenTheKeyCannotBeRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer srv.Close()
	if err := tokenExpiring(NewClient(srv.URL, "abcdefghij-secret", "p", "default")); err != nil {
		t.Errorf("got %v, want silence", err)
	}
	if err := tokenExpiring(NewClient(srv.URL, "no-dashes-here", "p", "default")); err != nil {
		t.Errorf("got %v, want silence", err)
	}
}
