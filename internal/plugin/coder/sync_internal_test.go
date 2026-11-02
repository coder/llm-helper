package coder

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/plugin"
	"github.com/coder/vllm-helper/internal/profile"
)

func profileWith(t *testing.T, body string) *profile.Profile {
	t.Helper()
	dir := t.TempDir()
	dirs := config.Dirs{Config: dir, Profiles: filepath.Join(dir, "models")}
	if err := os.MkdirAll(dirs.Profiles, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dirs.ProfileFile("ornith"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := profile.Load(dirs, "ornith")
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestModelConfigFromProfile(t *testing.T) {
	p := profileWith(t, "MODEL=deepreinforce-ai/Ornith-1.0-35B-FP8\nMAX_MODEL_LEN=65536\n")

	enabled, err := modelConfig(nil, p, true)
	if err != nil {
		t.Fatalf("modelConfig: %v", err)
	}
	if enabled.Model != "deepreinforce-ai/Ornith-1.0-35B-FP8" {
		t.Errorf("Model = %q", enabled.Model)
	}
	if enabled.DisplayName != "Ornith 1.0 35B FP8" {
		t.Errorf("DisplayName = %q", enabled.DisplayName)
	}
	if enabled.ContextLimit != 65536 {
		t.Errorf("ContextLimit = %d, want 65536", enabled.ContextLimit)
	}
	if !enabled.Enabled || !enabled.IsDefault {
		t.Error("the active model should be both enabled and the default")
	}

	disabled, err := modelConfig(nil, p, false)
	if err != nil {
		t.Fatalf("modelConfig: %v", err)
	}
	if disabled.Enabled || disabled.IsDefault {
		t.Error("a backfilled entry should be neither enabled nor the default")
	}
}

// Coder rejects context_limit <= 0, so a MAX_MODEL_LEN we can't turn into
// one has to be reported here. Sending it produced an HTTP 400 that said
// nothing about which local setting was wrong.
func TestModelConfigRejectsAContextLimitCoderWouldRefuse(t *testing.T) {
	for _, length := range []string{"", "lots", "0", "-1"} {
		t.Run(length, func(t *testing.T) {
			p := profileWith(t, "MODEL=some/model\nMAX_MODEL_LEN="+length+"\n")
			if _, err := modelConfig(nil, p, true); err == nil {
				t.Errorf("MAX_MODEL_LEN=%q was accepted, want an error naming the setting", length)
			}
		})
	}
}

// A non-vLLM profile has no MAX_MODEL_LEN, and failing sync for that would
// leave the previous model enabled in Coder after a switch. It gets the
// default context unless MAX_MODEL_LEN says otherwise.
func TestModelConfigForAnotherProvider(t *testing.T) {
	for _, tc := range []struct {
		name, profile string
		want          int64
	}{
		{"unset", "PROVIDER=mlx\nMODEL=mlx-community/m\nEXTRA_ARGS=\n", profile.DefaultMaxModelLen},
		{"set", "PROVIDER=llamacpp\nMODEL=org/m-GGUF:Q8_0\nMAX_MODEL_LEN=32768\n", 32768},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := modelConfig(nil, profileWith(t, tc.profile), true)
			if err != nil {
				t.Fatalf("modelConfig: %v", err)
			}
			if got.ContextLimit != tc.want {
				t.Errorf("ContextLimit = %d, want %d", got.ContextLimit, tc.want)
			}
		})
	}
	// vLLM is given MAX_MODEL_LEN as --max-model-len, so an empty one is a
	// broken profile rather than one to fill in. A PROVIDER nothing serves is
	// the same: it gets an error, not a context limit invented for it.
	if _, err := modelConfig(nil, profileWith(t, "PROVIDER=vllm\nMODEL=m\nMAX_MODEL_LEN=\n"), true); err == nil {
		t.Error("a vLLM profile with no MAX_MODEL_LEN was accepted")
	}
	if _, err := modelConfig(nil, profileWith(t, "PROVIDER=typo\nMODEL=m\n"), true); err == nil || !strings.Contains(err.Error(), "MAX_MODEL_LEN") {
		t.Errorf("unknown PROVIDER: err = %v, want a MAX_MODEL_LEN error, not a made-up limit", err)
	}
}

// Opting into an event without handling it would be a silent no-op, so every
// event this plugin advertises has to reach a real branch. An unconfigured
// Config means the handlers bail immediately, which is enough to tell "no
// handler" apart from "handled, then declined to do anything".
func TestEveryAdvertisedEventIsHandled(t *testing.T) {
	p := Plugin{}
	for _, e := range p.Events() {
		if err := p.Handle(context.Background(), e, plugin.State{Config: &config.Config{File: &config.EnvFile{}}}); err != nil {
			t.Errorf("Events() advertises %q but Handle rejected it: %v", e, err)
		}
	}
	if err := p.Handle(context.Background(), plugin.Event("invented"), plugin.State{}); err == nil {
		t.Error("an event with no handler was accepted, which would be a silent no-op")
	}
}

// Adoption maps a second profile for the same model onto the existing entry.
// Activating either profile must not then disable that entry on behalf of
// the other name, and removing one alias must not delete the row the other
// still uses.
func TestEntriesAreOwnedByID(t *testing.T) {
	var patched []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPatch {
			patched = append(patched, path.Base(r.URL.Path))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	t.Cleanup(srv.Close)
	client := NewClient(srv.URL, "token", "provider", "default")

	dir := t.TempDir()
	cfg := &config.Config{Dirs: config.Dirs{Config: dir, Profiles: filepath.Join(dir, "models"), State: dir}}
	if err := os.MkdirAll(cfg.Dirs.Profiles, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"alias-a", "alias-b", "other"} {
		if err := os.WriteFile(cfg.Dirs.ProfileFile(name), []byte("MODEL=m/"+name+"\nMAX_MODEL_LEN=1024\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	state := &config.EnvFile{}
	state.Set("alias-a", "shared-id")
	state.Set("alias-b", "shared-id")
	state.Set("other", "other-id")

	disableOthers(cfg, client, state, "alias-b")
	if strings.Join(patched, ",") != "other-id" {
		t.Errorf("disabled %v, want only other-id: the shared entry is the one just enabled", patched)
	}

	if got := sharedWith(state, "shared-id", "alias-a"); got != "alias-b" {
		t.Errorf("sharedWith = %q, want alias-b", got)
	}
	if got := sharedWith(state, "other-id", "other"); got != "" {
		t.Errorf("sharedWith = %q for an entry nobody else uses", got)
	}
}
