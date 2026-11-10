package warmup

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestPrepareRewritesForOneToken(t *testing.T) {
	raw := `{"model":"other","stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":900,"tools":[{"type":"function"}],"messages":[{"role":"system","content":"s"},{"role":"user","content":"Hi"}]}`
	out, err := Prepare([]byte(raw), "served")
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got["model"] != "served" || got["stream"] != false || got["max_tokens"] != float64(1) {
		t.Fatalf("not rewritten: %v", got)
	}
	for _, k := range []string{"stream_options", "max_completion_tokens"} {
		if _, ok := got[k]; ok {
			t.Errorf("%s kept", k)
		}
	}
	if _, ok := got["tools"]; !ok {
		t.Error("tools dropped")
	}
}

func TestPrepareRejectsBodiesWithoutMessages(t *testing.T) {
	for _, raw := range []string{`[]`, `{"model":"x"}`, `not json`} {
		if _, err := Prepare([]byte(raw), "m"); err == nil {
			t.Errorf("%s accepted", raw)
		}
	}
}

func TestRunSendsKeyAndReportsCachedTokens(t *testing.T) {
	var gotAuth, gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth, gotPath = r.Header.Get("Authorization"), r.URL.Path
		data, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(data, &gotBody)
		_, _ = w.Write([]byte(`{"usage":{"prompt_tokens":9400,"prompt_tokens_details":{"cached_tokens":9390}}}`))
	}))
	defer srv.Close()
	path := filepath.Join(t.TempDir(), "w.json")
	if err := os.WriteFile(path, []byte(`{"messages":[{"role":"user","content":"Hi"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	res, err := Run(context.Background(), srv.URL+"/v1", "k", "m", path)
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer k" || gotPath != "/v1/chat/completions" || gotBody["model"] != "m" {
		t.Fatalf("auth %q path %q body %v", gotAuth, gotPath, gotBody)
	}
	if res.PromptTokens != 9400 || res.CachedTokens != 9390 {
		t.Fatalf("got %+v", res)
	}
}

func TestSendWithoutCachedTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"usage":{"prompt_tokens":12}}`))
	}))
	defer srv.Close()
	res, err := Send(context.Background(), srv.Client(), srv.URL+"/v1/", "", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.PromptTokens != 12 || res.CachedTokens != -1 {
		t.Fatalf("got %+v", res)
	}
}

func TestSendReportsServerErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no such model", http.StatusNotFound)
	}))
	defer srv.Close()
	if _, err := Send(context.Background(), srv.Client(), srv.URL, "", []byte(`{}`)); err == nil {
		t.Fatal("no error")
	}
}

func TestUsageCountsResponsesAPI(t *testing.T) {
	var u Usage
	if err := json.Unmarshal([]byte(`{"input_tokens":50,"input_tokens_details":{"cached_tokens":40}}`), &u); err != nil {
		t.Fatal(err)
	}
	if p, c := u.Counts(); p != 50 || c != 40 {
		t.Fatalf("got %d %d", p, c)
	}
}

func TestFromCaptureKeepsSharedPrefix(t *testing.T) {
	raw := `{"model":"m","stream":true,"tools":[{"type":"function"}],"messages":[
		{"role":"system","content":"rules"},
		{"role":"user","content":"fix the bug in main.go"},
		{"role":"assistant","content":"ok"},
		{"role":"user","content":"thanks"}]}`
	out, err := FromCapture([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Tools    []any               `json:"tools"`
		Messages []map[string]string `json:"messages"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Tools) != 1 || len(got.Messages) != 2 {
		t.Fatalf("got %s", out)
	}
	if got.Messages[0]["content"] != "rules" || got.Messages[1]["content"] != Placeholder {
		t.Fatalf("got %s", out)
	}
	if _, err := FromCapture([]byte(`{"messages":[{"role":"system","content":"s"}]}`)); err == nil {
		t.Fatal("request without a user message accepted")
	}
}
