// Package warmup sends a client's fixed prompt prefix to a freshly started
// server so the server's prompt cache holds it before the first real request.
package warmup

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// Timeout bounds one warm-up request; a long prefix on a slow machine takes a while.
const Timeout = 5 * time.Minute

// Result is what the server reported for one warm-up request.
type Result struct {
	PromptTokens int
	// CachedTokens is -1 when the server did not report it.
	CachedTokens int
}

// Load reads a chat completions request body and rewrites it for the given
// served model so that it generates a single token.
func Load(path, model string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Prepare(raw, model)
}

// Prepare rewrites a chat completions request body for the given model so
// that it generates a single token without streaming.
func Prepare(raw []byte, model string) ([]byte, error) {
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("warm-up request is not a JSON object: %w", err)
	}
	msgs, _ := body["messages"].([]any)
	if len(msgs) == 0 {
		return nil, fmt.Errorf("warm-up request has no messages")
	}
	if model != "" {
		body["model"] = model
	}
	body["max_tokens"] = 1
	body["stream"] = false
	delete(body, "max_completion_tokens")
	delete(body, "stream_options")
	delete(body, "n")
	return json.Marshal(body)
}

// Send posts body to base + "/chat/completions". base is the server's
// OpenAI-compatible root, ending in /v1.
func Send(ctx context.Context, client *http.Client, base, key string, body []byte) (Result, error) {
	res := Result{CachedTokens: -1}
	url := strings.TrimSuffix(base, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return res, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := client.Do(req)
	if err != nil {
		return res, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return res, err
	}
	if resp.StatusCode != http.StatusOK {
		return res, fmt.Errorf("server answered %s: %s", resp.Status, strings.TrimSpace(string(data)))
	}
	var reply struct {
		Usage *Usage `json:"usage"`
	}
	if err := json.Unmarshal(data, &reply); err != nil {
		return res, fmt.Errorf("server reply is not JSON: %w", err)
	}
	if reply.Usage != nil {
		res.PromptTokens, res.CachedTokens = reply.Usage.Counts()
	}
	return res, nil
}

// Usage is the usage block of an OpenAI-compatible reply: chat completions
// name the fields prompt_*, the Responses API names them input_*.
type Usage struct {
	PromptTokens        *int     `json:"prompt_tokens"`
	InputTokens         *int     `json:"input_tokens"`
	PromptTokensDetails *details `json:"prompt_tokens_details"`
	InputTokensDetails  *details `json:"input_tokens_details"`
}

type details struct {
	CachedTokens *int `json:"cached_tokens"`
}

// Counts returns the prompt tokens and the cached prompt tokens, -1 for
// whichever the server did not report.
func (u *Usage) Counts() (prompt, cached int) {
	prompt, cached = -1, -1
	switch {
	case u.PromptTokens != nil:
		prompt = *u.PromptTokens
	case u.InputTokens != nil:
		prompt = *u.InputTokens
	}
	switch {
	case u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens != nil:
		cached = *u.PromptTokensDetails.CachedTokens
	case u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens != nil:
		cached = *u.InputTokensDetails.CachedTokens
	}
	return prompt, cached
}

// Run loads path and sends it once.
func Run(ctx context.Context, base, key, model, path string) (Result, error) {
	body, err := Load(path, model)
	if err != nil {
		return Result{CachedTokens: -1}, err
	}
	ctx, cancel := context.WithTimeout(ctx, Timeout)
	defer cancel()
	return Send(ctx, http.DefaultClient, base, key, body)
}

// Describe renders a result for people.
func (r Result) Describe() string {
	if r.CachedTokens < 0 {
		return fmt.Sprintf("%d prompt tokens (the server does not report cached tokens)", r.PromptTokens)
	}
	return fmt.Sprintf("%d prompt tokens, %d from cache", r.PromptTokens, r.CachedTokens)
}

// Placeholder is the user message a captured request ends with.
const Placeholder = "Hi"

// FromCapture turns a client's chat request into a warm-up request: it keeps
// what comes before the first user message, which every chat shares, and ends
// with Placeholder as the user message.
func FromCapture(raw []byte) ([]byte, error) {
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("captured request is not a JSON object: %w", err)
	}
	msgs, _ := body["messages"].([]any)
	first := -1
	for i, m := range msgs {
		if mm, ok := m.(map[string]any); ok && mm["role"] == "user" {
			first = i
			break
		}
	}
	if first < 0 {
		return nil, fmt.Errorf("captured request has no user message")
	}
	msgs = msgs[:first+1]
	msgs[first] = map[string]any{"role": "user", "content": Placeholder}
	body["messages"] = msgs
	return json.MarshalIndent(body, "", "  ")
}
