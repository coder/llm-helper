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
	body, err := parseObject(raw)
	if err != nil {
		return nil, fmt.Errorf("warm-up request is not a JSON object: %w", err)
	}
	var msgs []json.RawMessage
	if v, ok := body.get("messages"); ok {
		_ = json.Unmarshal(v, &msgs)
	}
	if len(msgs) == 0 {
		return nil, fmt.Errorf("warm-up request has no messages")
	}
	if model != "" {
		body.set("model", mustJSON(model))
	}
	body.set("max_tokens", json.RawMessage("1"))
	body.set("stream", json.RawMessage("false"))
	body.del("max_completion_tokens")
	body.del("stream_options")
	body.del("n")
	return body.marshal(), nil
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
// with Placeholder as the user message. Everything it keeps stays byte for
// byte, so the order of keys the chat template sees does not change.
func FromCapture(raw []byte) ([]byte, error) {
	body, err := parseObject(raw)
	if err != nil {
		return nil, fmt.Errorf("captured request is not a JSON object: %w", err)
	}
	var msgs []json.RawMessage
	if v, ok := body.get("messages"); ok {
		_ = json.Unmarshal(v, &msgs)
	}
	first := -1
	for i, m := range msgs {
		var role struct {
			Role string `json:"role"`
		}
		if json.Unmarshal(m, &role) == nil && role.Role == "user" {
			first = i
			break
		}
	}
	if first < 0 {
		return nil, fmt.Errorf("captured request has no user message")
	}
	msgs = msgs[:first+1]
	msgs[first] = mustJSON(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{"user", Placeholder})
	body.set("messages", joinArray(msgs))
	var out bytes.Buffer
	if err := json.Indent(&out, body.marshal(), "", "  "); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// object is a JSON object that keeps its keys in order and its values as
// they were written.
type object struct {
	keys []string
	vals map[string]json.RawMessage
}

func parseObject(raw []byte) (*object, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("not an object")
	}
	o := &object{vals: map[string]json.RawMessage{}}
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, _ := tok.(string)
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, err
		}
		o.set(key, v)
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data after the object")
	}
	return o, nil
}

func (o *object) get(key string) (json.RawMessage, bool) {
	v, ok := o.vals[key]
	return v, ok
}

func (o *object) set(key string, v json.RawMessage) {
	if _, ok := o.vals[key]; !ok {
		o.keys = append(o.keys, key)
	}
	o.vals[key] = v
}

func (o *object) del(key string) {
	if _, ok := o.vals[key]; !ok {
		return
	}
	delete(o.vals, key)
	for i, k := range o.keys {
		if k == key {
			o.keys = append(o.keys[:i], o.keys[i+1:]...)
			break
		}
	}
}

func (o *object) marshal() []byte {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, k := range o.keys {
		if i > 0 {
			b.WriteByte(',')
		}
		b.Write(mustJSON(k))
		b.WriteByte(':')
		b.Write(o.vals[k])
	}
	b.WriteByte('}')
	return b.Bytes()
}

func joinArray(items []json.RawMessage) json.RawMessage {
	var b bytes.Buffer
	b.WriteByte('[')
	for i, v := range items {
		if i > 0 {
			b.WriteByte(',')
		}
		b.Write(v)
	}
	b.WriteByte(']')
	return b.Bytes()
}

func mustJSON(v any) json.RawMessage {
	out, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return out
}
