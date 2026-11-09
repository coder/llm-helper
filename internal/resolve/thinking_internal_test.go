package resolve

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/vllm"
)

const kwargsFlag = "--default-chat-template-kwargs"

// serveThinking answers config.json with modelType and the chat template
// with template, or 404 when template is "".
func serveThinking(t *testing.T, modelType, template string) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/config.json"):
			fmt.Fprintf(w, `{"model_type":%q}`, modelType)
		case strings.HasSuffix(r.URL.Path, "chat_template.jinja") && template != "":
			fmt.Fprint(w, template)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(hf.SetBaseURLForTest(srv.URL))
}

func vllmProvider(t *testing.T) engine.Provider {
	t.Helper()
	prov, err := engine.Lookup(engine.VLLM)
	if err != nil {
		t.Fatal(err)
	}
	return prov
}

func TestAThinkingModelIsNotedOnItsProfile(t *testing.T) {
	for _, c := range []struct {
		name, modelType, template string
		want                      string
	}{
		{"switch in the template", "qwen3", "{%- if enable_thinking is defined and enable_thinking is false %}", ThinkingOn},
		{"no switch in the template", "qwen3", "{{ messages }}", ""},
		{"model type without a switch", "llama", "{{ enable_thinking }}", ""},
		{"switch off by default", "deepseek_v3", "{% if thinking %}<think>{% endif %}", ThinkingOff},
		{"engine-side formatting with no template", "deepseek_v4", "", ThinkingOn},
	} {
		t.Run(c.name, func(t *testing.T) {
			serveThinking(t, c.modelType, c.template)
			p := PrepareFor(context.Background(), config.Dirs{Data: t.TempDir()}, vllmProvider(t), "org/model", nil, vllm.Caps{}, 0)
			if p.Opts.Thinking != c.want {
				t.Errorf("Thinking = %q, want %q", p.Opts.Thinking, c.want)
			}
			if (p.Thinking != nil) != (c.want != "") {
				t.Errorf("plan Thinking = %+v, want it set only for a thinking model", p.Thinking)
			}
			noted := slices.ContainsFunc(p.Notes, func(n string) bool { return strings.Contains(n, "Thinking") })
			if noted != (c.want != "") {
				t.Errorf("Notes = %q, want a thinking note only for a thinking model", p.Notes)
			}
		})
	}
}

func TestSetThinkingWritesTheKwarg(t *testing.T) {
	serveThinking(t, "qwen3", "{{ enable_thinking }}")
	p := PrepareFor(context.Background(), config.Dirs{Data: t.TempDir()}, vllmProvider(t), "org/model", nil, vllm.Caps{}, 0)
	before := p.Opts.ExtraArgs
	if err := p.SetThinking(ThinkingOff, vllmProvider(t), vllm.Caps{}); err != nil {
		t.Fatal(err)
	}
	if p.Opts.Thinking != ThinkingOff {
		t.Errorf("Thinking = %q, want off", p.Opts.Thinking)
	}
	words, err := config.SplitArgs(p.Opts.ExtraArgs)
	if err != nil {
		t.Fatal(err)
	}
	n := len(words)
	if n < 2 || words[n-2] != kwargsFlag || words[n-1] != `{"enable_thinking":false}` {
		t.Errorf("ExtraArgs = %q, want it to end with %s and the kwarg", p.Opts.ExtraArgs, kwargsFlag)
	}
	if !strings.HasPrefix(p.Opts.ExtraArgs, before) {
		t.Errorf("ExtraArgs = %q, want %q kept", p.Opts.ExtraArgs, before)
	}
	if slices.ContainsFunc(p.Notes, func(n string) bool { return strings.HasPrefix(n, "Thinking model:") }) {
		t.Errorf("Notes = %q, want the note for thinking on replaced", p.Notes)
	}

	// Asking for what the model already does adds nothing.
	q := PrepareFor(context.Background(), config.Dirs{Data: t.TempDir()}, vllmProvider(t), "org/model", nil, vllm.Caps{}, 0)
	args := q.Opts.ExtraArgs
	if err := q.SetThinking(ThinkingOn, vllmProvider(t), vllm.Caps{}); err != nil {
		t.Fatal(err)
	}
	if q.Opts.ExtraArgs != args || q.Opts.Thinking != ThinkingOn {
		t.Errorf("ExtraArgs = %q, Thinking = %q, want the args unchanged and thinking on", q.Opts.ExtraArgs, q.Opts.Thinking)
	}
}

// The value goes through the env file, so it must read back as the JSON it
// was written as.
func TestTheKwargSurvivesTheEnvFile(t *testing.T) {
	args, err := withThinking(`--max-model-len 8192 --default-chat-template-kwargs '{"custom":"a b"}'`, kwargsFlag, "enable_thinking", false)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "p.env")
	if err := os.WriteFile(path, []byte("EXTRA_ARGS="+config.Quote(args)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := config.ParseEnvFile(path)
	if err != nil {
		t.Fatal(err)
	}
	words, err := config.SplitArgs(f.Get("EXTRA_ARGS"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--max-model-len", "8192", kwargsFlag, `{"custom":"a b","enable_thinking":false}`}
	if !slices.Equal(words, want) {
		t.Errorf("words = %q, want %q", words, want)
	}
}

func TestThinkingInTakesTheLastKwarg(t *testing.T) {
	for _, c := range []struct {
		args string
		def  bool
		want bool
	}{
		{"", true, true},
		{"--max-model-len 8192", false, false},
		{`--default-chat-template-kwargs {"enable_thinking":false}`, true, false},
		{`--default-chat-template-kwargs='{"enable_thinking":true}'`, false, true},
		{`--default-chat-template-kwargs {"enable_thinking":true} --default-chat-template-kwargs {"enable_thinking":false}`, true, false},
		{`--default-chat-template-kwargs {"other":1}`, true, true},
	} {
		if got := thinkingIn(c.args, kwargsFlag, "enable_thinking", c.def); got != c.want {
			t.Errorf("thinkingIn(%q, default %v) = %v, want %v", c.args, c.def, got, c.want)
		}
	}
}

func TestSetThinkingRefuses(t *testing.T) {
	serveThinking(t, "qwen3", "{{ enable_thinking }}")
	dirs := config.Dirs{Data: t.TempDir()}
	thinking := PrepareFor(context.Background(), dirs, vllmProvider(t), "org/model", nil, vllm.Caps{}, 0)
	mlx, err := engine.Lookup(engine.MLX)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		plan Plan
		mode string
		prov engine.Provider
		caps vllm.Caps
		want string
	}{
		{"a mode that is not on or off", thinking, "maybe", vllmProvider(t), vllm.Caps{}, "on or off"},
		{"a model with no switch", Plan{}, ThinkingOff, vllmProvider(t), vllm.Caps{}, "no thinking switch"},
		{"an engine with no kwargs flag", thinking, ThinkingOff, mlx, vllm.Caps{}, "no flag for chat template kwargs"},
		{"a vLLM without the flag", thinking, ThinkingOff, vllmProvider(t), vllm.Caps{Version: "0.8.0", Flags: []string{"--port"}}, "does not take"},
	} {
		p := c.plan
		if err := p.SetThinking(c.mode, c.prov, c.caps); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want one mentioning %q", c.name, err, c.want)
		}
	}
	p := thinking
	if err := p.SetThinking("", vllmProvider(t), vllm.Caps{}); err != nil || p.Opts.ExtraArgs != thinking.Opts.ExtraArgs {
		t.Errorf("an empty mode: err %v, ExtraArgs %q, want the plan left alone", err, p.Opts.ExtraArgs)
	}
}
