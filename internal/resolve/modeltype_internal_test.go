package resolve

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/modeltype"
	"github.com/coder/llm-helper/internal/vllm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// routeFetchesTo sends every file fetch to srv, whatever host it names.
func routeFetchesTo(t *testing.T, srv *httptest.Server) {
	t.Helper()
	target, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	old := httpClient
	httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		r = r.Clone(r.Context())
		r.URL.Scheme, r.URL.Host = target.Scheme, target.Host
		return http.DefaultTransport.RoundTrip(r)
	})}
	t.Cleanup(func() { httpClient = old })
}

var vocab = []modeltype.FileSet{{
	Dir:     "tiktoken-encodings",
	Env:     "TIKTOKEN_ENCODINGS_BASE",
	BaseURL: "https://example.invalid/encodings/",
	Names:   []string{"a.tiktoken", "b.tiktoken"},
	Why:     "The tokenizer reads these from disk.",
}}

func TestModelTypeFilesAreFetchedOnce(t *testing.T) {
	var mu sync.Mutex
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		asked = append(asked, r.URL.Path)
		mu.Unlock()
		w.Write([]byte("vocab bytes"))
	}))
	defer srv.Close()
	routeFetchesTo(t, srv)

	data := t.TempDir()
	env, notes := modelTypeFiles(data, vocab)
	dir := filepath.Join(data, "tiktoken-encodings")
	if len(env) != 1 || env[0] != "TIKTOKEN_ENCODINGS_BASE="+dir {
		t.Errorf("env = %v, want the variable pointing at %s", env, dir)
	}
	if len(notes) != 1 || !strings.Contains(notes[0], "a.tiktoken, b.tiktoken") {
		t.Errorf("notes = %v, want one naming both files", notes)
	}
	for _, name := range vocab[0].Names {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Errorf("%s was not written: %v", name, err)
		}
	}
	if asked[0] != "/encodings/a.tiktoken" {
		t.Errorf("fetched %s, want the base_url path", asked[0])
	}

	before := len(asked)
	if _, notes := modelTypeFiles(data, vocab); len(notes) != 0 {
		t.Errorf("refetched files already on disk: %v", notes)
	}
	if len(asked) != before {
		t.Errorf("made %d more requests for files already present", len(asked)-before)
	}
}

func TestAFailedFileFetchIsReported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusInternalServerError)
	}))
	defer srv.Close()
	routeFetchesTo(t, srv)

	_, notes := modelTypeFiles(t.TempDir(), vocab)
	if len(notes) != 2 || !strings.Contains(notes[0], "failed") || !strings.Contains(notes[0], vocab[0].Why) {
		t.Errorf("notes = %v, want each failure reported with the reason the file matters", notes)
	}
}

// spark is what a DGX Spark running vLLM 0.27.1 reports.
var spark = vllm.Caps{
	ToolParsers:      []string{"hermes", "openai", "pythonic"},
	ReasoningParsers: []string{"deepseek_r1", "openai_gptoss", "qwen3"},
	Flags:            []string{"--tool-call-parser", "--reasoning-parser", "--enable-auto-tool-choice"},
}

var harmonyArgs = []modeltype.Arg{
	{Flag: "--tool-call-parser", Values: []string{"openai", "openai_gptoss"}, With: []string{"--enable-auto-tool-choice"}, Why: "tools"},
	{Flag: "--reasoning-parser", Values: []string{"openai_gptoss", "gpt_oss"}, Why: "thinking"},
}

func TestModelTypeArgsFillGaps(t *testing.T) {
	flags, notes := modelTypeArgs(harmonyArgs, spark, "")
	got := strings.Join(flags, " ")
	want := "--tool-call-parser openai --enable-auto-tool-choice --reasoning-parser openai_gptoss"
	if got != want {
		t.Errorf("flags = %q, want %q", got, want)
	}
	if len(notes) != 2 || !strings.HasPrefix(notes[0], "Added --tool-call-parser openai and --enable-auto-tool-choice. ") {
		t.Errorf("notes = %v, want one per arg naming what was added", notes)
	}
}

// The card's author knows the model; a flag it already names is left alone.
func TestAnArgTheCardAlreadyNamedIsLeftAlone(t *testing.T) {
	flags, _ := modelTypeArgs(harmonyArgs, spark, "--tool-call-parser hermes")
	got := strings.Join(flags, " ")
	if strings.Contains(got, "--tool-call-parser") {
		t.Errorf("overrode the card's tool parser: %q", got)
	}
	if !strings.Contains(got, "--reasoning-parser openai_gptoss") {
		t.Errorf("the gap that was left should still be filled: %q", got)
	}
}

// A profile naming a parser this vLLM lacks will not start at all.
func TestNoValueIsNamedThatThisEngineDoesNotRegister(t *testing.T) {
	for _, caps := range []vllm.Caps{
		{},
		{Flags: []string{"--tool-call-parser", "--reasoning-parser"}},
		{ToolParsers: []string{"openai"}, ReasoningParsers: []string{"openai_gptoss"}},
	} {
		if flags, _ := modelTypeArgs(harmonyArgs, caps, ""); len(flags) > 0 {
			t.Errorf("named a parser without confirming it exists: %v (caps %+v)", flags, caps)
		}
	}
}

func TestShippedGPTOSSRulesProduceTheHarmonyFlags(t *testing.T) {
	rules := modeltype.Default().RulesFor("gpt_oss", "vllm")
	flags, _ := modelTypeArgs(rules.Args, spark, "")
	want := "--tool-call-parser openai --enable-auto-tool-choice --reasoning-parser openai_gptoss"
	if got := strings.Join(flags, " "); got != want {
		t.Errorf("flags = %q, want %q", got, want)
	}
	if len(rules.Files) != 1 || rules.Files[0].Env != "TIKTOKEN_ENCODINGS_BASE" {
		t.Errorf("files = %+v, want the tiktoken vocab", rules.Files)
	}
	for _, other := range [][2]string{{"qwen3", "vllm"}, {"gpt_oss", "mlx"}} {
		if r := modeltype.Default().RulesFor(other[0], other[1]); len(r.Args)+len(r.Files)+len(r.ChatTemplateFixes) > 0 {
			t.Errorf("%s on %s has rules %+v, want none", other[0], other[1], r)
		}
	}
}

func TestShippedDeepSeekV4RulesProduceItsTokenizerAndParsers(t *testing.T) {
	caps := vllm.Caps{
		ToolParsers:      []string{"deepseek_v4"},
		ReasoningParsers: []string{"deepseek_v4"},
		Flags:            []string{"--tokenizer-mode", "--tool-call-parser", "--reasoning-parser", "--enable-auto-tool-choice"},
	}
	flags, _ := modelTypeArgs(modeltype.Default().RulesFor("deepseek_v4", "vllm").Args, caps, "--trust-remote-code")
	want := "--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4"
	if got := strings.Join(flags, " "); got != want {
		t.Errorf("flags = %q, want %q", got, want)
	}
}

func TestAValueWithSpacesSurvivesTheProfile(t *testing.T) {
	caps := vllm.Caps{Flags: []string{"--download-dir"}}
	args := []modeltype.Arg{{Flag: "--download-dir", Values: []string{"/srv/Model Cache"}, Why: "w"}}
	flags, _ := modelTypeArgs(args, caps, "")
	got, err := config.SplitArgs(strings.Join(flags, " "))
	if err != nil || !slices.Equal(got, []string{"--download-dir", "/srv/Model Cache"}) {
		t.Errorf("EXTRA_ARGS splits to %q (%v), want the value kept whole", got, err)
	}
}
