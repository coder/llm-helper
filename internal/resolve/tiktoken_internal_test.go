package resolve

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/vllm"
)

// setup creates the encodings directory and points vLLM at it, but nothing
// filled it -- so a gpt-oss model started healthy and then failed every
// request with "invalid tiktoken vocab file". Measured on a GB10.
func TestHarmonyVocabIsFetchedWhenMissing(t *testing.T) {
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = append(asked, strings.TrimPrefix(r.URL.Path, "/"))
		w.Write([]byte("vocab bytes"))
	}))
	defer srv.Close()
	defer swapEncodingsURL(srv.URL + "/")()

	data := t.TempDir()
	if note := ensureHarmonyEncodings(data); note == "" {
		t.Error("fetched the vocab files without saying so")
	}
	for _, name := range harmonyEncodings {
		if _, err := os.Stat(filepath.Join(data, "tiktoken-encodings", name)); err != nil {
			t.Errorf("%s was not written: %v", name, err)
		}
	}

	// A second pull must not refetch: these are several megabytes and do
	// not change.
	before := len(asked)
	if note := ensureHarmonyEncodings(data); note != "" {
		t.Errorf("refetched files already on disk: %q", note)
	}
	if len(asked) != before {
		t.Errorf("made %d more requests for files already present", len(asked)-before)
	}
}

// A failed fetch has to say so: the server will start, look healthy, and then
// refuse every request.
func TestAFailedVocabFetchIsReported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusInternalServerError)
	}))
	defer srv.Close()
	defer swapEncodingsURL(srv.URL + "/")()

	note := ensureHarmonyEncodings(t.TempDir())
	if !strings.Contains(note, "harmony") || !strings.Contains(note, "error") {
		t.Errorf("note = %q, want it to name harmony and the failure", note)
	}
}

// Only gpt-oss-style models read a vocab from disk; nothing else should pay
// for the download.
func TestOnlyHarmonyModelsNeedTheVocab(t *testing.T) {
	for arch, want := range map[string]bool{
		"gpt_oss": true, "qwen3": false, "llama": false, "": false,
	} {
		if got := needsHarmony(arch); got != want {
			t.Errorf("needsHarmony(%q) = %v, want %v", arch, got, want)
		}
	}
}

// spark is what a DGX Spark running vLLM 0.27.1 reports, which is where the
// empty-message failure was measured.
var spark = vllm.Caps{
	ToolParsers:      []string{"hermes", "openai", "pythonic"},
	ReasoningParsers: []string{"deepseek_r1", "openai_gptoss", "qwen3"},
	Flags:            []string{"--tool-call-parser", "--reasoning-parser", "--enable-auto-tool-choice"},
}

// Without these flags vLLM has no way to read a harmony model's tool calls, so
// it returns content "", tool_calls null and finish_reason "stop" -- 47 tokens
// generated and nothing handed back. Plain chat still answers, which is what
// makes it read as the model ignoring the request.
func TestAHarmonyModelGetsTheParsersThatMakeItUsable(t *testing.T) {
	flags, notes := harmonyServeFlags(spark, "")
	got := strings.Join(flags, " ")
	want := "--tool-call-parser openai --enable-auto-tool-choice --reasoning-parser openai_gptoss"
	if got != want {
		t.Errorf("harmony flags = %q, want %q", got, want)
	}
	if len(notes) != 2 {
		t.Errorf("got %d notes, want one per flag pair: %v", len(notes), notes)
	}
}

// The card's author knows the model. Filling a gap is help; overriding a
// deliberate choice is a different thing, and would also pass the same flag
// twice.
func TestAParserTheCardAlreadyNamedIsLeftAlone(t *testing.T) {
	flags, _ := harmonyServeFlags(spark, "--tool-call-parser hermes")
	got := strings.Join(flags, " ")
	if strings.Contains(got, "--tool-call-parser") {
		t.Errorf("overrode the card's tool parser: %q", got)
	}
	if !strings.Contains(got, "--reasoning-parser openai_gptoss") {
		t.Errorf("the gap that was left should still be filled: %q", got)
	}
}

// A name is only worth writing if this vLLM registers it. Parser names move
// between releases, and a profile naming one that is gone will not start at
// all -- strictly worse than the empty answers it was meant to fix.
func TestNoParserIsNamedThatThisVLLMDoesNotRegister(t *testing.T) {
	for _, caps := range []vllm.Caps{
		{}, // introspection failed: we could not ask
		{Flags: []string{"--tool-call-parser", "--reasoning-parser"}},                  // registries empty
		{ToolParsers: []string{"openai"}, ReasoningParsers: []string{"openai_gptoss"}}, // flags unreadable
	} {
		if flags, _ := harmonyServeFlags(caps, ""); len(flags) > 0 {
			t.Errorf("named a parser without confirming it exists: %v (caps %+v)", flags, caps)
		}
	}
}
