package resolve

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/vllm"
)

// harmonyEncodings are the vocab files the harmony tokenizer reads, by the
// names it looks for under TIKTOKEN_ENCODINGS_BASE.
var harmonyEncodings = []string{"o200k_base.tiktoken", "cl100k_base.tiktoken"}

var encodingsURL = "https://openaipublic.blob.core.windows.net/encodings/"

// swapEncodingsURL points the fetch at a stand-in and returns a restore func.
func swapEncodingsURL(url string) func() {
	old := encodingsURL
	encodingsURL = url
	return func() { encodingsURL = old }
}

// needsHarmony reports whether a model tokenizes with harmony, which reads its
// vocab from disk rather than from the model repo.
func needsHarmony(modelType string) bool { return modelType == "gpt_oss" }

// Parser names for harmony output, best first. They have moved between
// releases, so which one is real is a question for the installed vLLM rather
// than something to hardcode.
var (
	harmonyToolParsers      = []string{"openai", "openai_gptoss", "gpt_oss"}
	harmonyReasoningParsers = []string{"openai_gptoss", "gpt_oss", "openai"}
)

// harmonyServeFlags reports the flags a harmony model needs before a client can
// use it, and says what it added.
//
// A harmony model emits its tool calls and its thinking as channels in its own
// format rather than as JSON. Without parsers for them vLLM has nothing to turn
// that into tool_calls, and drops the message: content "", tool_calls null,
// finish_reason "stop". Plain chat still answers normally, so it reads as the
// model ignoring the request rather than as a missing flag. Measured on a DGX
// Spark serving gpt-oss-120b, where a tool request generated 47 tokens and
// returned nothing at all.
//
// present is what has already been asked for. Anything named there wins: the
// card's author knows the model, and this is only filling a gap.
func harmonyServeFlags(caps vllm.Caps, present string) (flags []string, notes []string) {
	// Every flag below is gated on this vLLM registering both the option and
	// the parser name, so an unreadable registry adds nothing on its own and
	// needs no separate guard. Guessing a name into a profile would trade a
	// model that answers plainly for one that will not start.
	asked := map[string]bool{}
	if args, err := config.SplitArgs(present); err == nil {
		for _, a := range args {
			name, _, _ := strings.Cut(a, "=")
			asked[name] = true
		}
	}

	pick := func(candidates []string, registered func(string) bool) string {
		for _, name := range candidates {
			if registered(name) {
				return name
			}
		}
		return ""
	}

	if !asked["--tool-call-parser"] && caps.HasFlag("--tool-call-parser") {
		if name := pick(harmonyToolParsers, caps.HasToolParser); name != "" {
			flags = append(flags, "--tool-call-parser", name)
			// vLLM refuses to start with a tool parser and no way to reach it.
			if !asked["--enable-auto-tool-choice"] && caps.HasFlag("--enable-auto-tool-choice") {
				flags = append(flags, "--enable-auto-tool-choice")
			}
			notes = append(notes, "Added --tool-call-parser "+name+" and --enable-auto-tool-choice. "+
				"This model reports tool calls in its own format, and without a parser for it vLLM "+
				"returns an empty message instead, which looks like the model ignoring the request.")
		}
	}
	if !asked["--reasoning-parser"] && caps.HasFlag("--reasoning-parser") {
		if name := pick(harmonyReasoningParsers, caps.HasReasoningParser); name != "" {
			flags = append(flags, "--reasoning-parser", name)
			notes = append(notes, "Added --reasoning-parser "+name+
				", which separates this model's thinking from its answer.")
		}
	}
	return flags, notes
}

// ensureHarmonyEncodings downloads the vocab files if they are missing.
//
// setup creates this directory and points vLLM at it, but nothing filled it, so
// a gpt-oss model started healthy and then failed every request with "invalid
// tiktoken vocab file" -- a failure that looks like a broken model rather than
// a missing file. Returns a note for the user, empty when there was nothing to
// do or nothing useful to say.
func ensureHarmonyEncodings(dataDir string) string {
	dir := filepath.Join(dataDir, "tiktoken-encodings")
	var fetched []string
	for _, name := range harmonyEncodings {
		path := filepath.Join(dir, name)
		if st, err := os.Stat(path); err == nil && st.Size() > 0 {
			continue
		}
		if err := download(encodingsURL+name, path); err != nil {
			return fmt.Sprintf(
				"This model tokenizes with harmony, which needs %s in %s. "+
					"Fetching it failed (%v), and requests will return an error until it is there.",
				name, dir, err)
		}
		fetched = append(fetched, name)
	}
	if len(fetched) == 0 {
		return ""
	}
	return fmt.Sprintf("Fetched the harmony vocab files (%v) into %s -- this model reads them from disk.", fetched, dir)
}

func download(url, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	// Written beside the target and renamed, so an interrupted fetch cannot
	// leave a half file that later looks present and passes the size check.
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tiktoken-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
