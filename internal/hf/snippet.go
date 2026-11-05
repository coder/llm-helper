package hf

import (
	"regexp"
	"strings"

	"github.com/coder/llm-helper/internal/config"
)

// ownedFlags are set by vllm-helper itself from config and the profile, so a
// card's values for them would be overridden or would fight with it.
var ownedFlags = map[string]bool{
	"--model":                  true,
	"--max-model-len":          true, // captured separately, into MAX_MODEL_LEN
	"--gpu-memory-utilization": true,
	"--port":                   true,
	"--host":                   true,
	"--api-key":                true,
	"--served-model-name":      true,
}

// topologyFlags describe the machine the card's author happened to use.
// Copying "--tensor-parallel-size 8" onto a single-GPU box is not a smaller
// version of that deployment, it's a server that won't start.
var topologyFlags = map[string]bool{
	"--tensor-parallel-size":         true,
	"--pipeline-parallel-size":       true,
	"--data-parallel-size":           true,
	"-tp":                            true,
	"-pp":                            true,
	"--distributed-executor-backend": true,
	"--node-rank":                    true,
	"--master-addr":                  true,
	"--master-port":                  true,
}

// thinkingParsers default thinking=false per request in vLLM's raw
// DeepSeekV3ReasoningParser; when false it silently swaps in a no-op parser
// and raw <think> tags leak into content. No model card documents this.
var thinkingParsers = regexp.MustCompile(`^(deepseek_v3|poolside_v1)$`)

// flagsFromSnippet reads a card's "vllm serve" line and returns the flags
// worth adopting plus the context length it asked for.
//
// It carries over anything it doesn't recognise, rather than picking from a
// list of known flags. An allowlist silently drops whatever the list didn't
// anticipate -- memory-shaping flags like --max-num-seqs, and the plugin flag
// that explains an otherwise-unknown parser -- and the user is never told the
// served config differs from what the model's authors prescribed. This is the
// same shape `adopt` uses to take over a running server.
//
// Validation of the values is deliberately not done here: whether a parser
// name is real depends on the vLLM installed on this machine, which is not
// something a HuggingFace client should know about.
func flagsFromSnippet(snippet string) (extraArgs, maxModelLen string) {
	extraArgs, maxModelLen, _ = parseSnippet(snippet, "")
	return extraArgs, maxModelLen
}

// parseSnippet is flagsFromSnippet that also returns the --reasoning-parser it
// kept, replaced by parser when that is set.
func parseSnippet(snippet, parser string) (extraArgs, maxModelLen, reasoningParser string) {
	// Fold shell line continuations, then flatten what is left. The backslash
	// and the newline are separate concerns: a card can wrap without one, and
	// a bare newline left in place gets read as the previous flag's value.
	flat := strings.ReplaceAll(snippet, "\\\n", " ")
	flat = strings.NewReplacer("\r", " ", "\n", " ", "\t", " ").Replace(flat)
	args, err := config.SplitArgs(flat)
	if err != nil {
		// A card whose snippet isn't valid shell (prose wrapped mid-line,
		// an unbalanced quote) yields nothing rather than half of it.
		return "", "", ""
	}

	var kept []string
	var sawParser string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !strings.HasPrefix(a, "-") {
			continue // the model id, or a stray word
		}

		flag, value := a, ""
		if f, v, found := strings.Cut(a, "="); found {
			flag, value = f, v
		} else if i+1 < len(args) && !isFlagToken(args[i+1]) {
			i++
			value = args[i]
		}

		if flag == "--max-model-len" {
			maxModelLen = value
			continue
		}
		if ownedFlags[flag] || topologyFlags[flag] {
			continue
		}
		if looksSecret(flag) {
			continue
		}
		if flag == "--reasoning-parser" {
			if parser != "" {
				value = parser
			}
			sawParser = value
		}
		if value == "" {
			kept = append(kept, flag)
		} else {
			kept = append(kept, flag+" "+config.Quote(value))
		}
	}

	if thinkingParsers.MatchString(sawParser) {
		kept = append(kept, `--default-chat-template-kwargs {"thinking":true}`)
	}
	return strings.Join(kept, " "), maxModelLen, sawParser
}

var sglangSnippetRe = regexp.MustCompile(`sglang[._](?:launch_server|serve)(?:.*\\\n)*.*`)

func sglangReasoningParser(card string) string {
	found := ""
	for _, snippet := range sglangSnippetRe.FindAllString(card, -1) {
		name := reasoningParserIn(snippet)
		if name == "" {
			continue
		}
		if found != "" && found != name {
			return ""
		}
		found = name
	}
	return found
}

func reasoningParserIn(snippet string) string {
	flat := strings.ReplaceAll(snippet, "\\\n", " ")
	flat = strings.NewReplacer("\r", " ", "\n", " ", "\t", " ").Replace(flat)
	args, err := config.SplitArgs(flat)
	if err != nil {
		return ""
	}
	for i, a := range args {
		if v, ok := strings.CutPrefix(a, "--reasoning-parser="); ok {
			return v
		}
		if a == "--reasoning-parser" && i+1 < len(args) && !isFlagToken(args[i+1]) {
			return args[i+1]
		}
	}
	return ""
}

// looksSecret keeps a credential pasted into a card out of a profile file.
// Matching the trailing word spares real flags like --tokenizer-mode.
func looksSecret(flag string) bool {
	for _, suffix := range []string{"-key", "-token", "-secret", "-password"} {
		if strings.HasSuffix(strings.ToLower(flag), suffix) {
			return true
		}
	}
	return false
}

// isFlagToken reports whether a token starts a new flag rather than being a
// value. Negative numbers are values despite the dash.
func isFlagToken(s string) bool {
	if !strings.HasPrefix(s, "-") || s == "-" {
		return false
	}
	rest := strings.TrimLeft(s, "-")
	if rest == "" {
		return true
	}
	c := rest[0]
	return !(c >= '0' && c <= '9' || c == '.')
}
