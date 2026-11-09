package vllm

import (
	"fmt"
	"strings"

	"github.com/coder/llm-helper/internal/config"
)

// parserFlags are the flags whose values must name something in the
// installed vLLM's registry.
var parserFlags = map[string]func(Caps, string) bool{
	"--tool-call-parser": Caps.HasToolParser,
	"--reasoning-parser": Caps.HasReasoningParser,
}

// Accepts reports whether this vLLM is known to take flag with value: the
// flag is registered, and a parser flag's value names a registered parser.
func Accepts(caps Caps, flag, value string) bool {
	if !caps.HasFlag(flag) {
		return false
	}
	if has, ok := parserFlags[flag]; ok {
		return has(caps, value)
	}
	return true
}

// pluginFlags load a parser from a file shipped alongside a model. The file
// lives in the model's repo, not on this machine, so a path copied out of a
// card points at nothing here and vLLM exits on the missing file.
var pluginFlags = map[string]string{
	"--reasoning-parser-plugin": "--reasoning-parser",
	"--tool-parser-plugin":      "--tool-call-parser",
}

// Check inspects an EXTRA_ARGS string against the installed vLLM and returns
// a cleaned version plus anything worth telling the user.
//
// The policy is to drop what won't work and say so, never to substitute. A
// parser is not interchangeable with the one whose name looks closest:
// deepseek_r1, deepseek_v3 and deepseek_v4 are all real and all behave
// differently. Dropping degrades reasoning output in a way that is visible
// and reversible; guessing changes it silently. When something close does
// exist the message names it, so the choice stays with the person who can
// judge it.
//
// Caps that aren't Known() mean introspection failed. That returns the input
// untouched: refusing to serve because we couldn't ask would be worse than
// the problem.
func Check(extraArgs string, caps Caps) (cleaned string, warnings []string) {
	if extraArgs == "" || !caps.Known() {
		return extraArgs, nil
	}
	args, err := config.SplitArgs(extraArgs)
	if err != nil {
		// Unparseable EXTRA_ARGS is reported elsewhere, by the validation
		// that already exists for it.
		return extraArgs, nil
	}

	// A plugin flag explains why its companion parser is unknown, so find
	// them first to make the message specific rather than baffling.
	plugins := map[string]string{}
	for i := 0; i < len(args); i++ {
		if flag, value, ok := splitFlag(args, &i); ok {
			if owner, isPlugin := pluginFlags[flag]; isPlugin && value != "" {
				plugins[owner] = value
			}
		}
	}

	var kept []string
	for i := 0; i < len(args); i++ {
		flag, value, ok := splitFlag(args, &i)
		if !ok {
			kept = append(kept, args[i])
			continue
		}

		if _, isPlugin := pluginFlags[flag]; isPlugin {
			warnings = append(warnings, fmt.Sprintf(
				"dropped %s %s: the plugin file ships with the model card, not with this machine, and vLLM won't start without it",
				flag, value))
			continue
		}

		// A flag that no longer exists is fatal in a different way to a bad
		// value: vLLM exits on "unrecognized arguments" before it reads any
		// of them. Cards outlive the flags they were written for, and
		// --enable-reasoning is the live example -- real once, gone now.
		if caps.KnowsFlags() && !caps.HasFlag(flag) {
			warnings = append(warnings, fmt.Sprintf(
				"dropped %s: vLLM %s has no such flag (it may have been renamed or removed since the model card was written)",
				flag, caps.Version))
			continue
		}

		if known, isParser := parserFlags[flag]; isParser && value != "" {
			if !known(caps, value) {
				warnings = append(warnings, unknownParserMessage(flag, value, plugins[flag], caps))
				continue
			}
		}

		if value == "" {
			kept = append(kept, flag)
		} else {
			kept = append(kept, flag+" "+config.Quote(value))
		}
	}
	return strings.Join(kept, " "), warnings
}

// unknownParserMessage explains a rejected parser and, when the registry has
// a plausible neighbour, names it as a suggestion rather than applying it.
func unknownParserMessage(flag, value, pluginFile string, caps Caps) string {
	registry := caps.ToolParsers
	if flag == "--reasoning-parser" {
		registry = caps.ReasoningParsers
	}
	msg := fmt.Sprintf("dropped %s %s: vLLM %s doesn't have it", flag, value, caps.Version)
	if pluginFile != "" {
		msg += fmt.Sprintf(" as a built-in -- the card supplies it as a plugin (%s)", pluginFile)
	}
	if near := nearest(value, registry); near != "" {
		msg += fmt.Sprintf(". This build does have %q; if that's the right parser for this model, set it with 'llm-helper set <profile> EXTRA_ARGS=...' -- it is not applied automatically because parsers are not interchangeable", near)
	}
	return msg
}

// nearest finds a registered name plausibly meant by an unknown one. It is
// deliberately conservative: a shared leading or trailing word, not an edit
// distance, because deepseek_r1 and deepseek_v3 are one character apart
// and completely different parsers.
func nearest(unknown string, registry []string) string {
	parts := strings.FieldsFunc(strings.ToLower(unknown), func(r rune) bool { return r == '_' || r == '-' })
	if len(parts) == 0 {
		return ""
	}
	head, tail := parts[0], parts[len(parts)-1]
	if len(parts) < 2 {
		// A single word gives nothing to corroborate: "hermes" resembling
		// "harmony" is not evidence of anything.
		return ""
	}
	var best string
	var bestScore int
	for _, name := range registry {
		got := strings.FieldsFunc(strings.ToLower(name), func(r rune) bool { return r == '_' || r == '-' })
		if len(got) < 2 || got[len(got)-1] != tail {
			// The version-ish tail must agree, so nano_v3 can find
			// nemotron_v3 but never nemotron_v2.
			continue
		}
		// Prefer a candidate whose family name is also close. Without this
		// every *_v3 in the registry ties and the shortest name wins, which
		// is how nano_v3 ended up pointing at deepseek_v3.
		score := 0
		switch {
		case got[0] == head:
			score = 3
		case strings.HasPrefix(got[0], head[:1]) && len(head) > 2:
			// A shared first letter is weak, but it is the only signal
			// linking "nano" to "nemotron", and the tail already agrees.
			score = 2
		default:
			continue
		}
		if score > bestScore || (score == bestScore && best != "" && len(name) < len(best)) {
			best, bestScore = name, score
		}
	}
	return best
}

// splitFlag reads one "--flag value" or "--flag=value" at args[*i],
// advancing past a consumed value. Reports false for a non-flag token.
func splitFlag(args []string, i *int) (flag, value string, ok bool) {
	a := args[*i]
	if !strings.HasPrefix(a, "--") {
		return "", "", false
	}
	if f, v, found := strings.Cut(a, "="); found {
		return f, v, true
	}
	// A value is whatever follows that isn't itself a flag. Negative numbers
	// are values despite the dash, matching how adopt reads a live argv.
	if *i+1 < len(args) && !isFlag(args[*i+1]) {
		*i++
		return a, args[*i], true
	}
	return a, "", true
}

func isFlag(s string) bool {
	if !strings.HasPrefix(s, "-") || s == "-" {
		return false
	}
	// "-1" and "-0.5" are values.
	rest := strings.TrimLeft(s, "-")
	if rest == "" {
		return true
	}
	c := rest[0]
	return !(c >= '0' && c <= '9' || c == '.')
}
