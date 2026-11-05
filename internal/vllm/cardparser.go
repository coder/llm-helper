package vllm

import (
	"fmt"
	"strings"

	"github.com/coder/vllm-helper/internal/config"
)

// PreferSGLangParser replaces the card's vLLM --reasoning-parser with the one
// its SGLang command names, when this vLLM registers that exact name.
func PreferSGLangParser(extraArgs, sglangParser string, caps Caps) (string, string) {
	if sglangParser == "" || !caps.Known() || !caps.HasReasoningParser(sglangParser) {
		return extraArgs, ""
	}
	args, err := config.SplitArgs(extraArgs)
	if err != nil {
		return extraArgs, ""
	}
	var kept []string
	old := ""
	for i := 0; i < len(args); i++ {
		flag, value, ok := splitFlag(args, &i)
		switch {
		case !ok:
			kept = append(kept, args[i])
		case flag == "--reasoning-parser" && value != "" && value != sglangParser:
			old = value
			kept = append(kept, flag+" "+config.Quote(sglangParser))
		case value == "":
			kept = append(kept, flag)
		default:
			kept = append(kept, flag+" "+config.Quote(value))
		}
	}
	if old == "" {
		return extraArgs, ""
	}
	return strings.Join(kept, " "), fmt.Sprintf(
		"using --reasoning-parser %s instead of the card's %s: the card's SGLang command names %s, and vLLM %s has it",
		sglangParser, old, sglangParser, caps.Version)
}
