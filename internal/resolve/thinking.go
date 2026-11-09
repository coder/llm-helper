package resolve

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/modeltype"
	"github.com/coder/llm-helper/internal/vllm"
)

// Thinking is a model's chat template switch for reasoning before it answers.
type Thinking struct {
	Kwarg string `json:"kwarg"`
	// Default is whether the model thinks when nothing sets the kwarg.
	Default bool   `json:"default"`
	Why     string `json:"why"`
}

// What a new profile can ask for.
const (
	ThinkingOn  = "on"
	ThinkingOff = "off"
)

// detectThinking returns the switch modelType's catalogue entry describes,
// when the repo's chat template has it. fallbackRepo is read when repo has no
// template, as GGUF repos often do.
func detectThinking(ctx context.Context, modelType, repo, fallbackRepo string) *Thinking {
	e, ok := modeltype.Default().Lookup(modelType)
	if !ok || e.Thinking == nil {
		return nil
	}
	t := *e.Thinking
	var text string
	if t.Detect != modeltype.DetectAlways {
		text, _ = hf.TemplateText(ctx, repo)
		if text == "" && fallbackRepo != "" && fallbackRepo != repo {
			text, _ = hf.TemplateText(ctx, fallbackRepo)
		}
	}
	if !t.Applies(text) {
		return nil
	}
	return &Thinking{Kwarg: t.Kwarg, Default: t.Default, Why: t.Why}
}

// thinkingIn reports whether args turn thinking on through flag, and def
// when they do not set kwarg. The last occurrence wins, as it does for the
// server.
func thinkingIn(args, flag, kwarg string, def bool) bool {
	words, err := config.SplitArgs(args)
	if err != nil || flag == "" {
		return def
	}
	for i, w := range words {
		val, ok := strings.CutPrefix(w, flag+"=")
		if !ok && w == flag && i+1 < len(words) {
			val, ok = words[i+1], true
		}
		if !ok {
			continue
		}
		var m map[string]any
		if json.Unmarshal([]byte(val), &m) == nil {
			if b, isBool := m[kwarg].(bool); isBool {
				def = b
			}
		}
	}
	return def
}

// withThinking returns args with kwarg set to on in flag's JSON object,
// merged into the object args already pass.
func withThinking(args, flag, kwarg string, on bool) (string, error) {
	words, err := config.SplitArgs(args)
	if err != nil {
		return "", err
	}
	kwargs := map[string]any{}
	var kept []string
	for i := 0; i < len(words); i++ {
		w := words[i]
		val, ok := strings.CutPrefix(w, flag+"=")
		if !ok && w == flag && i+1 < len(words) {
			val, ok = words[i+1], true
			i++
		}
		if !ok {
			kept = append(kept, w)
			continue
		}
		if err := json.Unmarshal([]byte(val), &kwargs); err != nil {
			return "", fmt.Errorf("%s %s is not a JSON object", flag, val)
		}
	}
	kwargs[kwarg] = on
	obj, err := json.Marshal(kwargs)
	if err != nil {
		return "", err
	}
	out := make([]string, 0, len(kept)+2)
	for _, w := range kept {
		out = append(out, config.Quote(w))
	}
	out = append(out, flag, config.Quote(string(obj)))
	return strings.Join(out, " "), nil
}

// onOff renders a thinking state as the profile records it.
func onOff(on bool) string {
	if on {
		return ThinkingOn
	}
	return ThinkingOff
}

// note is the plan note for a thinking state.
func (t *Thinking) note(on bool) string {
	if on {
		return "Thinking model: replies begin after a reasoning pass. " + t.Why
	}
	return "Thinking is off for this profile. " + t.Why
}

// noteThinking records what the plan's args leave thinking at.
func (p *Plan) noteThinking(flag string) {
	if p.Thinking == nil {
		return
	}
	on := thinkingIn(p.Opts.ExtraArgs, flag, p.Thinking.Kwarg, p.Thinking.Default)
	p.Opts.Thinking = onOff(on)
	p.Notes = append(p.Notes, p.Thinking.note(on))
}

// SetThinking turns the model's thinking on or off in the profile, through
// the engine's flag for chat template kwargs. mode "" leaves the plan as it
// is.
func (p *Plan) SetThinking(mode string, prov engine.Provider, caps vllm.Caps) error {
	if mode == "" {
		return nil
	}
	if mode != ThinkingOn && mode != ThinkingOff {
		return fmt.Errorf("thinking must be %s or %s, not %q", ThinkingOn, ThinkingOff, mode)
	}
	if p.Thinking == nil {
		return fmt.Errorf("%s has no thinking switch llm-helper knows of", p.Opts.Repo)
	}
	flag := prov.Capabilities().ChatTemplateKwargsFlag
	if flag == "" {
		return fmt.Errorf("%s has no flag for chat template kwargs, so a profile cannot switch thinking", prov.Name())
	}
	if prov.Capabilities().VLLMArgs && caps.KnowsFlags() && !caps.HasFlag(flag) {
		return fmt.Errorf("the installed vLLM %s does not take %s, so a profile cannot switch thinking", caps.Version, flag)
	}
	on := mode == ThinkingOn
	if thinkingIn(p.Opts.ExtraArgs, flag, p.Thinking.Kwarg, p.Thinking.Default) == on {
		p.Opts.Thinking = mode
		return nil
	}
	args, err := withThinking(p.Opts.ExtraArgs, flag, p.Thinking.Kwarg, on)
	if err != nil {
		return err
	}
	p.Opts.ExtraArgs = args
	p.Opts.Thinking = mode
	stale := p.Thinking.note(!on)
	p.Notes = slices.DeleteFunc(p.Notes, func(n string) bool { return n == stale })
	p.Notes = append(p.Notes, p.Thinking.note(on), fmt.Sprintf("Thinking set %s with %s.", mode, flag))
	return nil
}
