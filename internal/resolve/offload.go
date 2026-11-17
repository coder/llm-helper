package resolve

import (
	"fmt"
	"strings"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/vllm"
)

// offloadPlan is how much of a model's weights a profile keeps in CPU memory.
type offloadPlan struct {
	GiB  int
	Args []string
	Env  []string
	Note string
}

// planOffload decides whether a mixture-of-experts model too big for the GPU
// is served with part of its experts in CPU memory. Only a GPU that reads CPU
// memory coherently has room for that, and only an engine that can is asked.
func planOffload(hw detect.Hardware, arch hf.Arch, weights int64, prov engine.Provider, caps vllm.Caps) offloadPlan {
	if prov == nil || !prov.Capabilities().WeightOffload || !arch.MoE() || weights <= hw.BudgetBytes {
		return offloadPlan{}
	}
	gib, ok := hw.OffloadFor(weights)
	if !ok {
		return offloadPlan{}
	}
	size := fmt.Sprintf("This model is %.0f GB, more than the %d GB its GPU holds", float64(weights)/1e9, hw.BudgetGB)
	args, env := prov.OffloadSettings(gib)
	if len(args) == 0 {
		return offloadPlan{Note: size + "; " + prov.Name() + " keeps its expert weights in CPU memory by itself, which decodes slower than from the GPU's own memory."}
	}
	quoted := make([]string, len(args))
	for i, a := range args {
		quoted[i] = config.Quote(a)
	}
	flags := strings.Join(quoted, " ")
	if prov.Capabilities().VLLMArgs {
		if _, dropped := vllm.Check(flags, caps); len(dropped) > 0 {
			return offloadPlan{Note: size + ", and the installed vLLM cannot keep experts in CPU memory: " + strings.Join(dropped, "; ")}
		}
	}
	return offloadPlan{GiB: gib, Args: quoted, Env: env, Note: fmt.Sprintf(
		"%s, so %d GiB of its expert weights stay in CPU memory (%s). "+
			"The GPU reads them from there on every token, which is slower than its own memory; a smaller copy that fits on the GPU decodes faster.",
		size, gib, flags)}
}
