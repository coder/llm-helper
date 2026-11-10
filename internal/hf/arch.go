package hf

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Arch is what a model's config.json says about its shape. Only the parts that
// decide whether it can be served: parsing more would mean breaking on fields
// we never use.
type Arch struct {
	ModelType string `json:"model_type"`
	// NumExperts is set by mixture-of-experts models under one of several
	// names, since the field was never standardised.
	NumExperts       int `json:"num_experts"`
	NumLocalExperts  int `json:"num_local_experts"`
	NRoutedExperts   int `json:"n_routed_experts"`
	NumExpertsPerTok int `json:"num_experts_per_tok"`

	// KV cache shape. HeadDim is read, never derived: some configs declare
	// 128 where hidden_size/num_attention_heads gives 64, and halving the
	// per-token cost is the direction that OOMs.
	NumHiddenLayers   int    `json:"num_hidden_layers"`
	NumAttentionHeads int    `json:"num_attention_heads"`
	NumKeyValueHeads  int    `json:"num_key_value_heads"`
	HeadDim           int    `json:"head_dim"`
	HiddenSize        int    `json:"hidden_size"`
	MaxPositionEmbed  int    `json:"max_position_embeddings"`
	TorchDtype        string `json:"torch_dtype"`
	// KVLoraRank and QKRopeHeadDim are set on multi-head latent attention
	// (DeepSeek V2/V3, GLM-5), which caches one compressed latent and the
	// rotary key per token and layer instead of K and V for every head.
	KVLoraRank    int `json:"kv_lora_rank"`
	QKRopeHeadDim int `json:"qk_rope_head_dim"`
	// IndexHeadDim is the key width of a DeepSeek sparse attention indexer,
	// which caches one indexer key per token beside the latent.
	IndexHeadDim int `json:"index_head_dim"`
	// LayerTypes names each layer for hybrids, where only some attend.
	LayerTypes []string `json:"layer_types"`
	// SlidingWindow caps how far a sliding-attention layer looks back, so
	// those layers hold that many tokens rather than the whole context.
	SlidingWindow int `json:"sliding_window"`
	// TextConfig holds the language model's shape on multimodal repos.
	TextConfig *Arch `json:"text_config"`
}

// text returns the sub-config holding the attention shape.
func (a Arch) text() Arch {
	if a.TextConfig != nil && a.TextConfig.NumHiddenLayers > 0 {
		return *a.TextConfig
	}
	return a
}

// attentionLayers counts the layers holding a KV cache, split by how much each
// holds. Hybrids interleave attention with Mamba or linear-attention layers
// that hold a fixed-size state rather than a cache that grows with the
// context, and sliding layers hold only their window -- gpt-oss is half full,
// half sliding at a window of 128, so treating them alike doubles the estimate.
func (a Arch) attentionLayers() (full, sliding int) {
	for _, t := range a.LayerTypes {
		switch t := strings.ToLower(t); {
		case strings.Contains(t, "linear"):
			// GLM-5.3 names its Kimi Delta layers linear_attention.
		case strings.Contains(t, "sliding"):
			sliding++
		case strings.Contains(t, "attention"):
			full++
		}
	}
	if full+sliding == 0 {
		full = a.NumHiddenLayers
	}
	return full, sliding
}

// Hybrid reports whether some layers are linear attention or a recurrent
// state rather than attention.
func (a Arch) Hybrid() bool {
	for _, t := range a.text().LayerTypes {
		t = strings.ToLower(t)
		if strings.Contains(t, "linear") || !strings.Contains(t, "attention") {
			return true
		}
	}
	return false
}

// kvDtypeBytes is the width of one cached element. 2 unless the cache itself
// is quantised, which is a separate flag we do not set.
func kvDtypeBytes(dtype string) int {
	switch strings.ToLower(dtype) {
	case "float32", "float", "fp32":
		return 4
	default:
		return 2
	}
}

// KVBytesPerToken is what one token of context costs, averaged over a request
// of maxLen: 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes, with
// sliding layers counted at their window rather than the full context.
// Reproduces vLLM's own allocation. Returns 0 when the config did not say
// enough, which callers must treat as unknown rather than free.
func (a Arch) KVBytesPerToken(maxLen int) int64 {
	t := a.text()
	full, sliding := t.attentionLayers()
	heads, dim := t.NumKeyValueHeads, t.HeadDim
	if heads == 0 {
		heads = t.NumAttentionHeads // no grouped-query attention
	}
	if dim == 0 && t.NumAttentionHeads > 0 && t.HiddenSize > 0 {
		// Last resort only; deriving over a declared value is the OOM above.
		dim = t.HiddenSize / t.NumAttentionHeads
	}
	if t.KVLoraRank > 0 {
		// Latent attention: vLLM and llama.cpp cache kv_lora_rank +
		// qk_rope_head_dim per token, whatever the head count, plus the
		// sparse-attention indexer's key where there is one (stored narrower
		// than this counts it). Counting heads overstates DeepSeek V3 about
		// 25 times.
		heads, dim = 1, t.KVLoraRank+t.QKRopeHeadDim+t.IndexHeadDim
	}
	if full+sliding <= 0 || heads <= 0 || dim <= 0 || maxLen <= 0 {
		return 0
	}
	perLayer := 2 * int64(heads) * int64(dim) * int64(kvDtypeBytes(t.TorchDtype))
	if t.KVLoraRank > 0 {
		// One latent per token, not a K and a V.
		perLayer /= 2
	}

	// A sliding layer never holds more than its window, so past that point
	// it stops costing anything per extra token of context.
	window := t.SlidingWindow
	if window <= 0 || window > maxLen {
		window = maxLen
	}
	total := perLayer * (int64(full)*int64(maxLen) + int64(sliding)*int64(window))
	return total / int64(maxLen)
}

// MaxContext is the longest context this model was trained for, or 0 if the
// config did not say.
func (a Arch) MaxContext() int { return a.text().MaxPositionEmbed }

// MoE reports whether this is a mixture-of-experts model. Three spellings
// because Qwen/OLMoE, Mixtral-derived and DeepSeek configs each chose one.
func (a Arch) MoE() bool {
	return a.NumExperts > 0 || a.NumLocalExperts > 0 || a.NRoutedExperts > 0
}

// Config reads a model's config.json. Returns a zero Arch and no error when
// the repo has none: GGUF repos typically ship none, which is a fact about the
// repo rather than a failure.
func Config(ctx context.Context, repo string) (Arch, error) {
	resp, err := get(ctx, fmt.Sprintf("%s/%s/raw/main/config.json", baseURL, repo))
	if err != nil {
		return Arch{}, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound, http.StatusUnauthorized, http.StatusForbidden:
		// Missing, or gated behind a licence click. Neither is something the
		// caller can act on, and neither means the model is unservable.
		return Arch{}, nil
	default:
		return Arch{}, fmt.Errorf("config.json: HTTP %d", resp.StatusCode)
	}
	var a Arch
	if err := json.NewDecoder(resp.Body).Decode(&a); err != nil {
		return Arch{}, err
	}
	return a, nil
}

// ggufFusedExpertSupport are the model types whose fused expert weights vLLM's
// GGUF plugin can map. Modern Transformers fuses MoE experts into single
// mlp.experts.gate_up_proj/down_proj parameters and only these two branches
// register that shape; every other MoE fails at load with "Failed to map GGUF
// parameters", after the service has already restarted. Read out of
// vllm-gguf-plugin 0.0.5 and unchanged on main. A name absent here is refused,
// which costs an explanation rather than an outage.
var ggufFusedExpertSupport = map[string]bool{
	"olmoe":      true,
	"minimax_m2": true,
}

// GGUFServable reports whether a GGUF build of this architecture can load, and
// why not when it cannot. Dense models are fine; the problem is experts.
func GGUFServable(a Arch) (bool, string) {
	if !a.MoE() || ggufFusedExpertSupport[strings.ToLower(a.ModelType)] {
		return true, ""
	}
	return false, "a mixture-of-experts model, which vLLM's GGUF plugin cannot map " +
		"(only OLMoE and MiniMax-M2 are supported) -- look for a safetensors quant instead"
}
