// Package hf is a minimal HuggingFace Hub client: model search, weight-size
// lookup, and auto-detection of vLLM serving flags from model cards.
package hf

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

var (
	httpClient = &http.Client{Timeout: 15 * time.Second}
	// baseURL is the Hub root; tests point it at a local server.
	baseURL = "https://huggingface.co"
)

// get issues a GET bound to ctx, so a caller that has given up stops the
// request instead of leaving it running.
func get(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	return httpClient.Do(req)
}

type Model struct {
	ID          string   `json:"id"`
	Downloads   int64    `json:"downloads"`
	Likes       int64    `json:"likes"`
	PipelineTag string   `json:"pipeline_tag"`
	CreatedAt   string   `json:"createdAt"`
	Tags        []string `json:"tags"`
	// CardData carries the card's YAML front matter. Only base_model is read
	// from it, and it is typed loosely because the Hub accepts either a
	// single repo id or a list of them.
	CardData struct {
		BaseModel json.RawMessage `json:"base_model"`
	} `json:"cardData"`
}

// BaseModel is the model a derivative was made from -- the unquantized repo a
// GGUF was converted from, in practice.
//
// It matters because vLLM's GGUF plugin resolves its HF config from the
// tokenizer repo, and GGUF publishers largely ship no config.json: verified
// live, bartowski/Qwen_Qwen3-32B-GGUF and both unsloth repos checked have
// none. Without --tokenizer pointing somewhere that does, the server cannot
// start. Deriving it from the repo name would be guesswork ("Qwen_Qwen3-32B"
// -> "Qwen/Qwen3-32B" only by convention), so the card is asked instead.
func (m Model) BaseModel() string {
	raw := m.CardData.BaseModel
	if len(raw) == 0 {
		return ""
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return one
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil && len(many) > 0 {
		return many[0]
	}
	return ""
}

// License extracts the license tag ("license:apache-2.0" -> "apache-2.0").
func (m Model) License() string {
	for _, t := range m.Tags {
		if rest, ok := strings.CutPrefix(t, "license:"); ok {
			return rest
		}
	}
	return "unknown"
}

// QuantTags returns quantization-related tags (fp8, gptq, awq, ...).
func (m Model) QuantTags() []string {
	known := map[string]bool{
		"fp4": true, "fp8": true, "nvfp4": true, "mxfp4": true, "awq": true,
		"gptq": true, "int4": true, "int8": true, "8-bit": true, "4-bit": true,
		"gguf": true, "bf16": true,
	}
	var out []string
	for _, t := range m.Tags {
		if known[strings.ToLower(t)] {
			out = append(out, t)
		}
	}
	return out
}

// Lookup fetches one model's metadata by repo id.
func Lookup(ctx context.Context, repo string) (Model, error) {
	var m Model
	resp, err := get(ctx, baseURL+"/api/models/"+repo)
	if err != nil {
		return m, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return m, fmt.Errorf("no model %q on HuggingFace (private, gated, or a typo?)", repo)
	}
	if resp.StatusCode != http.StatusOK {
		return m, fmt.Errorf("hub lookup: HTTP %d", resp.StatusCode)
	}
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return m, err
	}
	if m.ID == "" {
		m.ID = repo
	}
	return m, nil
}

// WhoAmI validates a Hub token and returns the account name it belongs to.
func WhoAmI(ctx context.Context, token string) (string, error) {
	// Built by hand rather than via get: this is the one request that
	// carries a header.
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/whoami-v2", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		return "", fmt.Errorf("token rejected (HTTP 401)")
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("whoami: HTTP %d", resp.StatusCode)
	}
	var body struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", err
	}
	return body.Name, nil
}

// Search queries the Hub for text-generation models, most-downloaded first.
func Search(ctx context.Context, query string, limit int) ([]Model, error) {
	u := fmt.Sprintf(
		baseURL+"/api/models?search=%s&filter=text-generation&sort=downloads&direction=-1&limit=%d",
		url.QueryEscape(query), limit)
	resp, err := get(ctx, u)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("hub search: HTTP %d", resp.StatusCode)
	}
	var models []Model
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		return nil, err
	}
	return models, nil
}

// Variant is one complete copy of a model within a repo -- typically one
// quantization, since publishers ship a dozen of them side by side.
type Variant struct {
	// Name is how the repo distinguishes this copy: a GGUF filename, a
	// quant subfolder, or "" for weights sitting at the repo root.
	Name string `json:"name"`
	// Label is Name reduced to the part worth reading, so a list of them
	// can be scanned: "Q8_0" rather than "Qwen_Qwen3-32B-Q8_0.gguf". It is
	// also the selector vLLM is given, so it cannot be abbreviated.
	Label string `json:"label"`
	// Display is Label shortened for a list or a dropdown. Some publishers
	// name a quant after every per-tensor override it applies, which runs to
	// 70 characters and hides the one part that differs.
	Display string  `json:"display,omitempty"`
	SizeGB  float64 `json:"size_gb"`
	// GGUF marks a variant that needs vLLM's out-of-tree GGUF plugin.
	GGUF bool `json:"gguf,omitempty"`
	// Blocked says why vLLM cannot be pointed at this variant, and is empty
	// when it can. Listing a quant that cannot be selected is worse than not
	// listing it, because the sizes make it look like a live option.
	Blocked string `json:"blocked,omitempty"`
}

// Selector is what to append to a repo id to serve this variant, and false
// when the variant is the repo's own default and needs no suffix.
//
// vLLM addresses a GGUF quant by name -- "repo:Q8_0" -- not by filename or
// path, so the label is what goes on the end.
func (v Variant) Selector() (string, bool) {
	if v.Name == "" || v.Name == "." || !v.GGUF {
		return "", false
	}
	return ":" + v.Label, true
}

// Servable reports whether this variant can be selected at all.
func (v Variant) Servable() bool { return v.Blocked == "" }

// Weights describes the weight files found in a model repo.
type Weights struct {
	SizeGB float64 // 0 when the repo has no weight files we recognise
	// Variants is how many independent weight groupings the repo holds.
	// >1 means SizeGB is the smallest of them; 1 means it is the whole story.
	Variants int
	// All lists those groupings, largest first.
	//
	// Kept rather than reduced to the winner because "which of these should
	// I run" is a question only the user can answer: the smallest is the
	// safe answer to "does anything here fit", and simultaneously the worst
	// thing to actually serve when the machine has room for more. A 32B repo
	// whose smallest is a 2-bit quant will happily run its 8-bit one.
	All []Variant `json:"all,omitempty"`
}

// quantLabel pulls the part of a variant name worth showing. GGUF publishers
// prefix every file with the model name, which is the same for all of them and
// so carries no information in a list of alternatives.
func quantLabel(name, repo string) string {
	if name == "" || name == "." {
		return "default"
	}
	label := strings.TrimSuffix(strings.TrimSuffix(name, ".gguf"), ".GGUF")
	if i := strings.LastIndex(label, "/"); i >= 0 {
		label = label[i+1:]
	}
	// Strip the repeated model-name prefix: everything up to the last dash
	// that precedes something quant-shaped.
	if m := quantSuffix.FindString(label); m != "" {
		return strings.TrimPrefix(m, "-")
	}

	if _, model, ok := strings.Cut(repo, "/"); ok {
		// HasPrefix, not TrimPrefix: a miss leaves the label untouched, and
		// trimming that would look like a successful strip.
		if strings.HasPrefix(label, model) {
			if t := strings.Trim(strings.TrimPrefix(label, model), "-_."); t != "" {
				return t
			}
		}
		// The prefix rarely matches exactly: a repo named deepseek-v4-gguf
		// holds files called DeepSeek-V4-Pro-... So drop leading words the
		// repo name already says, which by definition tell these variants
		// apart from nothing.
		if t := trimRepoWords(label, model); t != "" {
			return t
		}
	}
	return label
}

// trimRepoWords removes leading dash-separated words of label that also appear
// in the repo's model name, case-insensitively.
func trimRepoWords(label, model string) string {
	known := map[string]bool{}
	for _, w := range strings.FieldsFunc(strings.ToLower(model), isLabelSep) {
		known[w] = true
	}
	words := strings.Split(label, "-")
	i := 0
	for i < len(words)-1 && known[strings.ToLower(words[i])] {
		i++
	}
	if i == 0 {
		return ""
	}
	return strings.Join(words[i:], "-")
}

func isLabelSep(r rune) bool { return r == '-' || r == '_' || r == '.' }

// maxLabelWidth is how much of a label a list can show before the column stops
// being scannable.
const maxLabelWidth = 26

// shortLabel elides the middle of an over-long label. The ends are what differ
// between siblings -- the quant at the front, a date or revision at the back --
// so the middle is the part that can go.
func shortLabel(label string) string {
	if len(label) <= maxLabelWidth {
		return label
	}
	const ellipsis = "..."
	keep := maxLabelWidth - len(ellipsis)
	head := (keep + 1) / 2
	return label[:head] + ellipsis + label[len(label)-(keep-head):]
}

// quantSuffix matches the precision marker at the end of a variant name --
// Q4_K_M, IQ2_XXS, UD-Q2_K_XL, bf16, fp8. Anchored to the end so the model's
// own name cannot match it, and the leading dash is optional so a folder that
// is nothing but the quant matches as a whole.
//
// The UD- prefix is part of the name, not noise: unsloth's UD-Q2_K_XL is a
// different quantization from a plain Q2_K_XL, and dropping it would label two
// distinct choices identically.
var quantSuffix = regexp.MustCompile(`(?i)(^|-)(ud-)?(i?q\d[\w_]*|bf16|fp16|fp32|f16|f32|fp8|fp4|nvfp4|mxfp4|int4|int8|awq|gptq)$`)

// maxTreePages bounds how far we follow the Hub's cursor pagination. Pages hold
// 1000 entries, so this covers any real repo while keeping a broken or hostile
// Link header from looping forever.
const maxTreePages = 20

// treeEntry is one file or directory in a repo listing.
type treeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// ggufShard matches the "-00001-of-00003" that marks one piece of a split
// GGUF, so the pieces can be added back together.
var ggufShard = regexp.MustCompile(`(?i)-\d+-of-\d+\.gguf$`)

// quantFolder matches directory names that name a precision rather than a part
// of the model -- Q4_K_M, UD-IQ1_S, BF16, NVFP4. Sub-folders like these are
// alternatives to each other; ones like text_encoder/ or vae/ are not.
var quantFolder = regexp.MustCompile(`(?i)^(ud-)?(i?q\d|bf16|fp16|fp32|f16|f32|fp8|fp4|nvfp4|mxfp4|int4|int8|awq|gptq)`)

// companionFiles are weight files that are not a model you can serve: a vision
// projector, a LoRA adapter, or training leftovers. They matter because they
// are small, so left in they win "smallest" outright -- unsloth/gemma-3-27b-it-GGUF
// would report its 0.86 GB mmproj instead of the 6.5 GB smallest real quant.
// Matched anywhere in the name, since the conventions disagree about order:
// mmproj-F16.gguf and Qwen2-VL-7B-mmproj-f16.gguf are the same kind of file.
var companionFiles = []string{
	"mmproj", "mm_projector", "lora", "adapter",
	"training_args", "optimizer", "scheduler", "rng_state",
}

func isCompanionFile(p string) bool {
	base := strings.ToLower(p[strings.LastIndex(p, "/")+1:])
	for _, c := range companionFiles {
		if strings.Contains(base, c) {
			return true
		}
	}
	return false
}

// componentFolders hold one part of a model rather than a whole copy of it.
// Naming them is what stops a GGUF-quantized pipeline -- transformer/, vae/,
// text_encoder/ -- from looking like a set of alternatives, which would report
// the VAE as though it were the model.
var componentFolders = regexp.MustCompile(`(?i)^(text_encoder|tokenizer|scheduler|feature_extractor|safety_checker|image_encoder|vision_tower|audio_tower|language_model|mm_projector|transformer|unet|vae|controlnet)`)

// smallestPlausibleShare guards the same mistake from the other side. The full
// span of GGUF quants is about 9x, TQ1_0 at under two bits per weight against
// BF16 at sixteen (unsloth/DeepSeek-V3.1-GGUF runs 170 GB to 1.34 TB), so a
// twentieth is comfortably below any real quant while still discarding the
// components this is here to catch -- a VAE runs a few hundredths of its
// transformer. Set it any tighter and it starts dropping honest IQ1 quants.
const smallestPlausibleShare = 0.05

func isGGUF(p string) bool { return strings.HasSuffix(strings.ToLower(p), ".gguf") }

// oneCopyOfRoot sums the root-level tensor files that make up a single copy of
// the model. Repos often ship the same weights twice: Mistral keeps a
// single-file consolidated.safetensors next to the sharded copy, and older
// repos keep pytorch_model.bin next to model.safetensors. Adding both reports
// exactly double -- mistralai/Mistral-7B-Instruct-v0.3 measures 29.0 GB when it
// is 14.5 GB -- so we keep one set and drop the duplicate.
func oneCopyOfRoot(files []treeEntry) int64 {
	var safetensors, consolidated, other int64
	for _, f := range files {
		base := strings.ToLower(f.Path[strings.LastIndex(f.Path, "/")+1:])
		switch {
		case !strings.HasSuffix(base, ".safetensors"):
			other += f.Size
		case strings.HasPrefix(base, "consolidated"):
			consolidated += f.Size
		default:
			safetensors += f.Size
		}
	}
	// The largest set, not a preferred one. Preferring safetensors outright
	// would report a stray 0.3 GB file beside 140 GB of .bin shards and call
	// it a fit, and every copy here is the same model anyway.
	return max(safetensors, max(consolidated, other))
}

// WeightsSize measures the weight files (.safetensors/.bin/.gguf) in a model
// repo and reports what one loadable copy costs.
//
// The trap is that plenty of repos hold several complete models, and adding
// those together produces a number with no meaning. Two conventions do it:
// GGUF publishers put a whole quant in each sub-folder (unsloth/GLM-5.2-GGUF
// has 22, summing to ~10.6 TB) or as one flat file per quant at the root
// (bartowski/Qwen_Qwen3-32B-GGUF has 26, summing to 466 GB for a model whose
// smallest quant is 9 GB). Sharded weights, by contrast, genuinely are one
// model split across files and do have to be added up.
//
// So we group the weight files into candidates -- one complete set each -- and
// report the smallest, since that is the one with any chance of fitting.
// Variants says how many there were, so the caller can admit which it picked.
//
// Tensor files at the repo root settle it on their own: that is the model, and
// anything under a sub-folder is a second copy in another format (openai/gpt-oss-120b
// keeps whole metal/ and original/ copies beside it) rather than a smaller option.
//
// Sub-folders are only read as alternatives when they look like alternatives --
// they hold GGUFs, or they are named for a precision. Otherwise they are the
// components of one model, as in text_encoder/ plus vae/ plus transformer/, and
// have to be added up. Guessing wrong that way round would report a vision
// tower as if it were the whole model.
func WeightsSize(ctx context.Context, repo string) (Weights, error) {
	files, err := repoTree(ctx, repo)
	if err != nil {
		return Weights{}, err
	}

	var rootTensors []treeEntry
	candidates := map[string]int64{}
	// Grouping keys are lowered so shards collapse together; quant names are
	// conventionally upper case, so the name to show is kept alongside.
	display := map[string]string{}
	// parts counts how many files each root-GGUF candidate is made of, since
	// a split quant is one vLLM cannot address on any released plugin.
	parts := map[string]int{}
	subdirs := map[string]int64{}
	alternatives := map[string]bool{}
	for _, f := range files {
		if f.Type == "directory" || !isWeightFile(f.Path) || isCompanionFile(f.Path) {
			continue
		}
		dir, _, nested := strings.Cut(f.Path, "/")
		switch {
		case nested:
			subdirs[dir] += f.Size
			// A folder is an alternative when it is named for a precision, or
			// when it holds GGUFs and is not one of the known model parts --
			// a GGUF-quantized transformer/ plus vae/ pipeline is still one
			// model, and reading it as a choice would report the VAE.
			if quantFolder.MatchString(dir) || (isGGUF(f.Path) && !componentFolders.MatchString(dir)) {
				// Present means "an alternative"; true means "holds GGUF".
				// A folder named awq/ or fp16/ is an alternative too, and
				// calling it GGUF would answer the wrong question about it.
				alternatives[dir] = alternatives[dir] || isGGUF(f.Path)
			}
		case isGGUF(f.Path):
			// Root GGUFs are one quant per name, minus any shard suffix. The
			// name is lowered so a repo shipping .GGUF still groups its shards.
			key := ggufShard.ReplaceAllString(strings.ToLower(f.Path), ".gguf")
			candidates[key] += f.Size
			parts[key]++
			if _, seen := display[key]; !seen {
				display[key] = ggufShard.ReplaceAllString(f.Path, ".gguf")
			}
		default:
			rootTensors = append(rootTensors, f)
		}
	}

	// Anything at the repo root settles it, and sub-folders are then second
	// copies in another format (gpt-oss-120b keeps whole metal/ and original/
	// copies) rather than smaller options. Root tensors join the GGUFs as one
	// more candidate instead of short-circuiting, so a repo shipping both
	// still gets its smallest option compared.
	if n := oneCopyOfRoot(rootTensors); n > 0 {
		candidates[""] = n
	}
	// A folder named for a precision is an alternative even when something
	// also sits at the repo root: unsloth/DeepSeek-V3.1-GGUF ships its
	// smallest quant loose and the other 26 in folders, and gating these
	// behind an empty root hid all 26.
	//
	// Sub-folders that are second copies in another format -- gpt-oss-120b's
	// metal/ and original/ -- are not caught by this, because naming a format
	// is not naming a precision and they hold no GGUF. That is the distinction
	// the alternatives set already draws.
	for dir := range alternatives {
		candidates[dir] = subdirs[dir]
	}
	if len(candidates) == 0 {
		// Components of one model: the sum is the model.
		var total int64
		for _, size := range subdirs {
			total += size
		}
		if total > 0 {
			candidates["."] = total
		}
	}

	if len(candidates) == 0 {
		return Weights{}, nil
	}
	return pickSmallest(candidates, display, parts, alternatives, repo), nil
}

// pickSmallest reports the smallest candidate, which is the one with any chance
// of fitting, and lists them all so a caller can offer the choice. Candidates
// far below their siblings are dropped first: real quants of one model stay
// within a single order of magnitude of each other, so anything under that is a
// component or a stray file rather than a choice.
//
// SizeGB stays the smallest for compatibility with the fit check, which asks
// "could anything here run" and wants the conservative answer. Choosing what to
// actually serve is a different question and wants All.
func pickSmallest(candidates map[string]int64, display map[string]string, parts map[string]int, folders map[string]bool, repo string) Weights {
	var largest int64
	for _, size := range candidates {
		largest = max(largest, size)
	}
	floor := int64(float64(largest) * smallestPlausibleShare)

	w := Weights{}
	smallest := int64(-1)
	for name, size := range candidates {
		if size < floor {
			continue
		}
		shown := name
		if d, ok := display[name]; ok {
			shown = d
		}
		label := quantLabel(shown, repo)
		inFolder, isFolder := folders[name]
		v := Variant{
			Name:    shown,
			Label:   label,
			Display: shortLabel(label),
			SizeGB:  float64(size) / 1e9,
			GGUF:    isGGUF(shown) || inFolder,
		}
		// Which layouts vLLM can be pointed at is not a judgement about
		// quality: it addresses a GGUF quant by name, and on every released
		// plugin that lookup is a non-recursive glob over one file. So a
		// quant in a folder is fetched and then reported missing, and a
		// split one loads partial weights. Both are worse than refusing.
		switch {
		case isFolder && inFolder:
			v.Blocked = "in a subfolder, which vLLM's GGUF plugin cannot address"
		case isFolder:
			v.Blocked = "in a subfolder, which vLLM cannot be pointed at"
		case parts[name] > 1:
			v.Blocked = "split across files, which vLLM's GGUF plugin cannot load"
		}
		w.All = append(w.All, v)
		if smallest < 0 || size < smallest {
			smallest = size
		}
	}
	// Largest first: the best quality that fits is the interesting end of
	// the list, and it is the one a reader should land on.
	sort.Slice(w.All, func(i, j int) bool { return w.All[i].SizeGB > w.All[j].SizeGB })
	w.Variants = len(w.All)
	w.SizeGB = float64(smallest) / 1e9
	return w
}

// BestFit picks the variant to actually serve: the largest that fits the
// budget, since quantization trades quality for size and there is no reason to
// take a worse one than the machine can hold.
//
// ggufOK says whether this machine's vLLM can load GGUF at all. Both that and
// the per-variant Blocked reason filter the candidates first, because ranking
// on size alone reliably lands on something unloadable: the largest quant in a
// GGUF repo is the one most likely to be split across files, and
// bartowski/Qwen_Qwen3-32B-GGUF's biggest-that-fits is a bf16 that is both
// split and in a subfolder.
//
// Returns false when nothing servable fits, rather than falling back to the
// smallest -- "the least bad option is still too big" is the caller's decision
// to make and present, not something to paper over here.
//
// A budget of 0 means unknown, in which case the largest is not a safe default:
// with nothing to check it against, the conservative choice is the smallest.
func (w Weights) BestFit(budgetGB float64, ggufOK bool) (Variant, bool) {
	usable := make([]Variant, 0, len(w.All))
	for _, v := range w.All {
		if v.Servable() && (ggufOK || !v.GGUF) {
			usable = append(usable, v)
		}
	}
	if len(usable) == 0 {
		return Variant{}, false
	}
	if budgetGB <= 0 {
		return usable[len(usable)-1], true
	}
	for _, v := range usable { // largest first
		if v.SizeGB <= budgetGB {
			return v, true
		}
	}
	return Variant{}, false
}

func isWeightFile(path string) bool {
	p := strings.ToLower(path)
	return strings.HasSuffix(p, ".safetensors") ||
		strings.HasSuffix(p, ".bin") ||
		strings.HasSuffix(p, ".gguf")
}

// repoTree lists every file in a repo, following the Hub's cursor pagination.
// The listing is recursive because plenty of repos keep their weights one level
// down; WeightsSize decides what to do with the sub-folders.
//
// Entries are de-duplicated by path, because a cursor that fails to advance
// would otherwise have us add the same shards in again on every pass. Running
// out of pages is an error rather than a short answer: a truncated listing
// undercounts, and an undercount is the one wrong answer that tells someone a
// model fits when it does not.
func repoTree(ctx context.Context, repo string) ([]treeEntry, error) {
	next := baseURL + "/api/models/" + repo + "/tree/main?recursive=true"
	var all []treeEntry
	seen := map[string]bool{}
	for page := 0; page < maxTreePages; page++ {
		entries, link, err := treePage(ctx, next)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if !seen[e.Path] {
				seen[e.Path] = true
				all = append(all, e)
			}
		}
		if link == "" {
			return all, nil
		}
		next = link
	}
	return nil, fmt.Errorf("hub tree: %s has more than %d pages", repo, maxTreePages)
}

func treePage(ctx context.Context, u string) ([]treeEntry, string, error) {
	resp, err := get(ctx, u)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("hub tree: HTTP %d", resp.StatusCode)
	}
	var entries []treeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, "", err
	}
	return entries, nextPageURL(resp.Header.Get("Link"), baseURL), nil
}

// nextPageURL pulls the rel="next" target out of an RFC 8288 Link header,
// which is how the Hub hands back the cursor for the following page.
//
// The target has to stay on the host we asked, since otherwise a response gets
// to redirect the next request anywhere it likes. That is a parsed comparison
// of scheme and host rather than a string prefix, because prefixes are not
// host checks: https://huggingface.co.evil/ and https://huggingface.co@evil/
// both start with the Hub's URL and neither is the Hub.
func nextPageURL(header, want string) string {
	base, err := url.Parse(want)
	if err != nil {
		return ""
	}
	for _, link := range strings.Split(header, ",") {
		target, rest, ok := strings.Cut(strings.TrimSpace(link), ";")
		if !ok || !strings.Contains(rest, `rel="next"`) {
			continue
		}
		target = strings.TrimSpace(target)
		if !strings.HasPrefix(target, "<") || !strings.HasSuffix(target, ">") {
			continue
		}
		u, err := url.Parse(target[1 : len(target)-1])
		if err != nil || u.User != nil {
			continue
		}
		if u.Scheme == base.Scheme && u.Host == base.Host {
			return u.String()
		}
	}
	return ""
}

// DetectedFlags is what flag auto-detection found in a model card.
type DetectedFlags struct {
	ExtraArgs   string // e.g. "--enable-auto-tool-choice --tool-call-parser qwen3_coder"
	MaxModelLen string // from --max-model-len in the snippet, "" if absent
	// SGLangReasoningParser is the --reasoning-parser the card's SGLang
	// command names, "" if none or if its SGLang commands disagree.
	SGLangReasoningParser string
	// ReasoningParser is the --reasoning-parser the vllm serve line names.
	ReasoningParser string
	// SGLangExtraArgs is ExtraArgs rebuilt with SGLangReasoningParser, set
	// only when the two parsers differ.
	SGLangExtraArgs string
}

// DetectFlags fetches a model's README and pulls known vLLM serving flags
// out of any literal "vllm serve" snippet. Topology flags (parallelism,
// host, port, served-model-name) are deliberately ignored: they describe
// someone else's deployment, not the model.
func DetectFlags(ctx context.Context, repo string) (DetectedFlags, error) {
	var d DetectedFlags
	resp, err := get(ctx, baseURL+"/"+repo+"/raw/main/README.md")
	if err != nil {
		return d, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return d, fmt.Errorf("model card: HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return d, err
	}

	// Match "vllm serve ..." including backslash-continued lines. A card
	// without a snippet is common (facebook/opt-125m has none); it just
	// means there are no flags to adopt, not that we should skip the
	// context clamp below -- those models are exactly the ones whose real
	// context is far under the generic default.
	snippetRe := regexp.MustCompile(`vllm serve(?:.*\\\n)*.*`)
	d.SGLangReasoningParser = sglangReasoningParser(string(body))
	if snippet := snippetRe.FindString(string(body)); snippet != "" {
		d.ExtraArgs, d.MaxModelLen, d.ReasoningParser = parseSnippet(snippet, "")
		if d.ReasoningParser != "" && d.SGLangReasoningParser != "" && d.ReasoningParser != d.SGLangReasoningParser {
			d.SGLangExtraArgs, _, _ = parseSnippet(snippet, d.SGLangReasoningParser)
		}
	}

	// Model cards routinely paste a snippet from a bigger sibling model, so
	// the advertised context can exceed what this model actually supports.
	// vLLM refuses to start in that case, so clamp to the real ceiling.
	if limit, err := ContextLimit(ctx, repo); err == nil {
		d.MaxModelLen = clampContext(d.MaxModelLen, limit)
	}
	return d, nil
}

// ContextLimit reads max_position_embeddings from a model's config.json --
// the ceiling vLLM validates --max-model-len against. Returns 0 when the
// field is absent.
func ContextLimit(ctx context.Context, repo string) (int, error) {
	resp, err := get(ctx, baseURL+"/"+repo+"/raw/main/config.json")
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("model config: HTTP %d", resp.StatusCode)
	}
	var cfg struct {
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&cfg); err != nil {
		return 0, err
	}
	return cfg.MaxPositionEmbeddings, nil
}

// clampContext caps a model card's advertised context length at the model's
// real limit. An unparseable or absent value falls back to the limit itself,
// which is a better default than vLLM's.
func clampContext(value string, limit int) string {
	if limit <= 0 {
		return value
	}
	n, err := strconv.Atoi(value)
	if err != nil || n > limit {
		return strconv.Itoa(limit)
	}
	return value
}

var systemMessagePositionRe = regexp.MustCompile(
	`(\{%-\s*if\s+not\s+loop\.first\s*%\}\s*)` +
		`\{\{-\s*raise_exception\('System message must be at the beginning\.'\)\s*\}\}` +
		`(\s*\{%-\s*endif\s*%\})`,
)

// DetectChatTemplatePatch fetches a model's bundled chat template and checks
// whether it hard-crashes on a system-role message that isn't first in the
// conversation (found for real in some Qwen3.5-derived templates -- Ornith
// and Qwen3.5-122B both have it, byte-for-byte identical check). Agent
// harnesses -- Coder Agents included -- routinely inject periodic system-
// role "reminder" messages mid-conversation, which trips this. No vLLM flag
// controls it; the only lever is --chat-template <path> to override the
// whole file, so this returns a patched copy that drops a misplaced system
// message instead of crashing.
//
// ok is false (with empty patched and nil err) if the model's template
// doesn't have this issue -- the caller should just use the model's own
// bundled template as normal in that case, not add --chat-template at all.
func DetectChatTemplatePatch(ctx context.Context, repo string) (patched string, ok bool, err error) {
	resp, err := get(ctx, baseURL+"/"+repo+"/raw/main/chat_template.jinja")
	if err != nil {
		return "", false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", false, nil // no separate template file -- nothing to patch
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", false, err
	}
	return patchSystemMessageCheck(string(body))
}

// patchSystemMessageCheck is the pure (no I/O) half of
// DetectChatTemplatePatch, split out so the regex logic is unit-testable
// without mocking HTTP.
func patchSystemMessageCheck(text string) (patched string, ok bool, err error) {
	if !strings.Contains(text, "System message must be at the beginning") {
		return "", false, nil
	}
	result := systemMessagePositionRe.ReplaceAllString(text, "${1}{# llm-helper: dropped instead of raising #}${2}")
	if result == text {
		// The crash string is present but our regex didn't match its exact
		// surrounding shape -- don't silently claim a fix that didn't apply.
		return "", false, fmt.Errorf("found the crash string but this template's exact shape didn't match the patch regex -- patch by hand")
	}
	return result, true, nil
}
