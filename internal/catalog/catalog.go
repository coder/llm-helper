// Package catalog turns a HuggingFace repo into something this machine can
// serve: what a model would cost here, and the settings to serve it with.
//
// It sits under both the CLI and the web panel. Both need the same judgements
// -- whether a model fits, which flags the installed vLLM will actually accept
// -- and those judgements have a safe direction that is easy to lose track of.
// Keeping one copy is what stops the two interfaces from disagreeing about
// whether a 200 GB model fits in 84 GB.
package catalog

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/detect"
	"github.com/coder/vllm-helper/internal/hf"
	"github.com/coder/vllm-helper/internal/profile"
	"github.com/coder/vllm-helper/internal/provider"
	"github.com/coder/vllm-helper/internal/vllm"
)

// repoPattern is the shape of a HuggingFace repo id: one owner, one name.
//
// Worth being strict about, because the string becomes both a URL sent to the
// Hub and -- through profile.NameFromRepo -- part of a filename. The Hub
// itself allows only these characters, so nothing legitimate is turned away.
var repoPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)

// ValidRepo reports whether repo is a well-formed HuggingFace repo id.
//
// The dot-only names are excluded separately: they satisfy the pattern but
// NameFromRepo would turn "owner/.." into a profile name of "..", and a
// filename assembled from that is not something to find out about later.
func ValidRepo(repo string) bool {
	if !repoPattern.MatchString(repo) {
		return false
	}
	name := repo[strings.IndexByte(repo, '/')+1:]
	return strings.Trim(name, ".") != ""
}

// Result is the machine-readable shape of one search hit.
type Result struct {
	ID        string   `json:"id"`
	URL       string   `json:"url"`
	License   string   `json:"license"`
	Downloads int64    `json:"downloads"`
	Likes     int64    `json:"likes"`
	CreatedAt string   `json:"created_at"`
	QuantTags []string `json:"quant_tags"`
}

// Detail is everything we can learn about serving one model here.
type Detail struct {
	Result
	SizeGB float64 `json:"size_gb"` // 0 = unknown
	// SizeError says why the size is missing when the Hub could not be read,
	// as opposed to a repo that simply has no servable weights.
	SizeError string `json:"size_error,omitempty"`
	// Variants is how many complete copies of the model the repo holds, so
	// 1 means SizeGB is the whole story and more means it is the smallest of
	// them. Absent when SizeGB is unknown, since then there was nothing to
	// count.
	Variants int `json:"variants,omitempty"`
	// All lists those copies, largest first, so the choice is visible rather
	// than made silently. SizeGB being the smallest is the right answer to
	// "could anything here run" and the wrong one to show as the headline,
	// since the smallest quant in a repo is also its worst.
	All []hf.Variant `json:"all_variants,omitempty"`
	// Best is the largest copy that fits the budget -- what someone would
	// choose by hand, and what the verdict should be about. Absent when
	// nothing fits or there is no budget to judge against.
	Best *hf.Variant `json:"best_variant,omitempty"`
	// NeedsGGUF is true when every copy in this repo is GGUF, so the repo as
	// a whole is out of reach without vLLM's out-of-tree plugin.
	NeedsGGUF bool `json:"needs_gguf,omitempty"`
	// GGUFReady is whether this machine's vLLM can load GGUF at all.
	GGUFReady bool `json:"gguf_ready"`
	// ArchBlocked says why no GGUF build of this model can be served,
	// whatever the machine or the quant -- set when the architecture itself
	// is one the plugin cannot map.
	ArchBlocked string `json:"arch_blocked,omitempty"`
	// BaseModel is where a GGUF's tokenizer and config have to come from,
	// since GGUF repos largely ship no config.json of their own.
	BaseModel     string  `json:"base_model,omitempty"`
	BudgetGB      float64 `json:"budget_gb"`             // 0 = unknown (no config)
	FitsBudget    *bool   `json:"fits_budget,omitempty"` // absent when either side is unknown
	DetectedFlags string  `json:"detected_flags"`
	MaxModelLen   string  `json:"detected_max_model_len,omitempty"`
	// DroppedFlags explains anything the card asked for that the installed
	// vLLM won't take, so it's visible before pulling rather than as a
	// crash loop afterwards.
	DroppedFlags []string `json:"dropped_flags,omitempty"`
	// KVBytesPerToken is what one token of context costs in KV cache, and
	// MaxContext the longest the model was trained for. Both 0 when the
	// config could not be read.
	KVBytesPerToken int64 `json:"kv_bytes_per_token,omitempty"`
	MaxContext      int   `json:"max_context,omitempty"`
	// ContextTokens is how much context this machine can hold for the model
	// once weights and overhead are paid for, and KVCacheGB what that costs.
	ContextTokens int64   `json:"context_tokens,omitempty"`
	KVCacheGB     float64 `json:"kv_cache_gb,omitempty"`
}

// ToResult converts a Hub model into a search hit.
func ToResult(m hf.Model) Result {
	return Result{
		ID:        m.ID,
		URL:       "https://huggingface.co/" + m.ID,
		License:   m.License(),
		Downloads: m.Downloads,
		Likes:     m.Likes,
		CreatedAt: m.CreatedAt,
		QuantTags: m.QuantTags(),
	}
}

// Inspect runs the live checks for one model: weight size against the given
// budget, and whether the card's flags survive this vLLM. A budgetGB of 0
// means no budget is configured.
//
// A model whose size could not be read keeps FitsBudget unset rather than
// guessing, so nothing downstream claims it fits.
// contextTarget is how many full-length requests to size a KV cache for; 0
// takes detect's default.
func Inspect(ctx context.Context, m hf.Model, budgetGB float64, caps vllm.Caps, contextTarget int) Detail {
	d := Detail{
		Result:    ToResult(m),
		BudgetGB:  budgetGB,
		BaseModel: m.BaseModel(),
		// Only claim the plugin is missing when the registry was actually
		// readable. An unreadable one means "we could not ask", and refusing
		// a pull on that basis would be worse than letting it try.
		GGUFReady: !caps.KnowsQuantizations() || caps.SupportsGGUF(),
	}
	// Whether a GGUF build of this architecture can be loaded at all needs the
	// base model's config, which is a separate request. Started here and read
	// after the weights have been measured, so it costs no wall time: both are
	// network-bound and neither depends on the other.
	//
	// Only for repos that name a base model, which in practice means GGUF
	// ones -- a normal repo has nothing to look up and pays nothing.
	//
	// The same config carries the attention shape, so one request answers
	// both questions. The repo's own config wins: a released model often
	// credits a base trained for a different context length. Base is a
	// fallback for repos that ship none, which in practice means GGUF.
	archCh := make(chan hf.Arch, 1)
	go func() {
		a, _ := hf.Config(ctx, m.ID)
		if a.ModelType == "" && a.MaxContext() == 0 && d.BaseModel != "" {
			a, _ = hf.Config(ctx, d.BaseModel)
		}
		archCh <- a
	}()

	// A failed measurement is not the same as a repo with nothing to measure:
	// the pagination guards report an error precisely so a short listing is
	// not mistaken for a small model, and swallowing it here would undo that.
	if w, err := hf.WeightsSize(ctx, m.ID); err != nil {
		d.SizeError = err.Error()
	} else if w.SizeGB > 0 {
		d.SizeGB = w.SizeGB
		d.Variants = w.Variants
		d.All = w.All
		d.NeedsGGUF = allGGUF(w.All)

		// Mark what the architecture rules out before anything is ranked, so
		// a quant that cannot load is neither recommended nor selectable.
		// The alternative is finding out from a crash loop, by which point
		// the model that was serving has already been stopped.
		arch := <-archCh
		d.MaxContext = arch.MaxContext()
		d.KVBytesPerToken = arch.KVBytesPerToken(d.MaxContext)
		// Only a repo that publishes GGUF can be blocked by the plugin's
		// mapping, and a safetensors MoE is served by vLLM directly.
		if ok, why := hf.GGUFServable(arch); !ok && d.BaseModel != "" {
			d.ArchBlocked = why
			for i := range d.All {
				if d.All[i].GGUF && d.All[i].Servable() {
					d.All[i].Blocked = why
				}
			}
		}

		if d.BudgetGB > 0 {
			fits := w.SizeGB <= d.BudgetGB
			d.FitsBudget = &fits
			// Which one to actually run is a separate question from
			// whether any of them can, and only worth answering when
			// there is more than one to choose between.
			// GGUFReady, not SupportsGGUF: an unreadable registry means
			// "we could not ask", and every other path here offers those
			// variants. Asking the stricter question only here would rank
			// nothing on a machine that is about to be offered the lot.
			if best, ok := w.BestFit(d.BudgetGB, d.GGUFReady); ok && len(w.All) > 1 {
				d.Best = &best
			}
		}

		// How much context this machine can actually hold for it, which is
		// the question "does it fit" is usually standing in for.
		if d.KVBytesPerToken > 0 && d.MaxContext > 0 {
			size := d.SizeGB
			if d.Best != nil {
				size = d.Best.SizeGB
			}
			plan := detect.Probe().PlanFor(int64(size*1e9), d.KVBytesPerToken, d.MaxContext, contextTarget)
			d.ContextTokens = plan.Tokens
			d.KVCacheGB = float64(plan.KVCacheBytes) / 1e9
		}
	}
	if flags, err := hf.DetectFlags(ctx, m.ID); err == nil {
		// Validate here too: seeing that a card's parser won't work on this
		// machine is most useful before pulling tens of gigabytes.
		d.DetectedFlags, d.DroppedFlags = vllm.Check(flags.ExtraArgs, caps)
		d.MaxModelLen = flags.MaxModelLen
	}
	return d
}

// ForProvider reshapes what Inspect found for the server that will load the
// model, since Inspect judges GGUF by vLLM's plugin. A server that picks its
// own GGUF quant has none of the plugin's limits, so split and subfolder
// quants are offered, and copies that are not GGUF are not. One that loads no
// GGUF has every GGUF copy ruled out. Best is ranked again over what is left,
// so search and the panel never offer what the server cannot load. The card's
// flags are vllm serve's and this server is not taking them, so they stop
// reading as detected -- a pull would not have copied them either.
func ForProvider(d Detail, prov provider.Provider) Detail {
	caps := prov.Capabilities()
	if caps.VLLMArgs {
		return d
	}
	// GGUFReady and ArchBlocked are about vLLM's plugin, and installing it
	// would change nothing here.
	d.GGUFReady = true
	d.ArchBlocked = ""
	d.DetectedFlags = ""
	d.DroppedFlags = nil
	if !caps.GGUFQuant && d.NeedsGGUF {
		d.ArchBlocked = "only published as GGUF here, which " + prov.Name() + " does not load"
	}
	d.All = slices.Clone(d.All)
	for i := range d.All {
		v := &d.All[i]
		switch {
		case caps.GGUFQuant && v.GGUF:
			v.Blocked = ""
		case caps.GGUFQuant:
			v.Blocked = prov.Name() + " loads only GGUF"
		case v.GGUF:
			v.Blocked = prov.Name() + " does not load GGUF"
		}
	}
	d.Best = nil
	if d.BudgetGB > 0 && len(d.All) > 1 {
		if best, ok := (hf.Weights{All: d.All}).BestFit(d.BudgetGB, true); ok {
			d.Best = &best
		}
	}
	return d
}

// Plan is the settings a new profile should be created with, plus what was
// noticed while working them out. Notes are returned rather than printed so
// the CLI can put them on a terminal and the panel can put them in a browser.
type Plan struct {
	Opts  profile.CreateOpts
	Notes []string
}

// PrepareFor is PrepareProfile for whichever provider will serve the model.
// Everything PrepareProfile works out (card flags, parser checks, a sized KV
// cache, the template patch) is in vLLM's terms, so a profile for any other
// provider names the model and leaves the rest to that server's defaults.
func PrepareFor(ctx context.Context, dirs config.Dirs, prov provider.Provider, repo string, choice *Choice, caps vllm.Caps, contextTarget int) Plan {
	if prov.Capabilities().VLLMArgs {
		return PrepareProfile(ctx, dirs, repo, choice, caps, contextTarget)
	}
	p := Plan{
		Opts: profile.CreateOpts{
			Repo:     repo,
			Provider: prov.Name(),
			Source:   "llm-helper new",
			Status:   "NOT YET VERIFIED -- served by " + prov.Name() + " with its defaults",
		},
		Notes: []string{"Served by " + prov.Name() + ". The flags on a model card are vLLM's, so none were copied; " +
			"add any this server needs to EXTRA_ARGS."},
	}
	// Named the way PrepareProfile names a quant, so `new` and the panel
	// find the profile again under the name they derive.
	if choice != nil && prov.Capabilities().GGUFQuant {
		if sel, ok := choice.Variant.Selector(); ok {
			p.Opts.Repo = repo + sel
			p.Opts.Name = profile.NameFromRepo(repo) + "-" + strings.ToLower(choice.Variant.Label)
			p.Notes = append(p.Notes, "Serving the "+choice.Variant.Label+" quant.")
		}
	}
	return p
}

// PrepareProfile works out how to serve a model this machine has never run:
// which flags its card prescribes, which of those the installed vLLM will
// take, and whether its chat template needs the local patch.
//
// Nothing here is fatal. Every check that fails degrades to a note, because
// the alternative -- refusing to save a profile because the Hub was slow --
// helps nobody.
func PrepareProfile(ctx context.Context, dirs config.Dirs, repo string, choice *Choice, caps vllm.Caps, contextTarget int) Plan {
	var p Plan
	name := profile.NameFromRepo(repo)
	model := repo

	// A chosen quant becomes part of the model string, which is how vLLM
	// addresses one: "repo:Q8_0". Only GGUF works this way -- for everything
	// else the repo is the quant, so there is nothing to append.
	var extraFlags string
	if choice != nil {
		if sel, ok := choice.Variant.Selector(); ok {
			model += sel
			name = profile.NameFromRepo(repo) + "-" + strings.ToLower(choice.Variant.Label)
			p.Notes = append(p.Notes, "Serving the "+choice.Variant.Label+" quant.")
		}
		// vLLM's GGUF plugin resolves its HF config from the tokenizer repo,
		// and GGUF publishers largely ship no config.json, so without this
		// the server cannot start at all.
		if choice.Variant.GGUF && choice.BaseModel != "" {
			extraFlags = "--tokenizer " + choice.BaseModel
			p.Notes = append(p.Notes,
				"Pointing --tokenizer at "+choice.BaseModel+
					", since GGUF repos rarely ship the config vLLM needs.")
		}
	}

	detected, _ := hf.DetectFlags(ctx, repo)

	// Kept apart from extraFlags on purpose. These two end up in the same
	// ExtraArgs but they are not the same claim: one is what the model's
	// author prescribed, the other is what this tool added. Merging them
	// before reporting attributed our own --tokenizer to the card, and said
	// "flags auto-detected from README" about a repo whose README had none.
	cardArgs, dropped := vllm.Check(detected.ExtraArgs, caps)
	p.Notes = append(p.Notes, dropped...)

	if patched, needsPatch, err := hf.DetectChatTemplatePatch(ctx, repo); err != nil {
		p.Notes = append(p.Notes, "Chat template check failed, continuing without it: "+err.Error())
	} else if needsPatch {
		tmplPath := filepath.Join(dirs.Data, "chat-templates", name+".jinja")
		switch {
		case os.MkdirAll(filepath.Dir(tmplPath), 0o755) != nil:
			p.Notes = append(p.Notes, "Couldn't create the chat-templates directory, continuing without the patch.")
		case os.WriteFile(tmplPath, []byte(patched), 0o644) != nil:
			p.Notes = append(p.Notes, "Couldn't write the patched template, continuing without it.")
		default:
			p.Notes = append(p.Notes,
				"This model's bundled template crashes if a system-role message isn't first in "+
					"the conversation (common with agent harnesses like Coder Agents). Patched a "+
					"local copy: "+tmplPath)
			extraFlags = strings.TrimSpace(extraFlags + " --chat-template " + tmplPath)
		}
	}

	p.Opts = profile.CreateOpts{
		Repo:        model,
		Name:        name,
		MaxModelLen: detected.MaxModelLen,
		ExtraArgs:   strings.TrimSpace(extraFlags + " " + cardArgs),
		Source:      "llm-helper new",
	}

	// Size the cache rather than letting vLLM fill whatever is left.
	maxLen, _ := strconv.Atoi(p.Opts.MaxModelLen)
	if maxLen == 0 {
		maxLen = profile.DefaultMaxModelLen
	}
	arch, _ := hf.Config(ctx, configRepoFor(repo, choice))
	cache, clamped := planCache(detect.Probe(), arch, weightBytes(ctx, repo, choice), maxLen, contextTarget)
	if clamped != maxLen {
		p.Opts.MaxModelLen = strconv.Itoa(clamped)
	}
	if cache.KVCacheBytes > 0 {
		p.Opts.KVCacheBytes = cache.KVCacheBytes
		p.Notes = append(p.Notes, fmt.Sprintf(
			"Sized the KV cache at %.1f GB, which is %s of context. "+
				"Left to itself vLLM would claim everything free, which on a shared "+
				"memory pool is taken from the OS.",
			float64(cache.KVCacheBytes)/1e9, tokenCount(cache.Tokens)))
	}
	if cache.Note != "" {
		p.Notes = append(p.Notes, cache.Note)
	}

	// Harmony reads its vocab from disk rather than the model repo, and
	// without it the server starts healthy and fails every request.
	if needsHarmony(arch.ModelType) {
		if note := ensureHarmonyEncodings(dirs.Data); note != "" {
			p.Notes = append(p.Notes, note)
		}
		// Harmony is a wire format as well as a tokenizer: the parsers that
		// read it are what turn the model's output into an answer.
		if flags, notes := harmonyServeFlags(caps, p.Opts.ExtraArgs); len(flags) > 0 {
			p.Opts.ExtraArgs = strings.TrimSpace(p.Opts.ExtraArgs + " " + strings.Join(flags, " "))
			p.Notes = append(p.Notes, notes...)
		}
	}
	// Only the card's own flags decide this. What we added says nothing about
	// whether the author documented how to serve the model, which is what the
	// status is for.
	if cardArgs != "" {
		p.Notes = append(p.Notes, "Auto-detected from the model card: "+cardArgs)
		p.Opts.Status = "NOT YET VERIFIED -- flags auto-detected from README"
	} else {
		p.Notes = append(p.Notes,
			"No vllm serve snippet found in the model's README -- check its HF card by hand "+
				"and edit the profile if it needs specific flags.")
		p.Opts.Status = "NOT YET VERIFIED -- no flags auto-detected, check the model card by hand"
	}
	return p
}

// planCache sizes the KV cache for one model, and reports the context length it
// was sized for -- clamped to what the model was trained for, since asking for
// more is how vLLM refuses to start and a sliding-window model's cost per token
// depends on the answer.
//
// Pure so the sizing can be checked against a machine other than this one.
func planCache(hw detect.Hardware, arch hf.Arch, weights int64, maxLen, target int) (detect.Plan, int) {
	if n := arch.MaxContext(); n > 0 && n < maxLen {
		maxLen = n
	}
	kv := arch.KVBytesPerToken(maxLen)
	if kv <= 0 {
		// Nothing known about the shape, so leave it to vLLM's profiling
		// rather than guess: a wrong cache size is worse than a slow one.
		return detect.Plan{}, maxLen
	}
	return hw.PlanFor(weights, kv, maxLen, target), maxLen
}

// weightBytes is what the model will occupy. A chosen quant names its own size;
// otherwise it has to be measured, and a plan given 0 believes the weights are
// free and hands their room to the cache.
func weightBytes(ctx context.Context, repo string, choice *Choice) int64 {
	if choice != nil {
		return int64(choice.Variant.SizeGB * 1e9)
	}
	w, err := hf.WeightsSize(ctx, repo)
	if err != nil {
		return 0
	}
	return int64(w.SizeGB * 1e9)
}

// configRepoFor names the repo holding the config.json for a model's shape.
// GGUF builds keep theirs in the base model they were converted from.
func configRepoFor(repo string, choice *Choice) string {
	if choice != nil && choice.Variant.GGUF && choice.BaseModel != "" {
		return choice.BaseModel
	}
	return repo
}

// tokenCount renders a context length the way people say it.
func tokenCount(n int64) string {
	if n >= 1000 {
		return fmt.Sprintf("about %.0fk tokens", float64(n)/1000)
	}
	return fmt.Sprintf("%d tokens", n)
}

// allGGUF reports whether every copy in a repo is GGUF, which makes the repo
// itself unreachable without the plugin. A repo that also ships safetensors is
// not, since that copy serves natively.
func allGGUF(all []hf.Variant) bool {
	if len(all) == 0 {
		return false
	}
	for _, v := range all {
		if !v.GGUF {
			return false
		}
	}
	return true
}

// FindVariant resolves a user's quant choice against what a repo holds.
//
// Matching is case-insensitive on the label, since quant names are written
// however the publisher felt: Q8_0, q8_0, and UD-Q2_K_XL all appear.
func FindVariant(all []hf.Variant, label string) (hf.Variant, bool) {
	for _, v := range all {
		if strings.EqualFold(v.Label, label) {
			return v, true
		}
	}
	return hf.Variant{}, false
}

// Choice is a user's decision about which copy of a model to serve, together
// with the base model its tokenizer must come from.
type Choice struct {
	Variant   hf.Variant
	BaseModel string
}

// ChooseGGUFQuant resolves want against the GGUF files repo holds, for a
// provider other than vLLM that loads one itself. The limits of vLLM's GGUF
// plugin (split files, subfolders, unsupported architectures) are not that
// server's, so they are not checked here; the server reports what it cannot
// load when it starts.
func ChooseGGUFQuant(ctx context.Context, repo, want string) (*Choice, error) {
	w, err := hf.WeightsSize(ctx, repo)
	if err != nil {
		return nil, fmt.Errorf("looking up %s: %w", repo, err)
	}
	v, ok := FindVariant(w.All, want)
	if !ok {
		var have []string
		for _, x := range w.All {
			if x.GGUF {
				have = append(have, x.Label)
			}
		}
		if len(have) == 0 {
			return nil, fmt.Errorf("%s holds no GGUF files, so there is no quant to choose -- look for a GGUF repo of it", repo)
		}
		return nil, fmt.Errorf("%s has no quant called %q -- it has: %s", repo, want, strings.Join(have, ", "))
	}
	if _, ok := v.Selector(); !ok {
		return nil, fmt.Errorf("%s is not a GGUF quant, so it cannot be chosen by name", v.Label)
	}
	return &Choice{Variant: v}, nil
}
