package catalog

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/detect"
	"github.com/coder/vllm-helper/internal/hf"
	"github.com/coder/vllm-helper/internal/profile"
	"github.com/coder/vllm-helper/internal/provider"
	"github.com/coder/vllm-helper/internal/vllm"
)

// A repo id becomes both a URL sent to the Hub and, through NameFromRepo, part
// of a filename. The web panel takes it from a query string, so this is the
// check standing between a URL and the filesystem.
func TestValidRepo(t *testing.T) {
	valid := []string{
		"Qwen/Qwen3-0.6B",
		"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
		"unsloth/gemma-3-27b-it-GGUF",
		"bartowski/Qwen_Qwen3-32B-GGUF",
		"mistralai/Mistral-7B-Instruct-v0.3",
		"a/b",
	}
	for _, r := range valid {
		if !ValidRepo(r) {
			t.Errorf("ValidRepo(%q) = false, want true: this is a real repo id", r)
		}
	}

	invalid := []struct{ repo, why string }{
		{"", "empty"},
		{"noslash", "a repo id has an owner"},
		{"a/b/c", "two slashes is not a repo id"},
		{"../../etc/passwd", "traversal"},
		{"owner/..", "would become a profile name of '..'"},
		{"owner/.", "would become a profile name of '.'"},
		{"owner/...", "dots only"},
		{"owner/na me", "space"},
		{"owner/na\x00me", "NUL"},
		{"owner/name\n", "newline"},
		{"/name", "no owner"},
		{"owner/", "no name"},
		{"owner/sub/name", "nested"},
		{"-owner/name", "leading dash"},
		{"owner/../name", "traversal in the middle"},
		{"https://evil.test/x", "a URL"},
	}
	for _, c := range invalid {
		if ValidRepo(c.repo) {
			t.Errorf("ValidRepo(%q) = true, want false (%s)", c.repo, c.why)
		}
	}
}

// Inspect has to answer two different questions from one measurement, and
// they have opposite answers: "could anything in this repo run here" wants the
// smallest, while "which one should I run" wants the biggest that fits. Fusing
// them is what made a 32B repo look like a 2-bit-only proposition.
func TestInspectSeparatesTheFitCheckFromTheChoice(t *testing.T) {
	// Stands in for bartowski/Qwen_Qwen3-32B-GGUF against an 84 GB budget.
	w := hf.Weights{
		SizeGB:   9.0,
		Variants: 3,
		All: []hf.Variant{
			{Name: "m-Q8_0.gguf", Label: "Q8_0", SizeGB: 34.8},
			{Name: "m-Q4_K_M.gguf", Label: "Q4_K_M", SizeGB: 19.8},
			{Name: "m-IQ2_XXS.gguf", Label: "IQ2_XXS", SizeGB: 9.0},
		},
	}

	best, ok := w.BestFit(84, true)
	if !ok || best.Label != "Q8_0" {
		t.Fatalf("BestFit(84) = %q/%v, want Q8_0", best.Label, ok)
	}
	// The conservative figure is unchanged, because the fit check still has
	// to ask whether anything here could run at all.
	if w.SizeGB != 9.0 {
		t.Errorf("SizeGB = %v, want the smallest kept for the fit check", w.SizeGB)
	}
	// And the choice is a strictly better model than the fit check's answer.
	if best.SizeGB <= w.SizeGB {
		t.Errorf("best %.1f GB is not larger than the smallest %.1f GB", best.SizeGB, w.SizeGB)
	}

	// A budget that only the worst quant clears must not pretend otherwise.
	if got, _ := w.BestFit(10, true); got.Label != "IQ2_XXS" {
		t.Errorf("BestFit(10) = %q, want IQ2_XXS: nothing better fits", got.Label)
	}
}

// A chosen quant has to reach vLLM's command line, and for GGUF that means
// two things at once: the selector on the model string, and a tokenizer to
// read the config from, since GGUF repos rarely ship one.
func TestPrepareProfileAppliesAQuantChoice(t *testing.T) {
	dirs := config.Dirs{Data: t.TempDir()}
	choice := &Choice{
		Variant:   hf.Variant{Name: "m-Q8_0.gguf", Label: "Q8_0", GGUF: true},
		BaseModel: "Qwen/Qwen3-32B",
	}
	p := PrepareProfile(context.Background(), dirs, "bartowski/Qwen_Qwen3-32B-GGUF", choice, vllm.Caps{}, 0)

	if want := "bartowski/Qwen_Qwen3-32B-GGUF:Q8_0"; p.Opts.Repo != want {
		t.Errorf("Repo = %q, want %q", p.Opts.Repo, want)
	}
	// A name of its own, or a second quant of the same repo could not exist
	// alongside the first.
	if want := "qwen_qwen3-32b-gguf-q8_0"; p.Opts.Name != want {
		t.Errorf("Name = %q, want %q", p.Opts.Name, want)
	}
	if !strings.Contains(p.Opts.ExtraArgs, "--tokenizer Qwen/Qwen3-32B") {
		t.Errorf("ExtraArgs = %q, want a --tokenizer: GGUF repos rarely ship a config", p.Opts.ExtraArgs)
	}

	// No choice means the repo's default, unchanged.
	plain := PrepareProfile(context.Background(), dirs, "Qwen/Qwen3-32B", nil, vllm.Caps{}, 0)
	if plain.Opts.Repo != "Qwen/Qwen3-32B" {
		t.Errorf("Repo = %q, want the bare repo when nothing was chosen", plain.Opts.Repo)
	}
	if strings.Contains(plain.Opts.ExtraArgs, "--tokenizer") {
		t.Errorf("ExtraArgs = %q, want no tokenizer for a normal repo", plain.Opts.ExtraArgs)
	}
}

// Everything PrepareProfile reads off the Hub is in vLLM's terms. A profile for
// another provider must not carry any of it, and must not wait on the Hub to
// find that out.
func TestPrepareForAnotherProviderNamesOnlyTheModel(t *testing.T) {
	var hits sync.Map
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Store(r.URL.Path, true)
		fmt.Fprint(w, "```\nvllm serve x --tool-call-parser hermes\n```")
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	mlx, _ := provider.Lookup(provider.MLX)
	p := PrepareFor(context.Background(), config.Dirs{Data: t.TempDir()}, mlx,
		"mlx-community/Qwen3-4B-4bit", nil, vllm.Caps{}, 0)

	if p.Opts.Provider != provider.MLX || p.Opts.Repo != "mlx-community/Qwen3-4B-4bit" {
		t.Errorf("Opts = %+v, want the repo and PROVIDER=mlx", p.Opts)
	}
	if p.Opts.ExtraArgs != "" || p.Opts.KVCacheBytes != 0 || p.Opts.MaxModelLen != "" {
		t.Errorf("Opts = %+v, want no vLLM settings", p.Opts)
	}
	hits.Range(func(k, _ any) bool {
		t.Errorf("asked the Hub for %s while preparing an mlx profile", k)
		return true
	})

	// vLLM still gets the full treatment, and a profile that names no
	// provider, as every vLLM profile always has.
	vl, _ := provider.Lookup(provider.VLLM)
	v := PrepareFor(context.Background(), config.Dirs{Data: t.TempDir()}, vl,
		"Qwen/Qwen3-32B", nil, vllm.Caps{}, 0)
	if v.Opts.Provider != "" {
		t.Errorf("vLLM profile Provider = %q, want empty", v.Opts.Provider)
	}
	if !strings.Contains(v.Opts.ExtraArgs, "--tool-call-parser hermes") {
		t.Errorf("vLLM ExtraArgs = %q, want the card's flags", v.Opts.ExtraArgs)
	}
}

// llama.cpp loads one GGUF file out of a repo, including a split one that
// vLLM's plugin cannot, and the profile is named and addressed the way a vLLM
// quant profile is, so `new` finds it again under the name it derives.
func TestChooseGGUFQuantForLlamaCPP(t *testing.T) {
	const repo = "bartowski/Qwen_Qwen3-32B-GGUF"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var files []map[string]any
		switch r.URL.Path {
		case "/api/models/" + repo + "/tree/main":
			for name, size := range map[string]int64{
				"Qwen_Qwen3-32B-Q4_K_M.gguf":              19e9,
				"Qwen_Qwen3-32B-Q8_0.gguf":                34e9,
				"Qwen_Qwen3-32B-BF16-00001-of-00002.gguf": 33e9,
				"Qwen_Qwen3-32B-BF16-00002-of-00002.gguf": 33e9,
				"README.md": 1e3,
			} {
				files = append(files, map[string]any{"type": "file", "path": name, "size": size})
			}
		case "/api/models/Qwen/Qwen3-32B/tree/main":
			files = append(files, map[string]any{"type": "file", "path": "model.safetensors", "size": 65e9})
		default:
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(files)
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()
	ctx := context.Background()
	llama, _ := provider.Lookup(provider.LlamaCPP)

	choice, err := ChooseGGUFQuant(ctx, repo, "bf16")
	if err != nil {
		t.Fatalf("a split GGUF was refused: %v", err)
	}
	p := PrepareFor(ctx, config.Dirs{Data: t.TempDir()}, llama, repo, choice, vllm.Caps{}, 0)
	if p.Opts.Repo != repo+":BF16" || p.Opts.Name != "qwen_qwen3-32b-gguf-bf16" || p.Opts.Provider != provider.LlamaCPP {
		t.Errorf("Opts = %+v, want MODEL %s:BF16 in profile qwen_qwen3-32b-gguf-bf16", p.Opts, repo)
	}

	_, err = ChooseGGUFQuant(ctx, repo, "Q6_K")
	if err == nil || !strings.Contains(err.Error(), "Q8_0") || !strings.Contains(err.Error(), "Q4_K_M") {
		t.Errorf("unknown quant: err = %v, want it to list the quants the repo has", err)
	}
	_, err = ChooseGGUFQuant(ctx, "Qwen/Qwen3-32B", "Q8_0")
	if err == nil || !strings.Contains(err.Error(), "no GGUF") {
		t.Errorf("safetensors repo: err = %v, want it refused for holding no GGUF", err)
	}
}

// Search and the panel offer quants from what Inspect found, which is judged
// by vLLM's GGUF plugin. llama.cpp gets the split and subfolder quants the
// plugin refuses, on a machine whose vLLM has no plugin at all, and MLX gets
// no GGUF to pick, which PrepareFor would otherwise drop without a word.
func TestForProvider(t *testing.T) {
	t.Parallel()
	inspected := Detail{
		BudgetGB:    80,
		GGUFReady:   false,
		ArchBlocked: "a mixture-of-experts model",
		NeedsGGUF:   true,
		All: []hf.Variant{
			{Name: "BF16", Label: "BF16", SizeGB: 66, GGUF: true, Blocked: "split across files, which vLLM's GGUF plugin cannot load"},
			{Name: "Q8_0.gguf", Label: "Q8_0", SizeGB: 34, GGUF: true, Blocked: "a mixture-of-experts model"},
			{Name: "Q4_K_M.gguf", Label: "Q4_K_M", SizeGB: 19, GGUF: true, Blocked: "a mixture-of-experts model"},
		},
	}
	get := func(name string) provider.Provider {
		p, err := provider.Lookup(name)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}

	if got := ForProvider(inspected, get(provider.VLLM)); got.ArchBlocked == "" || got.GGUFReady || got.Best != nil {
		t.Errorf("vllm: %+v, want Inspect's view unchanged", got)
	}
	inspected.DetectedFlags = "--tokenizer Qwen/Qwen3-32B"
	inspected.DroppedFlags = []string{"--a-flag-this-vllm-lacks"}

	llama := ForProvider(inspected, get(provider.LlamaCPP))
	if !llama.GGUFReady || llama.ArchBlocked != "" {
		t.Errorf("llamacpp: GGUFReady %v, ArchBlocked %q, want the plugin's limits lifted", llama.GGUFReady, llama.ArchBlocked)
	}
	if llama.DetectedFlags != "" || llama.DroppedFlags != nil {
		t.Errorf("llamacpp: flags %q %v, want none: the card's flags are vllm serve's and none were copied", llama.DetectedFlags, llama.DroppedFlags)
	}
	for _, v := range llama.All {
		if !v.Servable() {
			t.Errorf("llamacpp: %s blocked as %q", v.Label, v.Blocked)
		}
	}
	if llama.Best == nil || llama.Best.Label != "BF16" {
		t.Errorf("llamacpp: Best = %+v, want BF16, the biggest that fits", llama.Best)
	}
	if inspected.All[0].Blocked == "" {
		t.Error("ForProvider changed the Detail it was given")
	}

	mlx := ForProvider(inspected, get(provider.MLX))
	if mlx.ArchBlocked == "" || mlx.Best != nil {
		t.Errorf("mlx: ArchBlocked %q, Best %+v, want the GGUF repo refused", mlx.ArchBlocked, mlx.Best)
	}
	for _, v := range mlx.All {
		if v.Servable() {
			t.Errorf("mlx: %s is offered, want every GGUF quant ruled out", v.Label)
		}
	}

	// A repo with both kinds: llama.cpp can only take the GGUF, and MLX only
	// the safetensors.
	mixed := Detail{BudgetGB: 80, All: []hf.Variant{
		{Name: ".", Label: "default", SizeGB: 65},
		{Name: "m-Q8_0.gguf", Label: "Q8_0", SizeGB: 34, GGUF: true},
	}}
	if got := ForProvider(mixed, get(provider.LlamaCPP)); got.Best == nil || got.Best.Label != "Q8_0" || got.All[0].Servable() {
		t.Errorf("llamacpp mixed repo: Best %+v, safetensors blocked %q, want Q8_0 and the safetensors ruled out", got.Best, got.All[0].Blocked)
	}
	if got := ForProvider(mixed, get(provider.MLX)); got.Best == nil || got.Best.Label != "default" || got.All[1].Servable() {
		t.Errorf("mlx mixed repo: Best %+v, want the safetensors and the GGUF ruled out", got.Best)
	}
}

// gpt-oss ships no vllm serve snippet, so nothing was detected and the profile
// was written with an empty EXTRA_ARGS. Serving it that way answers plain chat
// normally but returns an empty message for anything with tools, which reads as
// the model refusing to work rather than as a missing flag.
func TestAHarmonyProfileIsWrittenWithItsParsers(t *testing.T) {
	// The repo decides the architecture, so one server answers for both by
	// reading the path it was asked for.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		harmony := strings.Contains(r.URL.Path, "gpt-oss")
		switch {
		case strings.HasSuffix(r.URL.Path, "config.json") && harmony:
			fmt.Fprint(w, `{"model_type":"gpt_oss","num_hidden_layers":36,
			                "num_key_value_heads":8,"head_dim":64,
			                "max_position_embeddings":131072}`)
		case strings.HasSuffix(r.URL.Path, "config.json"):
			fmt.Fprint(w, `{"model_type":"qwen3","num_hidden_layers":28,
			                "num_key_value_heads":8,"head_dim":128,
			                "max_position_embeddings":40960}`)
		case strings.Contains(r.URL.Path, "/tree/"):
			fmt.Fprint(w, `[{"type":"file","path":"model.safetensors","size":1500000000}]`)
		default:
			fmt.Fprint(w, `# no serve snippet`)
		}
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()
	// A harmony model also fetches its vocab, which must not reach the real
	// internet from a test.
	defer swapEncodingsURL(srv.URL + "/encodings/")()

	dirs := config.Dirs{Data: t.TempDir()}
	p := PrepareProfile(context.Background(), dirs, "openai/gpt-oss-120b", nil, spark, 0)
	for _, want := range []string{
		"--tool-call-parser openai",
		"--enable-auto-tool-choice",
		"--reasoning-parser openai_gptoss",
	} {
		if !strings.Contains(p.Opts.ExtraArgs, want) {
			t.Errorf("ExtraArgs = %q, want it to contain %q", p.Opts.ExtraArgs, want)
		}
	}

	// A model that does not speak harmony has no use for them, and naming a
	// parser that does not match the model is how #23 produced empty reasoning.
	q := PrepareProfile(context.Background(), dirs, "Qwen/Qwen3-32B", nil, spark, 0)
	if strings.Contains(q.Opts.ExtraArgs, "parser") {
		t.Errorf("ExtraArgs = %q, want no parser flags on a model that isn't harmony", q.Opts.ExtraArgs)
	}
}

// Case matters to publishers and not to people.
func TestFindVariantIsCaseInsensitive(t *testing.T) {
	all := []hf.Variant{{Label: "Q8_0"}, {Label: "UD-Q2_K_XL"}}
	for _, in := range []string{"Q8_0", "q8_0", "Q8_0"} {
		if v, ok := FindVariant(all, in); !ok || v.Label != "Q8_0" {
			t.Errorf("FindVariant(%q) missed", in)
		}
	}
	if v, ok := FindVariant(all, "ud-q2_k_xl"); !ok || v.Label != "UD-Q2_K_XL" {
		t.Errorf("FindVariant lower-cased miss: %v/%v", v.Label, ok)
	}
	if _, ok := FindVariant(all, "Q4_K_M"); ok {
		t.Error("FindVariant should not invent a quant the repo lacks")
	}
}

// What the tool adds and what the model's author prescribed are different
// claims, and merging them before reporting made both wrong: a repo whose
// README contains no serve snippet was told "Auto-detected from the model
// card: --tokenizer ...", naming a flag this tool had just injected, and the
// profile was stamped "flags auto-detected from README" on that basis. The
// user-visible result was the same tokenizer reported twice, the second time
// falsely.
func TestNotesDoNotCreditTheCardForOurOwnFlags(t *testing.T) {
	dirs := config.Dirs{Data: t.TempDir()}
	choice := &Choice{
		Variant:   hf.Variant{Name: "m-Q8_0.gguf", Label: "Q8_0", GGUF: true},
		BaseModel: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
	}
	// A repo with no README snippet, which is the usual case for GGUF.
	p := PrepareProfile(context.Background(), dirs, "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", choice, vllm.Caps{}, 0)

	// The flag still has to reach vLLM -- the plugin cannot start without it.
	if !strings.Contains(p.Opts.ExtraArgs, "--tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct") {
		t.Errorf("ExtraArgs = %q, want the tokenizer we added", p.Opts.ExtraArgs)
	}

	var credited, mentions int
	for _, n := range p.Notes {
		if strings.HasPrefix(n, "Auto-detected from the model card:") {
			credited++
			if strings.Contains(n, "--tokenizer") {
				t.Errorf("note credits the card for a flag we added: %q", n)
			}
		}
		if strings.Contains(n, "--tokenizer") {
			mentions++
		}
	}
	if credited != 0 {
		t.Errorf("a card with no serve snippet produced %d auto-detected notes", credited)
	}
	if mentions != 1 {
		t.Errorf("the tokenizer is reported %d times, want exactly 1:\n  %s",
			mentions, strings.Join(p.Notes, "\n  "))
	}
	if strings.Contains(p.Opts.Status, "auto-detected from README") {
		t.Errorf("Status = %q, but the README prescribed nothing", p.Opts.Status)
	}
}

// The architecture check needs the base model's config, which is a second
// request. It runs concurrently with the weights measurement so it costs no
// wall time -- if it were sequential every GGUF inspect would get slower by a
// whole round trip, which is the thing that makes a picker unusable.
func TestInspectFetchesTheArchConcurrently(t *testing.T) {
	var mu sync.Mutex
	var overlapped bool
	inFlight := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		inFlight++
		if inFlight > 1 {
			overlapped = true
		}
		mu.Unlock()
		// Long enough that a sequential implementation cannot overlap by
		// accident, short enough not to slow the suite.
		time.Sleep(80 * time.Millisecond)
		mu.Lock()
		inFlight--
		mu.Unlock()

		switch {
		case strings.HasSuffix(r.URL.Path, "config.json"):
			fmt.Fprint(w, `{"model_type":"qwen3_moe","num_experts":128}`)
		case strings.Contains(r.URL.Path, "/tree/"):
			fmt.Fprint(w, `[{"type":"file","path":"m-Q8_0.gguf","size":34000000000},
			                {"type":"file","path":"m-Q4_K_M.gguf","size":19000000000}]`)
		default:
			fmt.Fprint(w, `# no serve snippet`)
		}
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	m := hf.Model{ID: "unsloth/demo-GGUF"}
	m.CardData.BaseModel = json.RawMessage(`"org/demo"`)

	d := Inspect(context.Background(), m, 84, vllm.Caps{}, 0)

	// Overlap is the whole claim, and it is deterministic: a sequential
	// implementation can never have two requests in flight at once. An
	// elapsed-time bound was here too and measured the same thing less
	// reliably -- it failed under a loaded test suite, where wall clock says
	// more about the other packages than about this one.
	if !overlapped {
		t.Error("the config fetch did not overlap the others -- it is costing a round trip")
	}

	// And the answer is the point: every quant is refused, with a reason
	// that names the way out.
	if d.ArchBlocked == "" {
		t.Fatal("a Qwen3-MoE GGUF repo was not flagged")
	}
	for _, v := range d.All {
		if v.Servable() {
			t.Errorf("%s is still offered despite the architecture being unloadable", v.Label)
		}
	}
	if d.Best != nil {
		t.Errorf("recommended %s from a repo where nothing can load", d.Best.Label)
	}
}

// An unreadable quantization registry means "we could not ask", and every
// other path here offers the repo's GGUF variants on that basis. Asking the
// stricter "does vLLM support GGUF" question only when ranking left a GGUF
// repo with no recommendation at all, on a machine that was about to be
// offered every one of those quants.
func TestBestIsStillChosenWhenTheQuantRegistryCouldNotBeRead(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "config.json"):
			fmt.Fprint(w, `{"model_type":"qwen3","num_hidden_layers":28,
			                "num_key_value_heads":8,"head_dim":128,
			                "max_position_embeddings":40960}`)
		case strings.Contains(r.URL.Path, "/tree/"):
			fmt.Fprint(w, `[{"type":"file","path":"m-Q8_0.gguf","size":34000000000},
			                {"type":"file","path":"m-Q4_K_M.gguf","size":19000000000},
			                {"type":"file","path":"m-IQ2_XXS.gguf","size":9000000000}]`)
		default:
			fmt.Fprint(w, `# no serve snippet`)
		}
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	m := hf.Model{ID: "someone/demo-GGUF"}
	m.CardData.BaseModel = json.RawMessage(`"org/demo"`)

	// An empty Caps is what a probe that could not run returns.
	d := Inspect(context.Background(), m, 84, vllm.Caps{}, 0)
	if !d.GGUFReady {
		t.Fatal("caps we could not read were taken as a missing GGUF plugin")
	}
	if len(d.All) != 3 {
		t.Fatalf("got %d variants, want the 3 the repo publishes", len(d.All))
	}
	if d.Best == nil {
		t.Fatal("no quant recommended from a repo whose quants are all on offer")
	}
	if d.Best.Label != "Q8_0" {
		t.Errorf("recommended %q, want Q8_0: the largest that fits the budget", d.Best.Label)
	}
}

// A repo that names a base model is still described by its own config. Reading
// the base model's instead reports the wrong shape: Qwen3-0.6B names
// Qwen3-0.6B-Base, which was trained for 32k of context where the released
// model does 40k, and sizing a KV cache to the wrong number is how a profile
// ends up either starved or wasteful.
func TestTheReposOwnConfigBeatsItsBaseModels(t *testing.T) {
	var mu sync.Mutex
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "config.json"):
			mu.Lock()
			asked = append(asked, r.URL.Path)
			mu.Unlock()
			if strings.Contains(r.URL.Path, "-Base") {
				fmt.Fprint(w, `{"model_type":"qwen3","num_hidden_layers":28,
				                "num_key_value_heads":8,"head_dim":128,
				                "max_position_embeddings":32768}`)
				return
			}
			fmt.Fprint(w, `{"model_type":"qwen3","num_hidden_layers":28,
			                "num_key_value_heads":8,"head_dim":128,
			                "max_position_embeddings":40960}`)
		case strings.Contains(r.URL.Path, "/tree/"):
			fmt.Fprint(w, `[{"type":"file","path":"model.safetensors","size":1500000000}]`)
		default:
			fmt.Fprint(w, `# no serve snippet`)
		}
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	m := hf.Model{ID: "Qwen/Qwen3-0.6B"}
	m.CardData.BaseModel = json.RawMessage(`["Qwen/Qwen3-0.6B-Base"]`)

	d := Inspect(context.Background(), m, 84, vllm.Caps{}, 0)
	if d.MaxContext != 40960 {
		t.Errorf("MaxContext = %d, want 40960 from the repo's own config", d.MaxContext)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, p := range asked {
		if strings.Contains(p, "-Base") {
			t.Errorf("fetched the base model's config (%s) when the repo has its own", p)
		}
	}
}

// GGUF repos genuinely ship no config, and there the base model is the only
// description of the model available.
func TestAConfiglessRepoFallsBackToItsBaseModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "config.json"):
			if !strings.Contains(r.URL.Path, "org/base") {
				http.NotFound(w, r)
				return
			}
			fmt.Fprint(w, `{"model_type":"qwen3","num_hidden_layers":28,
			                "num_key_value_heads":8,"head_dim":128,
			                "max_position_embeddings":40960}`)
		case strings.Contains(r.URL.Path, "/tree/"):
			fmt.Fprint(w, `[{"type":"file","path":"m-Q8_0.gguf","size":1500000000}]`)
		default:
			fmt.Fprint(w, `# no serve snippet`)
		}
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	m := hf.Model{ID: "someone/demo-GGUF"}
	m.CardData.BaseModel = json.RawMessage(`"org/base"`)

	d := Inspect(context.Background(), m, 84, vllm.Caps{}, 0)
	if d.KVBytesPerToken != 2*28*8*128*2 {
		t.Errorf("KVBytesPerToken = %d, want the base model's shape", d.KVBytesPerToken)
	}
	if d.MaxContext != 40960 {
		t.Errorf("MaxContext = %d, want 40960", d.MaxContext)
	}
}

// The write path, not just the estimate. A sliding-window model's cache has to
// come out sized for its real cost per token: gpt-oss-120b is 18 full-attention
// layers plus 18 sliding ones at a window of 128, and counting all 36 as full
// wrote a profile reserving twice what the context needed. Measured on a GB10,
// the doubled figure reserved 38.7 GB where 19.3 GB buys the same 4x131k.
func TestPrepareProfileSizesASlidingWindowModelCorrectly(t *testing.T) {
	types := make([]string, 36)
	for i := range types {
		if i%2 == 0 {
			types[i] = "sliding_attention"
		} else {
			types[i] = "full_attention"
		}
	}
	cfg, err := json.Marshal(map[string]any{
		"model_type": "gpt_oss", "num_hidden_layers": 36,
		"num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
		"sliding_window": 128, "layer_types": types,
		"max_position_embeddings": 131072, "torch_dtype": "bfloat16",
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "config.json") {
			w.Write(cfg)
			return
		}
		fmt.Fprint(w, "# no serve snippet")
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	var arch hf.Arch
	if err := json.Unmarshal(cfg, &arch); err != nil {
		t.Fatal(err)
	}
	// The machine this was measured on, not whichever runs the tests.
	hw := detect.Hardware{TotalMemBytes: 127600592 * 1024, Unified: true}
	const weights = 65_200_000_000 // gpt-oss-120b on disk

	plan, maxLen := planCache(hw, arch, weights, profile.DefaultMaxModelLen, 0)
	if plan.KVCacheBytes == 0 {
		t.Fatal("no KV cache size planned for a model whose shape is fully known")
	}
	if maxLen != 131072 {
		t.Errorf("sized for %d tokens of context, want the model's own 131072", maxLen)
	}

	// What one token really costs, half the layers capped at their window.
	perLayer := int64(2 * 8 * 64 * 2)
	perToken := perLayer * (18*int64(maxLen) + 18*128) / int64(maxLen)
	wantTokens := int64(detect.DefaultContextTarget) * int64(maxLen)

	if got := plan.KVCacheBytes / perToken; got < wantTokens*95/100 || got > wantTokens*105/100 {
		t.Errorf("reserved %d bytes = %d tokens, want about %d (%dx a full context)",
			plan.KVCacheBytes, got, wantTokens, detect.DefaultContextTarget)
	}
	// The bug: sizing as though every layer held the whole context. Measured
	// on a GB10, that reserved 38.7 GB where 19.3 GB buys the same context.
	if doubled := perLayer * 36 * wantTokens; plan.KVCacheBytes >= doubled {
		t.Errorf("reserved %d bytes, no better than counting all 36 layers full (%d)",
			plan.KVCacheBytes, doubled)
	}
	if plan.KVCacheBytes > 22e9 {
		t.Errorf("reserved %.1f GB, want about 19.3", float64(plan.KVCacheBytes)/1e9)
	}
}

// A plan given no weight believes the model is free and hands its room to the
// cache. Every pull without a quant choice used to do exactly that.
//
// It only shows up once the wanted context costs more than the machine has
// spare: below that the target decides the size and the weight never enters
// the arithmetic, which is why a 40k-context case would pass either way.
func TestTheModelsOwnWeightIsPartOfThePlan(t *testing.T) {
	const ctx = 262144
	arch := hf.Arch{
		NumHiddenLayers: 28, NumKeyValueHeads: 8, HeadDim: 128,
		MaxPositionEmbed: ctx, TorchDtype: "bfloat16",
	}
	hw := detect.Hardware{TotalMemBytes: 127600592 * 1024, Unified: true}

	small, _ := planCache(hw, arch, 2e9, ctx, 0)
	large, _ := planCache(hw, arch, 100e9, ctx, 0)
	if small.KVCacheBytes == 0 {
		t.Fatal("no cache planned for a 2 GB model on a 128 GB machine")
	}
	if large.KVCacheBytes >= small.KVCacheBytes {
		t.Errorf("a 100 GB model was planned %.1f GB of cache, no less than a 2 GB one at %.1f GB",
			float64(large.KVCacheBytes)/1e9, float64(small.KVCacheBytes)/1e9)
	}
	// And the weights plus their cache still have to fit the machine.
	committed := 100e9 + float64(large.KVCacheBytes)
	if ceiling := large.MemUtilization * float64(hw.TotalMemBytes); committed > ceiling {
		t.Errorf("100 GB model plus %.1f GB of cache exceeds the %.1f GB ceiling",
			float64(large.KVCacheBytes)/1e9, ceiling/1e9)
	}
}

// The planner above can only weigh a model it is told about. A pull with no
// quant choice has nothing naming a size, so the weights have to be measured
// -- otherwise the plan is handed 0 and reserves the model's own room for its
// cache. That was every safetensors pull.
func TestAPullWithoutAQuantChoiceStillMeasuresTheWeights(t *testing.T) {
	var mu sync.Mutex
	weighed := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/tree/"):
			mu.Lock()
			weighed = true
			mu.Unlock()
			fmt.Fprint(w, `[{"type":"file","path":"model.safetensors","size":65200000000}]`)
		case strings.HasSuffix(r.URL.Path, "config.json"):
			fmt.Fprint(w, `{"model_type":"qwen3","num_hidden_layers":28,
			                "num_key_value_heads":8,"head_dim":128,
			                "max_position_embeddings":262144}`)
		default:
			fmt.Fprint(w, "# no serve snippet")
		}
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	PrepareProfile(context.Background(), config.Dirs{Data: t.TempDir()}, "org/big-model", nil, vllm.Caps{}, 0)

	mu.Lock()
	defer mu.Unlock()
	if !weighed {
		t.Error("the pull planned a KV cache without ever measuring the model it has to fit around")
	}
}
