package cli

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/engine"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/resolve"
	"github.com/coder/llm-helper/internal/vllm"
)

// cmdSearch searches HuggingFace and walks the user through picking a
// model: a numbered result list, then a detail view with a live size check
// against this machine's budget, then pull-and-switch. With --json it
// prints the results and exits.
func newSearchCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:     "search <search terms>",
		Short:   "Search HuggingFace and pick a model to pull and run",
		GroupID: groupModels,
		RunE:    func(cmd *cobra.Command, args []string) error { return runSearch(cmd.Context(), args, jsonOut) },
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print results as JSON and exit (no prompts)")
	return c
}

func runSearch(ctx context.Context, args []string, jsonOut bool) error {
	if len(args) == 0 {
		return errors.New("usage: llm-helper search <search terms> [--json]")
	}
	query := strings.Join(args, " ")

	if !jsonOut {
		fmt.Printf("Searching HuggingFace for %q...\n", query)
	}
	models, err := hf.Search(ctx, query, 25)
	if err != nil {
		return fmt.Errorf("search request failed (network or rate limit): %w", err)
	}
	if jsonOut {
		out := make([]resolve.Result, 0, len(models))
		for _, m := range models {
			out = append(out, resolve.ToResult(m))
		}
		return printJSON(out)
	}
	if len(models) == 0 {
		fmt.Println("No text-generation models found. Try different terms.")
		return nil
	}
	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	fmt.Println()
	for i, m := range models {
		quant := strings.Join(m.QuantTags(), ",")
		if quant != "" {
			quant = "  [" + quant + "]"
		}
		fmt.Printf("  %2d) %-52s %9d dl  %5d likes%s\n", i+1, m.ID, m.Downloads, m.Likes, quant)
	}
	fmt.Println()

	// The picker needs a human on both ends; piped sessions get the list
	// (or --json) without prompts that would block forever.
	if !stdoutIsTTY() || !stdinIsTTY() {
		fmt.Println("(non-interactive session -- use --json for machine-readable results, or")
		fmt.Println("'llm-helper new <org/model>' to pull one of these directly)")
		return nil
	}

	for {
		pick := ask(fmt.Sprintf("Pick a model to inspect (1-%d, Enter to cancel)", len(models)), "")
		if pick == "" {
			fmt.Println("Cancelled.")
			return nil
		}
		n, err := strconv.Atoi(pick)
		if err != nil || n < 1 || n > len(models) {
			fmt.Println("Not a valid number, try again.")
			continue
		}
		m := models[n-1]
		fmt.Println()
		d := inspectModel(ctx, m, cfg)
		printModelDetail(d, cfg)
		fmt.Println()

		choice, cancelled := pickQuant(cfg, d)
		if cancelled {
			fmt.Println()
			continue
		}
		label := m.ID
		if choice != nil {
			label += " (" + choice.Variant.Label + ")"
		}
		if confirm(fmt.Sprintf("Pull %s and run it?", label), false) {
			return pullAndUp(ctx, cfg, m.ID, choice, "", watchOpts{follow: stdoutIsTTY(), wait: true})
		}
		fmt.Println()
	}
}

// cmdInspect shows the detail view for one repo without the search loop:
// vllm-helper inspect <org/model> [--json]
func newInspectCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:     "inspect <org/model-repo>",
		Short:   "Show a model's license, weight size against this machine, and detected flags",
		GroupID: groupModels,
		Args:    cobra.MaximumNArgs(1),
		RunE:    func(cmd *cobra.Command, args []string) error { return runInspect(cmd.Context(), args, jsonOut) },
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print the details as JSON")
	return c
}

func runInspect(ctx context.Context, args []string, jsonOut bool) error {
	if len(args) < 1 {
		return errors.New("usage: llm-helper inspect <org/model-repo> [--json]")
	}
	// A profile's MODEL names a quant as <repo>:<quant>, so inspect takes it
	// in that form too. The quant is not part of the repo id the Hub knows,
	// which answers an unknown id with a 401.
	repo, quant, _ := strings.Cut(args[0], ":")
	m, err := hf.Lookup(ctx, repo)
	if err != nil {
		return err
	}
	// Config is optional here: without it the size check just has no budget.
	cfg, err := loadConfig()
	if err != nil && !errors.Is(err, config.ErrNotConfigured) {
		return err
	}
	d := inspectModel(ctx, m, cfg)
	var picked *hf.Variant
	if quant != "" {
		v, ok := resolve.FindVariant(d.All, quant)
		if !ok {
			have := make([]string, 0, len(d.All))
			for _, x := range d.All {
				have = append(have, x.Label)
			}
			return fmt.Errorf("%s has no quant called %q -- it has: %s", repo, quant, strings.Join(have, ", "))
		}
		picked = &v
	}
	if jsonOut {
		return printJSON(d)
	}
	printModelDetail(d, cfg)
	if picked != nil {
		fit := ""
		switch {
		case !d.BudgetKnown:
		case picked.SizeGB > d.BudgetGB:
			fit = fmt.Sprintf(", too big for the ~%.0f GB budget", d.BudgetGB)
		default:
			fit = fmt.Sprintf(", fits the ~%.0f GB budget", d.BudgetGB)
		}
		fmt.Printf("\n  %s:%s is %.1f GB%s\n", d.ID, picked.Label, picked.SizeGB, fit)
	}
	return nil
}

// inspectModel measures a model against this machine, for the provider new
// profiles are served by. cfg may be nil, which just means there is no budget
// to measure against.
func inspectModel(ctx context.Context, m hf.Model, cfg *config.Config) resolve.Detail {
	var override string
	var target int
	prov, _ := engine.Lookup("")
	if cfg != nil {
		override = cfg.ModelBudgetGB()
		target = cfg.KVContextTarget()
		if p, err := engine.Lookup(cfg.Provider()); err == nil {
			prov = p
		}
	}
	// Only asked for when it is used: probing runs the vLLM binary.
	var caps vllm.Caps
	if prov.Capabilities().VLLMArgs {
		caps = vllmCaps(cfg)
	}
	return resolve.ForProvider(resolve.Inspect(ctx, m, detect.ModelBudget(override), caps, target), prov)
}

// firstClause keeps the head of a reason. Repeating a full sentence on every
// row buries the column that differs, and when a whole repo is blocked the
// reason has already been given once above the list.
func firstClause(reason string) string {
	for _, sep := range []string{", ", " -- "} {
		if head, _, ok := strings.Cut(reason, sep); ok {
			return head
		}
	}
	return reason
}

// display is the readable form of a variant's label. Never use it where the
// exact label is needed -- it is what vLLM is given.
func display(v hf.Variant) string {
	if v.Display != "" {
		return v.Display
	}
	return v.Label
}

// thousands renders a token count the way people say it: 262144 as "262k".
func thousands(n int64) string {
	if n < 1000 {
		return strconv.FormatInt(n, 10)
	}
	return fmt.Sprintf("%.0fk", float64(n)/1000)
}

func printModelDetail(d resolve.Detail, cfg *config.Config) {
	fmt.Println(d.ID)
	fmt.Println(d.URL)
	fmt.Printf("  license:    %s\n", d.License)
	fmt.Printf("  downloads:  %d   likes: %d   created: %.10s\n", d.Downloads, d.Likes, d.CreatedAt)
	if len(d.QuantTags) > 0 {
		fmt.Printf("  quant:      %s\n", strings.Join(d.QuantTags, ", "))
	} else {
		fmt.Println("  quant:      none detected in tags -- check the model card")
	}
	// Repos that ship several quants side by side get measured at their
	// smallest, so say so rather than letting the number look definitive.
	var of string
	if d.Variants > 1 {
		of = fmt.Sprintf(", smallest of %d quants in this repo", d.Variants)
	}
	switch {
	case d.SizeError != "":
		fmt.Printf("  size:       could not be measured: %s\n", d.SizeError)
	case d.SizeGB == 0:
		// Distinct from "we could not measure it": adapters and vision
		// projectors are weight files, they are just not a model you can
		// serve, and saying none were found would be the opposite of true.
		fmt.Println("  size:       unknown (no servable weights found -- an adapter or projector-only repo, or check the model card)")
	case d.FitsBudget == nil:
		fmt.Printf("  size:       %.1f GB  (no budget configured -- run 'llm-helper setup'%s)\n", d.SizeGB, of)
	case d.BudgetGB <= 0:
		// Measured, not missing: say why nothing fits rather than calling
		// this model in particular too big.
		fmt.Printf("  size:       %.1f GB  (this machine has no room for model weights: %s)\n", d.SizeGB, d.BudgetReason)
	case *d.FitsBudget && d.Best != nil:
		// Lead with the one worth running rather than the one that merely
		// proves something here fits, which is the smallest and so also the
		// worst quant in the repo.
		fmt.Printf("  size:       %.1f GB at %s  (biggest of %d quants that fits the ~%.0fGB budget; smallest is %.1f GB)\n",
			d.Best.SizeGB, display(*d.Best), d.Variants, d.BudgetGB, d.SizeGB)
	case *d.FitsBudget:
		fmt.Printf("  size:       %.1f GB  (fits under the ~%.0fGB budget%s)\n", d.SizeGB, d.BudgetGB, of)
	case d.Variants > 1:
		// Every quant was measured and the smallest still doesn't fit, so
		// there is no smaller one to go and look for.
		fmt.Printf("  size:       %.1f GB  (TOO BIG for this machine -- and it is the smallest of %d quants here, so try a different repo)\n", d.SizeGB, d.Variants)
	default:
		fmt.Printf("  size:       %.1f GB  (TOO BIG for this machine -- look for a smaller quant%s)\n", d.SizeGB, of)
	}
	// What "fits" actually buys, which is the question the size line is
	// usually standing in for.
	if d.ContextTokens > 0 {
		full := ""
		if d.MaxContext > 0 {
			full = fmt.Sprintf(", %.1fx its %s max", float64(d.ContextTokens)/float64(d.MaxContext), thousands(int64(d.MaxContext)))
		}
		fmt.Printf("  context:    ~%s tokens in %.1f GB of KV cache%s\n",
			thousands(d.ContextTokens), d.KVCacheGB, full)
	}
	if t := d.Thinking; t != nil {
		if t.Default {
			fmt.Println("  thinking:   on by default -- replies begin after a reasoning pass ('new --thinking off' skips it)")
		} else {
			fmt.Println("  thinking:   off by default ('new --thinking on' turns it on)")
		}
	}
	if d.HybridNote != "" {
		fmt.Printf("  hybrid:     %s\n", d.HybridNote)
	}
	if d.DetectedFlags != "" {
		fmt.Printf("  vllm flags: %s (auto-detected, will be pre-filled)\n", d.DetectedFlags)
	} else {
		fmt.Println("  vllm flags: no vllm serve snippet in README -- set flags by hand after pulling")
	}
	for _, n := range d.FlagNotes {
		fmt.Printf("  changed:    %s\n", n)
	}
	for _, w := range d.DroppedFlags {
		fmt.Printf("  dropped:    %s\n", w)
	}
	// A repo that is entirely GGUF on a machine without the plugin cannot be
	// served at all, whichever quant is picked, so lead with that rather than
	// leaving it to be inferred from every row saying the same thing.
	// The architecture is the harder stop of the two: no machine and no
	// plugin install changes it, so it is said first and instead.
	switch {
	case d.NeedsGGUF && d.ArchBlocked != "":
		fmt.Println()
		fmt.Println("  can't be served as GGUF: " + d.ArchBlocked)
	case d.NeedsGGUF && !d.GGUFReady:
		fmt.Println()
		fmt.Println("  " + ggufMissingMessage(cfg))
	}
	printVariants(d)
}

// printVariants lists the other copies of a model in a repo.
//
// Worth the space because the size above is the smallest, which is the safe
// answer to "could anything here run" and simultaneously the worst thing to
// actually serve: bartowski/Qwen_Qwen3-32B-GGUF measures 9.0 GB at IQ2_XXS
// while its Q8_0 is 34.8 GB and fits an 84 GB budget nearly twice over. A
// reader who only sees the smallest will write the repo off, or run a 2-bit
// quant without ever knowing there was a choice.
func printVariants(d resolve.Detail) {
	if len(d.All) < 2 {
		return
	}
	fmt.Println()
	fmt.Printf("  %d quants in this repo:\n", len(d.All))
	for _, v := range d.All {
		mark, note := "   ", ""
		switch {
		case !v.Servable():
			mark, note = "  -", "  "+firstClause(v.Blocked)
		case v.GGUF && !d.GGUFReady:
			mark, note = "  -", "  needs the GGUF plugin"
		case d.Best != nil && v.Name == d.Best.Name:
			mark = "  >"
		case d.BudgetKnown && v.SizeGB > d.BudgetGB:
			mark, note = "  x", "  too big for this machine"
		}
		fmt.Printf("%s %-26s %6.1f GB%s\n", mark, display(v), v.SizeGB, note)
	}
	// Only offer the flag when there is something it could pick. On a machine
	// that cannot load GGUF every row is struck through, and inviting a
	// choice there would be inviting a failure.
	if d.Best != nil {
		fmt.Printf("    > = the one 'new --quant %s' would serve\n", d.Best.Label)
		fmt.Printf("    Choose another with: llm-helper new %s --quant <name>\n", d.ID)
	}
}

// pickQuant asks which copy of a model to serve, when there is a choice worth
// offering. Returns nil to mean "the repo's default", and cancelled when the
// user wants to go back to the results rather than pull anything.
//
// The default is pre-selected and one keypress away, so the common case costs
// nothing -- but a repo holding a 2-bit and an 8-bit copy should not decide
// between them on the user's behalf and say nothing.
func pickQuant(cfg *config.Config, d resolve.Detail) (choice *resolve.Choice, cancelled bool) {
	servable := make([]hf.Variant, 0, len(d.All))
	for _, v := range d.All {
		if v.Servable() && (d.GGUFReady || !v.GGUF) {
			servable = append(servable, v)
		}
	}
	// One option, or none we could act on, is not a question.
	if len(servable) < 2 {
		if d.NeedsGGUF && !d.GGUFReady {
			fmt.Println(ggufMissingMessage(cfg))
			fmt.Println()
		}
		return nil, false
	}

	fmt.Println("Which quant?")
	for i, v := range servable {
		mark := " "
		if d.Best != nil && v.Name == d.Best.Name {
			mark = "*"
		}
		fmt.Printf(" %s %2d) %-26s %6.1f GB\n", mark, i+1, display(v), v.SizeGB)
	}
	// servable is ordered largest first, so the last row is the smallest.
	// That is the default when nothing was ranked -- an unknown budget, or
	// nothing that fits -- matching Weights.BestFit: with nothing to check a
	// size against, the conservative choice is the smallest. Defaulting to
	// row 1 there would put the largest quant one keypress away.
	def := len(servable)
	if d.Best != nil {
		for i, v := range servable {
			if v.Name == d.Best.Name {
				def = i + 1
			}
		}
		fmt.Printf("  * = biggest that fits this machine\n")
	}
	for {
		pick := ask(fmt.Sprintf("Pick one (1-%d, Enter for %d, 'b' to go back)", len(servable), def), "")
		switch {
		case pick == "":
			pick = strconv.Itoa(def)
		case pick == "b" || pick == "B":
			return nil, true
		}
		n, err := strconv.Atoi(pick)
		if err != nil || n < 1 || n > len(servable) {
			fmt.Println("Not a valid number, try again.")
			continue
		}
		v := servable[n-1]
		// Nothing to append for a repo whose single copy is the repo.
		if _, ok := v.Selector(); !ok {
			return nil, false
		}
		return &resolve.Choice{Variant: v, BaseModel: d.BaseModel}, false
	}
}
