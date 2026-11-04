package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"slices"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/recipe"
)

func newRecipesCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "recipes",
		Short: "Tested launch configurations for known-good models",
		Long: "Tested launch configurations: each pins a model, image and flags. 'recipes launch'\n" +
			"starts one as containers on the nodes you name, over SSH for any that are not this machine.\n" +
			"Recipes ship in the binary, and files in <config>/recipes.d can add more without a rebuild.",
		GroupID: groupModels,
	}
	c.AddCommand(newRecipesListCmd(), newRecipesShowCmd(), newRecipesLaunchCmd(), newRecipesStopCmd(),
		newRecipesInstallServiceCmd(), newRecipesHealthCheckCmd())
	return c
}

func newRecipesListCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:   "list",
		Short: "List tested recipes",
		Args:  cobra.NoArgs,
		RunE:  func(*cobra.Command, []string) error { return runRecipesList(jsonOut) },
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print recipes as JSON")
	return c
}

func newRecipesShowCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:   "show <id>",
		Short: "Show everything a tested recipe pins, including how it was verified",
		Args:  cobra.ExactArgs(1),
		RunE:  func(_ *cobra.Command, a []string) error { return runRecipesShow(a[0], jsonOut) },
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print the recipe as JSON")
	return c
}

func runRecipesList(jsonOut bool) error {
	for _, err := range recipe.LocalErrors() {
		fmt.Fprintf(os.Stderr, "warning: skipped a local recipe file: %v\n", err)
	}
	t, err := recipe.LoadTested()
	if err != nil {
		return err
	}
	if jsonOut {
		return json.NewEncoder(os.Stdout).Encode(t.Recipes)
	}
	if len(t.Recipes) == 0 {
		fmt.Println("No tested recipes shipped in this build.")
		return nil
	}
	// Each recipe was measured on its own hardware, which is not necessarily
	// this machine, so the heading must not claim otherwise.
	fmt.Println("TESTED -- measured on the hardware shown, pinned by image digest")
	if err := writeRecipeRows(os.Stdout, t.Recipes, overrideNote); err != nil {
		return err
	}
	fmt.Println()
	fmt.Println("Run 'llm-helper recipes show <id>' for the pinned image, flags and evidence.")
	return nil
}

// writeRecipeRows sizes every column to its longest entry. The display name
// had a fixed width of 22, so "DeepSeek V4 Flash Vision" pushed its own row's
// layout column out of line with the rest.
func writeRecipeRows(out io.Writer, recipes []recipe.TestedRecipe, note func(recipe.TestedRecipe) string) error {
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	for _, r := range recipes {
		line := "  " + r.ID + "\t" + r.DisplayName + "\t" + r.Engine().ID + "\t" + verifiedOn(r) + "\t" + r.Summary()
		if r.Local != "" {
			line += "  (local)"
		}
		if note(r) != "" {
			line += "  (+ local overrides)"
		}
		fmt.Fprintln(tw, line)
	}
	return tw.Flush()
}

// verifiedOn is the hardware a recipe was measured on, as it records it
// ("dgx_spark_gb10 x2"), since that is the id a node's hardware is
// compared with.
func verifiedOn(r recipe.TestedRecipe) string {
	if r.Verified.Hardware == "" {
		return "hardware not recorded"
	}
	return r.Verified.Hardware
}

func runRecipesShow(id string, jsonOut bool) error {
	t, err := recipe.LoadTested()
	if err != nil {
		return err
	}
	r, ok := t.Get(id)
	if !ok {
		ids := make([]string, 0, len(t.Recipes))
		for _, x := range t.Recipes {
			ids = append(ids, x.ID)
		}
		return fmt.Errorf("no tested recipe %q (have: %s)", id, strings.Join(ids, ", "))
	}
	if jsonOut {
		return json.NewEncoder(os.Stdout).Encode(r)
	}

	line := func(k, v string) {
		if v != "" {
			fmt.Printf("  %-14s %s", k, v)
			fmt.Println()
		}
	}
	fmt.Println(r.DisplayName + "  (" + r.ID + ")")
	fmt.Println()
	line("model", r.Model)
	line("engine", r.Engine().ID+" ("+r.Engine().DisplayName+")")
	line("layout", r.Summary())
	line("image", r.Image)
	line("port", fmt.Sprintf("%d", r.Port))
	line("requires", r.Requires)
	line("derived from", r.Source.Describe())
	line("source format", r.Source.Format)
	line("defined in", r.Local)
	line("local", overrideNote(r))
	fmt.Println()
	if len(r.Models) > 0 {
		fmt.Println("Pinned models (each node must have them in its Hugging Face cache)")
		for _, m := range r.Models {
			fmt.Printf("  %-8s %s@%s\n", m.Role, m.ID, m.Revision)
		}
		fmt.Println()
	}
	if devs := r.Source.Deviations; len(devs) > 0 {
		fmt.Println("Deviations from upstream")
		for _, d := range devs {
			fmt.Printf("  %s: %s\n", d.Setting, d.Change())
			fmt.Println("    " + d.Why)
		}
		fmt.Println()
	}
	if overrideNote(r) != "" {
		fmt.Println("Verified before those overrides -- they do not describe this launch")
	} else {
		fmt.Println("Verified")
	}
	v := r.Verified
	line("date", v.Date)
	line("hardware", v.Hardware)
	line("driver", v.Driver)
	line("vllm", v.VLLMVersion)
	if v.KVTokens > 0 {
		line("kv cache", fmt.Sprintf("%d tokens", v.KVTokens))
	}
	line("smoke test", v.Smoke)
	if len(v.Notes) > 0 {
		fmt.Println()
		fmt.Println("Notes")
		for _, n := range v.Notes {
			fmt.Println("  - " + n)
		}
	}
	fmt.Println()
	fmt.Println("Flags")
	fmt.Println("  " + strings.Join(r.Args, " "))
	if len(r.Params) > 0 {
		fmt.Println()
		fmt.Println("Engine params")
		keys := slices.Sorted(maps.Keys(r.Params))
		for _, k := range keys {
			fmt.Printf("  %s=%s\n", k, r.Params[k])
		}
	}
	if len(r.Env) > 0 {
		fmt.Println()
		fmt.Printf("Environment (%d vars; IFACE_NAME, IB_IF and NODE_IP are filled in per node)", len(r.Env))
		fmt.Println()
	}
	return nil
}
