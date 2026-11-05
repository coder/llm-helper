package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/modeltype"
)

func newModelTypesCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model-types",
		Short: "List what specific model architectures need to run correctly",
		Long: "A model type is the model_type in a HuggingFace repo's config.json, such as gpt_oss.\n" +
			"Its definition says what every repo of that type needs on each engine: files to\n" +
			"fetch, flags to add, chat template fixes. 'new' applies it to the profile it writes.\n" +
			"They ship in the binary, and a file in <config>/model-types.d adds one or replaces\n" +
			"a shipped one with the same id, without a rebuild.",
		GroupID: groupModels,
	}
	cmd.AddCommand(newModelTypesListCmd(), newModelTypesShowCmd())
	return cmd
}

func newModelTypesListCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:   "list",
		Short: "List the model types in effect",
		Long: "List the model types in effect: the shipped ones, and any added or replaced by\n" +
			"files in <config>/model-types.d, which are shown with the file they came from.",
		Aliases: []string{"ls"},
		Args:    cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return runModelTypesList(os.Stdout, os.Stderr, jsonOut)
		},
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print the model types as JSON")
	return c
}

func newModelTypesShowCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:   "show <id>",
		Short: "Show a model type's definition",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, a []string) error {
			return runModelTypesShow(os.Stdout, a[0], jsonOut)
		},
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print the definition as JSON")
	return c
}

// modelTypeRow is one model type as 'model-types list --json' prints it.
type modelTypeRow struct {
	modeltype.Entry
	// Source is the model-types.d file a local entry came from, or "shipped".
	Source string `json:"source"`
}

func modelTypeSource(e modeltype.Entry) string {
	if e.Source == "" {
		return "shipped"
	}
	return e.Source
}

func engineIDs(e modeltype.Entry) []string {
	ids := make([]string, 0, len(e.Engines))
	for id := range e.Engines {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

func runModelTypesList(out, errOut io.Writer, jsonOut bool) error {
	for _, err := range modeltype.LocalErrors() {
		fmt.Fprintf(errOut, "warning: skipped a local model type file: %v\n", err)
	}
	var rows []modelTypeRow
	for _, e := range modeltype.Default().List() {
		rows = append(rows, modelTypeRow{Entry: e, Source: modelTypeSource(e)})
	}
	if jsonOut {
		return printJSONTo(out, rows)
	}
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tENGINES\tNAME\tSOURCE")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", r.ID, strings.Join(engineIDs(r.Entry), ","), r.DisplayName, r.Source)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintf(out, "\nLocal model types go in %s.\n", filepath.Join(config.ResolveDirs().Config, modeltype.LocalDir))
	return nil
}

func runModelTypesShow(out io.Writer, id string, jsonOut bool) error {
	e, ok := modeltype.Default().Lookup(id)
	if !ok {
		var have []string
		for _, e := range modeltype.Default().List() {
			have = append(have, e.ID)
		}
		return fmt.Errorf("no model type %q (have: %s)", id, strings.Join(have, ", "))
	}
	if jsonOut {
		return printJSONTo(out, modelTypeRow{Entry: e, Source: modelTypeSource(e)})
	}
	fmt.Fprintf(out, "%s  (%s)\n\n", e.DisplayName, e.ID)
	fmt.Fprintf(out, "  %-14s %s\n\n", "source", modelTypeSource(e))
	fmt.Fprintln(out, e.Why)
	for _, id := range engineIDs(e) {
		r := e.Engines[id]
		fmt.Fprintf(out, "\nOn %s\n", id)
		for _, f := range r.Files {
			envNote := ""
			if f.Env != "" {
				envNote = ", " + f.Env + " points at it"
			}
			fmt.Fprintf(out, "  fetch %s from %s into <data>/%s%s\n", strings.Join(f.Names, ", "), f.BaseURL, f.Dir, envNote)
		}
		for _, a := range r.Args {
			arg := a.Flag
			if len(a.Values) > 0 {
				arg += " <first of " + strings.Join(a.Values, ", ") + ">"
			}
			if len(a.With) > 0 {
				arg += " with " + strings.Join(a.With, " ")
			}
			fmt.Fprintf(out, "  add %s\n", arg)
		}
		for _, t := range r.ChatTemplateFixes {
			fmt.Fprintf(out, "  fix chat templates containing %q\n", t.Contains)
		}
	}
	return nil
}
