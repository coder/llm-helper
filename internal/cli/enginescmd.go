package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/engine"
)

func newEnginesCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "engines",
		Short: "List the inference engines a profile or recipe can run on",
		Long: "Engines are the inference servers llm-helper can launch, such as vLLM or llama.cpp.\n" +
			"Each is a definition: how it is installed, its command line, how it takes the model\n" +
			"and the API key, and how its health is read. They ship in the binary, and files in\n" +
			"<config>/engines.d can add more without a rebuild. A profile picks one with PROVIDER,\n" +
			"and a recipe with its provider field.",
		GroupID: groupModels,
	}
	cmd.AddCommand(newEnginesListCmd(), newEnginesShowCmd())
	return cmd
}

func newEnginesListCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:   "list",
		Short: "List the engines in effect",
		Long: "List the engines in effect: the shipped ones, then any added by files in\n" +
			"<config>/engines.d, which are shown with the file they came from.",
		Aliases: []string{"ls"},
		Args:    cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return runEnginesList(os.Stdout, os.Stderr, jsonOut)
		},
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print the engines as JSON")
	return c
}

func newEnginesShowCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:   "show <id>",
		Short: "Show an engine's definition",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, a []string) error {
			return runEnginesShow(os.Stdout, a[0], jsonOut)
		},
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print the definition as JSON")
	return c
}

// engineRow is one engine as 'engines list --json' prints it.
type engineRow struct {
	engine.Engine
	// Source is the engines.d file a local engine came from, or "shipped".
	Source string `json:"source"`
}

func engineSource(e engine.Engine) string {
	if e.Source == "" {
		return "shipped"
	}
	return e.Source
}

func runEnginesList(out, errOut io.Writer, jsonOut bool) error {
	for _, err := range engine.LocalErrors() {
		fmt.Fprintf(errOut, "warning: skipped a local engine definition file: %v\n", err)
	}
	var rows []engineRow
	for _, e := range engine.Default().Engines() {
		rows = append(rows, engineRow{Engine: e, Source: engineSource(e)})
	}
	if jsonOut {
		return printJSONTo(out, rows)
	}
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tRUNTIME\tGPUS\tNAME\tSOURCE")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", r.ID, strings.Join(r.Runtime, ","), orDash(strings.Join(r.GPUVendors, ",")), r.DisplayName, r.Source)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintf(out, "\nLocal engines go in %s.\n", filepath.Join(config.ResolveDirs().Config, engine.LocalDir))
	return nil
}

func runEnginesShow(out io.Writer, id string, jsonOut bool) error {
	e, ok := engine.Default().Get(id)
	if !ok {
		return fmt.Errorf("no engine %q (have: %s)", id, strings.Join(engine.Names(), ", "))
	}
	if jsonOut {
		return printJSONTo(out, engineRow{Engine: e, Source: engineSource(e)})
	}
	line := func(k, v string) {
		if v != "" {
			fmt.Fprintf(out, "  %-14s %s\n", k, v)
		}
	}
	fmt.Fprintf(out, "%s  (%s)\n\n", e.DisplayName, e.ID)
	line("runtime", strings.Join(e.Runtime, ", "))
	line("gpus", strings.Join(e.GPUVendors, ", "))
	line("api key", e.APIKey)
	if e.Container != nil {
		line("container key", e.Container.APIKey)
	}
	line("health", e.Health.Path+" ("+e.Health.Mode+")")
	line("source", engineSource(e))
	fmt.Fprintln(out)
	fmt.Fprintln(out, e.Why)
	if len(e.Install) > 0 {
		fmt.Fprintln(out)
		fmt.Fprintln(out, "Install")
		for _, goos := range []string{"linux", "darwin", "any"} {
			if hint, ok := e.Install[goos]; ok {
				fmt.Fprintf(out, "  %-6s %s\n", goos, hint)
			}
		}
	}
	if len(e.Params) > 0 {
		fmt.Fprintln(out)
		fmt.Fprintln(out, "Params")
		for _, p := range e.Params {
			fmt.Fprintf(out, "  %s (default %q)\n    %s\n", p.Name, p.Default, p.Why)
		}
	}
	if len(e.Serve) > 0 {
		fmt.Fprintln(out)
		fmt.Fprintln(out, "Command")
		fmt.Fprintln(out, "  "+strings.Join(e.Serve, " "))
	}
	if c := e.Container; c != nil {
		for _, r := range c.Roles {
			fmt.Fprintln(out)
			fmt.Fprintf(out, "Container role %s (ranks %s)\n", r.Name, r.Ranks)
			fmt.Fprintln(out, "  "+strings.TrimSpace(c.Entrypoint+" "+strings.Join(r.Args, " ")))
		}
	}
	return nil
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
