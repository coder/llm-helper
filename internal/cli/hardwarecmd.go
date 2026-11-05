package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/hardware"
)

func newHardwareCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "hardware",
		Short: "List hardware definitions and report this machine's hardware",
		Long: "Hardware definitions tell llm-helper how a GPU's memory is shared, which architecture it is,\n" +
			"and which flags abort at startup there. They ship in the binary, and files in\n" +
			"<config>/hardware.d can add more without a rebuild.\n" +
			"'hardware report' prints what this machine reports, to attach to a new definition.",
		GroupID: groupMachine,
	}
	cmd.AddCommand(newHardwareListCmd(), newHardwareReportCmd())
	return cmd
}

func newHardwareListCmd() *cobra.Command {
	var jsonOut bool
	c := &cobra.Command{
		Use:   "list",
		Short: "List the hardware definitions in effect",
		Long: "List the hardware definitions in effect: the shipped ones, then any added by\n" +
			"files in <config>/hardware.d, which are shown with the file they came from.",
		Aliases: []string{"ls"},
		Args:    cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return runHardwareList(os.Stdout, os.Stderr, jsonOut)
		},
	}
	c.Flags().BoolVar(&jsonOut, "json", false, "print the definitions as JSON")
	return c
}

// hardwareRow is one definition as 'hardware list --json' prints it.
type hardwareRow struct {
	hardware.Definition
	// Source is the hardware.d file a local definition came from, or
	// "shipped".
	Source string `json:"source"`
}

func runHardwareList(out, errOut io.Writer, jsonOut bool) error {
	for _, err := range hardware.LocalErrors() {
		fmt.Fprintf(errOut, "warning: skipped a local hardware definition file: %v\n", err)
	}
	var rows []hardwareRow
	for _, d := range hardware.Default().Definitions() {
		src := d.Source
		if src == "" {
			src = "shipped"
		}
		rows = append(rows, hardwareRow{Definition: d, Source: src})
	}
	if jsonOut {
		return printJSONTo(out, rows)
	}
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tMEMORY\tARCH\tNAME\tSOURCE")
	for _, r := range rows {
		arch := r.Arch
		if arch == "" {
			arch = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", r.ID, r.Memory, arch, r.DisplayName, r.Source)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintf(out, "\nLocal definitions go in %s, one file each in %s/, %s/ or %s/.\n",
		filepath.Join(config.ResolveDirs().Config, hardware.LocalDir), hardware.KindMachine, hardware.KindArchitecture, hardware.KindVendor)
	return nil
}

func newHardwareReportCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "report",
		Short: "Print this machine's hardware readings as JSON",
		Long: "Print this machine's raw hardware readings and what llm-helper concluded from them, as JSON.\n" +
			"The report names no host: no hostname, serial number, GPU UUID or bus address.\n" +
			"Attach it when proposing a hardware definition; it becomes the definition's report in\n" +
			"catalogue/hardware/reports.",
		Args: cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return printJSONTo(os.Stdout, detect.ProbeReport("llm-helper "+Version+" hardware report"))
		},
	}
}
