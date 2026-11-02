// CLI reference generation: docs/reference/cli/ is rendered from this same
// command tree, so the reference cannot drift from what --help would say.
// Regenerate with `go run ./scripts/gendocs` (or `go generate ./...`); a test
// fails when the committed pages no longer match.
//
//go:generate go run ../../scripts/gendocs -dir ../../docs
package cli

import (
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/coder/vllm-helper/docs"
)

// CLIReferencePath is the manifest route whose children GenerateCLIRoutes
// owns.
const CLIReferencePath = "./reference/cli.md"

// GenerateCLIRoutes returns the manifest children for the CLI reference, in
// the same tree and order as the pages GenerateCLIReference writes. They are
// generated rather than kept by hand so that adding a command cannot leave a
// page the sidebar does not list.
func GenerateCLIRoutes() []docs.Route {
	var walk func(c *cobra.Command, parents []string) []docs.Route
	walk = func(c *cobra.Command, parents []string) []docs.Route {
		var out []docs.Route
		for _, sub := range c.Commands() {
			if sub.Hidden {
				continue
			}
			here := append(append([]string{}, parents...), sub.Name())
			out = append(out, docs.Route{
				Title:       cmdName + " " + strings.Join(here, " "),
				Description: sub.Short,
				Path:        "./" + path.Join(append([]string{"reference", "cli"}, here...)...) + ".md",
				Children:    walk(sub, here),
			})
		}
		return out
	}
	return walk(newRootCmd(), nil)
}

// generatedBy names the command that owns the reference, so a stale page says
// how to fix itself.
const generatedBy = "go run ./scripts/gendocs"

// GenerateCLIReference renders the command-line reference as a map of file
// paths (relative to the docs root) to page contents. The index is
// reference/cli.md; each visible command gets reference/cli/<name>.md, nested
// the way its parents are. Hidden commands are excluded: `start` is the
// service manager's ExecStart, not something to type.
func GenerateCLIReference() (map[string]string, error) {
	root := newRootCmd()
	out := make(map[string]string)

	var walk func(c *cobra.Command, parents []string) error
	walk = func(c *cobra.Command, parents []string) error {
		for _, sub := range c.Commands() {
			if sub.Hidden {
				continue
			}
			here := append(append([]string{}, parents...), sub.Name())
			key := path.Join(append([]string{"reference", "cli"}, here...)...) + ".md"
			page, err := commandPage(sub, here)
			if err != nil {
				return err
			}
			out[key] = page
			if err := walk(sub, here); err != nil {
				return err
			}
		}
		return nil
	}

	// The index describes the tree itself: usage, the tool's own long
	// description, and one row per command.
	var index strings.Builder
	index.WriteString("## Usage\n\n```console\n" + root.UseLine() + "\n```\n\n")
	index.WriteString("## Description\n\n```console\n" + strings.TrimSpace(root.Long) + "\n```\n\n")
	index.WriteString("## Subcommands\n\n")

	var subs []*cobra.Command
	for _, sub := range root.Commands() {
		if !sub.Hidden {
			subs = append(subs, sub)
		}
	}
	rows := [][]string{{"Name", "Purpose"}}
	for _, sub := range subs {
		link := "[" + codeSpan(sub.Name()) + "](" +
			path.Join(append([]string{"cli"}, sub.Name())...) + ".md)"
		rows = append(rows, []string{link, sub.Short})
	}
	index.WriteString(table(rows) + "\n")

	fm, err := frontmatter("CLI reference", root.Short)
	if err != nil {
		return nil, err
	}
	out["reference/cli.md"] = fm + index.String()

	if err := walk(root, nil); err != nil {
		return nil, err
	}
	return out, nil
}

// commandPage renders one command's reference page.
func commandPage(c *cobra.Command, parents []string) (string, error) {
	name := cmdName + " " + strings.Join(parents, " ")
	fm, err := frontmatter(name, c.Short)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	// The long description is the page's lede when the command has one.
	// Otherwise the frontmatter description (the short) is the lede the panel
	// shows, and the body starts at usage — repeating the short here too
	// would read it twice on the same screen.
	if long := strings.TrimSpace(c.Long); long != "" {
		b.WriteString(long + "\n\n")
	}

	b.WriteString("## Usage\n\n```console\n" + c.UseLine() + "\n```\n")

	flags := c.LocalFlags()
	names := make([]string, 0, 8)
	flags.VisitAll(func(f *pflag.Flag) {
		if !f.Hidden {
			names = append(names, f.Name)
		}
	})
	sort.Strings(names)
	if len(names) > 0 {
		b.WriteString("\n## Options\n")
	}
	for _, n := range names {
		f := flags.Lookup(n)
		title := "--" + f.Name
		if f.Shorthand != "" {
			title = "-" + f.Shorthand + ", " + title
		}
		rows := [][]string{{"", ""}, {"Type", codeSpan(f.Value.Type())}}
		if d := defaultOf(f); d != "" {
			rows = append(rows, []string{"Default", codeSpan(d)})
		}
		b.WriteString("\n### " + title + "\n\n")
		b.WriteString(table(rows) + "\n")
		if usage := strings.TrimSpace(f.Usage); usage != "" {
			b.WriteString("\n" + usage + "\n")
		}
	}
	return fm + b.String(), nil
}

// frontmatter renders the YAML header every generated page carries.
func frontmatter(title, description string) (string, error) {
	esc := func(s string) (string, error) {
		s = strings.ReplaceAll(s, `\`, `\\`)
		s = strings.ReplaceAll(s, `"`, `\"`)
		if strings.ContainsAny(s, "\n\r") {
			return "", fmt.Errorf("cannot put %q on one frontmatter line", s)
		}
		return s, nil
	}
	t, err := esc(title)
	if err != nil {
		return "", err
	}
	d, err := esc(description)
	if err != nil {
		return "", err
	}
	return "---\n" +
		"# Code generated by " + generatedBy + ". DO NOT EDIT.\n" +
		"title: \"" + t + "\"\n" +
		"description: \"" + d + "\"\n" +
		"---\n\n" +
		"<!-- DO NOT EDIT | GENERATED CONTENT -->\n\n", nil
}

// codeSpan renders s as inline code that is safe inside a table cell. The
// panel renders Markdown with raw HTML off, so <code> tags would be dropped;
// a backtick in s needs a longer fence, and a pipe would end the cell.
func codeSpan(s string) string {
	s = strings.ReplaceAll(s, "|", `\|`)
	if strings.Contains(s, "`") {
		return "`` " + s + " ``"
	}
	return "`" + s + "`"
}

// defaultOf renders a flag's default for the options table, empty when it is
// the zero value for its type — documenting "--json, default false" is noise.
func defaultOf(f *pflag.Flag) string {
	switch f.Value.Type() {
	case "bool":
		if f.DefValue == "false" {
			return ""
		}
	case "string":
		if f.DefValue == "" {
			return ""
		}
	case "int", "int64", "uint", "count":
		if f.DefValue == "0" {
			return ""
		}
	case "duration":
		if f.DefValue == "0s" {
			return ""
		}
	}
	return f.DefValue
}

// table renders a Markdown table the way prettier formats them: every cell
// padded to its column's widest content, and the delimiter row filled with
// dashes to the same width. The first row of an options table has empty
// headers on purpose.
func table(rows [][]string) string {
	widths := make([]int, len(rows[0]))
	for _, r := range rows {
		for i, cell := range r {
			if n := len([]rune(cell)); n > widths[i] {
				widths[i] = n
			}
		}
	}
	var b strings.Builder
	for i, r := range rows {
		if i == 1 {
			b.WriteString("|")
			for _, w := range widths {
				b.WriteString(" " + strings.Repeat("-", max(w, 3)) + " |")
			}
			b.WriteString("\n")
		}
		b.WriteString("|")
		for i, cell := range r {
			b.WriteString(" " + cell + strings.Repeat(" ", widths[i]-len([]rune(cell))) + " |")
		}
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}
