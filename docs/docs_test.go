package docs_test

import (
	"io/fs"
	"path"
	"strings"
	"testing"

	"github.com/coder/llm-helper/docs"
)

// The manifest is the docs engine's contract with the panel: one entry per
// page, each pointing at a file that exists. These checks keep the schema,
// the files, and the links between them honest.

func TestManifestParses(t *testing.T) {
	m, err := docs.ReadManifest(docs.FS())
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Routes) == 0 {
		t.Fatal("the manifest routes no pages")
	}
	if len(m.Versions) == 0 {
		t.Error("the manifest names no versions")
	}
}

// Every path the manifest names must exist in the embedded tree. A typo'd
// route otherwise surfaces as a 500 from the panel, on a machine the author
// is not sitting at.
func TestEveryManifestPathExists(t *testing.T) {
	m, err := docs.ReadManifest(docs.FS())
	if err != nil {
		t.Fatal(err)
	}
	var walk func(rs []docs.Route)
	walk = func(rs []docs.Route) {
		for _, r := range rs {
			if _, err := fs.ReadFile(docs.FS(), strings.TrimPrefix(r.Path, "./")); err != nil {
				t.Errorf("manifest route %q: %v", r.Path, err)
			}
			walk(r.Children)
		}
	}
	walk(m.Routes)
}

// Every Markdown file in the tree must be reachable from the manifest, or it
// is a page nobody can navigate to. This is also what keeps the generated
// CLI reference wired in: a command added without a manifest entry fails
// here.
func TestEveryMarkdownPageIsReachable(t *testing.T) {
	m, err := docs.ReadManifest(docs.FS())
	if err != nil {
		t.Fatal(err)
	}
	routed := map[string]bool{}
	var walk func(rs []docs.Route)
	walk = func(rs []docs.Route) {
		for _, r := range rs {
			routed[strings.TrimPrefix(r.Path, "./")] = true
			walk(r.Children)
		}
	}
	walk(m.Routes)

	err = fs.WalkDir(docs.FS(), ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(p, ".md") {
			return nil
		}
		if !routed[p] {
			t.Errorf("docs/%s is not routed by the manifest", p)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// Links between pages are written the GitHub way — "./install.md" — and must
// resolve to a file in the tree. Broken ones would render as dead panel
// routes; the serving tests check the route mapping itself.
func TestRelativeLinksResolve(t *testing.T) {
	err := fs.WalkDir(docs.FS(), ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(p, ".md") {
			return nil
		}
		source, err := fs.ReadFile(docs.FS(), p)
		if err != nil {
			return err
		}
		for _, dest := range markdownLinks(source) {
			if dest == "" || strings.HasPrefix(dest, "#") || strings.HasPrefix(dest, "/") {
				continue
			}
			if i := strings.Index(dest, "://"); i > 0 {
				continue
			}
			target, _, _ := strings.Cut(dest, "#")
			if !strings.HasSuffix(target, ".md") {
				continue
			}
			resolved := path.Join(path.Dir(p), target)
			if _, err := fs.Stat(docs.FS(), resolved); err != nil {
				t.Errorf("docs/%s links to %q, which does not exist", p, dest)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestRoutesMapOntoThePanel(t *testing.T) {
	for _, tc := range []struct {
		page, want string
	}{
		{"./README.md", "/docs"},
		{"./index.md", "/docs"},
		{"./install.md", "/docs/install"},
		{"./reference/cli.md", "/docs/reference/cli"},
		{"./reference/cli/list.md", "/docs/reference/cli/list"},
	} {
		if got := docs.RouteFor(tc.page); got != tc.want {
			t.Errorf("RouteFor(%q) = %q, want %q", tc.page, got, tc.want)
		}
	}
}

// markdownLinks pulls link destinations out of Markdown source. It is not
// a parser — it exists so the link checks can run over raw files without
// rendering them, which keeps docs/ free of a dependency on the panel.
// Fenced code blocks are skipped, since commands shown in them are not links.
func markdownLinks(source []byte) []string {
	var out []string
	inFence := false
	for _, line := range strings.Split(string(source), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		rest := line
		for {
			open := strings.Index(rest, "](")
			if open < 0 {
				break
			}
			close := strings.Index(rest[open:], ")")
			if close < 0 {
				break
			}
			out = append(out, rest[open+2:open+close])
			rest = rest[open+close:]
		}
	}
	return out
}
