// The panel's documentation: pages rendered from the embedded docs package
// under /docs, behind the same guard as everything else. The docs describe
// how to control this machine, so they are served by the same token; and they
// are rendered server-side because they are read, not operated.
//
// The same pages are also served without the panel by NewDocsSite, for the
// public docs website (docssite.go).
package web

import (
	"bytes"
	_ "embed"
	"fmt"
	"html/template"
	"io/fs"
	"net/http"
	"path"
	"regexp"
	"strings"
	"sync"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/extension"
	gparser "github.com/yuin/goldmark/parser"
	"github.com/yuin/goldmark/renderer"
	"github.com/yuin/goldmark/text"
	"github.com/yuin/goldmark/util"

	"github.com/coder/vllm-helper/docs"
)

//go:embed docs.tmpl
var docsTemplateSrc string

var docsTmpl = template.Must(template.New("docs").Parse(docsTemplateSrc))

// docsMarkdown turns a page's source into body HTML. GFM covers the tables
// and task lists the pages use; the two extenders below do the two things the
// panel needs that stock Markdown does not: callouts, and links between
// pages resolving to panel routes. Unsafe HTML stays off — the docs are
// content written in this repository, and nothing in them needs raw HTML.
var docsMarkdown = goldmark.New(
	goldmark.WithExtensions(
		extension.GFM,
		&docsAlerts{},
		&docsLinks{},
	),
	goldmark.WithParserOptions(gparser.WithAutoHeadingID()),
)

// tocEntry is one h2/h3 in a page's "on this page" list.
type tocEntry struct {
	Level int
	ID    string
	Text  string
}

// navEntry is one link in the sidebar tree.
type navEntry struct {
	Route    string
	Title    string
	Children []*navEntry
}

// sidebarEntry is navEntry as the template reads it, with the current page
// resolved so it can mark itself. Open says whether the current page is this
// entry or one of its descendants, which is what decides whether a folding
// group starts expanded.
type sidebarEntry struct {
	Route    string
	Title    string
	Current  bool
	Open     bool
	Children []sidebarEntry
}

// docsPage is one rendered page of the docs.
type docsPage struct {
	route       string
	file        string // path under docs/, e.g. "reference/cli/adopt.md"
	source      []byte // the file as written, frontmatter included
	markdown    []byte // the file without its frontmatter
	title       string
	description string
	body        template.HTML
	toc         []tocEntry
	nav         template.HTML // sidebar HTML, this page marked current
	prev, next  *navEntry
}

// docsSite is every page, the order prev/next follow, and the sidebar tree.
type docsSite struct {
	byRoute map[string]*docsPage
	byFile  map[string]*docsPage
	order   []*docsPage
	root    []*navEntry
	routes  []docs.Route
}

var (
	docsOnce  sync.Once
	docsReady *docsSite
	docsErr   error
)

// loadDocs renders every page once per process. The content is compiled in
// and immutable, so a request never re-parses what another request just did.
func loadDocs() (*docsSite, error) {
	docsOnce.Do(func() {
		docsReady, docsErr = buildDocs()
	})
	return docsReady, docsErr
}

func buildDocs() (*docsSite, error) {
	m, err := docs.ReadManifest(docs.FS())
	if err != nil {
		return nil, err
	}
	b := &docsBuilder{site: &docsSite{
		byRoute: make(map[string]*docsPage),
		byFile:  make(map[string]*docsPage),
		routes:  m.Routes,
	}}
	root, err := b.walk(m.Routes)
	if err != nil {
		return nil, err
	}
	b.site.root = root

	// Prev/next follow the manifest's depth-first order, which is the order
	// the sidebar reads in: the two stay consistent by construction.
	for i, p := range b.site.order {
		if i > 0 {
			p.prev = &navEntry{Route: b.site.order[i-1].route, Title: b.site.order[i-1].title}
		}
		if i+1 < len(b.site.order) {
			p.next = &navEntry{Route: b.site.order[i+1].route, Title: b.site.order[i+1].title}
		}
	}
	// The sidebar differs between pages only in which entry is current, but
	// that is per-page state, so each page keeps its own rendered copy.
	for _, p := range b.site.order {
		var buf bytes.Buffer
		err := docsTmpl.ExecuteTemplate(&buf, "docsSidebar", toSidebar(root, p.route))
		if err != nil {
			return nil, fmt.Errorf("rendering the docs sidebar: %w", err)
		}
		p.nav = template.HTML(buf.String())
	}
	return b.site, nil
}

type docsBuilder struct {
	site *docsSite
}

// walk renders every route depth-first, returning the sidebar tree.
func (b *docsBuilder) walk(routes []docs.Route) ([]*navEntry, error) {
	var out []*navEntry
	for i := range routes {
		r := &routes[i]
		page, err := b.page(r)
		if err != nil {
			return nil, err
		}
		children, err := b.walk(r.Children)
		if err != nil {
			return nil, err
		}
		out = append(out, &navEntry{
			Route:    page.route,
			Title:    r.Title,
			Children: children,
		})
	}
	return out, nil
}

func (b *docsBuilder) page(r *docs.Route) (*docsPage, error) {
	// io/fs paths have no "./" prefix; the manifest writes it to match
	// coder/coder's schema.
	rel := strings.TrimPrefix(r.Path, "./")
	source, err := fs.ReadFile(docs.FS(), rel)
	if err != nil {
		return nil, fmt.Errorf("docs page %s: %w", r.Path, err)
	}
	markdown, err := stripFrontmatter(source)
	if err != nil {
		return nil, fmt.Errorf("docs page %s: %w", r.Path, err)
	}
	body, toc, err := renderMarkdown(path.Dir(rel), markdown)
	if err != nil {
		return nil, fmt.Errorf("docs page %s: %w", r.Path, err)
	}
	page := &docsPage{
		route:       docs.RouteFor(r.Path),
		file:        rel,
		source:      source,
		markdown:    markdown,
		title:       r.Title,
		description: r.Description,
		body:        body,
		toc:         toc,
	}
	b.site.byRoute[page.route] = page
	b.site.byFile[page.file] = page
	b.site.order = append(b.site.order, page)
	return page, nil
}

func toSidebar(entries []*navEntry, current string) []sidebarEntry {
	out := make([]sidebarEntry, 0, len(entries))
	for _, e := range entries {
		children := toSidebar(e.Children, current)
		out = append(out, sidebarEntry{
			Route:    e.Route,
			Title:    e.Title,
			Current:  e.Route == current,
			Open:     e.Route == current || subtreeContainsCurrent(children),
			Children: children,
		})
	}
	return out
}

// subtreeContainsCurrent says whether any entry in the tree is the current
// page, or holds it below itself. A group is left open exactly when it does.
func subtreeContainsCurrent(entries []sidebarEntry) bool {
	for _, e := range entries {
		if e.Current || e.Open {
			return true
		}
	}
	return false
}

// handleDocs serves one documentation page, or says there is no such page.
func (s *Server) handleDocs(w http.ResponseWriter, r *http.Request) {
	site, err := loadDocs()
	if err != nil {
		http.Error(w, "documentation failed to load: "+err.Error(), http.StatusInternalServerError)
		return
	}
	page, ok := site.lookup(r.URL.Path)
	if !ok {
		http.Error(w, "no such documentation page: "+r.URL.Path, http.StatusNotFound)
		return
	}
	writeDocsPage(w, page, docsData{Build: s.build, Host: hostname()})
}

// lookup finds the page a request path names. "/docs" and "/docs/" are the
// introduction; a trailing slash anywhere else is tolerated the way a file
// server would.
func (site *docsSite) lookup(urlPath string) (*docsPage, bool) {
	page, ok := site.byRoute[strings.TrimSuffix(urlPath, "/")]
	return page, ok
}

// writeDocsPage renders one page. data carries the chrome around it -- the
// panel's build and host, or the public site's links -- and the page fields
// are filled in here.
func writeDocsPage(w http.ResponseWriter, page *docsPage, data docsData) {
	data.Title = page.title
	data.Description = page.description
	data.Body = page.body
	data.Nav = page.nav
	data.TOC = page.toc
	data.Prev = page.prev
	data.Next = page.next
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	// Past this point the response has started; a template error can only
	// truncate a page that tests have already proven well-formed.
	_ = docsTmpl.ExecuteTemplate(w, "docs", data)
}

type docsData struct {
	// Public renders the docs website's chrome instead of the panel's.
	Public bool
	// Canonical and Markdown are the page's absolute URL and the URL of its
	// Markdown source; the public site sets them, the panel does not.
	Canonical string
	Markdown  string

	Build       string
	Host        string
	Title       string
	Description string
	Body        template.HTML
	Nav         template.HTML
	TOC         []tocEntry
	Prev        *navEntry
	Next        *navEntry
}

// stripFrontmatter drops a page's leading YAML block. The manifest is where the
// panel reads a page's title and description; the frontmatter repeats them for
// GitHub and for coder/coder's doc engine. Left in, Markdown reads the closing
// "---" as a setext underline and renders the YAML as a heading.
func stripFrontmatter(source []byte) ([]byte, error) {
	rest, ok := bytes.CutPrefix(source, []byte("---\n"))
	if !ok {
		rest, ok = bytes.CutPrefix(source, []byte("---\r\n"))
	}
	if !ok {
		return source, nil
	}
	for len(rest) > 0 {
		line, after, _ := bytes.Cut(rest, []byte("\n"))
		if string(bytes.TrimRight(line, "\r")) == "---" {
			return after, nil
		}
		rest = after
	}
	return nil, fmt.Errorf("frontmatter is never closed with ---")
}

// renderMarkdown converts one page's source into body HTML plus its table of
// contents. Two passes over the same source keep the TOC's anchors and the
// body's ids in lockstep: both come from the same parse, so the same
// auto-generated id lands in each.
func renderMarkdown(dir string, source []byte) (template.HTML, []tocEntry, error) {
	ctx := gparser.NewContext()
	ctx.Set(pageDirKey, dir)

	doc := docsMarkdown.Parser().Parse(text.NewReader(source), gparser.WithContext(ctx))
	var body bytes.Buffer
	if err := docsMarkdown.Renderer().Render(&body, source, doc); err != nil {
		return "", nil, err
	}
	return template.HTML(body.String()), tableOfContents(doc, source), nil
}

// tableOfContents lists the h2 and h3 headings in document order, with the
// same ids the renderer wrote.
func tableOfContents(doc ast.Node, source []byte) []tocEntry {
	var toc []tocEntry
	ast.Walk(doc, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		h, ok := n.(*ast.Heading)
		if !ok || (h.Level != 2 && h.Level != 3) {
			return ast.WalkContinue, nil
		}
		id, _ := h.AttributeString("id")
		idb, _ := id.([]byte)
		toc = append(toc, tocEntry{
			Level: h.Level,
			ID:    string(idb),
			Text:  inlineText(h, source),
		})
		return ast.WalkSkipChildren, nil
	})
	return toc
}

// inlineText renders a heading the way the body will read it: the text of
// its inline nodes, without the markup.
func inlineText(n ast.Node, source []byte) string {
	var sb strings.Builder
	var collect func(ast.Node)
	collect = func(n ast.Node) {
		for c := n.FirstChild(); c != nil; c = c.NextSibling() {
			if t, ok := c.(*ast.Text); ok {
				sb.Write(t.Segment.Value(source))
			}
			collect(c)
		}
	}
	collect(n)
	return strings.TrimSpace(sb.String())
}

// pageDirKey carries the directory of the page being rendered into the link
// rewriter, so relative links resolve the way they would on disk.
var pageDirKey = gparser.NewContextKey()

// docsLinks rewrites relative ".md" links into panel routes. Links between
// pages are written the GitHub way — "./install.md", "../profiles.md" — so
// they work both places, and only the destination changes at render time.
type docsLinks struct{}

func (e *docsLinks) Extend(m goldmark.Markdown) {
	m.Parser().AddOptions(gparser.WithASTTransformers(util.Prioritized(&linkTransformer{}, 200)))
}

type linkTransformer struct{}

func (t *linkTransformer) Transform(node *ast.Document, reader text.Reader, pc gparser.Context) {
	dir, _ := pc.Get(pageDirKey).(string)
	if dir == "" {
		dir = "."
	}
	ast.Walk(node, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		switch l := n.(type) {
		case *ast.Link:
			l.Destination = rewriteDest(dir, l.Destination)
		case *ast.Image:
			l.Destination = rewriteDest(dir, l.Destination)
		}
		return ast.WalkContinue, nil
	})
}

// rewriteDest maps a relative .md destination onto its panel route, keeping
// any fragment. Anchors, absolute paths, and anything with a scheme — the
// links that already point somewhere real — pass through untouched.
func rewriteDest(dir string, dest []byte) []byte {
	d := string(dest)
	if d == "" || d[0] == '#' || d[0] == '/' {
		return dest
	}
	if i := strings.Index(d, "://"); i > 0 {
		return dest
	}
	frag := ""
	if i := strings.Index(d, "#"); i >= 0 {
		frag, d = d[i:], d[:i]
	}
	if !strings.HasSuffix(d, ".md") {
		return dest
	}
	return []byte(docs.RouteFor(path.Join(dir, d)) + frag)
}

// docsAlerts renders GitHub-style callouts: a blockquote whose first line is
// [!NOTE] and friends becomes a div the stylesheet can color. goldmark ships
// no such extension, so the blockquote is rewritten after the document is
// parsed and before it is rendered.
type docsAlerts struct{}

func (e *docsAlerts) Extend(m goldmark.Markdown) {
	m.Parser().AddOptions(gparser.WithASTTransformers(util.Prioritized(&alertTransformer{}, 100)))
	m.Renderer().AddOptions(renderer.WithNodeRenderers(util.Prioritized(&alertHTMLRenderer{}, 100)))
}

var (
	alertPattern = regexp.MustCompile(`^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*`)
	kindAlert    = ast.NewNodeKind("AlertBlock")

	alertTitles = map[string]string{
		"note":      "Note",
		"tip":       "Tip",
		"important": "Important",
		"warning":   "Warning",
		"caution":   "Caution",
	}
)

// alertBlock replaces a callout's blockquote and holds its remaining content.
type alertBlock struct {
	ast.BaseBlock
	kind string
}

func (a *alertBlock) Kind() ast.NodeKind { return kindAlert }
func (a *alertBlock) Dump(source []byte, level int) {
	ast.DumpHelper(a, source, level, map[string]string{"Kind": a.kind}, nil)
}

type alertTransformer struct{}

func (t *alertTransformer) Transform(node *ast.Document, reader text.Reader, _ gparser.Context) {
	// Collected first: a blockquote can hold another, and rewriting while
	// walking would walk into a tree that is being rearranged.
	var quotes []*ast.Blockquote
	ast.Walk(node, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if entering {
			if q, ok := n.(*ast.Blockquote); ok {
				quotes = append(quotes, q)
			}
		}
		return ast.WalkContinue, nil
	})
	for _, q := range quotes {
		applyAlert(q, reader.Source())
	}
}

// applyAlert rewrites a blockquote whose first line opens with a callout
// marker. The marker is stripped from the paragraph it opened; the rest of
// the quote becomes the callout's content. Anything else — a quote that
// quotes someone, say — is left a blockquote.
//
// The marker is matched across the paragraph's leading text nodes rather
// than in the first one: goldmark splits inline text at bracket boundaries,
// so `[!NOTE]` arrives as several nodes.
func applyAlert(q *ast.Blockquote, source []byte) {
	para, ok := q.FirstChild().(*ast.Paragraph)
	if !ok {
		return
	}
	// Gather the paragraph's first line, however many nodes it is split
	// across. A non-text node before any text means the line does not start
	// with a marker.
	var texts []*ast.Text
	var line []byte
	for c := para.FirstChild(); c != nil; c = c.NextSibling() {
		t, ok := c.(*ast.Text)
		if !ok {
			break
		}
		texts = append(texts, t)
		line = append(line, t.Segment.Value(source)...)
		if t.SoftLineBreak() {
			break
		}
	}
	m := alertPattern.FindSubmatch(line)
	if m == nil {
		return
	}

	// Consume the marker out of the leading nodes, removing the ones it
	// covers entirely.
	remaining := len(m[0])
	for _, t := range texts {
		if remaining <= 0 {
			break
		}
		if n := t.Segment.Len(); n <= remaining {
			para.RemoveChild(para, t)
			remaining -= n
			continue
		}
		t.Segment = t.Segment.WithStart(t.Segment.Start + remaining)
		remaining = 0
	}
	if para.ChildCount() == 0 {
		q.RemoveChild(q, para)
	}

	alert := &alertBlock{kind: strings.ToLower(string(m[1]))}
	for c := q.FirstChild(); c != nil; {
		next := c.NextSibling()
		alert.AppendChild(alert, c) // moves c out of the blockquote
		c = next
	}
	parent := q.Parent()
	parent.InsertBefore(parent, q, alert)
	parent.RemoveChild(parent, q)
}

type alertHTMLRenderer struct{}

func (r *alertHTMLRenderer) RegisterFuncs(reg renderer.NodeRendererFuncRegisterer) {
	reg.Register(kindAlert, r.render)
}

func (r *alertHTMLRenderer) render(w util.BufWriter, _ []byte, node ast.Node, entering bool) (ast.WalkStatus, error) {
	a := node.(*alertBlock)
	if entering {
		_, _ = w.WriteString(`<div class="markdown-alert markdown-alert-` + a.kind + `">` + "\n")
		_, _ = w.WriteString(`<p class="markdown-alert-title">` + alertTitles[a.kind] + "</p>\n")
		return ast.WalkContinue, nil
	}
	_, _ = w.WriteString("</div>\n")
	return ast.WalkContinue, nil
}
