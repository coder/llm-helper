package web

import (
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/coder/llm-helper/docs"
)

func getDocs(t *testing.T, s *Server, path string) *httptest.ResponseRecorder {
	t.Helper()
	h := s.Handler(Static())
	req := httptest.NewRequest("GET", path, nil)
	req.Header.Set(tokenHeader, s.token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// Every routed page must actually be served, as a document, behind the guard.
func TestDocsPagesAreServed(t *testing.T) {
	s := newTestServer(t)
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	if len(site.order) == 0 {
		t.Fatal("no docs pages were built")
	}
	for _, page := range site.order {
		rec := getDocs(t, s, page.route)
		if rec.Code != http.StatusOK {
			t.Errorf("GET %s = %d, want 200", page.route, rec.Code)
			continue
		}
		if !strings.Contains(rec.Body.String(), "<!doctype html>") {
			t.Errorf("GET %s did not return an HTML document", page.route)
		}
		// "/docs" and "/docs/" are the same page; so is a trailing slash.
		rec = getDocs(t, s, page.route+"/")
		if rec.Code != http.StatusOK {
			t.Errorf("GET %s/ = %d, want 200", page.route, rec.Code)
		}
	}
}

func TestDocsIndexIsServedAtTheRoot(t *testing.T) {
	s := newTestServer(t)
	for _, path := range []string{"/docs", "/docs/"} {
		rec := getDocs(t, s, path)
		if rec.Code != http.StatusOK {
			t.Errorf("GET %s = %d, want 200", path, rec.Code)
			continue
		}
		if !strings.Contains(rec.Body.String(), "Introduction") {
			t.Errorf("GET %s is not the introduction page", path)
		}
	}
}

func TestDocsUnknownPageIs404(t *testing.T) {
	s := newTestServer(t)
	rec := getDocs(t, s, "/docs/no-such-page")
	if rec.Code != http.StatusNotFound {
		t.Errorf("GET /docs/no-such-page = %d, want 404", rec.Code)
	}
}

// The docs describe how to control this machine, so they sit behind the same
// guard as everything else. A request without the token must not even reveal
// whether a page exists.
func TestDocsRequireTheToken(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(Static())
	for _, path := range []string{"/docs", "/docs/install", "/docs/reference/cli/list.md"} {
		req := httptest.NewRequest("GET", path, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("unauthenticated GET %s = %d, want 401", path, rec.Code)
		}
	}
}

// The manifest description is the page's lede: it is what says what the page
// is for before the body starts. The title comes with it.
func TestDocsShowTitleAndDescription(t *testing.T) {
	s := newTestServer(t)
	m, err := docs.ReadManifest(docs.FS())
	if err != nil {
		t.Fatal(err)
	}
	body := getDocs(t, s, "/docs/install").Body.String()
	if !strings.Contains(body, "<h1 class=\"docs-title\">Install and upgrade</h1>") {
		t.Error("the docs page does not show the manifest title")
	}
	if !strings.Contains(body, m.Routes[2].Description) {
		t.Errorf("the docs page does not show the manifest description %q", m.Routes[2].Description)
	}
}

// The sidebar is the manifest, nested, with the current page marked — so a
// reader can always see where they are and where else they could go.
func TestDocsSidebarNavigation(t *testing.T) {
	s := newTestServer(t)
	body := getDocs(t, s, "/docs/profiles").Body.String()
	if !strings.Contains(body, `href="/docs/quickstart"`) {
		t.Error("the docs sidebar does not link to /docs/quickstart")
	}
	if !strings.Contains(body, `href="/docs/reference/cli/list"`) {
		t.Error("the docs sidebar does not link into the CLI reference")
	}
	// The current page marks itself; another page must not be marked on it.
	if !strings.Contains(body, `href="/docs/profiles" aria-current="page"`) {
		t.Error("the current docs page is not marked in the sidebar")
	}
	if strings.Contains(body, `href="/docs/quickstart" aria-current="page"`) {
		t.Error("a page that is not current is marked in the sidebar")
	}
}

// The CLI reference is a long subtree, so it folds. It starts closed on a page
// outside it, and open on the CLI reference page or anything inside it, with
// the group's own link still clickable inside its summary.
func TestDocsSidebarCollapsesTheCLIReference(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	const closed = `<details class="docs-nav-group">`
	const open = `<details class="docs-nav-group" open>`

	profiles, ok := site.byRoute["/docs/profiles"]
	if !ok {
		t.Fatal("no /docs/profiles page")
	}
	if !strings.Contains(string(profiles.nav), closed) {
		t.Error("a group with children does not fold on a page outside it")
	}
	if strings.Contains(string(profiles.nav), open) {
		t.Error("the CLI reference is open on a page outside it")
	}
	if !strings.Contains(string(profiles.nav), `<summary><a href="/docs/reference/cli"`) {
		t.Error("the CLI reference link is not clickable inside its summary")
	}

	list, ok := site.byRoute["/docs/reference/cli/list"]
	if !ok {
		t.Fatal("no /docs/reference/cli/list page")
	}
	if n := strings.Count(string(list.nav), open); n != 1 {
		t.Errorf("/docs/reference/cli/list opens %d groups, want 1 (the CLI reference alone)", n)
	}

	add, ok := site.byRoute["/docs/reference/cli/node/add"]
	if !ok {
		t.Fatal("no /docs/reference/cli/node/add page")
	}
	if n := strings.Count(string(add.nav), open); n != 2 {
		t.Errorf("/docs/reference/cli/node/add opens %d groups, want 2 (the CLI reference and node)", n)
	}
}

// Links between pages are written the GitHub way and must arrive as panel
// routes; an anchor must land on an id the page actually contains.
func TestDocsLinksResolve(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	hrefRe := regexp.MustCompile(`href="([^"]+)"`)
	idRe := regexp.MustCompile(`id="([^"]+)"`)
	for _, page := range site.order {
		body := string(page.body)
		ids := map[string]bool{}
		for _, m := range idRe.FindAllStringSubmatch(body, -1) {
			ids[m[1]] = true
		}
		for _, m := range hrefRe.FindAllStringSubmatch(body, -1) {
			ref := m[1]
			switch {
			case strings.HasPrefix(ref, "http"), strings.HasPrefix(ref, "/"),
				strings.HasPrefix(ref, "#"):
			default:
				t.Errorf("docs page %s renders a non-route href %q", page.route, ref)
				continue
			}
			if strings.HasPrefix(ref, "/docs") {
				if _, ok := site.byRoute[strings.TrimSuffix(ref, "/")]; !ok {
					t.Errorf("docs page %s links to %q, which is not a route", page.route, ref)
				}
			}
			if strings.HasPrefix(ref, "#") && !ids[strings.TrimPrefix(ref, "#")] {
				t.Errorf("docs page %s links to anchor %q, which is not on the page", page.route, ref)
			}
		}
	}
}

// Prev and next are how a reader walks the docs end to end; the first page
// must have a next, the last a prev, and each pair must agree.
func TestDocsPrevNext(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	if len(site.order) < 2 {
		t.Fatal("fewer than two pages: prev/next cannot be exercised")
	}
	first, last := site.order[0], site.order[len(site.order)-1]
	if first.prev != nil {
		t.Errorf("%s has a previous page, but it is the first", first.route)
	}
	if last.next != nil {
		t.Errorf("%s has a next page, but it is the last", last.route)
	}
	for i, page := range site.order {
		if i > 0 && (page.prev == nil || page.prev.Route != site.order[i-1].route) {
			t.Errorf("%s prev does not point at %s", page.route, site.order[i-1].route)
		}
		if i+1 < len(site.order) && (page.next == nil || page.next.Route != site.order[i+1].route) {
			t.Errorf("%s next does not point at %s", page.route, site.order[i+1].route)
		}
	}
	body := getDocs(t, newTestServer(t), "/docs/install").Body.String()
	if !strings.Contains(body, "docs-prev") || !strings.Contains(body, "docs-next") {
		t.Error("a middle page does not offer prev and next links")
	}
}

// Callouts are written in GitHub's blockquote syntax and must arrive as the
// divs the stylesheet colors, with the marker line gone from the body.
func TestAlertsRender(t *testing.T) {
	src := []byte("" +
		"> [!NOTE]\n" +
		"> Something worth knowing.\n" +
		"\n" +
		"> [!WARNING] Read this first\n" +
		"> Because it bites.\n" +
		"\n" +
		"> A quote that is just a quote.\n")

	body, _, err := renderMarkdown(".", src)
	if err != nil {
		t.Fatal(err)
	}
	html := string(body)
	for _, tc := range []struct {
		class, title string
	}{
		{"markdown-alert-note", "Note"},
		{"markdown-alert-warning", "Warning"},
	} {
		if !strings.Contains(html, `<div class="markdown-alert `+tc.class+`">`) {
			t.Errorf("no %s callout rendered", tc.class)
		}
		if !strings.Contains(html, `<p class="markdown-alert-title">`+tc.title+"</p>") {
			t.Errorf("the %s callout has no %q title", tc.class, tc.title)
		}
	}
	if strings.Contains(html, "[!NOTE]") || strings.Contains(html, "[!WARNING]") {
		t.Error("the callout marker leaked into the rendered body")
	}
	if !strings.Contains(html, "Something worth knowing.") || !strings.Contains(html, "Because it bites.") {
		t.Error("callout content was lost")
	}
	if strings.Contains(html, "Read this first") == false || strings.Contains(html, "Because it bites.") == false {
		t.Error("an inline callout title was dropped")
	}
	if !strings.Contains(html, "<blockquote>") {
		t.Error("an ordinary blockquote stopped being one")
	}
}

// The "on this page" list is built from the same parse as the body, so its
// anchors must exist as ids in the rendered page.
func TestTableOfContentsAnchorsExist(t *testing.T) {
	s := newTestServer(t)
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	page := site.byRoute["/docs/profiles"]
	if page == nil {
		t.Fatal("profiles page missing")
	}
	if len(page.toc) == 0 {
		t.Fatal("profiles has no table of contents")
	}
	html := string(page.body)
	for _, e := range page.toc {
		if e.Level != 2 && e.Level != 3 {
			t.Errorf("toc entry %q is level %d, want 2 or 3", e.Text, e.Level)
		}
		if !strings.Contains(html, `id="`+e.ID+`"`) {
			t.Errorf("toc entry %q anchors at #%s, which the page does not contain", e.Text, e.ID)
		}
	}
	if !strings.Contains(string(getDocs(t, s, "/docs/profiles").Body.String()), "docs-toc") {
		t.Error("the served page carries no on-this-page list")
	}
}

// The frontmatter is for GitHub and coder/coder's engine; the panel takes the
// title and description from the manifest. If the YAML reaches Markdown, its
// closing "---" turns it into a heading at the top of every page and the TOC.
// Raw HTML is off, so markup the generator emits as tags would be dropped.
func TestDocsBodiesCarryNoFrontmatterOrDroppedHTML(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	yamlHeading := regexp.MustCompile(`<h[1-6][^>]*>(title|description):`)
	for _, page := range site.order {
		html := string(page.body)
		if yamlHeading.MatchString(html) {
			t.Errorf("%s renders its frontmatter as a heading", page.route)
		}
		for _, e := range page.toc {
			if strings.HasPrefix(e.Text, "title:") {
				t.Errorf("%s lists its frontmatter in the table of contents", page.route)
			}
		}
		if strings.Contains(html, "<td><!-- raw HTML omitted -->") {
			t.Errorf("%s has table cells whose HTML was dropped", page.route)
		}
	}
}

func TestStripFrontmatter(t *testing.T) {
	for _, tc := range []struct {
		in, want string
		wantErr  bool
	}{
		{in: "---\ntitle: A\n---\n\nbody\n", want: "\nbody\n"},
		{in: "---\r\ntitle: A\r\n---\r\nbody", want: "body"},
		{in: "# no frontmatter\n", want: "# no frontmatter\n"},
		{in: "---\ntitle: A\n", wantErr: true},
	} {
		got, err := stripFrontmatter([]byte(tc.in))
		if tc.wantErr {
			if err == nil {
				t.Errorf("stripFrontmatter(%q) succeeded, want an error", tc.in)
			}
			continue
		}
		if err != nil || string(got) != tc.want {
			t.Errorf("stripFrontmatter(%q) = %q, %v; want %q", tc.in, got, err, tc.want)
		}
	}
}

// The header nav on the other pages has to reach the docs, or they exist
// where nobody can get to them.
func TestDocsAreReachableFromEveryPage(t *testing.T) {
	for _, file := range pages {
		if !strings.Contains(readAsset(t, "static/"+file), `href="/docs"`) {
			t.Errorf("%s has no Docs link in its header nav", file)
		}
	}
	body := string(getDocs(t, newTestServer(t), "/docs/install").Body.String())
	for _, path := range []string{"/", "/models"} {
		if !strings.Contains(body, `href="`+path+`"`) {
			t.Errorf("the docs page does not link back to %s", path)
		}
	}
}
