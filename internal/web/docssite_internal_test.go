package web

import (
	"bytes"
	"encoding/xml"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/docs"
)

const testDocsBase = "https://docs.example.test"

func getDocsSite(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	h, err := NewDocsSite(DocsSiteOptions{BaseURL: testDocsBase + "/"})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
	return rec
}

// The public site needs no token, and shows none of the panel's controls.
func TestDocsSiteServesEveryPageWithoutAuth(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	for _, page := range site.order {
		rec := getDocsSite(t, page.route)
		if rec.Code != http.StatusOK {
			t.Errorf("GET %s = %d, want 200", page.route, rec.Code)
			continue
		}
		body := rec.Body.String()
		for _, panelOnly := range []string{`href="/models"`, `href="/">Machine`, `class="bar-host"`} {
			if strings.Contains(body, panelOnly) {
				t.Errorf("GET %s renders panel chrome %q", page.route, panelOnly)
			}
		}
		canonical := `<link rel="canonical" href="` + testDocsBase + page.route + `" />`
		if !strings.Contains(body, canonical) {
			t.Errorf("GET %s has no canonical link %s", page.route, canonical)
		}
		alternate := `<link rel="alternate" type="text/markdown" href="/docs/` + page.file + `" />`
		if !strings.Contains(body, alternate) {
			t.Errorf("GET %s does not point at its Markdown with %s", page.route, alternate)
		}
	}
}

// The panel keeps its own chrome, and gains none of the public site's links.
func TestPanelDocsKeepThePanelChrome(t *testing.T) {
	body := getDocs(t, newTestServer(t), "/docs").Body.String()
	if !strings.Contains(body, `href="/models"`) {
		t.Error("the panel's docs lost the Add model tab")
	}
	if strings.Contains(body, `rel="canonical"`) || strings.Contains(body, "View this page as Markdown") {
		t.Error("the panel's docs carry the public site's links")
	}
}

func TestDocsSiteRoutes(t *testing.T) {
	cases := []struct {
		path, wantType string
		wantCode       int
	}{
		{"/docs", "text/html", http.StatusOK},
		{"/app.css", "text/css", http.StatusOK},
		{"/healthz", "text/plain", http.StatusOK},
		{"/llms.txt", "text/plain", http.StatusOK},
		{"/llms-full.txt", "text/plain", http.StatusOK},
		{"/sitemap.xml", "application/xml", http.StatusOK},
		{"/robots.txt", "text/plain", http.StatusOK},
		{"/docs/no-such-page", "", http.StatusNotFound},
		{"/docs/no-such-page.md", "", http.StatusNotFound},
		{"/api/status", "", http.StatusNotFound},
		{"/models", "", http.StatusNotFound},
		{"/index.html", "", http.StatusNotFound},
	}
	for _, c := range cases {
		rec := getDocsSite(t, c.path)
		if rec.Code != c.wantCode {
			t.Errorf("GET %s = %d, want %d", c.path, rec.Code, c.wantCode)
			continue
		}
		if c.wantType != "" && !strings.HasPrefix(rec.Header().Get("Content-Type"), c.wantType) {
			t.Errorf("GET %s Content-Type = %q, want %s", c.path, rec.Header().Get("Content-Type"), c.wantType)
		}
		if rec.Header().Get("X-Content-Type-Options") != "nosniff" {
			t.Errorf("GET %s has no nosniff header", c.path)
		}
	}
}

func TestDocsSiteRootRedirectsToDocs(t *testing.T) {
	rec := getDocsSite(t, "/")
	if rec.Code != http.StatusFound || rec.Header().Get("Location") != "/docs" {
		t.Errorf("GET / = %d to %q, want 302 to /docs", rec.Code, rec.Header().Get("Location"))
	}
}

// Each page's Markdown is served byte for byte under its path in docs/, so
// relative links inside it resolve the way they do on GitHub.
func TestDocsSiteServesMarkdownSources(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	for _, page := range site.order {
		want, err := fs.ReadFile(docs.FS(), page.file)
		if err != nil {
			t.Fatal(err)
		}
		rec := getDocsSite(t, "/docs/"+page.file)
		if rec.Code != http.StatusOK {
			t.Errorf("GET /docs/%s = %d, want 200", page.file, rec.Code)
			continue
		}
		if !strings.HasPrefix(rec.Header().Get("Content-Type"), "text/markdown") {
			t.Errorf("GET /docs/%s Content-Type = %q", page.file, rec.Header().Get("Content-Type"))
		}
		if !bytes.Equal(rec.Body.Bytes(), want) {
			t.Errorf("GET /docs/%s is not the file as written", page.file)
		}
	}
}

// llms.txt links every page's Markdown, and every link it gives is served.
func TestLLMsTxtListsEveryPage(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	body := getDocsSite(t, "/llms.txt").Body.String()
	if !strings.HasPrefix(body, "# llm-helper\n\n> ") {
		t.Errorf("llms.txt does not open with a title and summary:\n%s", body)
	}
	for _, page := range site.order {
		link := "](" + testDocsBase + "/docs/" + page.file + ")"
		if strings.Count(body, link) != 1 {
			t.Errorf("llms.txt lists %s %d times, want once", link, strings.Count(body, link))
		}
	}
	for _, section := range []string{"\n## Docs\n", "\n## CLI reference\n"} {
		if !strings.Contains(body, section) {
			t.Errorf("llms.txt has no %q section", strings.TrimSpace(section))
		}
	}
}

func TestLLMsFullTxtHasEveryPage(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	body := getDocsSite(t, "/llms-full.txt").Body.String()
	for _, page := range site.order {
		if !strings.Contains(body, "\n# "+page.title+"\n") {
			t.Errorf("llms-full.txt has no heading for %s", page.title)
		}
		if !strings.Contains(body, "Source: "+testDocsBase+page.route+"\n") {
			t.Errorf("llms-full.txt has no source line for %s", page.route)
		}
	}
	if strings.Contains(body, "\ntitle: ") {
		t.Error("llms-full.txt carries a page's frontmatter")
	}
}

func TestSitemapListsEveryPage(t *testing.T) {
	site, err := loadDocs()
	if err != nil {
		t.Fatal(err)
	}
	var sm struct {
		URLs []struct {
			Loc string `xml:"loc"`
		} `xml:"url"`
	}
	if err := xml.Unmarshal(getDocsSite(t, "/sitemap.xml").Body.Bytes(), &sm); err != nil {
		t.Fatal(err)
	}
	if len(sm.URLs) != len(site.order) {
		t.Fatalf("sitemap has %d URLs, want %d", len(sm.URLs), len(site.order))
	}
	for i, page := range site.order {
		if want := testDocsBase + page.route; sm.URLs[i].Loc != want {
			t.Errorf("sitemap URL %d = %q, want %q", i, sm.URLs[i].Loc, want)
		}
	}
	robots := getDocsSite(t, "/robots.txt").Body.String()
	if !strings.Contains(robots, "Sitemap: "+testDocsBase+"/sitemap.xml") {
		t.Errorf("robots.txt does not point at the sitemap:\n%s", robots)
	}
}

// Without a base URL, links come from the request, honoring a proxy's
// X-Forwarded-Proto.
func TestDocsSiteOriginFromRequest(t *testing.T) {
	h, err := NewDocsSite(DocsSiteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("GET", "/robots.txt", nil)
	req.Host = "vh.up.railway.app"
	req.Header.Set("X-Forwarded-Proto", "https")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if !strings.Contains(rec.Body.String(), "Sitemap: https://vh.up.railway.app/sitemap.xml") {
		t.Errorf("robots.txt did not use the request's origin:\n%s", rec.Body.String())
	}
}
