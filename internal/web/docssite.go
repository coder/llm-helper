// The public docs website: the pages the panel serves under /docs, without
// the panel. Nothing here touches a machine, so nothing here is guarded.
//
// Besides the HTML pages it serves what lets a crawler or an LLM read the
// docs without scraping them: every page's Markdown source beside its route
// (under the same paths the files have in docs/, so their relative links keep
// working), /llms.txt and /llms-full.txt in the llmstxt.org format, and a
// sitemap.
package web

import (
	"encoding/xml"
	"fmt"
	"net/http"
	"strings"

	"github.com/coder/vllm-helper/docs"
)

// docsSiteSummary is the one-line description llms.txt leads with. It is the
// README's tagline.
const docsSiteSummary = "Serve large models with vLLM, mlx-lm or llama.cpp on one GPU machine or several, and switch models with one command."

// DocsSiteOptions configures NewDocsSite.
type DocsSiteOptions struct {
	// BaseURL is the site's public origin, such as https://docs.example.com.
	// Canonical links, the sitemap and llms.txt are built from it. Empty
	// derives the origin from each request's Host.
	BaseURL string
}

// NewDocsSite returns the handler for the public docs website. It renders
// every page up front, so a broken page fails here rather than on a request.
func NewDocsSite(opts DocsSiteOptions) (http.Handler, error) {
	site, err := loadDocs()
	if err != nil {
		return nil, err
	}
	d := &docsSiteHandler{site: site, base: strings.TrimRight(opts.BaseURL, "/")}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/docs", http.StatusFound)
	})
	mux.HandleFunc("GET /docs", d.handlePage)
	mux.HandleFunc("GET /docs/", d.handlePage)
	mux.HandleFunc("GET /app.css", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFileFS(w, r, Static(), "app.css")
	})
	mux.HandleFunc("GET /llms.txt", d.handleLLMs)
	mux.HandleFunc("GET /llms-full.txt", d.handleLLMsFull)
	mux.HandleFunc("GET /sitemap.xml", d.handleSitemap)
	mux.HandleFunc("GET /robots.txt", d.handleRobots)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = w.Write([]byte("ok\n"))
	})
	return docsSiteHeaders(mux), nil
}

type docsSiteHandler struct {
	site *docsSite
	base string
}

// docsSiteHeaders sets what every response from the public site carries. The
// pages are static content, so a short public cache is safe. The policy
// allows only this origin, plus inline style attributes, which is how
// goldmark aligns a table column.
func docsSiteHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "strict-origin-when-cross-origin")
		h.Set("Content-Security-Policy",
			"default-src 'self'; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
		if r.URL.Path != "/healthz" {
			h.Set("Cache-Control", "public, max-age=300")
		}
		next.ServeHTTP(w, r)
	})
}

// origin is the scheme and host links are built from.
func (d *docsSiteHandler) origin(r *http.Request) string {
	if d.base != "" {
		return d.base
	}
	scheme := "http"
	if r.TLS != nil || r.Header.Get("X-Forwarded-Proto") == "https" {
		scheme = "https"
	}
	return scheme + "://" + r.Host
}

// markdownPath is where a page's source is served: its path under docs/,
// below /docs/. That is not the route plus ".md" -- the introduction is
// /docs but README.md -- and it is chosen so a relative link in the source
// resolves against the served file the way it does on GitHub.
func markdownPath(page *docsPage) string { return "/docs/" + page.file }

func (d *docsSiteHandler) handlePage(w http.ResponseWriter, r *http.Request) {
	if strings.HasSuffix(r.URL.Path, ".md") {
		page, ok := d.site.byFile[strings.TrimPrefix(r.URL.Path, "/docs/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
		_, _ = w.Write(page.source)
		return
	}
	page, ok := d.site.lookup(r.URL.Path)
	if !ok {
		http.Error(w, "no such documentation page: "+r.URL.Path, http.StatusNotFound)
		return
	}
	origin := d.origin(r)
	writeDocsPage(w, page, docsData{
		Public:    true,
		Canonical: origin + page.route,
		Markdown:  markdownPath(page),
	})
}

// handleLLMs serves /llms.txt: a title, a summary, and a link to every page's
// Markdown with its description. Top-level pages are listed under "Docs"; a
// page's children get a section of their own, named after it.
func (d *docsSiteHandler) handleLLMs(w http.ResponseWriter, r *http.Request) {
	origin := d.origin(r)
	var sb strings.Builder
	fmt.Fprintf(&sb, "# llm-helper\n\n> %s\n\n", docsSiteSummary)
	fmt.Fprintf(&sb, "Each link below is the page's Markdown source. %s/llms-full.txt has every page in one file, and the rendered pages are under %s/docs.\n\n", origin, origin)

	sb.WriteString("## Docs\n\n")
	var sections []docs.Route
	for _, r := range d.site.routes {
		d.writeLLMsLine(&sb, origin, r)
		if len(r.Children) > 0 {
			sections = append(sections, r)
		}
	}
	for _, s := range sections {
		fmt.Fprintf(&sb, "\n## %s\n\n", s.Title)
		var walk func([]docs.Route)
		walk = func(routes []docs.Route) {
			for _, c := range routes {
				d.writeLLMsLine(&sb, origin, c)
				walk(c.Children)
			}
		}
		walk(s.Children)
	}
	writeText(w, "text/plain", sb.String())
}

func (d *docsSiteHandler) writeLLMsLine(sb *strings.Builder, origin string, r docs.Route) {
	page, ok := d.site.byRoute[docs.RouteFor(r.Path)]
	if !ok {
		return
	}
	fmt.Fprintf(sb, "- [%s](%s%s)", page.title, origin, markdownPath(page))
	if page.description != "" {
		fmt.Fprintf(sb, ": %s", page.description)
	}
	sb.WriteString("\n")
}

// handleLLMsFull serves /llms-full.txt: every page's Markdown, in sidebar
// order, each under its title and source URL.
func (d *docsSiteHandler) handleLLMsFull(w http.ResponseWriter, r *http.Request) {
	origin := d.origin(r)
	var sb strings.Builder
	fmt.Fprintf(&sb, "# llm-helper\n\n> %s\n", docsSiteSummary)
	for _, page := range d.site.order {
		fmt.Fprintf(&sb, "\n---\n\n# %s\n\n", page.title)
		if page.description != "" {
			fmt.Fprintf(&sb, "> %s\n\n", page.description)
		}
		fmt.Fprintf(&sb, "Source: %s%s\n\n", origin, page.route)
		sb.WriteString(strings.TrimSpace(string(page.markdown)))
		sb.WriteString("\n")
	}
	writeText(w, "text/plain", sb.String())
}

func (d *docsSiteHandler) handleSitemap(w http.ResponseWriter, r *http.Request) {
	origin := d.origin(r)
	var sb strings.Builder
	sb.WriteString(xml.Header)
	sb.WriteString(`<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">` + "\n")
	for _, page := range d.site.order {
		sb.WriteString("  <url><loc>")
		_ = xml.EscapeText(&sb, []byte(origin+page.route))
		sb.WriteString("</loc></url>\n")
	}
	sb.WriteString("</urlset>\n")
	writeText(w, "application/xml", sb.String())
}

func (d *docsSiteHandler) handleRobots(w http.ResponseWriter, r *http.Request) {
	writeText(w, "text/plain", "User-agent: *\nAllow: /\n\nSitemap: "+d.origin(r)+"/sitemap.xml\n")
}

func writeText(w http.ResponseWriter, contentType, body string) {
	w.Header().Set("Content-Type", contentType+"; charset=utf-8")
	_, _ = w.Write([]byte(body))
}
