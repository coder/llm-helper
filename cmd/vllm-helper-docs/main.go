// Command vllm-helper-docs serves the vllm-helper documentation as a public
// website: the pages the panel shows under Docs, without the panel, plus
// Markdown sources, /llms.txt and a sitemap.
//
// It is separate from the vllm-helper binary on purpose: that one controls a
// machine and guards every route, and it should not grow an unauthenticated
// mode. This one only reads the docs compiled into it.
//
// On Railway it listens on $PORT, and its links use $RAILWAY_PUBLIC_DOMAIN
// unless BASE_URL or -base-url names the origin; set one of those once the
// site has a custom domain.
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/coder/llm-helper/internal/web"
)

func main() {
	// Status goes to stdout and failures to stderr: hosts such as Railway file
	// everything on stderr as an error, which made a healthy start look like one.
	info := log.New(os.Stdout, "", log.LstdFlags)

	listen := flag.String("listen", defaultListen(), "address to serve on")
	baseURL := flag.String("base-url", defaultBaseURL(),
		"public origin for canonical links, the sitemap and llms.txt; empty uses each request's host")
	flag.Parse()

	handler, err := web.NewDocsSite(web.DocsSiteOptions{BaseURL: *baseURL})
	if err != nil {
		log.Fatalf("loading the docs: %v", err)
	}
	srv := &http.Server{
		Addr:              *listen,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	// Shutdown closes the listener at once, so ListenAndServe returns before
	// in-flight requests have drained; main waits on drained for the rest.
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdown); err != nil {
			log.Printf("shutting down: %v", err)
		}
	}()

	info.Printf("serving the docs on %s (base URL %q)", *listen, *baseURL)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	<-drained
}

// defaultListen is $PORT on every interface, as a platform like Railway
// expects, or localhost:8080 for a local preview.
func defaultListen() string {
	if port := os.Getenv("PORT"); port != "" {
		return ":" + port
	}
	return "localhost:8080"
}

// defaultBaseURL is BASE_URL, or the domain Railway assigned the service.
func defaultBaseURL() string {
	if u := os.Getenv("BASE_URL"); u != "" {
		return u
	}
	if domain := os.Getenv("RAILWAY_PUBLIC_DOMAIN"); domain != "" {
		return "https://" + domain
	}
	return ""
}
