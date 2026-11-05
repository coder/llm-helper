package cli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/nodes"
	"github.com/coder/llm-helper/internal/router"
)

func newPoolServeCmd() *cobra.Command {
	var (
		bind   string
		sticky string
		nlist  string
	)
	c := &cobra.Command{
		Use:   "serve --nodes n1,n2 [--bind :8011] [--sticky auto|authorization|header:Name]",
		Short: "Run a sticky reverse proxy over the given nodes",
		Args:  cobra.NoArgs,
		RunE: func(_ *cobra.Command, _ []string) error {
			if nlist == "" {
				return errors.New("--nodes is required")
			}
			r, err := nodes.Load()
			if err != nil {
				return err
			}
			var backends []string
			for _, name := range strings.Split(nlist, ",") {
				name = strings.TrimSpace(name)
				n, ok := r.Get(name)
				if !ok {
					return fmt.Errorf("unknown node %q -- add it with 'llm-helper node add'", name)
				}
				backends = append(backends, strings.TrimRight(n.Host, "/"))
			}
			// Stickiness function
			var stickyFn func(*http.Request) string
			switch {
			case sticky == "authorization":
				stickyFn = func(r *http.Request) string { return r.Header.Get("Authorization") }
			case strings.HasPrefix(sticky, "header:"):
				hdr := strings.TrimPrefix(sticky, "header:")
				stickyFn = func(r *http.Request) string { return r.Header.Get(hdr) }
			default: // auto
				stickyFn = func(r *http.Request) string {
					if v := r.Header.Get("X-Session-ID"); v != "" {
						return v
					}
					return r.Header.Get("Authorization")
				}
			}
			pool, err := router.NewPool(backends, stickyFn)
			if err != nil {
				return err
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			pool.StartHealth(ctx, 3*time.Second)

			mux := http.NewServeMux()
			// Proxy the OpenAI-compatible API path.
			mux.Handle("/v1/", pool)
			// Minimal status for humans.
			mux.HandleFunc("/pool/status", func(w http.ResponseWriter, r *http.Request) {
				st := pool.Status()
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(st)
			})

			ln, err := net.Listen("tcp", bind)
			if err != nil {
				return fmt.Errorf("listening on %s: %w", bind, err)
			}
			fmt.Println()
			fmt.Println("Pool router listening:")
			fmt.Printf("  http://%s/v1\n", ln.Addr().String())
			fmt.Println("Backends:")
			for _, b := range backends {
				fmt.Println("  ", b)
			}
			fmt.Println("Sticky:", sticky)

			srv := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 2 * time.Minute}
			errs := make(chan error, 1)
			go func() {
				if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
					errs <- err
				}
			}()
			stop := make(chan os.Signal, 1)
			signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
			select {
			case err := <-errs:
				return err
			case <-stop:
				_ = srv.Close()
				return nil
			}
		},
	}
	c.Flags().StringVar(&nlist, "nodes", "", "Comma-separated node names (from 'node ls')")
	c.Flags().StringVar(&bind, "bind", ":8011", "Bind address for the router")
	c.Flags().StringVar(&sticky, "sticky", "auto", "Stickiness (auto|authorization|header:Name)")
	return c
}
