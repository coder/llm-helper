package cli

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/web"
)

func newServeCmd() *cobra.Command {
	var addr string
	c := &cobra.Command{
		Use:     "serve",
		Short:   "Run a web panel for this machine",
		GroupID: groupServer,
		Long: "Serve a control panel: what model is loaded, whether it is healthy,\n" +
			"and what else could be loaded instead.\n\n" +
			"It listens on localhost unless told otherwise. The panel can restart the\n" +
			"server and change which model runs, so every request needs the token\n" +
			"printed below -- open the URL it gives you.\n\n" +
			"The panel drives the native service (systemd or launchd) only. While a\n" +
			"recipe container cluster serves, it shows the cluster's status and logs\n" +
			"but leaves switch, restart and stop to the command line.",
		Example: "  vllm-helper serve\n" +
			"  vllm-helper serve --listen 0.0.0.0:8080",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error { return runServe(cmd, addr) },
	}
	c.Flags().StringVar(&addr, "listen", "127.0.0.1:8080",
		"address to listen on (use 0.0.0.0:8080 to reach it from another machine)")
	return c
}

func runServe(cmd *cobra.Command, addr string) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	srv, err := web.New(cfg, describeBuild())
	if err != nil {
		return err
	}
	// The panel reports a recipe cluster instead of the idle native unit. The
	// explicit nil keeps a nil *clusterRunner from becoming a non-nil interface.
	srv.SetCluster(func() web.Cluster {
		if c := activeClusterRunner(cfg); c != nil {
			return c
		}
		return nil
	})

	handler := srv.Handler(web.Static())
	// A token in the query string is exchanged for a cookie on first load,
	// so it isn't repeated on every asset request. Once that cookie is set
	// the address is redirected to itself without the token: the page strips
	// it from the address bar too, but a redirect happens whether or not the
	// script runs and keeps the secret out of anything logging request lines.
	wrapped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Referrer-Policy", "no-referrer")
		if srv.SetCookie(w, r) && r.Method == http.MethodGet {
			http.Redirect(w, r, r.URL.Path, http.StatusSeeOther)
			return
		}
		handler.ServeHTTP(w, r)
	})

	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listening on %s: %w", addr, err)
	}

	fmt.Println(cmdName, "panel")
	fmt.Println()
	fmt.Println("  ", web.ListenURL(listener.Addr().String(), srv.Token()))
	fmt.Println()
	if !web.LoopbackOnly(addr) {
		// Worth saying plainly: this is reachable from other machines, the
		// token is the only thing between them and a stop button, and it
		// crosses the network in the clear. No TLS here on purpose -- a
		// self-signed certificate teaches people to click through warnings,
		// and a real one belongs to whatever already terminates it.
		fmt.Println("Reachable from the network over plain HTTP. Anyone who has that URL, or")
		fmt.Println("who can read traffic on the way to it, can switch models and stop the")
		fmt.Println("server. Treat it like a password, and put it behind a TLS proxy or a")
		fmt.Println("private network rather than exposing it directly.")
		fmt.Println()
	}
	fmt.Println("Ctrl+C to stop the panel. The model server is unaffected either way.")

	httpSrv := &http.Server{
		Handler: wrapped,
		// Long enough for a systemctl restart to return, which is the
		// slowest thing any handler waits on.
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      3 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}

	// Ctrl+C should end the panel, not orphan it.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	errs := make(chan error, 1)
	go func() {
		if err := httpSrv.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errs <- err
		}
	}()

	select {
	case err := <-errs:
		return err
	case <-stop:
		fmt.Println()
		fmt.Println("Stopping the panel. The model server keeps running.")
		return httpSrv.Close()
	}
}
