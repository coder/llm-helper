package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/router"
	"github.com/coder/llm-helper/internal/service"
	"github.com/coder/llm-helper/internal/warmup"
)

const defaultRouterPort = "8000"

func newRouterCmd() *cobra.Command {
	c := &cobra.Command{
		Use:     "router",
		Short:   "Serve every running model behind one port",
		Long:    "The router answers on one port and sends each request to the instance serving the model it names. Coder then offers every running model, not just one.",
		GroupID: groupServer,
	}
	c.AddCommand(newRouterServeCmd(), newRouterInstallCmd(), newRouterUninstallCmd())
	return c
}

// routerPort is the port the router uses: the flag, the installed one, or
// the default.
func routerPort(cfg *config.Config, flag string) string {
	if flag != "" {
		return flag
	}
	if p := cfg.RouterPort(); p != "" {
		return p
	}
	return defaultRouterPort
}

// routerBind keeps the router on loopback unless the instances check a key,
// the same rule the engines follow.
func routerBind(cfg *config.Config, port string) string {
	if cfg.APIKey() != "" {
		return net.JoinHostPort("", port)
	}
	return net.JoinHostPort("127.0.0.1", port)
}

func instanceBackends(cfg *config.Config) func() []router.Backend {
	return func() []router.Backend {
		list, err := control.ListInstances(cfg)
		if err != nil {
			return nil
		}
		out := make([]router.Backend, 0, len(list))
		for _, inst := range list {
			if u := instanceURL(cfg, inst); u != "" {
				out = append(out, router.Backend{URL: u, Model: inst.Env.Get("MODEL"), Warmup: warmupPath(cfg, inst)})
			}
		}
		return out
	}
}

// instanceURL is where the instance answers: its cluster endpoint, or its
// local port.
func instanceURL(cfg *config.Config, inst *control.Instance) string {
	if c := clusterRunnerForName(cfg, inst.Name); c != nil {
		if u, err := c.endpointURL(); err == nil {
			return u
		}
	}
	if inst.Port != "" {
		return "http://localhost:" + inst.Port
	}
	return ""
}

func newRouterServeCmd() *cobra.Command {
	var port, bind string
	c := &cobra.Command{
		Use:   "serve [--port N]",
		Short: "Run the router in the foreground",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			addr := bind
			if addr == "" {
				addr = routerBind(cfg, routerPort(cfg, port))
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			models := router.NewModels(instanceBackends(cfg), cfg.APIKey())
			models.OnWarm = func(model string, res warmup.Result, err error) {
				if err != nil {
					fmt.Fprintf(os.Stderr, "warm-up for %s failed: %v\n", model, err)
					return
				}
				fmt.Printf("Warmed up %s: %s\n", model, res.Describe())
			}
			models.StartRefresh(ctx, 5*time.Second)
			mux := routerMux(models, cfg.APIKey())

			ln, err := net.Listen("tcp", addr)
			if err != nil {
				return fmt.Errorf("listening on %s: %w", addr, err)
			}
			fmt.Printf("Router listening on http://%s/v1\n", ln.Addr())

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
	c.Flags().StringVar(&port, "port", "", "port to listen on (default: ROUTER_PORT, else "+defaultRouterPort+")")
	c.Flags().StringVar(&bind, "bind", "", "address to listen on, overriding --port (default: loopback, or every interface when VLLM_API_KEY is set)")
	return c
}

func routerMux(models *router.Models, key string) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/v1/", router.RequireKey(key, models))
	mux.Handle("/router/status", router.RequireKey(key, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(models.Routes())
	})))
	mux.Handle("/router/stats", router.RequireKey(key, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(models.Report())
	})))
	mux.Handle("/router/capture", router.RequireKey(key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model := r.URL.Query().Get("model")
		if _, ok := models.Routes()[model]; !ok {
			http.Error(w, fmt.Sprintf("the router does not serve %q", model), http.StatusNotFound)
			return
		}
		body, err := models.Capture(r.Context(), model)
		if err != nil {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	})))
	return mux
}

type routerService interface {
	Install(self string) (string, error)
	Restart() error
	Uninstall() error
}

func routerUnit(cfg *config.Config, port string) service.Service {
	return service.Service{
		Name:           service.RouterName,
		Scope:          cfg.ServiceScope(),
		NonInteractive: !stdinIsTTY(),
		Args:           []string{"router", "serve", "--port", port},
	}
}

// installRouter checks port, installs and restarts svc, and records the port
// so integrations point at the router.
func installRouter(cfg *config.Config, port, self string, svc routerService) error {
	if err := control.CheckRouterPort(cfg, port); err != nil {
		return err
	}
	path, err := svc.Install(self)
	if err != nil {
		return err
	}
	if err := svc.Restart(); err != nil {
		return err
	}
	cfg.File.Set("ROUTER_PORT", port)
	if err := cfg.File.Write(cfg.Dirs.ConfigFile(), 0o644); err != nil {
		return err
	}
	fmt.Printf("Installed %s\n", path)
	fmt.Printf("Router: http://localhost:%s/v1\n", port)
	fmt.Println("Point your Coder AI provider's base URL at this port.")
	return nil
}

func newRouterInstallCmd() *cobra.Command {
	var port string
	c := &cobra.Command{
		Use:   "install-service [--port N]",
		Short: "Run the router as a service, so Coder can offer every running model",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			self, err := os.Executable()
			if err != nil {
				return err
			}
			p := routerPort(cfg, port)
			if err := installRouter(cfg, p, self, routerUnit(cfg, p)); err != nil {
				return err
			}
			control.NotifyDefault(context.Background(), cfg, os.Stderr)
			return nil
		},
	}
	c.Flags().StringVar(&port, "port", "", "port to listen on (default: ROUTER_PORT, else "+defaultRouterPort+")")
	return c
}

func uninstallRouter(cfg *config.Config, svc routerService) error {
	if err := svc.Uninstall(); err != nil {
		return err
	}
	cfg.File.Delete("ROUTER_PORT")
	return cfg.File.Write(cfg.Dirs.ConfigFile(), 0o644)
}

func newRouterUninstallCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "uninstall-service",
		Short: "Remove the router service; Coder then offers only the model on PORT",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			if err := uninstallRouter(cfg, routerUnit(cfg, routerPort(cfg, ""))); err != nil {
				return err
			}
			fmt.Println("Router removed.")
			control.NotifyDefault(context.Background(), cfg, os.Stderr)
			return nil
		},
	}
}
