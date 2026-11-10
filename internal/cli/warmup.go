package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/router"
	"github.com/coder/llm-helper/internal/warmup"
)

func newWarmupCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "warmup [profile]",
		Short: "Send a profile's warm-up request so its prompt is cached",
		Long: "Sends the chat request named by the profile's " + profile.WarmupKey + " twice and reports how much of each prompt the server served from its cache.\n" +
			"The warm-up also runs on its own after up, restart and switch, and whenever the router sees the server come back.",
		Args:    cobra.MaximumNArgs(1),
		GroupID: groupServer,
		RunE: func(_ *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			inst, err := pickInstance(cfg, args)
			if err != nil {
				return err
			}
			path := warmupPath(cfg, inst)
			if path == "" {
				return fmt.Errorf("'%s' has no %s; record one with '%s warmup capture %s'", inst.Name, profile.WarmupKey, cmdName, inst.Name)
			}
			base, model := warmupTarget(cfg, inst)
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
			defer stop()
			for _, label := range []string{"First send", "Second send"} {
				res, err := warmup.Run(ctx, base, cfg.APIKey(), model, path)
				if err != nil {
					return err
				}
				fmt.Printf("%s: %s\n", label, res.Describe())
			}
			return nil
		},
	}
	c.AddCommand(newWarmupCaptureCmd())
	return c
}

func newWarmupCaptureCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "capture [profile]",
		Short: "Record the next chat request through the router as the profile's warm-up",
		Long: "Waits for the next chat request the router passes to the profile's model, keeps everything before its first user message, and saves that as the profile's warm-up request.\n" +
			"Needs the router; start a chat from your client while this waits.",
		Args: cobra.MaximumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			inst, err := pickInstance(cfg, args)
			if err != nil {
				return err
			}
			_, model := warmupTarget(cfg, inst)
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
			defer stop()
			fmt.Printf("Waiting for the next chat request to %s through the router (Ctrl+C to stop)...\n", model)
			raw, err := captureRequest(ctx, cfg, model)
			if err != nil {
				return err
			}
			body, err := warmup.FromCapture(raw)
			if err != nil {
				return err
			}
			path, err := saveWarmup(cfg, inst.Name, body)
			if err != nil {
				return err
			}
			fmt.Printf("Saved %s\n", path)
			fmt.Printf("%s: %s=%s\n", inst.Name, profile.WarmupKey, path)
			return nil
		},
	}
}

// warmupPath is the instance's warm-up request: the profile's current value,
// so a file set after the instance started still counts.
func warmupPath(cfg *config.Config, inst *control.Instance) string {
	path := inst.Env.Get(profile.WarmupKey)
	if p, err := profile.Load(cfg.Dirs, inst.Name); err == nil {
		path = p.Warmup()
	}
	return expandHome(strings.TrimSpace(path))
}

// warmupTarget is the instance's OpenAI-compatible root and the model id it
// serves, which can differ from the profile's MODEL.
func warmupTarget(cfg *config.Config, inst *control.Instance) (base, model string) {
	root := instanceURL(cfg, inst)
	model = inst.Env.Get("MODEL")
	if ep, ok := detect.ProbeURL(root, cfg.APIKey()); ok && len(ep.Models) > 0 {
		model = ep.Models[0]
	}
	return strings.TrimSuffix(root, "/") + "/v1", model
}

// warmAfterBoot sends the warm-up once the server is healthy. A failure only
// warns: the server is up either way.
func warmAfterBoot(cfg *config.Config, inst *control.Instance) {
	path := warmupPath(cfg, inst)
	if path == "" {
		return
	}
	base, model := warmupTarget(cfg, inst)
	fmt.Println("Warming up the prompt cache...")
	res, err := warmup.Run(context.Background(), base, cfg.APIKey(), model, path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warm-up failed (the server is still up): %v\n", err)
		return
	}
	fmt.Printf("Warmed up: %s\n", res.Describe())
}

func routerURL(cfg *config.Config, path string) string {
	return "http://localhost:" + routerPort(cfg, "") + path
}

func routerRequest(ctx context.Context, cfg *config.Config, path string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, routerURL(cfg, path), nil)
	if err != nil {
		return nil, err
	}
	if key := cfg.APIKey(); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	return http.DefaultClient.Do(req)
}

func captureRequest(ctx context.Context, cfg *config.Config, model string) ([]byte, error) {
	resp, err := routerRequest(ctx, cfg, "/router/capture?model="+url.QueryEscape(model))
	if err != nil {
		if errors.Is(ctx.Err(), context.Canceled) {
			return nil, errors.New("stopped before a request arrived")
		}
		return nil, fmt.Errorf("the router is not answering on port %s (start it with '%s router install-service'): %w", routerPort(cfg, ""), cmdName, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("router answered %s: %s", resp.Status, strings.TrimSpace(string(data)))
	}
	return data, nil
}

// saveWarmup writes body under the data directory and points the profile at it.
func saveWarmup(cfg *config.Config, name string, body []byte) (string, error) {
	p, err := profile.Load(cfg.Dirs, name)
	if err != nil {
		return "", err
	}
	path := filepath.Join(cfg.Dirs.Data, "warmup", name+".json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(body, '\n'), 0o600); err != nil {
		return "", err
	}
	p.File.Set(profile.WarmupKey, path)
	if err := p.File.Write(cfg.Dirs.ProfileFile(name), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// routerCache is the router's prompt-cache counts for each instance it has
// passed replies for, by profile name. It is empty when the router is not
// running.
func routerCache(cfg *config.Config, list []*control.Instance) map[string]router.CacheStats {
	out := map[string]router.CacheStats{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	stats, err := router.FetchStats(ctx, routerURL(cfg, ""), cfg.APIKey())
	if err != nil {
		return out
	}
	for _, inst := range list {
		if s, ok := router.StatsAt(stats, instanceURL(cfg, inst)); ok && s.Reported+s.Unknown > 0 {
			out[inst.Name] = s
		}
	}
	return out
}

func describeCache(s router.CacheStats) string {
	if p := s.Percent(); p >= 0 {
		return fmt.Sprintf("%d%% of %d prompt tokens came from the cache (router traffic only)", p, s.PromptTokens)
	}
	return "the server does not report cached tokens"
}
