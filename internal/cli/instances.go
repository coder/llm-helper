package cli

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strconv"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/profile"
)

type upOpts struct {
	port    string
	memUtil string
	kvCache string
}

func newUpCmd() *cobra.Command {
	var w waitFlags
	var o upOpts
	c := &cobra.Command{
		Use:     "up <profile>",
		Short:   "Start serving a profile alongside anything already running",
		GroupID: groupModels,
		Args:    cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			if len(args) == 0 {
				printProfiles(cfg)
				fmt.Println()
				fmt.Println("usage: " + cmdName + " up <profile> [--port N] [--mem-util F | --kv-cache SIZE]")
				return errors.New("missing profile name")
			}
			p, err := profile.Load(cfg.Dirs, args[0])
			if err != nil {
				return err
			}
			return upProfile(cfg, p, o, w.opts())
		},
	}
	c.Flags().StringVar(&o.port, "port", "", "port to serve on (default: the next free one from PORT)")
	c.Flags().StringVar(&o.memUtil, "mem-util", "", "fraction of GPU memory this model may use, e.g. 0.4")
	c.Flags().StringVar(&o.kvCache, "kv-cache", "", "fixed KV cache size in bytes, e.g. 20G")
	c.MarkFlagsMutuallyExclusive("mem-util", "kv-cache")
	w.register(c)
	return c
}

// upProfile starts p as an instance. Starting one that already runs restarts
// it with the new settings.
func upProfile(cfg *config.Config, p *profile.Profile, o upOpts, opts watchOpts) error {
	if err := profile.ValidInstanceName(p.Name); err != nil {
		return fmt.Errorf("can't run '%s': %w; rename the profile", p.Name, err)
	}
	if o.memUtil != "" {
		if f, err := strconv.ParseFloat(o.memUtil, 64); err != nil || f <= 0 || f > 1 {
			return fmt.Errorf("--mem-util %q: want a fraction between 0 and 1", o.memUtil)
		}
	}
	if legacyActive(cfg) {
		return fmt.Errorf("this machine still runs a model the old single-model way; run '%s migrate' first", cmdName)
	}
	var cluster *clusterRunner
	if p.Generated() {
		c, err := clusterRunnerFor(cfg, p)
		if err != nil {
			return err
		}
		if err := c.preflight(cfg); err != nil {
			return err
		}
		cluster = c
	}

	if cluster == nil && (o.memUtil != "" || o.kvCache == "") {
		if err := control.CheckFitFor(cfg, p.Name, o.memUtil, p.File, cfg.File); err != nil {
			return err
		}
	}

	want := o.port
	if want == "" && cluster != nil {
		want = strconv.Itoa(cluster.recipe.Port)
	}
	port, err := control.AllocatePort(cfg, p.Name, want)
	if err != nil {
		return err
	}
	if err := control.WriteInstanceEnv(cfg, p, control.InstanceOpts{Port: port, MemUtil: o.memUtil, KVCache: o.kvCache}); err != nil {
		return err
	}
	inst, err := control.LoadInstance(cfg, p.Name)
	if err != nil {
		return err
	}
	if cluster == nil {
		self, err := os.Executable()
		if err != nil {
			return err
		}
		if _, err := instanceService(cfg, p.Name).Install(self); err != nil {
			_ = control.RemoveInstance(cfg, p.Name)
			return err
		}
	}
	if prov, err := profileProvider(p); err == nil && prov.Capabilities().VLLMArgs {
		warnExtraArgs(p, vllmCaps(cfg))
	}

	fmt.Printf("Starting profile %s on port %s\n", p.Name, port)
	fmt.Println()
	for i, line := range p.File.LeadingComments() {
		if i >= 8 {
			break
		}
		fmt.Println(line)
	}
	fmt.Println()
	run := instanceRunner(cfg, p.Name)
	restart := func() error {
		fmt.Println("Loading the model (this can take several minutes, longer if it")
		fmt.Println("isn't cached on disk yet)...")
		return run.Restart()
	}
	if err := control.NotifyAndRestart(context.Background(), cfg, p, inst, os.Stderr, restart); err != nil {
		return err
	}
	if !opts.follow && !opts.wait {
		fmt.Println()
		fmt.Println("Not waiting for the model to load. Check progress with 'llm-helper status'")
		fmt.Println("or 'llm-helper logs " + p.Name + "'.")
		return nil
	}
	fmt.Println()
	return watchBoot(cfg, p.Name, opts)
}

func newDownCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "down [profile]",
		Aliases: []string{"stop"},
		Short:   "Stop serving a profile (the only one running when none is named)",
		GroupID: groupServer,
		Args:    cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			inst, err := pickInstance(cfg, args)
			if len(args) == 0 && errors.Is(err, errNoInstances) {
				fmt.Println("No model is running; nothing to stop.")
				return nil
			}
			if err != nil {
				return err
			}
			if err := downInstance(cfg, inst.Name); err != nil {
				return err
			}
			fmt.Printf("Stopped %s.\n", inst.Name)
			return nil
		},
	}
}

func downInstance(cfg *config.Config, name string) error {
	if c := clusterRunnerForName(cfg, name); c != nil {
		if err := c.Stop(); err != nil {
			return err
		}
	} else if err := instanceService(cfg, name).Uninstall(); err != nil {
		return err
	}
	if err := control.RemoveInstance(cfg, name); err != nil {
		return err
	}
	control.NotifyStopped(context.Background(), cfg, name, os.Stderr)
	return nil
}

func newDefaultCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "default [profile]",
		Short:   "Show or set the model Coder uses by default",
		GroupID: groupModels,
		Args:    cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			live := control.LiveProfiles(cfg)
			if len(args) == 0 {
				if d := control.DefaultInstance(cfg, live); d != "" {
					fmt.Println(d)
					return nil
				}
				return errNoInstances
			}
			name := args[0]
			if _, err := control.LoadInstance(cfg, name); err != nil {
				if errors.Is(err, fs.ErrNotExist) {
					return fmt.Errorf("'%s' is not running; start it with '%s up %s'", name, cmdName, name)
				}
				return err
			}
			cfg.File.Set("DEFAULT_PROFILE", name)
			if err := cfg.File.Write(cfg.Dirs.ConfigFile(), 0o644); err != nil {
				return err
			}
			control.NotifyDefault(context.Background(), cfg, os.Stderr)
			fmt.Printf("%s is now the default model.\n", name)
			return nil
		},
	}
}

// legacyActive reports a model still recorded the pre-instance way.
func legacyActive(cfg *config.Config) bool {
	_, err := os.Stat(cfg.Dirs.LegacyActiveEnvFile())
	return err == nil
}

func newMigrateCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "migrate",
		Short:   "Move a model started by an older release to the per-model services",
		GroupID: groupServer,
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			return runMigrate(cfg)
		},
	}
}

func runMigrate(cfg *config.Config) error {
	name := cfg.LegacyActiveProfile()
	if !legacyActive(cfg) && name == "" {
		fmt.Println("Nothing to migrate.")
		return nil
	}
	legacy := legacyService(cfg)
	wasRunning := legacy.Installed() && legacy.IsActive() == "active"
	if name != "" {
		p, err := profile.Load(cfg.Dirs, name)
		if err != nil {
			return err
		}
		port := cfg.Port()
		if old, err := config.ParseEnvFile(cfg.Dirs.LegacyActiveEnvFile()); err == nil && old.Get("PORT") != "" {
			port = old.Get("PORT")
		}
		if err := profile.ValidInstanceName(p.Name); err != nil {
			return fmt.Errorf("can't migrate '%s': %w; rename the profile and run '%s up' on it", p.Name, err, cmdName)
		}
		if err := control.WriteInstanceEnv(cfg, p, control.InstanceOpts{Port: port}); err != nil {
			return err
		}
		if !p.Generated() {
			self, err := os.Executable()
			if err != nil {
				return err
			}
			if legacy.Installed() {
				if err := legacy.Uninstall(); err != nil {
					return err
				}
			}
			if _, err := instanceService(cfg, p.Name).Install(self); err != nil {
				return err
			}
			if wasRunning {
				if err := instanceService(cfg, p.Name).Restart(); err != nil {
					return err
				}
			}
		}
		fmt.Printf("%s now runs as its own service on port %s.\n", p.Name, port)
	} else if legacy.Installed() {
		if err := legacy.Uninstall(); err != nil {
			return err
		}
	}
	for _, f := range []string{cfg.Dirs.LegacyActiveEnvFile(), cfg.Dirs.LegacyActiveProfileFile()} {
		if err := os.Remove(f); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
	}
	return nil
}

// newSwitchCmd is the pre-instance verb: stop everything else, run this.
func newSwitchCmd() *cobra.Command {
	var w waitFlags
	c := &cobra.Command{
		Use:        "switch <profile>",
		Short:      "Stop every other model and run this one",
		Hidden:     true,
		Deprecated: "use 'up' and 'down'; several models can run at once",
		Args:       cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			p, err := profile.Load(cfg.Dirs, args[0])
			if err != nil {
				return err
			}
			list, err := control.ListInstances(cfg)
			if err != nil {
				return err
			}
			for _, inst := range list {
				if inst.Name == p.Name {
					continue
				}
				if err := downInstance(cfg, inst.Name); err != nil {
					return err
				}
				fmt.Printf("Stopped %s.\n", inst.Name)
			}
			return upProfile(cfg, p, upOpts{}, w.opts())
		},
	}
	w.register(c)
	return c
}
