package cli

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/spf13/cobra"
)

const defaultRecipeServiceName = "vllm-cluster.service"

// A big model takes minutes to load and the unit only settles once the
// endpoint answers, so systemd's default 90s start timeout would kill a
// healthy launch part-way through.
const defaultRecipeServiceTimeout = 1800

func newRecipesInstallServiceCmd() *cobra.Command {
	var (
		name      string
		user      string
		timeout   int
		now       bool
		supersede bool
		watchdog  bool
		interval  int
	)
	c := &cobra.Command{
		Use:   "install-service",
		Short: "Install a systemd service that keeps the active recipe serving",
		Long: "Installs a unit that runs 'llm-helper restart' at boot and 'llm-helper stop' on\n" +
			"shutdown. The unit names no recipe: which one runs comes from the active profile,\n" +
			"so switching recipes does not mean editing the unit. Linux hosts only.",
		Args: cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			// The unit this replaces is whichever one config names, so a
			// machine that renamed its service still has the right one stood
			// down rather than a guess.
			native := ""
			if cfg, err := loadConfig(); err == nil {
				native = cfg.ServiceName()
			}
			return installRecipeService(recipeServiceOptions{
				Name:      name,
				User:      userOrInvoker(user),
				Timeout:   timeout,
				Start:     now,
				Supersede: supersede,
				Native:    native,
				Watchdog:  watchdog,
				Interval:  interval,
				Service:   defaultRecipeServiceRunner{},
			})
		},
	}
	c.Flags().StringVar(&name, "name", defaultRecipeServiceName, "systemd unit name")
	c.Flags().StringVar(&user, "user", "", "user the launcher runs as (default: $USER)")
	c.Flags().IntVar(&timeout, "timeout", defaultRecipeServiceTimeout, "seconds to allow for the model to load")
	c.Flags().BoolVar(&now, "now", false, "enable and start the service immediately")
	c.Flags().BoolVar(&supersede, "supersede-native", true, "stop and disable the single-node unit this replaces")
	c.Flags().BoolVar(&watchdog, "watchdog", true, "also install a timer that relaunches the cluster when the endpoint stops answering")
	c.Flags().IntVar(&interval, "watchdog-interval", defaultHealthInterval, "minutes between health checks")
	return c
}

type recipeServiceOptions struct {
	Name      string
	User      string
	Timeout   int
	Start     bool
	Supersede bool
	Native    string
	Watchdog  bool
	Interval  int
	Service   recipeServiceRunner
}

// recipeServiceRunner is the host side, injected so the rendering and the
// supersede decision can be tested without a systemd host.
type recipeServiceRunner interface {
	Self() (string, error)
	Home(string) (string, error)
	InstallSystemd(name, unit string) (string, error)
	EnableStart(name string) error
	Supersede(name string) error
}

func installRecipeService(o recipeServiceOptions) error {
	if runtime.GOOS != "linux" {
		return errors.New("cluster services are implemented for Linux systemd hosts")
	}
	if strings.TrimSpace(o.User) == "" {
		return errors.New("--user is required")
	}
	if strings.TrimSpace(o.Name) == "" {
		return errors.New("--name is required")
	}
	if o.Timeout <= 0 {
		return errors.New("--timeout must be a number of seconds")
	}
	self, err := o.Service.Self()
	if err != nil {
		return err
	}
	home, err := o.Service.Home(o.User)
	if err != nil {
		return err
	}
	unit := renderRecipeSystemdUnit(recipeSystemdUnit{
		User:    o.User,
		Home:    home,
		Self:    self,
		Timeout: o.Timeout,
	})
	path, err := o.Service.InstallSystemd(o.Name, unit)
	if err != nil {
		return err
	}
	fmt.Printf("Installed cluster unit at %s\n", path)

	// The single-node unit cannot serve a container recipe, and left enabled
	// it races this one for the port at boot. A unit that failed months ago
	// also keeps the whole host reported as degraded.
	if o.Supersede && o.Native != "" && o.Native != o.Name {
		if err := o.Service.Supersede(o.Native); err != nil {
			fmt.Fprintf(os.Stderr, "Could not stand down %s: %v\n", o.Native, err)
		} else {
			fmt.Printf("Stopped, disabled and cleared %s; it cannot serve a container recipe.\n", o.Native)
		}
	}
	// The watchdog is what makes this unattended: a cluster that dies at 3am
	// otherwise stays dead until somebody looks. It restarts the unit rather
	// than the containers, so recovery goes through the same path as a boot.
	healthTimer := ""
	if o.Watchdog {
		interval := o.Interval
		if interval <= 0 {
			interval = defaultHealthInterval
		}
		healthService, timer := healthUnitNames(o.Name)
		healthTimer = timer
		svcUnit, timerUnit := renderHealthUnits(recipeSystemdUnit{Home: home, Self: self}, o.Name, interval)
		if _, err := o.Service.InstallSystemd(healthService, svcUnit); err != nil {
			return err
		}
		if _, err := o.Service.InstallSystemd(timer, timerUnit); err != nil {
			return err
		}
		fmt.Printf("Installed %s, checking every %d minutes after %d misses\n", timer, interval, defaultHealthMisses)
	}

	if o.Start {
		if err := o.Service.EnableStart(o.Name); err != nil {
			return err
		}
		fmt.Printf("Enabled and started %s\n", o.Name)
		if healthTimer != "" {
			if err := o.Service.EnableStart(healthTimer); err != nil {
				return err
			}
			fmt.Printf("Enabled %s\n", healthTimer)
		}
		return nil
	}
	// The timer is a unit of its own, so a hint naming only the service would
	// leave the advertised watchdog installed but never running.
	units := o.Name
	if healthTimer != "" {
		units += " " + healthTimer
	}
	fmt.Printf("Not started. Launch a recipe first so the profile records its nodes, then: sudo systemctl enable --now %s\n", units)
	return nil
}

type recipeSystemdUnit struct {
	User    string
	Home    string
	Self    string
	Timeout int
}

// renderRecipeSystemdUnit builds the unit.
//
// Type=oneshot with RemainAfterExit is right because the work is a launch
// that returns once the endpoint answers; the containers it starts outlive
// the command. ExecStop hands the same binary the teardown so both nodes are
// cleared, not just the head.
func renderRecipeSystemdUnit(u recipeSystemdUnit) string {
	return fmt.Sprintf("[Unit]\n"+
		"Description=LLM Helper cluster (the active recipe)\n"+
		"After=network-online.target docker.service\n"+
		"Wants=network-online.target\n\n"+
		"[Service]\n"+
		"Type=oneshot\n"+
		"RemainAfterExit=yes\n"+
		"User=%s\n"+
		"Environment=HOME=%s\n"+
		"WorkingDirectory=%s\n"+
		"ExecStart=%s restart\n"+
		"ExecStop=%s stop\n"+
		"TimeoutStartSec=%d\n\n"+
		"[Install]\n"+
		"WantedBy=multi-user.target\n",
		u.User, u.Home, u.Home, u.Self, u.Self, u.Timeout)
}

type defaultRecipeServiceRunner struct{}

func (defaultRecipeServiceRunner) Self() (string, error) { return defaultPoolServiceRunner{}.Self() }

func (defaultRecipeServiceRunner) Home(user string) (string, error) {
	return defaultPoolServiceRunner{}.Home(user)
}

func (defaultRecipeServiceRunner) InstallSystemd(name, unit string) (string, error) {
	return defaultPoolServiceRunner{}.InstallSystemd(name, unit)
}

func (defaultRecipeServiceRunner) EnableStart(name string) error {
	return runPoolCommand(os.Stdout, "sudo", "systemctl", "enable", "--now", name)
}

// Supersede stands the old unit down without deleting it: native mode is
// still the only way most machines serve anything, so the unit file stays as
// the way back. reset-failed clears a stale failure, which otherwise keeps
// systemctl reporting the whole host as degraded.
func (defaultRecipeServiceRunner) Supersede(name string) error {
	_ = runPoolCommand(nil, "sudo", "systemctl", "stop", name)
	if err := runPoolCommand(nil, "sudo", "systemctl", "disable", name); err != nil {
		return err
	}
	_ = runPoolCommand(nil, "sudo", "systemctl", "reset-failed", name)
	return nil
}
