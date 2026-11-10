package cli

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/profile"
)

const defaultRecipeServiceName = "vllm-cluster.service"

// A big model takes minutes to load and the unit only settles once the
// endpoint answers, so systemd's default 90s start timeout would kill a
// healthy launch part-way through.
const defaultRecipeServiceTimeout = 1800

func newRecipesInstallServiceCmd() *cobra.Command {
	var (
		name      string
		prof      string
		user      string
		timeout   int
		now       bool
		supersede bool
		watchdog  bool
		interval  int
	)
	c := &cobra.Command{
		Use:   "install-service",
		Short: "Install a systemd service that keeps a recipe profile serving",
		Long: "Installs a unit that brings a recipe profile back at boot and stops its containers\n" +
			"on shutdown. --profile names it; without it, the only recipe profile running is used.\n" +
			"Install one unit per recipe profile, each with its own --name. Linux hosts only.",
		Args: cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			prof, err := recipeServiceProfile(cfg, prof)
			if err != nil {
				return err
			}
			return installRecipeService(recipeServiceOptions{
				Name:      name,
				Profile:   prof,
				User:      userOrInvoker(user),
				Timeout:   timeout,
				Start:     now,
				Supersede: supersede,
				Native:    cfg.ServiceName(),
				Watchdog:  watchdog,
				Interval:  interval,
				Service:   defaultRecipeServiceRunner{},
			})
		},
	}
	c.Flags().StringVar(&name, "name", defaultRecipeServiceName, "systemd unit name")
	c.Flags().StringVar(&prof, "profile", "", "recipe profile the unit serves (default: the only one running)")
	c.Flags().StringVar(&user, "user", "", "user the launcher runs as (default: $USER)")
	c.Flags().IntVar(&timeout, "timeout", defaultRecipeServiceTimeout, "seconds to allow for the model to load")
	c.Flags().BoolVar(&now, "now", false, "enable and start the service immediately")
	c.Flags().BoolVar(&supersede, "supersede-native", true, "stop and disable the single-node unit this replaces")
	c.Flags().BoolVar(&watchdog, "watchdog", true, "also install a timer that relaunches the cluster when the endpoint stops answering")
	c.Flags().IntVar(&interval, "watchdog-interval", defaultHealthInterval, "minutes between health checks")
	return c
}

// recipeServiceProfile resolves the profile a recipe unit serves.
func recipeServiceProfile(cfg *config.Config, name string) (string, error) {
	if name != "" {
		p, err := profile.Load(cfg.Dirs, name)
		if err != nil {
			return "", err
		}
		if !p.Generated() {
			return "", fmt.Errorf("'%s' is not a recipe profile; '%s up %s' already installs its service", name, cmdName, name)
		}
		return name, nil
	}
	list, err := control.ListInstances(cfg)
	if err != nil {
		return "", err
	}
	var found []string
	for _, inst := range list {
		if clusterRunnerForName(cfg, inst.Name) != nil {
			found = append(found, inst.Name)
		}
	}
	switch len(found) {
	case 1:
		return found[0], nil
	case 0:
		return "", errors.New("no recipe profile is running; launch one first or pass --profile")
	}
	return "", fmt.Errorf("%d recipe profiles are running (%s); pass --profile", len(found), strings.Join(found, ", "))
}

type recipeServiceOptions struct {
	Name      string
	Profile   string
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
	if strings.TrimSpace(o.Profile) == "" {
		return errors.New("--profile is required")
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
		Profile: o.Profile,
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
		svcUnit, timerUnit := renderHealthUnits(recipeSystemdUnit{Home: home, Self: self, Profile: o.Profile}, o.Name, interval)
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
	Profile string
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
		"Description=LLM Helper cluster (%s)\n"+
		"After=network-online.target docker.service\n"+
		"Wants=network-online.target\n\n"+
		"[Service]\n"+
		"Type=oneshot\n"+
		"RemainAfterExit=yes\n"+
		"User=%s\n"+
		"Environment=HOME=%s\n"+
		"WorkingDirectory=%s\n"+
		"ExecStart=%s recipes service-start %s\n"+
		"ExecStop=%s recipes service-stop %s\n"+
		"TimeoutStartSec=%d\n\n"+
		"[Install]\n"+
		"WantedBy=multi-user.target\n",
		u.Profile, u.User, u.Home, u.Home, u.Self, u.Profile, u.Self, u.Profile, u.Timeout)
}

// newRecipesServiceStartCmd is the unit's ExecStart. A profile taken down on
// purpose stays down: there is no instance left to bring back.
func newRecipesServiceStartCmd() *cobra.Command {
	return &cobra.Command{
		Use:    "service-start <profile>",
		Hidden: true,
		Args:   cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			if _, err := control.LoadInstance(cfg, args[0]); errors.Is(err, fs.ErrNotExist) {
				fmt.Printf("%s is not running; leaving it stopped.\n", args[0])
				return nil
			}
			return runRestart(args, watchOpts{wait: true})
		},
	}
}

// newRecipesServiceStopCmd is the unit's ExecStop. It clears the containers
// but keeps the instance, so the next boot brings it back.
func newRecipesServiceStopCmd() *cobra.Command {
	return &cobra.Command{
		Use:    "service-stop <profile>",
		Hidden: true,
		Args:   cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			c := clusterRunnerForName(cfg, args[0])
			if c == nil {
				return fmt.Errorf("'%s' is not a recipe profile", args[0])
			}
			return c.Stop()
		},
	}
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
