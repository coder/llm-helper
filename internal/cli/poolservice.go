package cli

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

const defaultPoolServiceName = "vllm-pool-router.service"

func newPoolInstallServiceCmd() *cobra.Command {
	var (
		nlist  string
		bind   string
		sticky string
		name   string
		user   string
		now    bool
	)
	c := &cobra.Command{
		Use:   "install-service --nodes n1,n2",
		Short: "Install a systemd service for the pool router",
		Long:  "Writes a systemd unit that runs 'llm-helper pool serve' with the given nodes. Linux hosts only.",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return installPoolService(poolServiceOptions{
				Nodes:   nlist,
				Bind:    bind,
				Sticky:  sticky,
				Name:    name,
				User:    userOrInvoker(user),
				Start:   now,
				Service: defaultPoolServiceRunner{},
			})
		},
	}
	c.Flags().StringVar(&nlist, "nodes", "", "Comma-separated node names")
	c.Flags().StringVar(&bind, "bind", ":8011", "Bind address for the router")
	c.Flags().StringVar(&sticky, "sticky", "auto", "Stickiness (auto|authorization|header:Name)")
	c.Flags().StringVar(&name, "name", defaultPoolServiceName, "systemd unit name")
	c.Flags().StringVar(&user, "user", "", "user the router runs as (default: $USER)")
	c.Flags().BoolVar(&now, "now", true, "enable and restart the service immediately")
	return c
}

type poolServiceOptions struct {
	Nodes   string
	Bind    string
	Sticky  string
	Name    string
	User    string
	Start   bool
	Service poolServiceRunner
}

type poolServiceRunner interface {
	Self() (string, error)
	Home(string) (string, error)
	InstallSystemd(name, unit string) (string, error)
	EnableRestart(name string) error
}

func installPoolService(o poolServiceOptions) error {
	if runtime.GOOS != "linux" {
		return errors.New("pool services are implemented for Linux systemd hosts")
	}
	if strings.TrimSpace(o.Nodes) == "" {
		return errors.New("--nodes is required")
	}
	if strings.TrimSpace(o.User) == "" {
		return errors.New("--user is required")
	}
	if strings.TrimSpace(o.Name) == "" {
		return errors.New("--name is required")
	}
	self, err := o.Service.Self()
	if err != nil {
		return err
	}
	home, err := o.Service.Home(o.User)
	if err != nil {
		return err
	}
	unit := renderPoolSystemdUnit(poolSystemdUnit{
		User:   o.User,
		Home:   home,
		Self:   self,
		Nodes:  normalizeCSV(o.Nodes),
		Bind:   o.Bind,
		Sticky: o.Sticky,
	})
	path, err := o.Service.InstallSystemd(o.Name, unit)
	if err != nil {
		return err
	}
	fmt.Printf("Installed pool router unit at %s\n", path)
	if o.Start {
		if err := o.Service.EnableRestart(o.Name); err != nil {
			return err
		}
		fmt.Printf("Enabled and restarted %s\n", o.Name)
	}
	return nil
}

type poolSystemdUnit struct {
	User   string
	Home   string
	Self   string
	Nodes  string
	Bind   string
	Sticky string
}

func renderPoolSystemdUnit(u poolSystemdUnit) string {
	return fmt.Sprintf("[Unit]\n"+
		"Description=LLM Helper Pool Router (sticky replicas)\n"+
		"After=network-online.target\n"+
		"Wants=network-online.target\n\n"+
		"[Service]\n"+
		"Type=simple\n"+
		"User=%s\n"+
		"Environment=HOME=%s\n"+
		"WorkingDirectory=%s\n"+
		"ExecStart=%s pool serve --nodes %s --bind %s --sticky %s\n"+
		"Restart=always\n"+
		"RestartSec=3\n\n"+
		"[Install]\n"+
		"WantedBy=multi-user.target\n",
		u.User, u.Home, u.Home, u.Self, u.Nodes, u.Bind, u.Sticky)
}

func normalizeCSV(v string) string {
	var out []string
	for _, p := range strings.Split(v, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, ",")
}

type defaultPoolServiceRunner struct{}

func (defaultPoolServiceRunner) Self() (string, error) {
	self, err := os.Executable()
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(self); err == nil {
		self = resolved
	}
	return self, nil
}

// userOrInvoker resolves --user at run time rather than as the flag default,
// so help and the generated reference don't print whoever built them.
func userOrInvoker(flag string) string {
	if strings.TrimSpace(flag) != "" {
		return flag
	}
	return os.Getenv("USER")
}

func (defaultPoolServiceRunner) Home(user string) (string, error) {
	out, err := exec.Command("getent", "passwd", user).Output()
	if err == nil {
		fields := strings.Split(string(bytes.TrimSpace(out)), ":")
		if len(fields) >= 6 && fields[5] != "" {
			return fields[5], nil
		}
	}
	if user == os.Getenv("USER") {
		return os.UserHomeDir()
	}
	return "", fmt.Errorf("could not find home for user %q", user)
}

func runPoolCommand(out *os.File, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdin, cmd.Stderr = os.Stdin, os.Stderr
	if out != nil {
		cmd.Stdout = out
	}
	return cmd.Run()
}

func (defaultPoolServiceRunner) InstallSystemd(name, unit string) (string, error) {
	path := filepath.Join("/etc/systemd/system", name)
	tmp, err := os.CreateTemp("", "vllm-pool-router-*.service")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.WriteString(unit); err != nil {
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if err := runPoolCommand(os.Stdout, "sudo", "cp", tmp.Name(), path); err != nil {
		return "", fmt.Errorf("installing unit (needs sudo): %w", err)
	}
	_ = runPoolCommand(nil, "sudo", "chmod", "644", path)
	_ = runPoolCommand(nil, "sudo", "systemctl", "daemon-reload")
	return path, nil
}

func (defaultPoolServiceRunner) EnableRestart(name string) error {
	if err := runPoolCommand(os.Stdout, "sudo", "systemctl", "enable", name); err != nil {
		return err
	}
	return runPoolCommand(os.Stdout, "sudo", "systemctl", "restart", name)
}

func newPoolHTTPStatusCmd() *cobra.Command {
	var url string
	c := &cobra.Command{
		Use:   "status",
		Short: "Show pool router backend health",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			client := &http.Client{Timeout: 3 * time.Second}
			resp, err := client.Get(url)
			if err != nil {
				return err
			}
			defer resp.Body.Close()
			if _, err := io.Copy(os.Stdout, resp.Body); err != nil {
				return err
			}
			fmt.Println()
			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				return fmt.Errorf("pool status returned HTTP %d", resp.StatusCode)
			}
			return nil
		},
	}
	c.Flags().StringVar(&url, "url", "http://127.0.0.1:8011/pool/status", "pool status URL")
	return c
}
