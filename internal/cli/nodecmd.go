package cli

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/nodes"
)

func newNodeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "node",
		Short:   "Register machines by HTTP URL and SSH target",
		Long:    "Register machines by HTTP URL (--host) and SSH target (--ssh).\nRecipes, pools and sharding name them with --nodes.",
		GroupID: groupMultiNode,
	}
	cmd.AddCommand(newNodeAddCmd(), newNodeListCmd(), newNodeRemoveCmd(), newNodeProbeCmd())
	return cmd
}

func newNodeAddCmd() *cobra.Command {
	var host, sshFlag, key string
	var noProbe bool
	c := &cobra.Command{Use: "add <name>", Args: cobra.ExactArgs(1), Short: "Add a node",
		Long: "Add a node, then describe its hardware over SSH (memory and NVIDIA GPUs) so 'node ls' can show it.\n" +
			"A node that cannot be reached is still added; describe it later with 'node probe'.\n" +
			"Adding a node that is already registered updates its --host, and its SSH target only when --ssh is given.",
		RunE: func(cmd *cobra.Command, args []string) error {
			if host == "" {
				return errors.New("--host is required")
			}
			r, err := nodes.Load()
			if err != nil {
				return err
			}
			// Re-adding a node is how its URL gets corrected, and that must
			// not cost what was learned about it before: a re-add without
			// --ssh used to drop the only SSH target that reached it.
			n, existed := r.Get(args[0])
			n.Name, n.Host = args[0], host
			if cmd.Flags().Changed("ssh") {
				n.SSH = sshFlag
			}
			if !noProbe {
				if caps, err := probeNode(n, expandHome(key)); err != nil {
					fmt.Fprintf(os.Stderr, "warning: could not describe %s's hardware: %v\n  run '%s node probe %s' once it is reachable\n", n.Name, err, cmdName, n.Name)
				} else {
					caps.NICs = n.Caps.NICs
					n.Caps = caps
				}
			}
			r.Add(n)
			if err := r.Save(); err != nil {
				return err
			}
			verb := "Added"
			if existed {
				verb = "Updated"
			}
			fmt.Printf("%s %s -> %s\n", verb, args[0], host)
			if t, err := sshTarget(n); err == nil {
				fmt.Printf("  ssh %s\n", t)
			}
			if n.Caps.Accelerator != "" {
				fmt.Printf("  hardware: %s\n", describeCaps(n.Caps))
			}
			return nil
		},
	}
	c.Flags().StringVar(&host, "host", "", "base URL (scheme://host:port)")
	c.Flags().StringVar(&sshFlag, "ssh", "", "SSH target for orchestration (user@host)")
	c.Flags().StringVar(&key, "ssh-key", "", "SSH private key for describing the node")
	c.Flags().BoolVar(&noProbe, "no-probe", false, "add the node without describing its hardware")
	return c
}

func newNodeProbeCmd() *cobra.Command {
	var key string
	c := &cobra.Command{Use: "probe [name...]", Short: "Describe nodes' hardware again (all nodes when none are named)",
		RunE: func(_ *cobra.Command, args []string) error {
			r, err := nodes.Load()
			if err != nil {
				return err
			}
			var list []nodes.Node
			if len(args) == 0 {
				list = r.List()
			}
			for _, name := range args {
				n, ok := r.Get(name)
				if !ok {
					return fmt.Errorf("no node %q in the registry; add it with '%s node add'", name, cmdName)
				}
				list = append(list, n)
			}
			if len(list) == 0 {
				fmt.Println("No nodes.")
				return nil
			}
			var failed []string
			for _, n := range list {
				caps, err := probeNode(n, expandHome(key))
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s: %v\n", n.Name, err)
					failed = append(failed, n.Name)
					continue
				}
				// What was learned another way (NICs) is not this probe's
				// to discard.
				caps.NICs = n.Caps.NICs
				n.Caps = caps
				r.Add(n)
				fmt.Printf("%s: %s\n", n.Name, describeCaps(caps))
			}
			if err := r.Save(); err != nil {
				return err
			}
			if len(failed) > 0 {
				return fmt.Errorf("could not describe %s (pass --ssh-key if it needs one)", strings.Join(failed, ", "))
			}
			return nil
		},
	}
	c.Flags().StringVar(&key, "ssh-key", "", "SSH private key for reaching the nodes")
	return c
}

// nodeProbeTimeout bounds one node's description. It is a few shell reads
// and one nvidia-smi call, so anything slower is a node that is not answering.
const nodeProbeTimeout = 30 * time.Second

// probeNode describes a node's hardware. This machine is probed directly, so
// it honours the server's CUDA_VISIBLE_DEVICES; another is read over SSH.
func probeNode(n nodes.Node, key string) (nodes.Capabilities, error) {
	if n.IsLocal() {
		return capsFrom(detect.Probe()), nil
	}
	target, err := sshTarget(n)
	if err != nil {
		return nodes.Capabilities{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), nodeProbeTimeout)
	defer cancel()
	args := []string{"-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"}
	if key != "" {
		args = append(args, "-i", key)
	}
	args = append(args, target, "bash", "-s")
	cmd := exec.CommandContext(ctx, "ssh", args...)
	cmd.Stdin = strings.NewReader(detect.RemoteScript())
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		if msg := strings.TrimSpace(errb.String()); msg != "" {
			return nodes.Capabilities{}, fmt.Errorf("ssh %s: %w: %s", target, err, msg)
		}
		return nodes.Capabilities{}, fmt.Errorf("ssh %s: %w", target, err)
	}
	hw := detect.ParseRemote(out.String())
	if hw.TotalMemBytes <= 0 {
		return nodes.Capabilities{}, fmt.Errorf("%s did not report its memory (not a Linux machine?)", target)
	}
	return capsFrom(hw), nil
}

// capsFrom summarizes a node's hardware for the registry.
func capsFrom(hw detect.Hardware) nodes.Capabilities {
	c := nodes.Capabilities{Accelerator: "cpu", GPU: hw.GPUSummary(), GPUs: len(hw.GPUs), HardwareID: hw.HardwareID, Arch: hw.Arch}
	switch {
	case hw.CUDA:
		c.Accelerator = "cuda"
	case hw.Metal:
		c.Accelerator = "metal"
	}
	if pool, _ := hw.Pool(); pool > 0 {
		c.PoolBytes = uint64(pool)
	}
	return c
}

// describeCaps renders a node's hardware in one line, or "-" before it has
// been probed.
func describeCaps(c nodes.Capabilities) string {
	if c.Accelerator == "" {
		return "-"
	}
	parts := []string{c.Accelerator}
	if c.GPU != "" {
		parts = append(parts, c.GPU)
	}
	if c.PoolBytes > 0 {
		pool := fmt.Sprintf("%d GB", int(float64(c.PoolBytes)/1e9))
		// PoolBytes is what one server loads into, which on a box of
		// discrete cards is one card, not the box.
		if c.GPUs > 1 {
			pool += " per server"
		}
		parts = append(parts, pool)
	}
	return strings.Join(parts, ", ")
}

func newNodeListCmd() *cobra.Command {
	return &cobra.Command{Use: "ls", Short: "List nodes", RunE: func(*cobra.Command, []string) error {
		r, err := nodes.Load()
		if err != nil {
			return err
		}
		list := r.List()
		if len(list) == 0 {
			fmt.Println("No nodes.")
			return nil
		}
		printNodes(os.Stdout, list)
		return nil
	}}
}

func newNodeRemoveCmd() *cobra.Command {
	return &cobra.Command{Use: "rm <name>", Args: cobra.ExactArgs(1), Short: "Remove a node", RunE: func(_ *cobra.Command, args []string) error {
		r, err := nodes.Load()
		if err != nil {
			return err
		}
		r.Remove(args[0])
		return r.Save()
	}}
}

// printNodes lists each node with where it serves and how it is reached. The
// SSH column is what orchestration actually dials, including the target
// derived from --host when no --ssh was given, so a wrong target shows up
// here rather than as a failed launch.
func printNodes(w io.Writer, list []nodes.Node) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tHOST\tSSH\tHARDWARE")
	for _, n := range list {
		target, err := sshTarget(n)
		switch {
		case err != nil:
			target = "-"
		case n.SSH == "":
			target += " (from --host)"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", n.Name, n.Host, target, describeCaps(n.Caps))
	}
	_ = tw.Flush()
}
