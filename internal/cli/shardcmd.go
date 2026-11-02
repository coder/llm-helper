package cli

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/nodes"
	"github.com/coder/vllm-helper/internal/profile"
)

func newShardCmd() *cobra.Command {
	c := &cobra.Command{Use: "shard", Short: "Split one model across machines (Ray or vLLM multiprocessing)", GroupID: groupMultiNode}
	c.AddCommand(newShardInstallRayCmd(), newShardStartCmd(), newShardStopCmd(), newShardStatusCmd())
	return c
}

type shardCommon struct {
	nodesCSV string
	head     string
	sshKey   string
	dryRun   bool
}

func (s *shardCommon) register(c *cobra.Command) {
	c.Flags().StringVar(&s.nodesCSV, "nodes", "", "comma-separated node names")
	c.Flags().StringVar(&s.head, "head", "", "head node name (defaults to first --nodes entry)")
	c.Flags().StringVar(&s.sshKey, "ssh-key", "", "SSH private key for non-head nodes")
	c.Flags().BoolVar(&s.dryRun, "dry-run", false, "print commands instead of running them")
}

const (
	shardBackendRay = "ray"
	shardBackendMP  = "mp"
	shardBackendAll = "all"
)

func newShardInstallRayCmd() *cobra.Command {
	var common shardCommon
	var python string
	c := &cobra.Command{
		Use:   "install-ray --nodes n1,n2",
		Short: "Install Ray into the vLLM virtualenv on cluster nodes",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			return shardInstallRay(cfg, common, python)
		},
	}
	common.register(c)
	c.Flags().StringVar(&python, "python", "", "python executable to install Ray with (defaults next to VLLM_BIN)")
	return c
}

func newShardStartCmd() *cobra.Command {
	var common shardCommon
	var o shardStartOptions
	c := &cobra.Command{
		Use:   "start --profile <profile> --nodes n1,n2",
		Short: "Start a sharded vLLM server in the foreground",
		Args:  cobra.NoArgs,
		RunE:  func(*cobra.Command, []string) error { return runShardStart(common, o) },
	}
	common.register(c)
	c.Flags().StringVar(&o.profileName, "profile", "", "profile to serve")
	c.Flags().StringVar(&o.backend, "backend", shardBackendRay, "distributed executor backend: ray or mp")
	c.Flags().StringVar(&o.headIP, "head-ip", "", "IP workers use to reach the distributed head")
	c.Flags().StringVar(&o.iface, "iface", "", "network interface for distributed traffic")
	c.Flags().StringVar(&o.port, "port", "", "OpenAI API port (defaults to config PORT)")
	c.Flags().IntVar(&o.pp, "pp", 2, "pipeline parallel size")
	c.Flags().IntVar(&o.tp, "tp", 1, "tensor parallel size")
	c.Flags().IntVar(&o.rayPort, "ray-port", 6379, "ray head port")
	c.Flags().IntVar(&o.masterPort, "master-port", 29501, "vLLM multiprocessing master port")
	c.Flags().IntVar(&o.distributedTimeoutSeconds, "distributed-timeout-seconds", 0, "vLLM distributed init timeout (0 uses vLLM default)")
	c.Flags().StringVar(&o.serviceName, "service-name", "vllm.service", "vLLM service to stop when --stop-services is set")
	c.Flags().BoolVar(&o.stopServices, "stop-services", false, "stop existing vLLM services on all nodes before starting")
	c.Flags().BoolVar(&o.installRay, "install-ray", false, "install Ray into the vLLM virtualenv before starting")
	return c
}

func newShardStopCmd() *cobra.Command {
	var common shardCommon
	var backend string
	var serviceName string
	var startServices bool
	c := &cobra.Command{
		Use:   "stop --nodes n1,n2",
		Short: "Stop sharded runtimes on cluster nodes",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			backend, err := normalizeShardBackend(backend, true)
			if err != nil {
				return err
			}
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			members, head, err := loadShardMembers(common)
			if err != nil {
				return err
			}
			var firstErr error
			if startServices {
				if err := preflightServiceControl(members, head, common); err != nil {
					return err
				}
			}
			if backend == shardBackendAll || backend == shardBackendRay {
				if err := stopShardRay(cfg, members, head, common); err != nil && firstErr == nil {
					firstErr = err
				}
			}
			if backend == shardBackendAll || backend == shardBackendMP {
				if err := stopShardMP(members, head, common); err != nil && firstErr == nil {
					firstErr = err
				}
			}
			if startServices {
				for _, n := range members {
					if err := runOnNode(n, head.Name, common.sshKey, "sudo systemctl start "+shellQuote(serviceName), common.dryRun); err != nil && firstErr == nil {
						firstErr = err
					}
				}
			}
			return firstErr
		},
	}
	common.register(c)
	c.Flags().StringVar(&backend, "backend", shardBackendAll, "backend to stop: all, ray, or mp")
	c.Flags().StringVar(&serviceName, "service-name", "vllm.service", "service to start with --start-services")
	c.Flags().BoolVar(&startServices, "start-services", false, "start the normal vLLM service after stopping sharded runtimes")
	return c
}

func newShardStatusCmd() *cobra.Command {
	var common shardCommon
	var backend string
	var port string
	c := &cobra.Command{
		Use:   "status --nodes n1,n2",
		Short: "Show sharded endpoint and runtime status",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			backend, err := normalizeShardBackend(backend, true)
			if err != nil {
				return err
			}
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			if port == "" {
				port = cfg.Port()
			}
			members, head, err := loadShardMembers(common)
			if err != nil {
				return err
			}
			fmt.Print("endpoint health: ")
			_ = runLocal("curl -fsS -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:"+shellQuote(port)+"/health || true", common.dryRun)
			if backend == shardBackendAll || backend == shardBackendRay {
				if err := statusShardRay(cfg, members, head, common); err != nil {
					return err
				}
			}
			if backend == shardBackendAll || backend == shardBackendMP {
				if err := statusShardMP(members, head, common); err != nil {
					return err
				}
			}
			return nil
		},
	}
	common.register(c)
	c.Flags().StringVar(&backend, "backend", shardBackendAll, "backend to inspect: all, ray, or mp")
	c.Flags().StringVar(&port, "port", "", "API port to probe")
	return c
}

type shardStartOptions struct {
	profileName               string
	backend                   string
	headIP                    string
	iface                     string
	port                      string
	pp                        int
	tp                        int
	rayPort                   int
	masterPort                int
	distributedTimeoutSeconds int
	serviceName               string
	stopServices              bool
	installRay                bool
}

func runShardStart(common shardCommon, o shardStartOptions) error {
	backend, err := normalizeShardBackend(o.backend, false)
	if err != nil {
		return err
	}
	if backend == shardBackendMP && o.installRay {
		return errors.New("--install-ray only applies to --backend ray")
	}
	if backend == shardBackendMP {
		return runShardStartMP(common, o)
	}
	return runShardStartRay(common, o)
}

func preflightShardNodes(members []nodes.Node, head nodes.Node, common shardCommon) error {
	if common.dryRun {
		return nil
	}
	for _, n := range members {
		if n.Name == head.Name {
			continue
		}
		if err := runOnNode(n, head.Name, common.sshKey, "true", false); err != nil {
			return fmt.Errorf("cannot reach node %s over SSH (pass --ssh-key if it needs one): %w", n.Name, err)
		}
	}
	return nil
}

func preflightServiceControl(members []nodes.Node, head nodes.Node, common shardCommon) error {
	if common.dryRun {
		return nil
	}
	for _, n := range members {
		if err := runOnNode(n, head.Name, common.sshKey, "sudo -n true", false); err != nil {
			return fmt.Errorf("cannot control services on node %s: passwordless sudo is required for --stop-services/--start-services: %w", n.Name, err)
		}
	}
	return nil
}

func runShardStartRay(common shardCommon, o shardStartOptions) error {
	if o.profileName == "" {
		return errors.New("--profile is required")
	}
	if o.pp <= 0 || o.tp <= 0 {
		return errors.New("--pp and --tp must be positive")
	}
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	p, err := profile.Load(cfg.Dirs, o.profileName)
	if err != nil {
		return err
	}
	if err := requireVLLM(p, "shard"); err != nil {
		return err
	}
	members, head, err := loadShardMembers(common)
	if err != nil {
		return err
	}
	if err := preflightShardNodes(members, head, common); err != nil {
		return err
	}
	if o.installRay {
		if err := shardInstallRay(cfg, common, ""); err != nil {
			return err
		}
	}
	if o.stopServices {
		if err := preflightServiceControl(members, head, common); err != nil {
			return err
		}
		for _, n := range members {
			if err := runOnNode(n, head.Name, common.sshKey, "sudo systemctl stop "+shellQuote(o.serviceName)+" || true", common.dryRun); err != nil {
				return err
			}
		}
	}
	ray := rayBin(cfg)
	if o.headIP == "" && o.iface != "" {
		o.headIP, err = localInterfaceIP(o.iface)
		if err != nil {
			return err
		}
	}
	if o.headIP == "" {
		return errors.New("--head-ip is required unless --iface names a local interface with an IPv4 address")
	}

	for _, n := range members {
		_ = runOnNode(n, head.Name, common.sshKey, shellQuote(ray)+" stop --force || true", common.dryRun)
	}
	headStart := fmt.Sprintf("%s %s start --head --node-ip-address=%s --port=%d --disable-usage-stats", shardEnvPrefix(o.iface, o.headIP), shellQuote(ray), shellQuote(o.headIP), o.rayPort)
	if err := runOnNode(head, head.Name, common.sshKey, headStart, common.dryRun); err != nil {
		return err
	}
	for _, n := range members {
		if n.Name == head.Name {
			continue
		}
		workerStart := fmt.Sprintf("%s %s start --address=%s:%d --disable-usage-stats", shardEnvPrefix(o.iface, ""), shellQuote(ray), shellQuote(o.headIP), o.rayPort)
		if o.iface != "" {
			workerStart = fmt.Sprintf("node_ip=$(ip -o -4 addr show dev %s | awk '{split($4,a,\"/\"); print a[1]; exit}'); test -n \"$node_ip\" || { echo \"no IPv4 address on interface %s\" >&2; exit 1; }; %s %s start --address=%s:%d --node-ip-address=$node_ip --disable-usage-stats", shellQuote(o.iface), shellQuote(o.iface), shardWorkerEnvPrefix(o.iface), shellQuote(ray), shellQuote(o.headIP), o.rayPort)
		}
		if err := runOnNode(n, head.Name, common.sshKey, workerStart, common.dryRun); err != nil {
			return err
		}
	}

	spec, err := serveSpecFromProfile(cfg, p, o.port)
	if err != nil {
		return err
	}
	args, err := spec.args(shardRayServeArgs(o)...)
	if err != nil {
		return err
	}
	if common.dryRun {
		fmt.Println(shellJoin(redactServeArgs(args)))
		return nil
	}
	cmd := exec.Command(spec.Bin, args[1:]...)
	cmd.Env = shardEnv(serveEnvFromProfile(cfg, p), o.iface, o.headIP)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func shardInstallRay(cfg *config.Config, common shardCommon, python string) error {
	members, head, err := loadShardMembers(common)
	if err != nil {
		return err
	}
	if python == "" {
		python = filepath.Join(filepath.Dir(cfg.VLLMBin()), "python")
	}
	for _, n := range members {
		cmd := shellQuote(python) + " -m pip install 'ray[default]'"
		if err := runOnNode(n, head.Name, common.sshKey, cmd, common.dryRun); err != nil {
			return err
		}
	}
	return nil
}

func rayBin(cfg *config.Config) string { return filepath.Join(filepath.Dir(cfg.VLLMBin()), "ray") }

func loadShardMembers(common shardCommon) ([]nodes.Node, nodes.Node, error) {
	if strings.TrimSpace(common.nodesCSV) == "" {
		return nil, nodes.Node{}, errors.New("--nodes is required")
	}
	reg, err := nodes.Load()
	if err != nil {
		return nil, nodes.Node{}, err
	}
	var members []nodes.Node
	for _, name := range strings.Split(common.nodesCSV, ",") {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		n, ok := reg.Get(name)
		if !ok {
			return nil, nodes.Node{}, fmt.Errorf("unknown node %q -- add it with 'vllm-helper node add'", name)
		}
		members = append(members, n)
	}
	if len(members) == 0 {
		return nil, nodes.Node{}, errors.New("--nodes did not name any nodes")
	}
	headName := common.head
	if headName == "" {
		headName = members[0].Name
	}
	for _, n := range members {
		if n.Name == headName {
			return members, n, nil
		}
	}
	return nil, nodes.Node{}, fmt.Errorf("head node %q is not in --nodes", headName)
}

func runOnNode(n nodes.Node, headName, key, command string, dry bool) error {
	if n.Name == headName {
		return runLocal(command, dry)
	}
	target, err := sshTarget(n)
	if err != nil {
		return err
	}
	sshArgs := []string{"-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"}
	if key != "" {
		sshArgs = append(sshArgs, "-i", expandHome(key))
	}
	if dry {
		shown := append(append([]string{}, sshArgs...), target)
		fmt.Printf("ssh %s bash -s <<'SCRIPT'\n%s\nSCRIPT\n", shellJoin(shown), command)
		return nil
	}
	args := append(append([]string{}, sshArgs...), target, "bash", "-s")
	cmd := exec.Command("ssh", args...)
	cmd.Stdin = strings.NewReader(command + "\n")
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd.Run()
}

func runLocal(command string, dry bool) error {
	if dry {
		fmt.Println(command)
		return nil
	}
	cmd := exec.Command("bash", "-lc", command)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func sshTarget(n nodes.Node) (string, error) {
	if n.SSH != "" {
		return n.SSH, nil
	}
	u, err := url.Parse(n.Host)
	if err != nil {
		return "", err
	}
	host := u.Hostname()
	if host == "" {
		return "", fmt.Errorf("node %s has no SSH target and host %q is not a URL", n.Name, n.Host)
	}
	user := os.Getenv("USER")
	if user == "" {
		return host, nil
	}
	return user + "@" + host, nil
}

func expandHome(path string) string {
	if strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}

func localInterfaceIP(iface string) (string, error) {
	out, err := exec.Command("bash", "-lc", "ip -o -4 addr show dev "+shellQuote(iface)+" | awk '{split($4,a,\"/\"); print a[1]; exit}'").Output()
	if err != nil {
		return "", err
	}
	ip := strings.TrimSpace(string(out))
	if ip == "" {
		return "", fmt.Errorf("no IPv4 address on interface %s", iface)
	}
	return ip, nil
}

func shardToolchainEnv(base []string) []string {
	env := append([]string{}, base...)
	cudaHome := envValue(env, "CUDA_HOME")
	if cudaHome == "" {
		if info, err := os.Stat("/usr/local/cuda"); err == nil && info.IsDir() {
			cudaHome = "/usr/local/cuda"
			env = append(env, "CUDA_HOME="+cudaHome)
		}
	}
	if cudaHome != "" {
		bin := filepath.Join(cudaHome, "bin")
		if info, err := os.Stat(bin); err == nil && info.IsDir() {
			env = prependPath(env, bin)
		}
	}
	return env
}

func envValue(env []string, key string) string {
	prefix := key + "="
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			return strings.TrimPrefix(kv, prefix)
		}
	}
	return ""
}

func prependPath(env []string, dir string) []string {
	for i, kv := range env {
		if strings.HasPrefix(kv, "PATH=") {
			path := strings.TrimPrefix(kv, "PATH=")
			if path == "" {
				env[i] = "PATH=" + dir
			} else {
				env[i] = "PATH=" + dir + string(os.PathListSeparator) + path
			}
			return env
		}
	}
	return append(env, "PATH="+dir)
}

func shardToolchainShell() string {
	return `if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then export CUDA_HOME=/usr/local/cuda; fi; if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/bin" ]; then export PATH="$CUDA_HOME/bin:$PATH"; fi`
}

func shardEnv(base []string, iface, headIP string) []string {
	env := shardToolchainEnv(base)
	env = append(env, "RAY_ADDRESS=auto")
	if headIP != "" {
		env = append(env, "VLLM_HOST_IP="+headIP)
	}
	if iface != "" {
		env = append(env, "NCCL_SOCKET_IFNAME="+iface, "GLOO_SOCKET_IFNAME="+iface)
	}
	return env
}

func shardEnvPrefix(iface, hostIP string) string {
	parts := []string{"env"}
	if hostIP != "" {
		parts = append(parts, "VLLM_HOST_IP="+shellQuote(hostIP))
	}
	if iface != "" {
		parts = append(parts, "NCCL_SOCKET_IFNAME="+shellQuote(iface), "GLOO_SOCKET_IFNAME="+shellQuote(iface))
	}
	return strings.Join(parts, " ")
}

func shardWorkerEnvPrefix(iface string) string {
	parts := []string{"env", "VLLM_HOST_IP=$node_ip"}
	if iface != "" {
		parts = append(parts, "NCCL_SOCKET_IFNAME="+shellQuote(iface), "GLOO_SOCKET_IFNAME="+shellQuote(iface))
	}
	return strings.Join(parts, " ")
}

func shellQuote(s string) string {
	if s == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'"
}

func shellJoin(args []string) string {
	quoted := make([]string, 0, len(args))
	for _, a := range args {
		quoted = append(quoted, shellQuote(a))
	}
	return strings.Join(quoted, " ")
}
