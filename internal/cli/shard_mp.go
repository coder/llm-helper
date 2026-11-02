package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/nodes"
	"github.com/coder/vllm-helper/internal/profile"
)

func normalizeShardBackend(backend string, allowAll bool) (string, error) {
	backend = strings.ToLower(strings.TrimSpace(backend))
	if backend == "" {
		if allowAll {
			return shardBackendAll, nil
		}
		return shardBackendRay, nil
	}
	switch backend {
	case shardBackendRay, shardBackendMP:
		return backend, nil
	case "multiprocessing":
		return shardBackendMP, nil
	case shardBackendAll:
		if allowAll {
			return shardBackendAll, nil
		}
	}
	if allowAll {
		return "", errors.New("--backend must be one of: all, ray, mp")
	}
	return "", errors.New("--backend must be one of: ray, mp")
}

func shardRayServeArgs(o shardStartOptions) []string {
	args := []string{
		"--distributed-executor-backend", shardBackendRay,
		"--pipeline-parallel-size", strconv.Itoa(o.pp),
		"--tensor-parallel-size", strconv.Itoa(o.tp),
	}
	if o.distributedTimeoutSeconds > 0 {
		args = append(args, "--distributed-timeout-seconds", strconv.Itoa(o.distributedTimeoutSeconds))
	}
	return args
}

func runShardStartMP(common shardCommon, o shardStartOptions) error {
	if o.profileName == "" {
		return errors.New("--profile is required")
	}
	if o.pp <= 0 || o.tp <= 0 {
		return errors.New("--pp and --tp must be positive")
	}
	if o.masterPort <= 0 {
		return errors.New("--master-port must be positive")
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
	if len(members) < 2 {
		return errors.New("--backend mp requires at least two nodes")
	}
	if members[0].Name != head.Name {
		return errors.New("--backend mp requires the head node to be first in --nodes so it can run node-rank 0")
	}
	if err := preflightShardNodes(members, head, common); err != nil {
		return err
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
		_ = runOnNode(n, head.Name, common.sshKey, shardMPStopCommand(), common.dryRun)
	}

	spec, err := serveSpecFromProfile(cfg, p, o.port)
	if err != nil {
		return err
	}
	startedWorkers := false
	for rank, n := range members {
		if n.Name == head.Name {
			continue
		}
		args, err := shardMPServeArgs(spec, len(members), rank, o, true)
		if err != nil {
			return err
		}
		commandArgs := args
		if common.dryRun {
			commandArgs = redactServeArgs(args)
		}
		if err := runOnNode(n, head.Name, common.sshKey, shardMPWorkerStartCommand(commandArgs, o.iface), common.dryRun); err != nil {
			if startedWorkers && !common.dryRun {
				_ = stopShardMP(members, head, common)
			}
			return err
		}
		startedWorkers = true
	}

	headArgs, err := shardMPServeArgs(spec, len(members), 0, o, false)
	if err != nil {
		if startedWorkers && !common.dryRun {
			_ = stopShardMP(members, head, common)
		}
		return err
	}
	if common.dryRun {
		fmt.Println(shardEnvPrefix(o.iface, o.headIP), shellJoin(redactServeArgs(headArgs)))
		return nil
	}
	defer func() { _ = stopShardMP(members, head, common) }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	cmd := exec.CommandContext(ctx, spec.Bin, headArgs[1:]...)
	cmd.Env = shardMPEnv(serveEnvFromProfile(cfg, p), o.iface, o.headIP)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func shardMPServeArgs(spec serveSpec, nnodes, rank int, o shardStartOptions, headless bool) ([]string, error) {
	args := []string{
		"--distributed-executor-backend", shardBackendMP,
		"--pipeline-parallel-size", strconv.Itoa(o.pp),
		"--tensor-parallel-size", strconv.Itoa(o.tp),
		"--nnodes", strconv.Itoa(nnodes),
		"--node-rank", strconv.Itoa(rank),
		"--master-addr", o.headIP,
		"--master-port", strconv.Itoa(o.masterPort),
	}
	if o.distributedTimeoutSeconds > 0 {
		args = append(args, "--distributed-timeout-seconds", strconv.Itoa(o.distributedTimeoutSeconds))
	}
	if headless {
		args = append(args, "--headless")
	}
	return spec.args(args...)
}

func stopShardRay(cfg *config.Config, members []nodes.Node, head nodes.Node, common shardCommon) error {
	ray := rayBin(cfg)
	for _, n := range members {
		if err := runOnNode(n, head.Name, common.sshKey, shellQuote(ray)+" stop --force || true", common.dryRun); err != nil {
			return err
		}
	}
	return nil
}

func statusShardRay(cfg *config.Config, members []nodes.Node, head nodes.Node, common shardCommon) error {
	ray := rayBin(cfg)
	for _, n := range members {
		fmt.Printf("\n== %s ray ==\n", n.Name)
		cmd := shellQuote(ray) + " status 2>/dev/null | head -n 25 || true"
		if err := runOnNode(n, head.Name, common.sshKey, cmd, common.dryRun); err != nil {
			return err
		}
	}
	return nil
}

func stopShardMP(members []nodes.Node, head nodes.Node, common shardCommon) error {
	for _, n := range members {
		if err := runOnNode(n, head.Name, common.sshKey, shardMPStopCommand(), common.dryRun); err != nil {
			return err
		}
	}
	return nil
}

func statusShardMP(members []nodes.Node, head nodes.Node, common shardCommon) error {
	for _, n := range members {
		fmt.Printf("\n== %s mp ==\n", n.Name)
		if err := runOnNode(n, head.Name, common.sshKey, shardMPStatusCommand(), common.dryRun); err != nil {
			return err
		}
	}
	return nil
}

func shardMPEnv(base []string, iface, headIP string) []string {
	env := shardToolchainEnv(base)
	if headIP != "" {
		env = append(env, "VLLM_HOST_IP="+headIP)
	}
	if iface != "" {
		env = append(env, "NCCL_SOCKET_IFNAME="+iface, "GLOO_SOCKET_IFNAME="+iface)
	}
	return env
}

func shardMPWorkerStartCommand(args []string, iface string) string {
	prefix := shardEnvPrefix(iface, "")
	lines := []string{
		shardMPStateVars(),
		shardToolchainShell(),
		"if [ -f \"$pid\" ] && kill -0 \"$(cat \"$pid\")\" 2>/dev/null; then echo \"shard worker already running: $(cat \"$pid\")\" >&2; exit 1; fi",
		"rm -f \"$pid\"",
	}
	if iface != "" {
		lines = append(lines, nodeIPv4Shell(iface))
		prefix = shardWorkerEnvPrefix(iface)
	}
	lines = append(lines,
		fmt.Sprintf("nohup %s %s >\"$log\" 2>&1 &", prefix, shellJoin(args)),
		"echo $! >\"$pid\"",
		"sleep 2",
		"if ! kill -0 \"$(cat \"$pid\")\" 2>/dev/null; then echo \"shard worker exited; last log:\" >&2; tail -n 80 \"$log\" >&2 || true; rm -f \"$pid\"; exit 1; fi",
		"echo \"started shard worker pid $(cat \"$pid\") log $log\"",
	)
	return strings.Join(lines, "\n")
}

func shardMPStopCommand() string {
	return strings.Join([]string{
		shardMPStateVars(),
		"if [ -f \"$pid\" ]; then old=\"$(cat \"$pid\")\"; if kill -0 \"$old\" 2>/dev/null; then kill \"$old\" 2>/dev/null || true; for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 \"$old\" 2>/dev/null || break; sleep 1; done; kill -0 \"$old\" 2>/dev/null && kill -9 \"$old\" 2>/dev/null || true; fi; rm -f \"$pid\"; fi",
		"for old in $(pgrep -f 'v[l]lm serve .*--distributed-executor-backend mp.*--nnodes' || true); do kill \"$old\" 2>/dev/null || true; done",
	}, "\n")
}

func shardMPStatusCommand() string {
	return strings.Join([]string{
		shardMPStateVars(),
		"if [ -f \"$pid\" ] && kill -0 \"$(cat \"$pid\")\" 2>/dev/null; then echo \"mp worker: running pid $(cat \"$pid\")\"; else echo \"mp worker: stopped\"; fi",
		"ids=\"$(pgrep -f 'v[l]lm serve .*--distributed-executor-backend mp.*--nnodes' || true)\"; if [ -n \"$ids\" ]; then echo \"$ids\" | paste -sd, - | xargs -r ps -o pid,etime,stat,comm -p; fi",
		"test ! -f \"$log\" || { echo \"last log lines:\"; tail -n 5 \"$log\" || true; }",
	}, "\n")
}

func shardMPStateVars() string {
	return "state=\"${XDG_STATE_HOME:-$HOME/.local/state}/vllm-helper/shard\"; mkdir -p \"$state\"; pid=\"$state/mp-worker.pid\"; log=\"$state/mp-worker.log\""
}

func nodeIPv4Shell(iface string) string {
	return fmt.Sprintf("node_ip=$(ip -o -4 addr show dev %s | awk '{split($4,a,\"/\"); print a[1]; exit}'); test -n \"$node_ip\" || { echo \"no IPv4 address on interface %s\" >&2; exit 1; };", shellQuote(iface), shellQuote(iface))
}
