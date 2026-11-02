package cli

import (
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/nodes"
	"github.com/coder/vllm-helper/internal/provider"
)

func TestShardRayEnvPrefixesUseNodeLocalHostIP(t *testing.T) {
	head := shardEnvPrefix("enp1s0f1np1", "10.44.0.1")
	for _, want := range []string{
		"VLLM_HOST_IP='10.44.0.1'",
		"NCCL_SOCKET_IFNAME='enp1s0f1np1'",
		"GLOO_SOCKET_IFNAME='enp1s0f1np1'",
	} {
		if !strings.Contains(head, want) {
			t.Fatalf("head env prefix missing %q: %s", want, head)
		}
	}

	worker := shardWorkerEnvPrefix("enp1s0f1np1")
	for _, want := range []string{
		"VLLM_HOST_IP=$node_ip",
		"NCCL_SOCKET_IFNAME='enp1s0f1np1'",
		"GLOO_SOCKET_IFNAME='enp1s0f1np1'",
	} {
		if !strings.Contains(worker, want) {
			t.Fatalf("worker env prefix missing %q: %s", want, worker)
		}
	}
	if strings.Contains(worker, "'$node_ip'") {
		t.Fatalf("worker node IP must be expanded by the remote shell: %s", worker)
	}

	if got := shardEnvPrefix("", ""); got != "env" {
		t.Fatalf("empty env prefix = %q, want env", got)
	}
}

func TestShardServeEnvUsesHeadIP(t *testing.T) {
	env := shardEnv([]string{"BASE=x"}, "enp1s0f1np1", "10.44.0.1")
	for _, want := range []string{
		"BASE=x",
		"RAY_ADDRESS=auto",
		"VLLM_HOST_IP=10.44.0.1",
		"NCCL_SOCKET_IFNAME=enp1s0f1np1",
		"GLOO_SOCKET_IFNAME=enp1s0f1np1",
	} {
		if !hasString(env, want) {
			t.Fatalf("env missing %q: %v", want, env)
		}
	}
}

func TestRunOnNodeDryRunUsesScriptStdin(t *testing.T) {
	out, err := captureStdout(t, func() error {
		return runOnNode(nodes.Node{Name: "worker", SSH: "coder@example"}, "head", "/tmp/key with space", "echo \"$HOME\"; true", true)
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"ssh '-o' 'BatchMode=yes' '-o' 'StrictHostKeyChecking=accept-new' '-i' '/tmp/key with space' 'coder@example' bash -s <<'SCRIPT'",
		"echo \"$HOME\"; true",
		"SCRIPT",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("dry-run output missing %q: %s", want, out)
		}
	}
	if strings.Contains(out, "bash -lc") {
		t.Fatalf("remote commands should be piped to bash -s, not embedded in bash -lc: %s", out)
	}
}

func TestShardBackendAliases(t *testing.T) {
	tests := map[string]string{
		"":                shardBackendRay,
		"ray":             shardBackendRay,
		"mp":              shardBackendMP,
		"multiprocessing": shardBackendMP,
	}
	for in, want := range tests {
		got, err := normalizeShardBackend(in, false)
		if err != nil {
			t.Fatalf("normalizeShardBackend(%q): %v", in, err)
		}
		if got != want {
			t.Fatalf("normalizeShardBackend(%q) = %q, want %q", in, got, want)
		}
	}
	if got, err := normalizeShardBackend("", true); err != nil || got != shardBackendAll {
		t.Fatalf("empty stop/status backend = %q, %v; want all", got, err)
	}
}

func TestShardMPServeArgs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("serveSpec executable-bit validation is POSIX-oriented")
	}
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	spec := serveSpec{Spec: provider.Spec{
		Bin:            bin,
		Model:          "org/model",
		Port:           "8021",
		MaxModelLen:    "4096",
		MemUtilization: "0.80",
		APIKey:         "secret",
	}}
	args, err := shardMPServeArgs(spec, 2, 1, shardStartOptions{
		pp:                        1,
		tp:                        2,
		headIP:                    "10.44.0.1",
		masterPort:                29501,
		distributedTimeoutSeconds: 600,
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	joined := shellJoin(args)
	for _, want := range []string{
		"'--distributed-executor-backend' 'mp'",
		"'--nnodes' '2'",
		"'--node-rank' '1'",
		"'--master-addr' '10.44.0.1'",
		"'--master-port' '29501'",
		"'--distributed-timeout-seconds' '600'",
		"'--headless'",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("MP args missing %q: %s", want, joined)
		}
	}
}

func TestShardMPWorkerStartCommand(t *testing.T) {
	cmd := shardMPWorkerStartCommand([]string{"/opt/vllm", "serve", "org/model", "--api-key", "<redacted>"}, "enp1s0f1np1")
	for _, want := range []string{
		"mp-worker.pid",
		"mp-worker.log",
		"node_ip=$(ip -o -4 addr show dev",
		"VLLM_HOST_IP=$node_ip",
		"nohup env",
		"--api-key",
		"<redacted>",
	} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("worker command missing %q: %s", want, cmd)
		}
	}
	if strings.Contains(cmd, "'$node_ip'") {
		t.Fatalf("worker node IP must be expanded by the remote shell: %s", cmd)
	}
}

func TestShardMPShellSnippetsParse(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bash parser check is for target platforms")
	}
	for name, script := range map[string]string{
		"worker-start": shardMPWorkerStartCommand([]string{"/opt/vllm", "serve", "org/model"}, "enp1s0f1np1"),
		"stop":         shardMPStopCommand(),
		"status":       shardMPStatusCommand(),
	} {
		t.Run(name, func(t *testing.T) {
			cmd := exec.Command("bash", "-n")
			cmd.Stdin = strings.NewReader(script)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("bash -n: %v\n%s\nscript:\n%s", err, out, script)
			}
		})
	}
}

func hasString(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

func captureStdout(t *testing.T, fn func() error) (string, error) {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { os.Stdout = old }()
	os.Stdout = w

	callErr := fn()
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	os.Stdout = old
	b, readErr := io.ReadAll(r)
	if closeErr := r.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if readErr != nil {
		t.Fatal(readErr)
	}
	return string(b), callErr
}
