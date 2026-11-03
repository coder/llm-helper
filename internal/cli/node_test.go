package cli_test

import (
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/nodes"
)

// Re-adding a node is how its URL gets corrected. It used to replace the
// whole entry, so a re-add without --ssh dropped the only SSH target that
// reached the node, along with everything a probe had learned about it.
func TestRunNodeAddKeepsWhatItIsNotTold(t *testing.T) {
	newTree(t)
	reg, err := nodes.Load()
	if err != nil {
		t.Fatal(err)
	}
	reg.Add(nodes.Node{
		Name: "spark2", Host: "http://192.168.1.234:8001", SSH: "coder@10.44.0.2",
		Labels: map[string]string{"rack": "a"},
		Caps: nodes.Capabilities{Accelerator: "cuda", GPU: "NVIDIA GB10", GPUs: 1, PoolBytes: 130e9,
			NICs: []nodes.NIC{{Name: "enp1s0f1np1", Speed: "200G", MTU: 9000}}},
	})
	if err := reg.Save(); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := runCLI(t, "node", "add", "spark2", "--host", "http://192.168.1.235:8001", "--no-probe")
	if code != 0 {
		t.Fatalf("node add exited %d: %s", code, stderr)
	}
	if !strings.Contains(stdout, "Updated spark2") {
		t.Errorf("stdout = %q, want it to say the node was updated", stdout)
	}
	n := loadNode(t, "spark2")
	if n.Host != "http://192.168.1.235:8001" || n.SSH != "coder@10.44.0.2" || n.Labels["rack"] != "a" ||
		n.Caps.GPU != "NVIDIA GB10" || len(n.Caps.NICs) != 1 {
		t.Errorf("after a re-add without --ssh: %+v", n)
	}

	// --ssh given, even empty, is an instruction.
	if code, _, stderr := runCLI(t, "node", "add", "spark2", "--host", "http://192.168.1.235:8001", "--ssh", "", "--no-probe"); code != 0 {
		t.Fatalf("node add --ssh '' exited %d: %s", code, stderr)
	}
	if n := loadNode(t, "spark2"); n.SSH != "" {
		t.Errorf("--ssh '' left SSH = %q", n.SSH)
	}

	if code, stdout, _ := runCLI(t, "node", "add", "new", "--host", "http://10.0.0.9:8001", "--no-probe"); code != 0 || !strings.Contains(stdout, "Added new") {
		t.Errorf("a new node: exit %d, stdout %q", code, stdout)
	}
}

func loadNode(t *testing.T, name string) nodes.Node {
	t.Helper()
	reg, err := nodes.Load()
	if err != nil {
		t.Fatal(err)
	}
	n, ok := reg.Get(name)
	if !ok {
		t.Fatalf("node %q is gone", name)
	}
	return n
}
