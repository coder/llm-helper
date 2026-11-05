package detect_test

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/detect"
)

func TestScanContainersAgainstTheRealRuntime(t *testing.T) {
	t.Parallel()

	var runtimes []string
	for _, r := range []string{"docker", "podman"} {
		if _, err := exec.LookPath(r); err == nil {
			runtimes = append(runtimes, r)
		}
	}
	if len(runtimes) == 0 {
		t.Skip("no container runtime installed")
	}

	for _, c := range detect.ScanContainers() {
		if c.Runtime != "docker" && c.Runtime != "podman" {
			t.Errorf("Runtime = %q, want the runtime that reported it", c.Runtime)
		}
		blob := strings.ToLower(c.Image + " " + c.Command + " " + c.Name)
		if !strings.Contains(blob, "vllm") {
			t.Errorf("container %+v doesn't look vLLM-related and shouldn't have been returned", c)
		}
	}
}

// A runtime that isn't installed is not a failure: doctor still has to run.
func TestInspectArgsWithoutARuntime(t *testing.T) {
	t.Parallel()

	if got := detect.InspectArgs("definitely-not-a-container-runtime", "abc123"); got != nil {
		t.Errorf("InspectArgs = %#v, want nil", got)
	}
}
