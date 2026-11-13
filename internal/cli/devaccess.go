package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/coder/llm-helper/internal/detect"
)

// amdDeviceWarnings warns when an amdgpu GPU is on the bus but this user
// cannot open the device nodes a server reaches it through: /dev/kfd for
// ROCm and /dev/dri/renderD* for ROCm and Vulkan. Both are root:render 0660
// on the usual distributions, so a user outside the render group gets a
// server that finds no GPU ("no ROCm-capable device is detected") rather than
// a permission error. A node that does not exist is not reported: a kernel
// without KFD still serves through Vulkan.
//
// devDir stands in for /dev and canOpen for the access check, so the rule can
// be tested without the devices.
func amdDeviceWarnings(accels []detect.Accelerator, devDir string, canOpen func(path string) bool) []string {
	if !slices.ContainsFunc(accels, func(a detect.Accelerator) bool { return a.Driver == "amdgpu" }) {
		return nil
	}
	nodes, _ := filepath.Glob(filepath.Join(devDir, "dri", "renderD*"))
	nodes = append([]string{filepath.Join(devDir, "kfd")}, nodes...)
	var denied []string
	for _, n := range nodes {
		if _, err := os.Stat(n); err == nil && !canOpen(n) {
			denied = append(denied, n)
		}
	}
	if len(denied) == 0 {
		return nil
	}
	return []string{fmt.Sprintf(
		"an amdgpu GPU is present, but this user cannot open %s, so a server on ROCm or Vulkan will not find it. "+
			"Add the user to the groups that own them with 'sudo usermod -aG render,video $USER', then log out and back in "+
			"(with SERVICE_SCOPE=user and lingering, reboot so the user's service manager picks up the groups).",
		strings.Join(denied, ", "))}
}
