package cli

import (
	"syscall"

	"github.com/coder/llm-helper/internal/detect"
)

// gpuDeviceWarnings checks this machine's GPU device nodes for the user
// running doctor.
func gpuDeviceWarnings(accels []detect.Accelerator) []string {
	return amdDeviceWarnings(accels, "/dev", canReadWrite)
}

// canReadWrite asks the kernel whether this process's user and groups may
// open path for reading and writing, without opening it: opening /dev/kfd
// sets up a GPU context.
func canReadWrite(path string) bool {
	const rOK, wOK = 4, 2
	return syscall.Access(path, rOK|wOK) == nil
}
