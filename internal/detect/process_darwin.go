package detect

import (
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// listProcesses shells out to ps to enumerate the machine, since macOS has no
// /proc. -ww stops ps from truncating long command lines, which vLLM's always
// are.
//
// ps joins argv with spaces, so an argument that contains one can't be
// recovered from it -- and profiles legitimately hold such values, like a
// --chat-template path under a directory with a space in the name. For the
// processes that matter we re-read argv from the kernel, which keeps the
// boundaries intact; ps's split stands in when that isn't permitted.
func listProcesses() []rawProc {
	out, err := exec.Command("ps", "-Awwo", "pid=,command=").Output()
	if err != nil {
		return nil
	}
	var procs []rawProc
	for line := range strings.SplitSeq(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		pid, err := strconv.Atoi(fields[0])
		if err != nil {
			continue
		}
		args := fields[1:]
		// Joining argv with spaces doesn't change whether "vllm serve"
		// appears in it, so this filter is safe to apply to the lossy form,
		// and it keeps the syscall off every other process on the machine.
		if looksLikeVLLM(args) {
			if exact, err := procArgs(pid); err == nil {
				args = exact
			}
		}
		procs = append(procs, rawProc{PID: pid, Args: args})
	}
	return procs
}

// enrichPlatform looks for a launchd job owning the process. Containers are
// never visible here: on macOS they run inside a Linux VM, so anything ps can
// see is a host process. ScanContainers covers the rest.
func enrichPlatform(s *Server) {
	if s.VLLMBin == "" {
		for _, a := range s.Cmdline {
			// The venv's python sits next to its console scripts.
			if cand := filepath.Join(filepath.Dir(a), "vllm"); strings.Contains(a, "/bin/") && isExecutable(cand) {
				s.VLLMBin = cand
				break
			}
		}
	}
	if label := launchdLabelFor(s.PID); label != "" {
		s.SystemdUnit = label
		s.UserScope = true // launchctl list only reports the caller's domain
	} else if label := systemLabelFor(s.PID); label != "" {
		s.SystemdUnit = label
	}
}

// systemLabelFor returns the LaunchDaemon label running pid, or "". 'launchctl
// list' cannot see the system domain without root, but 'launchctl print
// system' lists its services, with their pids, to anyone.
func systemLabelFor(pid int) string {
	out, err := exec.Command("launchctl", "print", "system").Output()
	if err != nil {
		return ""
	}
	return labelInServices(string(out), pid)
}

// labelInServices finds pid in the "services = { ... }" block of 'launchctl
// print' output, whose lines are "<pid> <last exit> <label>".
func labelInServices(text string, pid int) string {
	want := strconv.Itoa(pid)
	in := false
	for line := range strings.SplitSeq(text, "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "services = {":
			in = true
			continue
		case in && trimmed == "}":
			return ""
		case !in:
			continue
		}
		fields := strings.Fields(trimmed)
		if len(fields) == 3 && fields[0] == want {
			return fields[2]
		}
	}
	return ""
}

// launchdLabelFor returns the launchd job label running pid, or "" if the
// process isn't managed by launchd in this user's domain.
func launchdLabelFor(pid int) string {
	out, err := exec.Command("launchctl", "list").Output()
	if err != nil {
		return ""
	}
	want := strconv.Itoa(pid)
	for line := range strings.SplitSeq(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 3 || fields[0] != want {
			continue
		}
		return fields[2]
	}
	return ""
}
