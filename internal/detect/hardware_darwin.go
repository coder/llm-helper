package detect

import (
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

func sysctlStr(key string) string {
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// totalMemBytes reads hw.memsize. macOS has no /proc/meminfo, and without this
// the weights budget comes out as 0 and `search` calls every model too big.
func totalMemBytes() int64 {
	n, err := strconv.ParseInt(sysctlStr("hw.memsize"), 10, 64)
	if err != nil {
		return 0
	}
	return n
}

var pageSizeRe = regexp.MustCompile(`page size of (\d+) bytes`)

// AvailableMemMB reports memory that can be allocated right now, counted the
// way psutil -- and therefore vLLM's CPU backend -- counts it on macOS: free
// pages plus the reclaimable ones.
func AvailableMemMB() int {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return 0
	}
	text := string(out)

	pageSize := int64(4096)
	if m := pageSizeRe.FindStringSubmatch(text); m != nil {
		if n, err := strconv.ParseInt(m[1], 10, 64); err == nil && n > 0 {
			pageSize = n
		}
	}

	var pages int64
	for _, key := range []string{"Pages free", "Pages inactive", "Pages speculative"} {
		m := regexp.MustCompile(regexp.QuoteMeta(key) + `:\s+(\d+)`).FindStringSubmatch(text)
		if m == nil {
			continue
		}
		if n, err := strconv.ParseInt(m[1], 10, 64); err == nil {
			pages += n
		}
	}
	return int(pages * pageSize / 1024 / 1024)
}

// platformFacts takes the macOS readings beyond memory. An Apple Silicon
// SoC's brand string names its GPU too; an Intel Mac's names only its CPU,
// so only the former is recorded, and the apple_silicon definition decides
// what it means.
func platformFacts(f *facts) {
	f.os, f.arch = runtime.GOOS, runtime.GOARCH
	f.system = sysctlStr("hw.model")
	if name := sysctlStr("machdep.cpu.brand_string"); strings.HasPrefix(name, "Apple ") {
		f.cpuBrand = name
	}
}
