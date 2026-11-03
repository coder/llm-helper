package detect

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// totalMemBytes reports installed RAM. Bytes, not gigabytes: the budget
// arithmetic mixes this with sizes read from the Hub, which are decimal, and
// rounding to whole binary gigabytes here is how a 7.4% error crept into the
// fit check.
func totalMemBytes() int64 {
	return meminfoBytes("MemTotal:")
}

func meminfoBytes(key string) int64 {
	data, err := os.ReadFile(rooted("proc/meminfo"))
	if err != nil {
		return 0
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		if rest, ok := strings.CutPrefix(line, key); ok {
			fields := strings.Fields(rest)
			if len(fields) >= 1 {
				kb, _ := strconv.ParseInt(fields[0], 10, 64)
				return kb * 1024
			}
		}
	}
	return 0
}

// AvailableMemMB reports memory that can be allocated right now.
func AvailableMemMB() int {
	return int(meminfoBytes("MemAvailable:") / (1 << 20))
}

// sysRoot is where /sys and /proc are read from, so a test can stand in a
// machine of its own.
var sysRoot = "/"

func rooted(path string) string { return filepath.Join(sysRoot, path) }

// readTrim reads a small sysfs or procfs file, "" when it is not there.
func readTrim(path string) string {
	b, err := os.ReadFile(rooted(path))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(strings.TrimRight(string(b), "\x00"))
}

// platformFacts takes the Linux readings beyond memory and nvidia-smi.
func platformFacts(f *facts) {
	f.os, f.arch = runtime.GOOS, runtime.GOARCH
	vendor, product := readTrim("sys/class/dmi/id/sys_vendor"), readTrim("sys/class/dmi/id/product_name")
	f.system = strings.TrimSpace(vendor + " " + product)
	f.memLimit = cgroupMemLimit()
	f.pci = scanPCI()
	f.tegra, f.tegraModel = tegraFacts()
}

// tegraFacts reports a Jetson the way vLLM recognises one, by the files the
// L4T platform installs, and names it from the device tree.
func tegraFacts() (bool, string) {
	for _, p := range []string{"etc/nv_tegra_release", "sys/class/tegra-firmware"} {
		if _, err := os.Stat(rooted(p)); err == nil {
			return true, readTrim("proc/device-tree/model")
		}
	}
	return false, ""
}

// scanPCI lists the display controllers and processing accelerators on the
// PCI bus. Only those classes are kept: the rest are bridges, NICs and
// disks, and a report full of them would hide what matters.
func scanPCI() []PCIDevice {
	dir := rooted("sys/bus/pci/devices")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []PCIDevice
	for _, e := range entries {
		base := filepath.Join(dir, e.Name())
		read := func(name string) string {
			b, err := os.ReadFile(filepath.Join(base, name))
			if err != nil {
				return ""
			}
			return strings.ToLower(strings.TrimSpace(string(b)))
		}
		class := read("class")
		if !strings.HasPrefix(class, "0x03") && !strings.HasPrefix(class, "0x12") {
			continue
		}
		d := PCIDevice{Vendor: read("vendor"), Device: read("device"), Class: class}
		if link, err := os.Readlink(filepath.Join(base, "driver")); err == nil {
			d.Driver = filepath.Base(link)
		}
		if n, err := strconv.ParseInt(read("mem_info_vram_total"), 10, 64); err == nil && n > 0 {
			d.VRAMBytes = n
		}
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Vendor+out[i].Device+out[i].Class < out[j].Vendor+out[j].Device+out[j].Class
	})
	return out
}

// cgroupMemLimit is the tightest cgroup v2 memory.max over this process's
// cgroup and its ancestors, or 0 when none is set. A container's limit is
// what vLLM started inside it can use, whatever the host has installed.
func cgroupMemLimit() int64 {
	data, err := os.ReadFile(rooted("proc/self/cgroup"))
	if err != nil {
		return 0
	}
	var path string
	for line := range strings.SplitSeq(string(data), "\n") {
		if rest, ok := strings.CutPrefix(line, "0::"); ok {
			path = rest
		}
	}
	if path == "" {
		return 0
	}
	var limit int64
	for p := filepath.Clean("/" + path); ; p = filepath.Dir(p) {
		v := readTrim(filepath.Join("sys/fs/cgroup", p, "memory.max"))
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 && (limit == 0 || n < limit) {
			limit = n
		}
		if p == "/" {
			break
		}
	}
	return limit
}
