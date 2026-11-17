//go:build linux

package detect

import (
	"os"
	"path/filepath"
	"testing"
)

// fakeRoot stands in a machine's /proc and /sys. Not parallel: sysRoot is
// read by every probe in the process.
func fakeRoot(t *testing.T, files map[string]string, links map[string]string) {
	t.Helper()
	root := t.TempDir()
	for path, body := range files {
		full := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for path, target := range links {
		full := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, full); err != nil {
			t.Fatal(err)
		}
	}
	old := sysRoot
	sysRoot = root
	t.Cleanup(func() { sysRoot = old })
}

// The readings a Linux probe takes beyond nvidia-smi, from the files the
// kernel and the L4T platform really put there.
func TestPlatformFactsReadsSysfs(t *testing.T) {
	const amd = "sys/bus/pci/devices/0000:03:00.0/"
	const nic = "sys/bus/pci/devices/0000:04:00.0/"
	fakeRoot(t, map[string]string{
		"proc/self/cgroup":                                "0::/system.slice/app.scope\n",
		"sys/fs/cgroup/system.slice/memory.max":           "34359738368\n",
		"sys/fs/cgroup/system.slice/app.scope/memory.max": "max\n",
		"sys/class/dmi/id/sys_vendor":                     "NVIDIA\n",
		"sys/class/dmi/id/product_name":                   "NVIDIA_DGX_Spark\n",
		"etc/nv_tegra_release":                            "# R36 (release)\n",
		"proc/device-tree/model":                          "NVIDIA Jetson AGX Orin Developer Kit\x00",
		amd + "vendor":                                    "0x1002\n",
		amd + "device":                                    "0x744c\n",
		amd + "class":                                     "0x030000\n",
		amd + "mem_info_vram_total":                       "25753026560\n",
		amd + "mem_info_gtt_total":                        "33554432000\n",
		nic + "vendor":                                    "0x15b3\n",
		nic + "device":                                    "0x1021\n",
		nic + "class":                                     "0x020000\n",
	}, map[string]string{
		amd + "driver": "../../../bus/pci/drivers/amdgpu",
	})

	var f facts
	platformFacts(&f)
	if f.memLimit != 32<<30 {
		t.Errorf("memLimit = %d, want the tightest ancestor's 32 GiB", f.memLimit)
	}
	if f.system != "NVIDIA NVIDIA_DGX_Spark" {
		t.Errorf("system = %q", f.system)
	}
	if !f.tegra || f.tegraModel != "NVIDIA Jetson AGX Orin Developer Kit" {
		t.Errorf("tegra = %v, %q", f.tegra, f.tegraModel)
	}
	if len(f.pci) != 1 || f.pci[0] != (PCIDevice{Vendor: "0x1002", Device: "0x744c", Class: "0x030000", Driver: "amdgpu", VRAMBytes: 25753026560, GTTBytes: 33554432000}) {
		t.Errorf("pci = %+v, want the AMD card and not the NIC", f.pci)
	}
}

// No limit anywhere up the tree is no limit, not a limit of zero.
func TestCgroupMemLimitWithoutALimit(t *testing.T) {
	fakeRoot(t, map[string]string{
		"proc/self/cgroup":                    "0::/user.slice\n",
		"sys/fs/cgroup/user.slice/memory.max": "max\n",
	}, nil)
	if got := cgroupMemLimit(); got != 0 {
		t.Errorf("cgroupMemLimit = %d, want 0", got)
	}
}

// A DGX Station GB300 in NUMA mode, as its sysfs reads: node 0 is the Grace
// CPU and its LPDDR5X, node 1 the GB300's HBM with no CPUs, and nodes 2-8
// empty placeholders with no CPUs and no memory.
func TestCPULessNodeMemCountsOnlyNodesWithoutCPUs(t *testing.T) {
	files := map[string]string{
		"sys/devices/system/node/node0/cpulist": "0-71\n",
		"sys/devices/system/node/node0/meminfo": "Node 0 MemTotal:       518528000 kB\nNode 0 MemFree:        489770000 kB\n",
		"sys/devices/system/node/node1/cpulist": "\n",
		"sys/devices/system/node/node1/meminfo": "Node 1 MemTotal:       261619712 kB\nNode 1 MemFree:        261618688 kB\n",
		"sys/devices/system/node/node2/cpulist": "\n",
		"sys/devices/system/node/node2/meminfo": "Node 2 MemTotal:              0 kB\n",
	}
	fakeRoot(t, files, nil)
	if got, want := cpulessNodeMem(), int64(261619712)*1024; got != want {
		t.Errorf("cpulessNodeMem() = %d, want %d (node 1 only)", got, want)
	}

	fakeRoot(t, map[string]string{"sys/devices/system/node/node0/cpulist": "0-7\n"}, nil)
	if got := cpulessNodeMem(); got != 0 {
		t.Errorf("one node with CPUs: cpulessNodeMem() = %d, want 0", got)
	}
}
