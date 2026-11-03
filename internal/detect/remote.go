package detect

import (
	"strconv"
	"strings"
)

// RemoteScript returns a shell script that prints the readings Probe takes,
// for a Linux machine reached over SSH. It needs nothing installed beyond what
// Probe itself would run, so a node is described whether or not llm-helper is
// on it, and the parsing stays here, where it matches the local probe's.
//
// The cgroup limit is not read: an SSH session's cgroup says nothing about
// the one the node's server runs in.
func RemoteScript() string {
	return `echo "memtotal $(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null)"
echo "memavail $(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)"
if [ -e /dev/nvidiactl ]; then echo "nvidiactl 1"; fi
if out=$(nvidia-smi ` + strings.Join(nvidiaSMIArgs, " ") + ` 2>/dev/null); then
  echo "smi ok"
  printf '%s\n' "$out" | sed 's/^/gpu /'
  if caps=$(nvidia-smi ` + strings.Join(nvidiaSMICapArgs, " ") + ` 2>/dev/null) || caps=$(nvidia-smi ` + strings.Join(nvidiaSMICapFallback, " ") + ` 2>/dev/null); then
    printf '%s\n' "$caps" | sed 's/^/gpucap /'
  fi
fi
if [ -e /etc/nv_tegra_release ] || [ -e /sys/class/tegra-firmware ]; then
  echo "tegra $(tr -d '\000' </proc/device-tree/model 2>/dev/null)"
fi
for d in /sys/bus/pci/devices/*; do
  class=$(cat "$d/class" 2>/dev/null)
  case "$class" in 0x03*|0x12*) ;; *) continue ;; esac
  drv=-
  if [ -L "$d/driver" ]; then drv=$(basename "$(readlink "$d/driver")"); fi
  echo "pci $(cat "$d/vendor") $(cat "$d/device") $class $drv $(cat "$d/mem_info_vram_total" 2>/dev/null || echo 0)"
done
`
}

// ParseRemote derives a node's hardware from RemoteScript's output. The
// node's server environment is not known from here, so every GPU nvidia-smi
// lists counts as visible.
func ParseRemote(out string) Hardware {
	f := facts{os: "linux"}
	var rows, caps []string
	for line := range strings.SplitSeq(out, "\n") {
		key, val, _ := strings.Cut(strings.TrimSpace(line), " ")
		switch key {
		case "memtotal":
			f.totalMem = kibibytes(val)
		case "memavail":
			f.availMem = kibibytes(val)
		case "nvidiactl":
			f.nvidiaDevice = true
		case "smi":
			f.smiOK = true
		case "gpu":
			rows = append(rows, val)
		case "gpucap":
			caps = append(caps, val)
		case "tegra":
			f.tegra, f.tegraModel = true, strings.TrimSpace(val)
		case "pci":
			if d, ok := parsePCILine(val); ok {
				f.pci = append(f.pci, d)
			}
		}
	}
	f.gpus = parseNVIDIASMI(strings.Join(rows, "\n"))
	applyGPUCaps(f.gpus, strings.Join(caps, "\n"))
	return f.hardware(nil)
}

// parsePCILine reads "vendor device class driver vram" as RemoteScript
// prints it, with "-" for no driver.
func parsePCILine(s string) (PCIDevice, bool) {
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return PCIDevice{}, false
	}
	d := PCIDevice{Vendor: strings.ToLower(fields[0]), Device: strings.ToLower(fields[1]), Class: strings.ToLower(fields[2])}
	if len(fields) > 3 && fields[3] != "-" {
		d.Driver = fields[3]
	}
	if len(fields) > 4 {
		if n, err := strconv.ParseInt(fields[4], 10, 64); err == nil && n > 0 {
			d.VRAMBytes = n
		}
	}
	return d, true
}

func kibibytes(s string) int64 {
	n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil || n < 0 {
		return 0
	}
	return n * 1024
}
