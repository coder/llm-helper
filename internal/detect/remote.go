package detect

import (
	"strconv"
	"strings"
)

// RemoteScript returns a shell script that prints the readings Probe takes,
// for a Linux machine reached over SSH. It needs nothing installed beyond what
// Probe itself would run, so a node is described whether or not llm-helper is
// on it, and the parsing stays here, where it matches the local probe's.
func RemoteScript() string {
	return `echo "memtotal $(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null)"
echo "memavail $(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)"
if [ -e /dev/nvidiactl ]; then echo "nvidiactl 1"; fi
if out=$(nvidia-smi ` + strings.Join(nvidiaSMIArgs, " ") + ` 2>/dev/null); then
  echo "smi ok"
  printf '%s\n' "$out" | sed 's/^/gpu /'
fi
`
}

// ParseRemote derives a node's hardware from RemoteScript's output. The
// node's server environment is not known from here, so every GPU nvidia-smi
// lists counts as visible.
func ParseRemote(out string) Hardware {
	var f facts
	var rows []string
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
		}
	}
	f.smiOut = strings.Join(rows, "\n")
	return f.hardware(nil, nil)
}

func kibibytes(s string) int64 {
	n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil || n < 0 {
		return 0
	}
	return n * 1024
}
