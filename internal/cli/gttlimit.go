package cli

import (
	"fmt"

	"github.com/coder/llm-helper/internal/detect"
)

// GTT is how much system memory amdgpu lets an integrated GPU map. The ttm
// kernel module sets it from its pages_limit option, which is read when the
// module loads, so raising it means a modprobe.d file, a rebuilt initramfs
// and a reboot. llm-helper only says how: writing a modprobe file needs root,
// and a wrong value is only found out at the next boot.
const (
	// gttSuggestShare is the share of RAM doctor suggests for the limit.
	// On a 128 GB Ryzen AI Max, 90% (112 GiB) served a 103 GB model with
	// about 15 GB of RAM still free.
	gttSuggestShare = 0.9
	// gttMinRaise is the smallest raise worth suggesting, since it costs a
	// reboot.
	gttMinRaise = 4 << 30
	// ttmConf is where the pages_limit option goes.
	ttmConf = "/etc/modprobe.d/ttm.conf"
)

// suggestGTTLimit is the GTT limit to suggest for a machine with ramBytes of
// RAM, rounded down to whole GiB, or 0 when RAM is unknown.
func suggestGTTLimit(ramBytes int64) int64 {
	if ramBytes <= 0 {
		return 0
	}
	return int64(float64(ramBytes)*gttSuggestShare) >> 30 << 30
}

// doctorGTTLines are the HARDWARE lines doctor prints for an integrated GPU's
// GTT limit, nil on any other GPU. pageSize is the kernel's page size, the
// unit pages_limit is in.
func doctorGTTLines(hw detect.Hardware, pageSize int) []string {
	if hw.GTTBytes <= 0 {
		return nil
	}
	line := fmt.Sprintf("  gtt:        %d GB the GPU can map (amdgpu's GTT limit)", hw.GTTGB)
	switch c := hw.GPUCarveOutBytes; {
	case c >= 1e9:
		line += fmt.Sprintf(", plus a %d GB carve-out", c/1e9)
	case c > 0:
		line += fmt.Sprintf(", plus a %d MB carve-out", c/1e6)
	}
	lines := []string{line}
	want := suggestGTTLimit(hw.TotalMemBytes)
	if pageSize <= 0 || want-hw.GTTBytes < gttMinRaise {
		return lines
	}
	return append(lines,
		fmt.Sprintf("              raise it to %d GiB with 'options ttm pages_limit=%d' in %s,", want>>30, want/int64(pageSize), ttmConf),
		"              then 'sudo update-initramfs -u' and a reboot (see docs/memory.md)")
}
