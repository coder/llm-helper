package control

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
)

// MaxUnifiedMemUtilization is the most of a shared CPU/GPU pool all models
// together may ask for.
const MaxUnifiedMemUtilization = 0.85

// Share is the fraction of GPU memory one running instance asked for.
type Share struct {
	Name string
	Frac float64
}

// OtherInstances lists the running instances except name.
func OtherInstances(cfg *config.Config, name string) ([]*Instance, error) {
	list, err := ListInstances(cfg)
	if err != nil {
		return nil, err
	}
	out := list[:0]
	for _, inst := range list {
		if inst.Name != name {
			out = append(out, inst)
		}
	}
	return out, nil
}

// Shares is the memory share of each instance in list.
func Shares(list []*Instance) []Share {
	out := make([]Share, 0, len(list))
	for _, inst := range list {
		out = append(out, Share{Name: inst.Name, Frac: ParseFrac(MemUtil(inst.Env))})
	}
	return out
}

// MemUtil is the first GPU_MEMORY_UTILIZATION set among files.
func MemUtil(files ...*config.EnvFile) string {
	for _, f := range files {
		if f == nil {
			continue
		}
		if v := f.Get("GPU_MEMORY_UTILIZATION"); v != "" {
			return v
		}
	}
	return ""
}

// ParseFrac reads a positive fraction, or 0.
func ParseFrac(s string) float64 {
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil || f <= 0 {
		return 0
	}
	return f
}

// CheckFit refuses to start name when its share of GPU memory plus what the
// running instances hold exceeds the pool. Unknown shares count as 0.
func CheckFit(hw detect.Hardware, running []Share, name string, want float64) error {
	var limit float64
	switch {
	case hw.Unified:
		limit = MaxUnifiedMemUtilization
	case hw.GPUName != "" || hw.GPUMemGB > 0:
		limit = 1
	default:
		return nil
	}
	if want == 0 {
		return nil
	}
	total := want
	var held []string
	for _, r := range running {
		if r.Frac == 0 {
			continue
		}
		total += r.Frac
		held = append(held, fmt.Sprintf("%s %.2f", r.Name, r.Frac))
	}
	if total <= limit+1e-9 {
		return nil
	}
	sort.Strings(held)
	return fmt.Errorf("'%s' wants %.2f of GPU memory, but the running models already use %.2f of the %.2f available (%s); "+
		"stop one first, or give this one a smaller share (--mem-util or --kv-cache)",
		name, want, total-want, limit, strings.Join(held, ", "))
}

// CheckFitFor runs CheckFit for p against every other running instance. want
// is the requested share; "" uses the profile's own.
func CheckFitFor(cfg *config.Config, name string, want string, profileFiles ...*config.EnvFile) error {
	others, err := OtherInstances(cfg, name)
	if err != nil {
		return err
	}
	if want == "" {
		want = MemUtil(profileFiles...)
	}
	return CheckFit(detect.Probe(), Shares(others), name, ParseFrac(want))
}
