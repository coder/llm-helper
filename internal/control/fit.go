package control

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/nodes"
	"github.com/coder/llm-helper/internal/profile"
)

// MaxUnifiedMemUtilization is the most of a shared CPU/GPU pool all models
// together may ask for.
const MaxUnifiedMemUtilization = 0.85

// Share is the fraction of GPU memory one instance asks for, on the GPUs it
// can see. Nil GPUs means every GPU.
type Share struct {
	Name string
	Frac float64
	GPUs []string
}

func gpuList(value string, set bool) []string {
	if !set {
		return nil
	}
	out := []string{}
	for _, g := range strings.Split(value, ",") {
		if g = strings.TrimSpace(g); g != "" {
			out = append(out, g)
		}
	}
	return out
}

func (s Share) overlaps(o Share) bool {
	if s.GPUs == nil || o.GPUs == nil {
		return true
	}
	for _, a := range s.GPUs {
		for _, b := range o.GPUs {
			if a == b {
				return true
			}
		}
	}
	return false
}

// runsElsewhere is a recipe instance whose recorded nodes do not include
// this machine.
func runsElsewhere(inst *Instance) bool {
	if strings.TrimSpace(inst.Env.Get(profile.RecipeKey)) == "" {
		return false
	}
	names := strings.Split(inst.Env.Get(profile.NodesKey), ",")
	reg, err := nodes.Load()
	if err != nil {
		return false
	}
	known := false
	for _, name := range names {
		n, ok := reg.Get(strings.TrimSpace(name))
		if !ok {
			continue
		}
		known = true
		if n.IsLocal() {
			return false
		}
	}
	return known
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

// Shares is the memory share of each instance in list that runs on this
// machine.
func Shares(list []*Instance) []Share {
	out := make([]Share, 0, len(list))
	for _, inst := range list {
		if runsElsewhere(inst) {
			continue
		}
		out = append(out, Share{Name: inst.Name, Frac: ParseFrac(MemUtil(inst.Env)), GPUs: gpuList(CUDAVisible(inst.Env))})
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

// CheckFit refuses to start want when its share of GPU memory plus what the
// running instances on the same GPUs hold exceeds the pool. Unknown shares
// count as 0.
func CheckFit(hw detect.Hardware, running []Share, want Share) error {
	var limit float64
	switch {
	case hw.Metal:
		limit = detect.MetalMemUtilization
	case hw.Unified:
		limit = MaxUnifiedMemUtilization
	case hw.GPUName != "" || hw.GPUMemGB > 0:
		limit = 1
	default:
		return nil
	}
	if want.Frac == 0 {
		return nil
	}
	total := want.Frac
	var held []string
	for _, r := range running {
		if r.Frac == 0 || !r.overlaps(want) {
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
		want.Name, want.Frac, total-want.Frac, limit, strings.Join(held, ", "))
}

// CheckFitFor runs CheckFit for name against every other running instance.
// want is the requested share; "" uses the first file that sets one.
// profileFiles go most specific first.
func CheckFitFor(cfg *config.Config, name string, want string, profileFiles ...*config.EnvFile) error {
	others, err := OtherInstances(cfg, name)
	if err != nil {
		return err
	}
	if want == "" {
		want = MemUtil(profileFiles...)
	}
	layered := make([]*config.EnvFile, 0, len(profileFiles))
	for i := len(profileFiles) - 1; i >= 0; i-- {
		layered = append(layered, profileFiles[i])
	}
	share := Share{Name: name, Frac: ParseFrac(want), GPUs: gpuList(CUDAVisible(layered...))}
	return CheckFit(detect.Probe(), Shares(others), share)
}
