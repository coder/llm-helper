// Package hardware is what vllm-helper knows about specific machines, kept as
// data rather than code: which GPUs share system memory, which architecture a
// machine is, and which flags abort at startup there.
//
// The knowledge lives in descriptors.json, embedded in the binary. Adding a
// machine is a matter of adding an entry there, with a why for every claim,
// not of editing the detection or launch code that reads it. Probing the
// machine stays in package detect; this package only says what a probe's
// findings mean.
//
// The file is parsed strictly. An unknown key, a duplicate id, a pattern that
// does not compile or an architecture nobody defined is an error, because a
// typo in safety data must not quietly turn into a missing safety check.
package hardware

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

//go:embed descriptors.json
var descriptorsJSON []byte

// Registry is a validated set of architectures and descriptors.
type Registry struct {
	archs []Architecture
	descs []Descriptor
}

// Architecture is a GPU instruction set and what is known to break on it.
// Forbidden flags belong here rather than to a machine, because a kernel that
// aborts on sm121 aborts on every machine built on sm121.
type Architecture struct {
	Arch          string         `json:"arch"`       // e.g. "sm121"
	Generation    string         `json:"generation"` // e.g. "blackwell"
	ForbiddenArgs []ForbiddenArg `json:"forbidden_args,omitempty"`
}

// ForbiddenArg is a flag and value that abort vLLM at startup on an
// architecture. Why is required: an entry nobody can explain cannot be
// reviewed, and cannot be removed when the reason stops holding.
type ForbiddenArg struct {
	Flag  string `json:"flag"`
	Value string `json:"value"`
	Why   string `json:"why"`
}

// Descriptor is one kind of machine: how to recognise it from a probe, and
// what that recognition implies.
type Descriptor struct {
	// ID is the stable name recipes record as the hardware they were
	// verified on, such as "dgx_spark_gb10".
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	Match       Match  `json:"match"`
	// Unified reports that the GPU shares one memory pool with the CPU. A
	// matched descriptor is authoritative either way, so a discrete card
	// whose VRAM happens to equal system RAM can say so.
	Unified bool `json:"unified"`
	// Arch names an entry in the architectures list, or is empty when no
	// architecture rules apply to this machine.
	Arch string `json:"arch,omitempty"`
	Why  string `json:"why"`

	gpuName *regexp.Regexp
}

// Match is how a descriptor recognises a machine. GPUName is a regular
// expression matched case-insensitively anywhere in the GPU name the probe
// reported, such as "NVIDIA GB10".
type Match struct {
	GPUName string `json:"gpu_name"`
}

// file is the on-disk shape of descriptors.json.
type file struct {
	Architectures []Architecture `json:"architectures"`
	Descriptors   []Descriptor   `json:"descriptors"`
}

var idPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_]*$`)

// Parse reads and validates a descriptors file. Descriptors are matched in
// file order, so a more specific entry has to come before a broader one.
func Parse(data []byte) (*Registry, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var f file
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("parse hardware descriptors: %w", err)
	}

	var errs []error
	archs := make(map[string]bool, len(f.Architectures))
	for _, a := range f.Architectures {
		switch {
		case a.Arch == "":
			errs = append(errs, errors.New("an architecture has no arch"))
			continue
		case archs[a.Arch]:
			errs = append(errs, fmt.Errorf("architecture %q is defined twice", a.Arch))
		case a.Generation == "":
			errs = append(errs, fmt.Errorf("architecture %q has no generation", a.Arch))
		}
		archs[a.Arch] = true
		for _, fa := range a.ForbiddenArgs {
			if !strings.HasPrefix(fa.Flag, "--") {
				errs = append(errs, fmt.Errorf("architecture %q: forbidden flag %q does not start with --", a.Arch, fa.Flag))
			}
			if strings.TrimSpace(fa.Why) == "" {
				errs = append(errs, fmt.Errorf("architecture %q: forbidden flag %s has no why", a.Arch, fa.Flag))
			}
		}
	}

	ids := make(map[string]bool, len(f.Descriptors))
	for i := range f.Descriptors {
		d := &f.Descriptors[i]
		if !idPattern.MatchString(d.ID) {
			errs = append(errs, fmt.Errorf("descriptor id %q must be lowercase letters, digits and underscores", d.ID))
		}
		if ids[d.ID] {
			errs = append(errs, fmt.Errorf("descriptor %q is defined twice", d.ID))
		}
		ids[d.ID] = true
		if d.DisplayName == "" {
			errs = append(errs, fmt.Errorf("descriptor %q has no display_name", d.ID))
		}
		if strings.TrimSpace(d.Why) == "" {
			errs = append(errs, fmt.Errorf("descriptor %q has no why", d.ID))
		}
		if d.Arch != "" && !archs[d.Arch] {
			errs = append(errs, fmt.Errorf("descriptor %q names architecture %q, which is not defined", d.ID, d.Arch))
		}
		if d.Match.GPUName == "" {
			errs = append(errs, fmt.Errorf("descriptor %q has no match rule", d.ID))
			continue
		}
		re, err := regexp.Compile("(?i)" + d.Match.GPUName)
		if err != nil {
			errs = append(errs, fmt.Errorf("descriptor %q: match.gpu_name: %w", d.ID, err))
			continue
		}
		d.gpuName = re
	}
	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("invalid hardware descriptors: %w", err)
	}
	return &Registry{archs: f.Architectures, descs: f.Descriptors}, nil
}

// defaultRegistry is the shipped set. The data is compiled in, so a parse
// failure is a build defect in the same way a bad regexp.MustCompile pattern
// is, and every test in every package that imports this one catches it.
var defaultRegistry = func() *Registry {
	r, err := Parse(descriptorsJSON)
	if err != nil {
		panic(err)
	}
	return r
}()

// Default returns the descriptors shipped in the binary.
func Default() *Registry { return defaultRegistry }

// Descriptors returns every descriptor, in match order.
func (r *Registry) Descriptors() []Descriptor { return slices.Clone(r.descs) }

// ByID looks a descriptor up by its id.
func (r *Registry) ByID(id string) (Descriptor, bool) {
	for _, d := range r.descs {
		if d.ID == id {
			return d, true
		}
	}
	return Descriptor{}, false
}

// MatchGPU returns the first descriptor that recognises a GPU name. An empty
// name matches nothing: a GPU nobody could describe is not evidence of any
// particular machine.
func (r *Registry) MatchGPU(name string) (Descriptor, bool) {
	if name == "" {
		return Descriptor{}, false
	}
	for _, d := range r.descs {
		if d.gpuName != nil && d.gpuName.MatchString(name) {
			return d, true
		}
	}
	return Descriptor{}, false
}

// Architecture looks an architecture up by name.
func (r *Registry) Architecture(arch string) (Architecture, bool) {
	for _, a := range r.archs {
		if a.Arch == arch {
			return a, true
		}
	}
	return Architecture{}, false
}
