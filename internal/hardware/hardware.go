// Package hardware is what llm-helper knows about specific GPUs, kept as data
// rather than code: which PCI vendor makes an accelerator, which architecture
// a GPU is and which flags abort at startup there, and which machines share
// their GPU's memory with the CPU.
//
// The knowledge lives in catalogue/hardware, embedded in the binary, with one
// file per entry in its vendors, architectures and machines folders. Adding
// something means adding a file there, with a why for every claim, not
// editing the detection or launch code that reads it. A GPU no machine file
// names still gets its architecture's rules from the compute capability it
// reports. A machine owner can add entries without a rebuild by putting files
// in the same folders under hardware.d in the config directory; see UseLocal.
// Probing the machine stays in package detect; this package only says what a
// probe's findings mean.
//
// The data is parsed strictly. An unknown key, a duplicate id, a file not
// named after its id, a pattern that does not compile or an architecture
// nobody defined is an error, because a typo in safety data must not quietly
// turn into a missing safety check.
package hardware

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/coder/llm-helper/catalogue"
)

// SchemaVersion is the definitions format this build reads. A file written
// for another version is refused rather than half understood.
const SchemaVersion = 1

// Memory is how a GPU's memory relates to the CPU's.
type Memory string

const (
	// MemoryUnified is one pool shared with the CPU, with nothing of the
	// GPU's own: a DGX Spark, a Jetson, Apple Silicon. Weights come out of
	// system memory.
	MemoryUnified Memory = "unified"
	// MemoryDiscrete is memory of the GPU's own, apart from the CPU's.
	MemoryDiscrete Memory = "discrete"
	// MemoryCoherent is a GPU with memory of its own that the CPU's memory
	// is coherently attached to, as on Grace Hopper and Grace Blackwell.
	// It is sized as unified until that has been measured on the hardware.
	MemoryCoherent Memory = "coherent"
)

func (m Memory) valid() bool {
	return m == MemoryUnified || m == MemoryDiscrete || m == MemoryCoherent
}

// Registry is a validated set of vendors, architectures and definitions.
type Registry struct {
	vendors []Vendor
	archs   []Architecture
	defs    []Definition
}

// Vendor is a maker of accelerators, and the PCI vendor ids its devices
// report, so a GPU no driver describes can still be named.
type Vendor struct {
	ID     string   `json:"id"`   // e.g. "nvidia"
	Name   string   `json:"name"` // e.g. "NVIDIA"
	PCIIDs []string `json:"pci_ids,omitempty"`
}

// Architecture is a GPU instruction set and what is known to break on it.
// Forbidden flags belong here rather than to a machine, because a kernel that
// aborts on sm121 aborts on every machine built on sm121.
type Architecture struct {
	Arch       string `json:"arch"`       // e.g. "sm121"
	Generation string `json:"generation"` // e.g. "blackwell"
	// ComputeCap is the CUDA compute capability nvidia-smi reports for it,
	// such as "12.1", so a GPU no definition names still gets its
	// architecture's rules.
	ComputeCap    string         `json:"compute_cap,omitempty"`
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

// Definition is one kind of machine: how to recognise it from a probe, and
// what that recognition implies.
type Definition struct {
	// ID is the stable name recipes record as the hardware they were
	// verified on, such as "dgx_spark_gb10". It never changes once shipped.
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	Match       Match  `json:"match"`
	// Memory is how the GPU's memory relates to the CPU's. A matched
	// definition is authoritative either way, so a discrete card whose
	// VRAM happens to equal system RAM can say so.
	Memory Memory `json:"memory"`
	// Arch names an entry in the architectures list, or is empty when the
	// definition covers GPUs of more than one architecture.
	Arch string `json:"arch,omitempty"`
	Why  string `json:"why"`

	// Source is the file a local definition came from, and "" for one
	// shipped in the binary.
	Source string `json:"-"`

	gpuName *regexp.Regexp
}

// Unified reports whether weights for this machine come out of system
// memory, which is true of a coherent pool too until that is measured.
func (d Definition) Unified() bool {
	return d.Memory == MemoryUnified || d.Memory == MemoryCoherent
}

// Match is how a definition recognises a GPU. Every rule given has to hold,
// and a definition needs a gpu_name or pci_devices.
type Match struct {
	// Vendor is a vendor id, such as "nvidia", or "" for any vendor.
	Vendor string `json:"vendor,omitempty"`
	// GPUName is a regular expression matched case-insensitively anywhere
	// in the name the probe reported, such as "NVIDIA GB10".
	GPUName string `json:"gpu_name"`
	// PCIDevices are PCI device ids, such as "0x1586", for a GPU found on
	// the PCI bus that no driver gives a name: an AMD GPU under amdgpu
	// reports only its ids. Device ids are only unique within a vendor, so
	// a definition that lists them names its vendor too.
	PCIDevices []string `json:"pci_devices,omitempty"`
}

// Observed is what a probe saw of one GPU, which is all a definition is
// matched against.
type Observed struct {
	Vendor  string // vendor id, e.g. "nvidia"
	GPUName string
	// PCIDevice is the GPU's PCI device id, such as "0x1586", or "" when
	// the GPU was not found on the PCI bus.
	PCIDevice string
}

// Kind is one kind of hardware entry. Each kind is a folder of its own, in
// the catalogue and in hardware.d, holding one file per entry.
type Kind string

const (
	KindVendor       Kind = "vendors"
	KindArchitecture Kind = "architectures"
	KindMachine      Kind = "machines"
)

// kinds is the order the folders load in: a machine names a vendor and an
// architecture, so those have to be known first.
var kinds = []Kind{KindVendor, KindArchitecture, KindMachine}

// The on-disk shape of one entry's file: the entry's fields and the format's
// version.
type (
	vendorFile struct {
		SchemaVersion int `json:"schema_version"`
		Vendor
	}
	architectureFile struct {
		SchemaVersion int `json:"schema_version"`
		Architecture
	}
	machineFile struct {
		SchemaVersion int `json:"schema_version"`
		Definition
	}
)

// set is entries of every kind, not yet validated.
type set struct {
	Vendors       []Vendor
	Architectures []Architecture
	Definitions   []Definition
}

var (
	idPattern         = regexp.MustCompile(`^[a-z0-9][a-z0-9_]*$`)
	pciIDPattern      = regexp.MustCompile(`^0x[0-9a-f]{4}$`)
	computeCapPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+$`)
)

// decode reads one JSON object strictly into v.
func decode(data []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	// A second object after the first would otherwise be dropped without a
	// word, and whatever it defined would silently not exist.
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("unexpected data after the top-level object")
	}
	return nil
}

// add reads the file for one entry of kind and adds the entry to s. The file
// has to be named after the entry's id, so the folder lists what is defined.
func (s *set) add(kind Kind, name string, data []byte) error {
	var version int
	var id string
	switch kind {
	case KindVendor:
		var f vendorFile
		if err := decode(data, &f); err != nil {
			return err
		}
		version, id = f.SchemaVersion, f.ID
		s.Vendors = append(s.Vendors, f.Vendor)
	case KindArchitecture:
		var f architectureFile
		if err := decode(data, &f); err != nil {
			return err
		}
		version, id = f.SchemaVersion, f.Arch
		s.Architectures = append(s.Architectures, f.Architecture)
	case KindMachine:
		var f machineFile
		if err := decode(data, &f); err != nil {
			return err
		}
		version, id = f.SchemaVersion, f.ID
		s.Definitions = append(s.Definitions, f.Definition)
	default:
		return fmt.Errorf("unknown kind %q", kind)
	}
	if version != SchemaVersion {
		return fmt.Errorf("schema_version is %d; this llm-helper reads version %d", version, SchemaVersion)
	}
	if base := strings.TrimSuffix(filepath.Base(name), ".json"); id != base {
		return fmt.Errorf("the file is named %s.json but defines %q; name it after the id it defines", base, id)
	}
	return nil
}

// Load reads and validates the hardware in fsys: a vendors, an architectures
// and a machines folder, each holding one *.json file per entry, named after
// its id. Machines are tried in file name order, and no two may match the
// same GPU, which the tests check against every report in the catalogue.
func Load(fsys fs.FS) (*Registry, error) {
	var s set
	var errs []error
	for _, k := range kinds {
		names, err := fs.Glob(fsys, string(k)+"/*.json")
		if err != nil {
			return nil, err
		}
		for _, name := range names {
			data, err := fs.ReadFile(fsys, name)
			if err == nil {
				err = s.add(k, name, data)
			}
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", name, err))
			}
		}
	}
	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("parse hardware definitions: %w", err)
	}
	r, err := build(s)
	if err != nil {
		return nil, fmt.Errorf("invalid hardware definitions: %w", err)
	}
	return r, nil
}

// build validates f and compiles its patterns.
func build(f set) (*Registry, error) {
	var errs []error

	vendors := make(map[string]bool, len(f.Vendors))
	pci := map[string]string{}
	for _, v := range f.Vendors {
		switch {
		case !idPattern.MatchString(v.ID):
			errs = append(errs, fmt.Errorf("vendor id %q must be lowercase letters, digits and underscores", v.ID))
		case vendors[v.ID]:
			errs = append(errs, fmt.Errorf("vendor %q is defined twice", v.ID))
		case v.Name == "":
			errs = append(errs, fmt.Errorf("vendor %q has no name", v.ID))
		}
		vendors[v.ID] = true
		for _, id := range v.PCIIDs {
			if !pciIDPattern.MatchString(id) {
				errs = append(errs, fmt.Errorf("vendor %q: PCI id %q must look like 0x10de", v.ID, id))
			}
			if other, ok := pci[id]; ok {
				errs = append(errs, fmt.Errorf("PCI id %s belongs to both %q and %q", id, other, v.ID))
			}
			pci[id] = v.ID
		}
	}

	archs := make(map[string]bool, len(f.Architectures))
	caps := map[string]string{}
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
		if a.ComputeCap != "" {
			if !computeCapPattern.MatchString(a.ComputeCap) {
				errs = append(errs, fmt.Errorf("architecture %q: compute_cap %q must look like 12.1", a.Arch, a.ComputeCap))
			}
			if other, ok := caps[a.ComputeCap]; ok {
				errs = append(errs, fmt.Errorf("compute_cap %s belongs to both %q and %q", a.ComputeCap, other, a.Arch))
			}
			caps[a.ComputeCap] = a.Arch
		}
		for _, fa := range a.ForbiddenArgs {
			if !strings.HasPrefix(fa.Flag, "--") {
				errs = append(errs, fmt.Errorf("architecture %q: forbidden flag %q does not start with --", a.Arch, fa.Flag))
			}
			if strings.TrimSpace(fa.Why) == "" {
				errs = append(errs, fmt.Errorf("architecture %q: forbidden flag %s has no why", a.Arch, fa.Flag))
			}
		}
	}

	ids := make(map[string]bool, len(f.Definitions))
	for i := range f.Definitions {
		d := &f.Definitions[i]
		if !idPattern.MatchString(d.ID) {
			errs = append(errs, fmt.Errorf("definition id %q must be lowercase letters, digits and underscores", d.ID))
		}
		if ids[d.ID] {
			errs = append(errs, fmt.Errorf("definition %q is defined twice", d.ID))
		}
		ids[d.ID] = true
		if d.DisplayName == "" {
			errs = append(errs, fmt.Errorf("definition %q has no display_name", d.ID))
		}
		if strings.TrimSpace(d.Why) == "" {
			errs = append(errs, fmt.Errorf("definition %q has no why", d.ID))
		}
		if !d.Memory.valid() {
			errs = append(errs, fmt.Errorf("definition %q: memory %q must be unified, discrete or coherent", d.ID, d.Memory))
		}
		if d.Arch != "" && !archs[d.Arch] {
			errs = append(errs, fmt.Errorf("definition %q names architecture %q, which is not defined", d.ID, d.Arch))
		}
		if d.Match.Vendor != "" && !vendors[d.Match.Vendor] {
			errs = append(errs, fmt.Errorf("definition %q names vendor %q, which is not defined", d.ID, d.Match.Vendor))
		}
		for _, id := range d.Match.PCIDevices {
			if !pciIDPattern.MatchString(id) {
				errs = append(errs, fmt.Errorf("definition %q: PCI device %q must look like 0x1586", d.ID, id))
			}
		}
		if len(d.Match.PCIDevices) > 0 && d.Match.Vendor == "" {
			errs = append(errs, fmt.Errorf("definition %q matches PCI devices but names no vendor; device ids are only unique within one", d.ID))
		}
		if d.Match.GPUName == "" {
			if len(d.Match.PCIDevices) == 0 {
				errs = append(errs, fmt.Errorf("definition %q has no match rule", d.ID))
			}
			continue
		}
		re, err := regexp.Compile("(?i)" + d.Match.GPUName)
		if err != nil {
			errs = append(errs, fmt.Errorf("definition %q: match.gpu_name: %w", d.ID, err))
			continue
		}
		d.gpuName = re
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return &Registry{vendors: f.Vendors, archs: f.Architectures, defs: f.Definitions}, nil
}

// Extend returns a registry with the entry in one local file added after
// this one's. kind is the folder the file is in, and path names the file in
// errors and on a machine it adds.
//
// A local file can only add: a new vendor, a new architecture, a new
// machine, or forbidden flags on an existing architecture. It cannot
// redefine anything already here, so a stale or mistaken file can teach
// llm-helper about a new machine but cannot change what it says about a
// known one, and cannot drop a safety check.
func (r *Registry) Extend(kind Kind, path string, data []byte) (*Registry, error) {
	var local set
	if err := local.add(kind, path, data); err != nil {
		return nil, err
	}
	var errs []error
	merged := set{
		Vendors:       slices.Clone(r.vendors),
		Architectures: make([]Architecture, len(r.archs)),
		Definitions:   slices.Clone(r.defs),
	}
	for i, a := range r.archs {
		a.ForbiddenArgs = slices.Clone(a.ForbiddenArgs)
		merged.Architectures[i] = a
	}

	for _, v := range local.Vendors {
		if _, ok := r.Vendor(v.ID); ok {
			errs = append(errs, fmt.Errorf("vendor %q is already defined; a local file can only add new ones", v.ID))
			continue
		}
		merged.Vendors = append(merged.Vendors, v)
	}
	for _, a := range local.Architectures {
		i := slices.IndexFunc(merged.Architectures, func(x Architecture) bool { return x.Arch == a.Arch })
		if i < 0 || a.Arch == "" {
			merged.Architectures = append(merged.Architectures, a)
			continue
		}
		have := &merged.Architectures[i]
		if (a.Generation != "" && a.Generation != have.Generation) || (a.ComputeCap != "" && a.ComputeCap != have.ComputeCap) {
			errs = append(errs, fmt.Errorf("architecture %q is already defined; a local file can only add forbidden_args to it", a.Arch))
			continue
		}
		for _, fa := range a.ForbiddenArgs {
			if !slices.ContainsFunc(have.ForbiddenArgs, func(x ForbiddenArg) bool { return x.Flag == fa.Flag && x.Value == fa.Value }) {
				have.ForbiddenArgs = append(have.ForbiddenArgs, fa)
			}
		}
	}
	for _, d := range local.Definitions {
		if _, ok := r.ByID(d.ID); ok {
			errs = append(errs, fmt.Errorf("definition %q is already defined; a local file can only add new ids", d.ID))
			continue
		}
		d.Source = path
		merged.Definitions = append(merged.Definitions, d)
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	out, err := build(merged)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// shippedRegistry is the set compiled into the binary. A parse failure is a
// build defect in the same way a bad regexp.MustCompile pattern is, and every
// test in every package that imports this one catches it.
var shippedRegistry = func() *Registry {
	r, err := Load(catalogue.Hardware)
	if err != nil {
		panic(err)
	}
	return r
}()

// Shipped returns the definitions compiled into the binary, without any
// local additions.
func Shipped() *Registry { return shippedRegistry }

// Definitions returns every definition, in match order.
func (r *Registry) Definitions() []Definition { return slices.Clone(r.defs) }

// Architectures returns every architecture.
func (r *Registry) Architectures() []Architecture { return slices.Clone(r.archs) }

// Vendors returns every vendor.
func (r *Registry) Vendors() []Vendor { return slices.Clone(r.vendors) }

// ByID looks a definition up by its id.
func (r *Registry) ByID(id string) (Definition, bool) {
	for _, d := range r.defs {
		if d.ID == id {
			return d, true
		}
	}
	return Definition{}, false
}

// Match returns the definition that recognises a GPU: the first one, with
// shipped machines tried before local ones. A GPU with neither a name nor a
// PCI device id matches nothing: a GPU nobody could describe is not evidence
// of any particular machine.
func (r *Registry) Match(o Observed) (Definition, bool) {
	if m := r.Matches(o); len(m) > 0 {
		return m[0], true
	}
	return Definition{}, false
}

// Matches returns every definition that recognises a GPU, in the order Match
// tries them. More than one means two definitions claim the same GPU. The
// tests keep the shipped ones from ever doing that, so only a local file can.
func (r *Registry) Matches(o Observed) []Definition {
	device := strings.ToLower(strings.TrimSpace(o.PCIDevice))
	if o.GPUName == "" && device == "" {
		return nil
	}
	var out []Definition
	for _, d := range r.defs {
		if d.Match.Vendor != "" && d.Match.Vendor != o.Vendor {
			continue
		}
		if d.gpuName != nil && (o.GPUName == "" || !d.gpuName.MatchString(o.GPUName)) {
			continue
		}
		if len(d.Match.PCIDevices) > 0 && (device == "" || !slices.Contains(d.Match.PCIDevices, device)) {
			continue
		}
		if d.gpuName == nil && len(d.Match.PCIDevices) == 0 {
			continue
		}
		out = append(out, d)
	}
	return out
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

// ArchitectureByComputeCap looks an architecture up by the compute
// capability nvidia-smi reports, such as "12.1". The match is exact: 12.0
// and 12.1 are different chips with different kernels.
func (r *Registry) ArchitectureByComputeCap(cc string) (Architecture, bool) {
	cc = strings.TrimSpace(cc)
	if cc == "" {
		return Architecture{}, false
	}
	for _, a := range r.archs {
		if a.ComputeCap == cc {
			return a, true
		}
	}
	return Architecture{}, false
}

// Vendor looks a vendor up by id.
func (r *Registry) Vendor(id string) (Vendor, bool) {
	for _, v := range r.vendors {
		if v.ID == id {
			return v, true
		}
	}
	return Vendor{}, false
}

// VendorByPCI looks a vendor up by the PCI vendor id sysfs reports, such as
// "0x10de".
func (r *Registry) VendorByPCI(id string) (Vendor, bool) {
	id = strings.ToLower(strings.TrimSpace(id))
	for _, v := range r.vendors {
		if slices.Contains(v.PCIIDs, id) {
			return v, true
		}
	}
	return Vendor{}, false
}
