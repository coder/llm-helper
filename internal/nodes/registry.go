package nodes

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/coder/llm-helper/internal/config"
)

// NIC describes a network interface.
type NIC struct {
	Name  string `json:"name"`
	Speed string `json:"speed"`
	MTU   int    `json:"mtu"`
}

// Capabilities is a short summary of a node's serving capacity.
type Capabilities struct {
	// Accelerator is how a server there runs: cuda, metal or cpu, and ""
	// until the node is probed. A GPU vLLM cannot use, such as one the
	// driver does not see, leaves it cpu.
	Accelerator string `json:"accelerator"`
	GPU         string `json:"gpu,omitempty"` // e.g. "NVIDIA GB10" or "4x NVIDIA H100 80GB HBM3"
	GPUs        int    `json:"gpus"`
	// PoolBytes is the memory one server loads weights into: one GPU's
	// VRAM on a node of discrete cards, since that is what a server
	// without --tensor-parallel-size uses, and system memory otherwise.
	PoolBytes uint64 `json:"pool_bytes"`
	// HardwareID is the hardware definition the node matched, such as
	// "dgx_spark_gb10", and "" when none did or the node predates it.
	HardwareID string `json:"hardware_id,omitempty"`
	// Arch is the GPU architecture, such as "sm121", whose forbidden flags
	// a command line for this node is checked against.
	Arch string `json:"arch,omitempty"`
	NICs []NIC  `json:"nics"`
}

// Node is one machine in the cluster registry.
type Node struct {
	Name   string            `json:"name"`
	Host   string            `json:"host"`          // scheme://host:port for API routing
	SSH    string            `json:"ssh,omitempty"` // user@host for orchestration
	Labels map[string]string `json:"labels"`
	Caps   Capabilities      `json:"caps"`
}

// Registry persists known nodes (config-scoped, not per-profile).
type Registry struct {
	path string
	mu   sync.RWMutex
	m    map[string]Node
}

func pathDefault() string { return filepath.Join(config.ResolveDirs().Config, "nodes.json") }

// Load reads ~/.config/vllm-helper/nodes.json.
func Load() (*Registry, error) {
	p := pathDefault()
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return &Registry{path: p, m: map[string]Node{}}, nil
	}
	if err != nil {
		return nil, err
	}
	var list []Node
	if err := json.Unmarshal(b, &list); err != nil {
		return nil, err
	}
	r := &Registry{path: p, m: map[string]Node{}}
	for _, n := range list {
		r.m[n.Name] = n
	}
	return r, nil
}

func (r *Registry) Save() error {
	r.mu.RLock()
	defer r.mu.RUnlock()
	b, err := json.MarshalIndent(r.sorted(), "", "  ")
	if err != nil {
		return err
	}
	_ = os.MkdirAll(filepath.Dir(r.path), 0o755)
	return os.WriteFile(r.path, b, 0o644)
}

func (r *Registry) Add(n Node) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.m == nil {
		r.m = map[string]Node{}
	}
	r.m[n.Name] = n
}

func (r *Registry) Remove(name string) { r.mu.Lock(); delete(r.m, name); r.mu.Unlock() }

func (r *Registry) Get(name string) (Node, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	n, ok := r.m[name]
	return n, ok
}

// List returns the nodes sorted by name.
func (r *Registry) List() []Node {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.sorted()
}

// sorted orders the nodes by name. The registry is a map, and reading one in
// iteration order made 'node ls' and nodes.json reshuffle between runs.
// Callers hold the lock.
func (r *Registry) sorted() []Node {
	out := make([]Node, 0, len(r.m))
	for _, n := range r.m {
		out = append(out, n)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
