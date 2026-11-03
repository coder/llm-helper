package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/detect"
	"github.com/coder/vllm-helper/internal/nodes"
	"github.com/coder/vllm-helper/internal/profile"
	"github.com/coder/vllm-helper/internal/provider"
	"github.com/coder/vllm-helper/internal/recipe"
)

type serveSpec struct {
	// Provider is nil for vLLM, so a spec built without one keeps meaning
	// what it meant before there was a choice.
	Provider provider.Provider
	provider.Spec
}

func serveSpecFromActive(active *config.EnvFile) (serveSpec, error) {
	prov, err := provider.Lookup(active.Get(provider.Key))
	if err != nil {
		return serveSpec{}, err
	}
	return serveSpec{Provider: prov, Spec: provider.Spec{
		Bin:            active.Get(prov.BinKey()),
		Model:          active.Get("MODEL"),
		Port:           active.Get("PORT"),
		MaxModelLen:    active.Get("MAX_MODEL_LEN"),
		MemUtilization: active.Get("GPU_MEMORY_UTILIZATION"),
		APIKey:         active.Get("VLLM_API_KEY"),
		KVCacheBytes:   active.Get("KV_CACHE_MEMORY_BYTES"),
		ExtraArgs:      active.Get("EXTRA_ARGS"),
	}}, nil
}

func serveSpecFromProfile(cfg *config.Config, p *profile.Profile, port string) (serveSpec, error) {
	if port == "" {
		port = cfg.Port()
	}
	prov, err := profileProvider(p)
	if err != nil {
		return serveSpec{}, err
	}
	return serveSpec{Provider: prov, Spec: provider.Spec{
		Bin:            cfg.Lookup(prov.BinKey(), ""),
		Model:          p.Model(),
		Port:           port,
		MaxModelLen:    p.File.Get("MAX_MODEL_LEN"),
		MemUtilization: p.File.Get("GPU_MEMORY_UTILIZATION"),
		APIKey:         cfg.APIKey(),
		KVCacheBytes:   p.File.Get("KV_CACHE_MEMORY_BYTES"),
		ExtraArgs:      p.ExtraArgs(),
	}}, nil
}

// profileProvider is the provider a profile is served by. Only the profile's
// own PROVIDER counts: config.env's names the default for new profiles, and
// letting it reach an existing one would hand a vLLM profile's flags to
// another server.
func profileProvider(p *profile.Profile) (provider.Provider, error) {
	return provider.Lookup(p.Provider())
}

func (s serveSpec) provider() provider.Provider {
	if s.Provider != nil {
		return s.Provider
	}
	p, _ := provider.Lookup(provider.VLLM)
	return p
}

func (s serveSpec) validate() error {
	return provider.Validate(s.provider(), s.Spec)
}

// args is the provider's command line with extra appended. Validation comes
// first so a broken spec is reported before the memory fit prints anything.
func (s serveSpec) args(extra ...string) ([]string, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	prov := s.provider()
	ps := s.Spec
	if prov.Capabilities().VLLMArgs {
		ps.MemUtilization = fitMemUtilization(s.MemUtilization, s.KVCacheBytes != "")
	}
	args, err := prov.Command(ps)
	if err != nil {
		return nil, err
	}
	args = append(args, extra...)
	if prov.Capabilities().VLLMArgs {
		if err := recipe.CheckForbidden(args, localArch()); err != nil {
			return nil, fmt.Errorf("refusing to start: %w", err)
		}
	}
	return args, nil
}

// localArch is this machine's GPU architecture, whose forbidden flags a
// command line run here is checked against. A variable so tests can name one.
var localArch = func() string { return detect.Probe().Arch }

// checkNodes refuses the command line on any node whose GPU architecture is
// known to abort on one of its flags. It runs before a shard touches its
// nodes, so a flag that would crash a worker is refused while whatever is
// serving still serves. The head runs here, so it is checked against this
// machine's hardware as probed now rather than what the registry recorded,
// which may be nothing. Another node probed before architectures were
// recorded has none and is not checked.
func (s serveSpec) checkNodes(members []nodes.Node, head nodes.Node) error {
	prov := s.provider()
	if !prov.Capabilities().VLLMArgs {
		return nil
	}
	args, err := prov.Command(s.Spec)
	if err != nil {
		return err
	}
	for _, n := range members {
		arch := n.Caps.Arch
		if n.Name == head.Name {
			if live := localArch(); live != "" {
				arch = live
			}
		}
		if err := recipe.CheckForbidden(args, arch); err != nil {
			return fmt.Errorf("refusing to start on node %s: %w", n.Name, err)
		}
	}
	return nil
}

// requireVLLM refuses a command built from vLLM's own flags, such as a shard's
// Ray or multiprocessing executor, for a profile another provider serves.
func requireVLLM(p *profile.Profile, what string) error {
	prov, err := profileProvider(p)
	if err != nil {
		return err
	}
	if prov.Name() != provider.VLLM {
		return fmt.Errorf("%s runs vLLM, but profile '%s' is served by %s", what, p.Name, prov.Name())
	}
	return nil
}

func serveEnvFromActive(active *config.EnvFile) []string {
	env := os.Environ()
	for _, k := range active.Keys() {
		env = append(env, k+"="+active.Get(k))
	}
	return withPCIBusOrder(env)
}

// withPCIBusOrder numbers the server's GPUs in PCI bus order, replacing any
// CUDA_DEVICE_ORDER it would otherwise get. CUDA's default numbers them
// fastest first, so on a machine mixing GPU models an index in
// CUDA_VISIBLE_DEVICES would name a different card than the one nvidia-smi
// gives that index, and detection, which can only read nvidia-smi, would
// size and check the wrong GPU.
func withPCIBusOrder(env []string) []string {
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		if !strings.HasPrefix(kv, "CUDA_DEVICE_ORDER=") {
			out = append(out, kv)
		}
	}
	return append(out, "CUDA_DEVICE_ORDER=PCI_BUS_ID")
}

// withProviderEnv adds the provider's own environment entries to env. Each one
// replaces an inherited entry of the same name, because a duplicate would
// leave the server reading whichever copy its libc finds first.
func withProviderEnv(env []string, s serveSpec) []string {
	ep, ok := s.provider().(provider.EnvProvider)
	if !ok {
		return env
	}
	add := ep.Env(s.Spec)
	if len(add) == 0 {
		return env
	}
	out := make([]string, 0, len(env)+len(add))
	for _, kv := range env {
		name, _, _ := strings.Cut(kv, "=")
		replaced := false
		for _, a := range add {
			if n, _, _ := strings.Cut(a, "="); n == name {
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, kv)
		}
	}
	return append(out, add...)
}

func serveEnvFromProfile(cfg *config.Config, p *profile.Profile) []string {
	return withPCIBusOrder(append(os.Environ(), config.MergedEnv(cfg.File, cfg.Secrets, p.File)...))
}

// redactServeArgs hides credentials in a command about to be printed. Both
// spellings vLLM accepts have to be covered: EXTRA_ARGS is free text, and
// "--api-key=sk-..." there would otherwise reach a dry run verbatim.
func redactServeArgs(args []string) []string {
	out := append([]string{}, args...)
	for i := 0; i < len(out); i++ {
		if !strings.HasPrefix(out[i], "--") {
			continue
		}
		if flag, _, inline := strings.Cut(out[i], "="); inline {
			if detect.LooksSecret(flag) {
				out[i] = flag + "=<redacted>"
			}
			continue
		}
		if detect.LooksSecret(out[i]) && i+1 < len(out) {
			out[i+1] = "<redacted>"
			i++
		}
	}
	return out
}
