package cli

import (
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/engine"
)

func newSetupCmd() *cobra.Command {
	var providerName string
	c := &cobra.Command{
		Use:     "setup",
		Short:   "First-time setup: detect hardware and the inference server, write config, offer to install the service",
		GroupID: groupMachine,
		Args:    cobra.NoArgs,
		RunE:    func(*cobra.Command, []string) error { return runSetup(providerName) },
	}
	c.Flags().StringVar(&providerName, "provider", engine.VLLM,
		"inference server new profiles use: "+strings.Join(engine.ProcessNames(), ", ")+" (only written to a new config)")
	return c
}

// runSetup is first-time setup for binary installs: detect hardware and any
// existing vLLM, write config + secrets, install the service unit, and
// offer to adopt a running server. Idempotent; never overwrites existing
// config and never restarts anything.
func runSetup(providerName string) error {
	prov, err := engine.Lookup(providerName)
	if err != nil {
		return err
	}
	if !prov.Has(engine.RuntimeProcess) {
		return fmt.Errorf("%s runs only in a container, so the service cannot serve it; launch it with 'llm-helper recipes launch'", prov.Name())
	}
	dirs := config.ResolveDirs()
	fmt.Printf("== %s setup ==\n", cmdName)
	fmt.Println()

	// Found before probing, because on a Mac a vllm decides whether memory
	// is budgeted for the GPU (vllm-metal) or the CPU backend.
	_, statErr := os.Stat(dirs.ConfigFile())
	bin := ""
	if statErr != nil {
		bin = lookBin(prov)
		if prov.Name() == engine.VLLM {
			useMetalProbe(bin)
		}
	} else if found, filled, err := fillMissingVLLMBin(dirs.ConfigFile(), findVLLM); err != nil {
		return err
	} else if filled {
		fmt.Printf("Detected vllm at %s and recorded it as VLLM_BIN in %s\n\n", found, dirs.ConfigFile())
		useMetalProbe(found)
	}

	hw := detect.Probe()
	fmt.Printf("Hardware:  %d GB memory", hw.TotalMemGB)
	if hw.GPUName != "" {
		kind := "discrete VRAM"
		if hw.Unified {
			kind = "unified memory"
		}
		fmt.Printf(", %s (%s)", hw.GPUSummary(), kind)
	}
	if hw.Metal {
		fmt.Print(", serving on the GPU through vllm-metal")
		if hw.MetalWorkingSetGB > 0 {
			fmt.Printf(" (%d GB Metal working set)", hw.MetalWorkingSetGB)
		}
	}
	fmt.Printf("\nBudget:    ~%d GB of model weights (%s)\n\n", hw.BudgetGB, hw.BudgetReason)

	for _, d := range []string{dirs.Config, dirs.Profiles, dirs.State} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
	}
	if err := os.Chmod(dirs.Config, 0o700); err != nil {
		return err
	}

	if statErr == nil {
		fmt.Println("Config already exists at", dirs.ConfigFile(), "-- keeping it.")
		if prov.Name() != engine.VLLM {
			fmt.Printf("--provider only shapes a new config; set PROVIDER=%s and %s in it by hand.\n", prov.Name(), prov.BinKey())
		}
	} else {
		content := fmt.Sprintf(`# llm-helper global config. Sourced (exported) into the server's
# environment whenever a model starts -- values here apply to every profile.

# First port a model's OpenAI-compatible endpoint listens on. Each model
# running at the same time takes the next free port from here.
PORT=8001

# The inference server new profiles are made for: %s. Each profile
# records its own, so changing this leaves existing profiles alone.
PROVIDER=%s

# Full path to the %s executable (usually inside a venv). REQUIRED.
%s=%s

# How each model's %s unit is managed: "system" (needs sudo) or "user".
SERVICE_SCOPE=%s

# Budget for model weights, used by 'llm-helper search' to flag models that
# won't fit. Measured from this machine on every run -- currently ~%d GB, being
# %s. Uncomment only to override that.
#MODEL_BUDGET_GB=%d

# How many full-length requests each profile's KV cache is sized to hold.
# Defaults to %d. A property of the workload rather than the machine: lower it
# on a single-user box, raise it for a shared endpoint.
#KV_CONTEXT_TARGET=%d

# --- Optional: Coder Agents sync -------------------------------------------
# If all three are set (token in secrets.env), a Coder deployment's AI-model
# picker is kept in sync with the models running here.
#CODER_URL=https://coder.example.com
#CODER_AI_PROVIDER_ID=
# Only needed if the deployment has more than one organization.
#CODER_ORGANIZATION=default
`, strings.Join(engine.ProcessNames(), ", "), prov.Name(), prov.BinName(), prov.BinKey(), bin,
			config.ServiceManager, config.DefaultServiceScope,
			hw.BudgetGB, hw.BudgetReason, hw.BudgetGB,
			detect.DefaultContextTarget, detect.DefaultContextTarget)
		if err := os.WriteFile(dirs.ConfigFile(), []byte(content), 0o644); err != nil {
			return err
		}
		fmt.Println("Wrote", dirs.ConfigFile())
		if bin == "" {
			fmt.Println()
			if prov.Name() == engine.VLLM {
				for _, line := range vllmInstallHint {
					fmt.Println(line)
				}
			} else {
				fmt.Printf("WARNING: no '%s' executable found on PATH. Install it (e.g.\n", prov.BinName())
				fmt.Printf("'%s'), then set %s\n", prov.InstallHint(runtime.GOOS), prov.BinKey())
				fmt.Println("in config.env to its full path.")
			}
			for _, name := range engine.ProcessNames() {
				other, _ := engine.Lookup(name)
				if name == prov.Name() || lookBin(other) == "" {
					continue
				}
				fmt.Printf("%s is installed, though: to serve with it, set PROVIDER=%s and\n", other.BinName(), other.Name())
				fmt.Printf("%s=%s in config.env.\n", other.BinKey(), lookBin(other))
			}
			fmt.Println()
		} else {
			fmt.Printf("Detected %s at %s\n", prov.BinName(), bin)
		}
		if !prov.Capabilities().APIKey {
			fmt.Printf("Note: %s checks no API key, so anything that can reach its port can use it.\n", prov.BinName())
			fmt.Println("See docs/providers.md before letting it listen beyond this machine.")
		}
	}

	if _, err := os.Stat(dirs.SecretsFile()); err == nil {
		fmt.Println("Secrets already exist at", dirs.SecretsFile(), "-- keeping them.")
	} else {
		buf := make([]byte, 32)
		if _, err := rand.Read(buf); err != nil {
			return err
		}
		key := "sk-" + base64.RawURLEncoding.EncodeToString(buf)
		content := fmt.Sprintf(`# Secrets for llm-helper. Never commit this file anywhere.

# API key clients must send to the vLLM endpoint (generated at setup).
VLLM_API_KEY=%s

# Only needed if Coder sync is configured in config.env:
#CODER_API_TOKEN=
`, key)
		if err := os.WriteFile(dirs.SecretsFile(), []byte(content), 0o600); err != nil {
			return err
		}
		fmt.Println("Generated a new VLLM_API_KEY into", dirs.SecretsFile())
	}

	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	// Offer to adopt a server that is already running.
	fmt.Println()
	if err := adoptInto(cfg); err != nil && !errors.Is(err, os.ErrNotExist) {
		fmt.Fprintln(os.Stderr, "adoption scan failed (continuing):", err)
	}

	fmt.Println()
	fmt.Println("Setup complete. Next: 'llm-helper new <org/model>' or 'llm-helper search <query>'.")
	fmt.Println("'llm-helper up <profile>' installs the service that runs each model.")
	return nil
}

// fillMissingVLLMBin records a vllm installed since config.env was written
// without one. The first setup leaves VLLM_BIN empty when it finds nothing and
// tells the user to install vLLM and run setup again, so the rerun has to fill
// it in. A VLLM_BIN already set is never replaced, and a config made for
// another provider is left alone. It returns the executable it recorded and
// whether it wrote config.env.
func fillMissingVLLMBin(path string, find func() string) (string, bool, error) {
	f, err := config.ParseEnvFile(path)
	if err != nil {
		return "", false, fmt.Errorf("reading %s: %w", path, err)
	}
	if prov, err := engine.Lookup(f.Get(engine.Key)); err != nil || prov.Name() != engine.VLLM {
		return "", false, nil
	}
	if strings.TrimSpace(f.Get("VLLM_BIN")) != "" {
		return "", false, nil
	}
	bin := find()
	if bin == "" {
		return "", false, nil
	}
	f.Set("VLLM_BIN", bin)
	if err := f.Write(path, 0o644); err != nil {
		return "", false, fmt.Errorf("recording VLLM_BIN=%s in %s: %w", bin, path, err)
	}
	return bin, true, nil
}

// lookBin finds a provider's server, "" when there is none. vLLM's is looked
// for by findVLLM, which resolves symlinks because its venv's python is looked
// up beside it. Any other server is kept as found on PATH, so a package
// manager's stable link is what config.env records and an upgrade does not
// break it.
func lookBin(prov engine.Provider) string {
	if prov.Name() == engine.VLLM {
		return findVLLM()
	}
	p, err := exec.LookPath(prov.BinName())
	if err != nil {
		return ""
	}
	return p
}
