package config_test

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/coder/llm-helper/internal/config"
)

// writeFile creates a file and the directories leading to it.
func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

// newDirs points every directory at a fresh temp tree.
func newDirs(t *testing.T) config.Dirs {
	t.Helper()
	root := t.TempDir()
	t.Setenv("VLLM_HELPER_CONFIG_DIR", filepath.Join(root, "config"))
	t.Setenv("VLLM_HELPER_STATE_DIR", filepath.Join(root, "state"))
	t.Setenv("VLLM_HELPER_DATA_DIR", filepath.Join(root, "data"))
	return config.ResolveDirs()
}

func TestResolveDirsPrefersExplicitOverride(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", filepath.Join(root, "home"))
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(root, "xdg"))
	t.Setenv("VLLM_HELPER_CONFIG_DIR", filepath.Join(root, "explicit"))

	if got, want := config.ResolveDirs().Config, filepath.Join(root, "explicit"); got != want {
		t.Errorf("Config = %q, want the explicit override %q", got, want)
	}
}

func TestResolveDirsFallsBackThroughXDGToHome(t *testing.T) {
	root := t.TempDir()
	t.Setenv("VLLM_HELPER_CONFIG_DIR", "")
	t.Setenv("HOME", root)

	t.Run("xdg set", func(t *testing.T) {
		t.Setenv("XDG_CONFIG_HOME", filepath.Join(root, "xdg"))
		if got, want := config.ResolveDirs().Config, filepath.Join(root, "xdg", "vllm-helper"); got != want {
			t.Errorf("Config = %q, want %q", got, want)
		}
	})

	t.Run("xdg unset", func(t *testing.T) {
		t.Setenv("XDG_CONFIG_HOME", "")
		if got, want := config.ResolveDirs().Config, filepath.Join(root, ".config", "vllm-helper"); got != want {
			t.Errorf("Config = %q, want %q", got, want)
		}
	})
}

// TestDirsLayout pins paths that existing installs already use.
func TestDirsLayout(t *testing.T) {
	t.Parallel()

	d := config.Dirs{Config: "/c", State: "/s", Data: "/d", Profiles: "/c/models"}
	tests := []struct {
		name string
		got  string
		want string
	}{
		{"config file", d.ConfigFile(), "/c/config.env"},
		{"secrets file", d.SecretsFile(), "/c/secrets.env"},
		{"active env", d.ActiveEnvFile(), "/s/active.env"},
		{"active profile", d.ActiveProfileFile(), "/s/active-profile"},
		{"coder ids", d.CoderIDsFile(), "/s/coder-ids.env"},
		{"profile file", d.ProfileFile("qwen"), "/c/models/qwen.env"},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %q, want %q", tt.name, tt.got, tt.want)
		}
	}
}

func TestLoadReadsConfigAndSecrets(t *testing.T) {
	dirs := newDirs(t)
	writeFile(t, dirs.ConfigFile(), "PORT=9001\nVLLM_BIN=/opt/vllm\n")
	writeFile(t, dirs.SecretsFile(), "VLLM_API_KEY=sk-secret\n")

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got, want := cfg.Port(), "9001"; got != want {
		t.Errorf("Port() = %q, want %q", got, want)
	}
	if got, want := cfg.VLLMBin(), "/opt/vllm"; got != want {
		t.Errorf("VLLMBin() = %q, want %q", got, want)
	}
	if got, want := cfg.APIKey(), "sk-secret"; got != want {
		t.Errorf("APIKey() = %q, want %q", got, want)
	}
}

func TestLoadWithoutConfigFails(t *testing.T) {
	newDirs(t)
	if _, err := config.Load(); err == nil {
		t.Fatal("expected an error when config.env is absent")
	}
}

func TestDefaultsMatchThePlatform(t *testing.T) {
	dirs := newDirs(t)
	writeFile(t, dirs.ConfigFile(), "# nothing set\n")

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tests := []struct {
		name string
		got  string
		want string
	}{
		{"port", cfg.Port(), "8001"},
		// Empty rather than a number: an unset budget means "measure this
		// machine", not "assume 90 GB", so that a config written on one box
		// cannot quietly mis-describe another.
		{"model budget", cfg.ModelBudgetGB(), ""},
		{"service name", cfg.ServiceName(), config.DefaultServiceName},
		{"service scope", cfg.ServiceScope(), config.DefaultServiceScope},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %q, want %q", tt.name, tt.got, tt.want)
		}
	}
	if cfg.APIKey() != "" {
		t.Errorf("APIKey() = %q, want empty when no secrets file exists", cfg.APIKey())
	}
}

func TestSecretsOverrideConfig(t *testing.T) {
	dirs := newDirs(t)
	writeFile(t, dirs.ConfigFile(), "HF_TOKEN=from-config\n")
	writeFile(t, dirs.SecretsFile(), "HF_TOKEN=from-secrets\n")

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got, want := cfg.HFToken(), "from-secrets"; got != want {
		t.Errorf("HFToken() = %q, want %q -- secrets.env should win", got, want)
	}
}

func TestCoderConfiguredNeedsEveryField(t *testing.T) {
	const (
		url   = "CODER_URL=https://coder.example.com\n"
		token = "CODER_API_TOKEN=tok\n"
		id    = "CODER_AI_PROVIDER_ID=prov\n"
	)
	tests := []struct {
		name    string
		content string
		want    bool
	}{
		{"all three", url + token + id, true},
		{"missing url", token + id, false},
		{"missing token", url + id, false},
		{"missing provider", url + token, false},
		{"none", "# empty\n", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dirs := newDirs(t)
			writeFile(t, dirs.ConfigFile(), tt.content)
			cfg, err := config.Load()
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if got := cfg.CoderConfigured(); got != tt.want {
				t.Errorf("CoderConfigured() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestActiveProfileReadsStateFile(t *testing.T) {
	dirs := newDirs(t)
	writeFile(t, dirs.ConfigFile(), "PORT=8001\n")

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := cfg.ActiveProfile(); got != "" {
		t.Errorf("ActiveProfile() = %q, want empty before anything is switched to", got)
	}

	// Trailing newline, as "echo name >" leaves.
	writeFile(t, dirs.ActiveProfileFile(), "qwen3-0.6b\n")
	if got, want := cfg.ActiveProfile(), "qwen3-0.6b"; got != want {
		t.Errorf("ActiveProfile() = %q, want %q", got, want)
	}
}

func TestMergedEnvLastFileWins(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	parse := func(name, content string) *config.EnvFile {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		f, err := config.ParseEnvFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	base := parse("base.env", "PORT=8001\nMODEL=a/b\n")
	over := parse("over.env", "MODEL=c/d\n")

	got := config.MergedEnv(base, nil, over)
	want := []string{"MODEL=c/d", "PORT=8001"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MergedEnv = %#v, want %#v", got, want)
	}
}
