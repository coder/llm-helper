package profile_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/profile"
)

func loadFrom(t *testing.T, body string) *profile.Profile {
	t.Helper()
	path := filepath.Join(t.TempDir(), "p.env")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := config.ParseEnvFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return &profile.Profile{Name: "p", File: f}
}

// The launch topology is what lets status, logs and restart act on the same
// cluster later without flags, so parsing has to survive the way people write
// these files by hand: spaces after commas, a trailing comma, no master port.
func TestLaunchTopologyRoundTrips(t *testing.T) {
	p := loadFrom(t, "NODES=\" spark1, spark2 ,\"\nCONTAINER=vllm-node\nMASTER_PORT=29501\nSSH_KEY=/home/u/.ssh/k\n")
	got := p.Nodes()
	if len(got) != 2 || got[0] != "spark1" || got[1] != "spark2" {
		t.Errorf("Nodes() = %q, want [spark1 spark2]", got)
	}
	if p.Container() != "vllm-node" {
		t.Errorf("Container() = %q", p.Container())
	}
	if p.MasterPort() != 29501 {
		t.Errorf("MasterPort() = %d", p.MasterPort())
	}
	if p.SSHKey() != "/home/u/.ssh/k" {
		t.Errorf("SSHKey() = %q", p.SSHKey())
	}

	// A profile that predates this recording, or a native one, must read as
	// "no topology" rather than as a one-node cluster called "".
	empty := loadFrom(t, "MODEL=x\n")
	if len(empty.Nodes()) != 0 {
		t.Errorf("Nodes() on a profile without NODES = %q, want none", empty.Nodes())
	}
	if empty.MasterPort() != 0 {
		t.Errorf("MasterPort() unset = %d, want 0 so callers can default it", empty.MasterPort())
	}
}

// Generated is what makes a profile the override layer of a recipe, and what
// sends start/switch to "recipes launch", so it must not be fooled by whitespace.
func TestGeneratedTracksTheRecipeKey(t *testing.T) {
	cases := []struct {
		name, body, wantRecipe string
		wantGenerated          bool
	}{
		{"hand written", "MODEL=org/m\n", "", false},
		{"empty key", "MODEL=org/m\nRECIPE=\n", "", false},
		{"whitespace only", "MODEL=org/m\nRECIPE=\"   \"\n", "", false},
		{"generated", "MODEL=org/m\nRECIPE=deepseek-v4-flash-2spark\n", "deepseek-v4-flash-2spark", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := loadFrom(t, c.body)
			if got := p.Generated(); got != c.wantGenerated {
				t.Errorf("Generated() = %v, want %v", got, c.wantGenerated)
			}
			if got := p.Recipe(); got != c.wantRecipe {
				t.Errorf("Recipe() = %q, want %q", got, c.wantRecipe)
			}
		})
	}
}

// A recipe publishes its own label; deriving from the repo would render
// "DeepSeek-V4-Flash-0731" as "DeepSeek V4 Flash 0731".
func TestDisplayNamePrefersTheExplicitKey(t *testing.T) {
	derived := loadFrom(t, "MODEL=deepseek-ai/DeepSeek-V4-Flash-0731\n")
	if got := derived.DisplayName(); got != "DeepSeek V4 Flash 0731" {
		t.Errorf("derived DisplayName() = %q", got)
	}
	explicit := loadFrom(t, "MODEL=deepseek-ai/DeepSeek-V4-Flash-0731\nDISPLAY_NAME=\"DeepSeek V4 Flash\"\n")
	if got := explicit.DisplayName(); got != "DeepSeek V4 Flash" {
		t.Errorf("explicit DisplayName() = %q", got)
	}
}
