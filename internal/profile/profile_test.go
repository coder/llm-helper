package profile_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/profile"
)

func newDirs(t *testing.T) config.Dirs {
	t.Helper()
	root := t.TempDir()
	d := config.Dirs{
		Config:   filepath.Join(root, "config"),
		State:    filepath.Join(root, "state"),
		Data:     filepath.Join(root, "data"),
		Profiles: filepath.Join(root, "config", "models"),
	}
	if err := os.MkdirAll(d.Profiles, 0o755); err != nil {
		t.Fatal(err)
	}
	return d
}

func writeProfile(t *testing.T, dirs config.Dirs, name, content string) {
	t.Helper()
	if err := os.WriteFile(dirs.ProfileFile(name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestNameFromRepo(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		repo string
		want string
	}{
		{"org and model", "Qwen/Qwen3.5-27B", "qwen3.5-27b"},
		{"bare model", "Ornith-1.0-35B-FP8", "ornith-1.0-35b-fp8"},
		{"nested path keeps the last segment", "a/b/c-D", "c-d"},
		{"empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := profile.NameFromRepo(tt.repo); got != tt.want {
				t.Errorf("NameFromRepo(%q) = %q, want %q", tt.repo, got, tt.want)
			}
		})
	}
}

func TestDisplayName(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		model string
		want  string
	}{
		{"strips the org and swaps dashes", "deepreinforce-ai/Ornith-1.0-35B-FP8", "Ornith 1.0 35B FP8"},
		{"no org", "Llama-3-8B", "Llama 3 8B"},
		{"no model set", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			dirs := newDirs(t)
			writeProfile(t, dirs, "p", "MODEL="+tt.model+"\n")
			p, err := profile.Load(dirs, "p")
			if err != nil {
				t.Fatal(err)
			}
			if got := p.DisplayName(); got != tt.want {
				t.Errorf("DisplayName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLoadReadsEveryField(t *testing.T) {
	t.Parallel()

	dirs := newDirs(t)
	writeProfile(t, dirs, "laguna", `# Status: TESTED AND WORKING
# second comment
MODEL=poolside/Laguna-S-2.1-NVFP4
MAX_MODEL_LEN=131072
EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser poolside_v1"
`)
	p, err := profile.Load(dirs, "laguna")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tests := []struct {
		name string
		got  string
		want string
	}{
		{"name", p.Name, "laguna"},
		{"model", p.Model(), "poolside/Laguna-S-2.1-NVFP4"},
		{"max model len", p.MaxModelLen(), "131072"},
		{"extra args", p.ExtraArgs(), "--enable-auto-tool-choice --tool-call-parser poolside_v1"},
		{"status", p.Status(), "TESTED AND WORKING"},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %q, want %q", tt.name, tt.got, tt.want)
		}
	}
}

func TestLoadUnknownProfileNamesThePathItLookedAt(t *testing.T) {
	t.Parallel()

	dirs := newDirs(t)
	_, err := profile.Load(dirs, "nope")
	if err == nil {
		t.Fatal("expected an error for a profile that doesn't exist")
	}
	if !strings.Contains(err.Error(), dirs.ProfileFile("nope")) {
		t.Errorf("error %q should name the path it looked at", err)
	}
}

func TestStatusIsEmptyWithoutTheHeader(t *testing.T) {
	t.Parallel()

	dirs := newDirs(t)
	writeProfile(t, dirs, "p", "# just a note\nMODEL=a/b\n")
	p, err := profile.Load(dirs, "p")
	if err != nil {
		t.Fatal(err)
	}
	if got := p.Status(); got != "" {
		t.Errorf("Status() = %q, want empty without a '# Status:' line", got)
	}
}

func TestListSortsAndSkipsNonProfiles(t *testing.T) {
	t.Parallel()

	dirs := newDirs(t)
	for _, name := range []string{"zeta", "alpha", "mid"} {
		writeProfile(t, dirs, name, "MODEL=org/"+name+"\n")
	}
	if err := os.WriteFile(filepath.Join(dirs.Profiles, "notes.txt"), []byte("ignored"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dirs.Profiles, "sub.env"), 0o755); err != nil {
		t.Fatal(err)
	}

	profiles, err := profile.List(dirs)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	var got []string
	for _, p := range profiles {
		got = append(got, p.Name)
	}
	want := []string{"alpha", "mid", "zeta"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("List = %v, want %v", got, want)
	}
}

func TestListWithoutTheProfilesDirIsEmpty(t *testing.T) {
	t.Parallel()

	dirs := newDirs(t)
	if err := os.RemoveAll(dirs.Profiles); err != nil {
		t.Fatal(err)
	}
	profiles, err := profile.List(dirs)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(profiles) != 0 {
		t.Errorf("List = %v, want empty before any profile exists", profiles)
	}
}

func TestCreateDefaultsAndOverrides(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		opts        profile.CreateOpts
		wantLen     string
		wantMemUtil string
	}{{
		name:        "defaults",
		opts:        profile.CreateOpts{Repo: "Qwen/Qwen3-8B"},
		wantLen:     "131072",
		wantMemUtil: config.DefaultMemUtilization,
	}, {
		name: "explicit values win",
		opts: profile.CreateOpts{
			Repo:           "Qwen/Qwen3-8B",
			MaxModelLen:    "40960",
			MemUtilization: "0.92",
		},
		wantLen:     "40960",
		wantMemUtil: "0.92",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			dirs := newDirs(t)
			p, err := profile.Create(dirs, tt.opts)
			if err != nil {
				t.Fatalf("Create: %v", err)
			}
			if p.Name != "qwen3-8b" {
				t.Errorf("Name = %q, want %q", p.Name, "qwen3-8b")
			}
			if got := p.MaxModelLen(); got != tt.wantLen {
				t.Errorf("MAX_MODEL_LEN = %q, want %q", got, tt.wantLen)
			}
			if got := p.File.Get("GPU_MEMORY_UTILIZATION"); got != tt.wantMemUtil {
				t.Errorf("GPU_MEMORY_UTILIZATION = %q, want %q", got, tt.wantMemUtil)
			}
		})
	}
}

// A profile for another provider records which one, and leaves out vLLM's
// sizing keys: written there, they read as settings that do something.
func TestCreateForAProvider(t *testing.T) {
	t.Parallel()
	dirs := newDirs(t)
	p, err := profile.Create(dirs, profile.CreateOpts{
		Repo: "mlx-community/Qwen3-4B-4bit", Provider: "mlx", ExtraArgs: "--max-tokens 8192",
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if got := p.Provider(); got != "mlx" {
		t.Errorf("Provider() = %q, want mlx", got)
	}
	if p.Model() != "mlx-community/Qwen3-4B-4bit" || p.ExtraArgs() != "--max-tokens 8192" {
		t.Errorf("MODEL/EXTRA_ARGS = %q / %q", p.Model(), p.ExtraArgs())
	}
	for _, k := range []string{"MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION", "KV_CACHE_MEMORY_BYTES"} {
		if p.File.Has(k) {
			t.Errorf("%s was written into a non-vLLM profile", k)
		}
	}

	// And a vLLM profile is the one it always was: no PROVIDER line.
	v, err := profile.Create(dirs, profile.CreateOpts{Repo: "Qwen/Qwen3-8B"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if v.File.Has("PROVIDER") || v.Provider() != "" {
		t.Error("a vLLM profile gained a PROVIDER line")
	}
}

func TestCreateRefusesToOverwrite(t *testing.T) {
	t.Parallel()

	dirs := newDirs(t)
	opts := profile.CreateOpts{Repo: "Qwen/Qwen3-8B", Status: "first"}
	if _, err := profile.Create(dirs, opts); err != nil {
		t.Fatalf("Create: %v", err)
	}
	before, err := os.ReadFile(dirs.ProfileFile("qwen3-8b"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := profile.Create(dirs, profile.CreateOpts{Repo: "Qwen/Qwen3-8B", Status: "second"}); err == nil {
		t.Fatal("expected an error when the profile already exists")
	}
	after, err := os.ReadFile(dirs.ProfileFile("qwen3-8b"))
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Error("a rejected Create must leave the existing profile untouched")
	}
}

// A profile written by Create has to read back identically, including the
// quoted JSON that real EXTRA_ARGS carry.
func TestCreateLoadRoundTrip(t *testing.T) {
	t.Parallel()

	dirs := newDirs(t)
	const extra = `--enable-auto-tool-choice --default-chat-template-kwargs {"thinking":true}`
	created, err := profile.Create(dirs, profile.CreateOpts{
		Repo:        "deepreinforce-ai/Ornith-1.0-35B-FP8",
		MaxModelLen: "65536",
		ExtraArgs:   extra,
		Status:      "ADOPTED from a running server",
		Source:      "vllm-helper adopt",
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	loaded, err := profile.Load(dirs, created.Name)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tests := []struct {
		name string
		got  string
		want string
	}{
		{"model", loaded.Model(), "deepreinforce-ai/Ornith-1.0-35B-FP8"},
		{"max model len", loaded.MaxModelLen(), "65536"},
		{"extra args", loaded.ExtraArgs(), extra},
		{"display name", loaded.DisplayName(), "Ornith 1.0 35B FP8"},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %q, want %q", tt.name, tt.got, tt.want)
		}
	}
	if !strings.HasPrefix(loaded.Status(), "ADOPTED from a running server") {
		t.Errorf("Status() = %q, want the status passed to Create", loaded.Status())
	}
	raw, err := os.ReadFile(dirs.ProfileFile(created.Name))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "vllm-helper adopt") {
		t.Errorf("the header should record the source, got:\n%s", raw)
	}
}

func TestRemove(t *testing.T) {
	t.Parallel()

	dirs := newDirs(t)
	writeProfile(t, dirs, "gone", "MODEL=a/b\n")
	if err := profile.Remove(dirs, "gone"); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if _, err := profile.Load(dirs, "gone"); err == nil {
		t.Error("the profile should be gone after Remove")
	}
	if err := profile.Remove(dirs, "gone"); err == nil {
		t.Error("removing a profile twice should report the second attempt")
	}
}
