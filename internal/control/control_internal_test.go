package control

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/profile"
)

// active.env is sourced by the service, so what goes into it decides the argv
// vLLM runs with. This locks the three properties a second implementation had
// already broken -- it sorted the keys, dropped empty values, and re-quoted
// values that were fine as written -- because the symptom of getting any of
// them wrong is a server that serves something subtly different from what the
// profile says, with nothing in any log to say so.
func TestWriteActiveEnvCopiesLinesThrough(t *testing.T) {
	root := t.TempDir()
	dirs := config.Dirs{
		Config:   filepath.Join(root, "config"),
		State:    filepath.Join(root, "state"),
		Data:     filepath.Join(root, "data"),
		Profiles: filepath.Join(root, "config", "models"),
	}
	for _, d := range []string{dirs.Config, dirs.State, dirs.Profiles} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write := func(path, body string) {
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// ZZZ first and AAA last, so sorting would be visible.
	write(dirs.ConfigFile(), "# a comment\nZZZ_FIRST=1\nPORT=8001\nEMPTY_ON_PURPOSE=\n")
	write(dirs.SecretsFile(), "VLLM_API_KEY=sk-secret\n")
	write(dirs.ProfileFile("demo"),
		"MODEL=org/demo\nEXTRA_ARGS=--tool-call-parser qwen3_coder\nAAA_LAST=z\n")

	cfgFile, err := config.ParseEnvFile(dirs.ConfigFile())
	if err != nil {
		t.Fatal(err)
	}
	secrets, err := config.ParseEnvFile(dirs.SecretsFile())
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{Dirs: dirs, File: cfgFile, Secrets: secrets}
	p, err := profile.Load(dirs, "demo")
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteActiveEnv(cfg, p); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(dirs.ActiveEnvFile())
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)

	want := "ZZZ_FIRST=1\nPORT=8001\nEMPTY_ON_PURPOSE=\nVLLM_API_KEY=sk-secret\n" +
		"MODEL=org/demo\nEXTRA_ARGS=--tool-call-parser qwen3_coder\nAAA_LAST=z\n"
	if got != want {
		t.Errorf("active.env mismatch\n got:\n%s\nwant:\n%s", got, want)
	}

	// Said individually, so a failure names which property broke.
	if strings.Contains(got, `EXTRA_ARGS="`) {
		t.Error("EXTRA_ARGS was re-quoted; it must be copied through as written")
	}
	if !strings.Contains(got, "EMPTY_ON_PURPOSE=\n") {
		t.Error("an empty value was dropped; set-to-empty is not the same as unset")
	}
	if strings.Index(got, "ZZZ_FIRST") > strings.Index(got, "AAA_LAST") {
		t.Error("keys were sorted; file order carries the config/secrets/profile layering")
	}
	if strings.Contains(got, "# a comment") {
		t.Error("a comment was copied into a file that gets sourced")
	}

	// Secrets land here, so the file must not be readable by other users.
	info, err := os.Stat(dirs.ActiveEnvFile())
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("active.env mode = %o, want 600: it contains every secret on the machine", perm)
	}

	if b, err := os.ReadFile(dirs.ActiveProfileFile()); err != nil {
		t.Fatal(err)
	} else if strings.TrimSpace(string(b)) != "demo" {
		t.Errorf("active profile = %q, want demo", strings.TrimSpace(string(b)))
	}
}

// config.env's PROVIDER is the default for new profiles. It must not reach an
// existing profile through config.env's lines, or setting PROVIDER=mlx on a
// Mac would hand every vLLM profile's flags to mlx_lm.server.
func TestWriteActiveEnvPinsTheProfilesProvider(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		config  string
		profile string
		want    string // the PROVIDER the service will read, "" for none at all
		wantErr bool
	}{
		{name: "no default anywhere", config: "PORT=8001\n", profile: "MODEL=m\n", want: ""},
		{name: "old profile under an mlx default", config: "PROVIDER=mlx\n", profile: "MODEL=m\n", want: "vllm"},
		{name: "mlx profile under a vllm default", config: "PROVIDER=vllm\n", profile: "PROVIDER=mlx\nMODEL=m\n", want: "mlx"},
		{name: "mlx profile, no default", config: "PORT=8001\n", profile: "PROVIDER=mlx\nMODEL=m\n", want: "mlx"},
		{name: "unknown provider", config: "PORT=8001\n", profile: "PROVIDER=nope\nMODEL=m\n", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			dirs := config.Dirs{
				Config:   filepath.Join(root, "config"),
				State:    filepath.Join(root, "state"),
				Profiles: filepath.Join(root, "config", "models"),
			}
			if err := os.MkdirAll(dirs.Profiles, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(dirs.ConfigFile(), []byte(tt.config), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(dirs.ProfileFile("p"), []byte(tt.profile), 0o600); err != nil {
				t.Fatal(err)
			}
			cfgFile, err := config.ParseEnvFile(dirs.ConfigFile())
			if err != nil {
				t.Fatal(err)
			}
			cfg := &config.Config{Dirs: dirs, File: cfgFile}
			p, err := profile.Load(dirs, "p")
			if err != nil {
				t.Fatal(err)
			}

			err = WriteActiveEnv(cfg, p)
			if tt.wantErr {
				if err == nil {
					t.Fatal("an unknown provider was written into active.env")
				}
				if _, statErr := os.Stat(dirs.ActiveEnvFile()); statErr == nil {
					t.Error("active.env was written for a profile that can't be served")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			active, err := config.ParseEnvFile(dirs.ActiveEnvFile())
			if err != nil {
				t.Fatal(err)
			}
			if got := active.Get("PROVIDER"); got != tt.want {
				t.Errorf("PROVIDER in active.env = %q, want %q", got, tt.want)
			}
			if tt.want == "" && active.Has("PROVIDER") {
				t.Error("an install that never set PROVIDER got one in active.env")
			}
		})
	}
}

func TestLastMeaningfulLine(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		logs string
		want string
	}{
		{"empty", "", ""},
		{"only blank lines", "\n  \n\t\n", ""},
		{"single line", "boom", "boom"},
		{"trailing newline is ignored", "first\nlast\n", "last"},
		{"trailing blank lines are ignored", "the real error\n\n   \n", "the real error"},
		{
			// The failure from issue #13: the exception is the last line
			// of the traceback, which is what people need to see.
			name: "python traceback ends with the exception",
			logs: "Traceback (most recent call last):\n" +
				"  File \"/usr/lib/vllm/engine.py\", line 1, in <module>\n" +
				"ValueError: User-specified max_model_len (131072) is greater than the derived max_model_len (2048)\n",
			want: "ValueError: User-specified max_model_len (131072) is greater than the derived max_model_len (2048)",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := LastMeaningfulLine(tt.logs); got != tt.want {
				t.Errorf("LastMeaningfulLine() = %q, want %q", got, tt.want)
			}
		})
	}
}
