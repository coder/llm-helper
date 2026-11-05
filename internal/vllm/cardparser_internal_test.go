package vllm

import (
	"strings"
	"testing"
)

func TestPreferSGLangParser(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, args, sglang, want string
		caps                     Caps
		changed                  bool
	}{
		{"qwen3 hybrid", "--enable-reasoning --reasoning-parser deepseek_r1", "qwen3", "--enable-reasoning --reasoning-parser qwen3", caps0271, true},
		{"equals form", "--reasoning-parser=deepseek_r1 --enforce-eager", "qwen3", "--reasoning-parser qwen3 --enforce-eager", caps0271, true},
		{"sglang name not registered", "--reasoning-parser deepseek_r1", "deepseek-r1", "--reasoning-parser deepseek_r1", caps0271, false},
		{"same parser", "--reasoning-parser qwen3", "qwen3", "--reasoning-parser qwen3", caps0271, false},
		{"no reasoning parser", "--enforce-eager", "qwen3", "--enforce-eager", caps0271, false},
		{"no sglang parser", "--reasoning-parser deepseek_r1", "", "--reasoning-parser deepseek_r1", caps0271, false},
		{"caps unknown", "--reasoning-parser deepseek_r1", "qwen3", "--reasoning-parser deepseek_r1", Caps{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, note := PreferSGLangParser(tt.args, tt.sglang, tt.caps)
			if got != tt.want {
				t.Errorf("args = %q, want %q", got, tt.want)
			}
			if tt.changed != (note != "") {
				t.Errorf("note = %q, want changed=%v", note, tt.changed)
			}
			if tt.changed && !strings.Contains(note, "deepseek_r1") {
				t.Errorf("note %q does not name the card's parser", note)
			}
		})
	}
}
