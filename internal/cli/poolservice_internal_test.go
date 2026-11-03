package cli

import (
	"strings"
	"testing"
)

func TestNormalizeCSV(t *testing.T) {
	if got := normalizeCSV(" spark1, spark2 ,, "); got != "spark1,spark2" {
		t.Fatalf("normalizeCSV = %q", got)
	}
}

func TestRenderPoolSystemdUnit(t *testing.T) {
	unit := renderPoolSystemdUnit(poolSystemdUnit{
		User:   "coder",
		Home:   "/home/coder",
		Self:   "/usr/local/bin/vllm-helper",
		Nodes:  "spark1,spark2",
		Bind:   "0.0.0.0:8011",
		Sticky: "auto",
	})
	for _, want := range []string{
		"Description=LLM Helper Pool Router (sticky replicas)",
		"User=coder",
		"Environment=HOME=/home/coder",
		"ExecStart=/usr/local/bin/vllm-helper pool serve --nodes spark1,spark2 --bind 0.0.0.0:8011 --sticky auto",
		"Restart=always",
	} {
		if !strings.Contains(unit, want) {
			t.Fatalf("unit missing %q: %s", want, unit)
		}
	}
}
