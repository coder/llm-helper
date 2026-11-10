package control

import (
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/detect"
)

func TestCheckFit(t *testing.T) {
	unified := detect.Hardware{Unified: true}
	discrete := detect.Hardware{GPUName: "NVIDIA RTX 6000", GPUMemGB: 96}
	cases := []struct {
		name    string
		hw      detect.Hardware
		running []Share
		want    float64
		refuse  bool
	}{
		{"first model", unified, nil, 0.85, false},
		{"two halves", unified, []Share{{"a", 0.4}}, 0.4, false},
		{"over the unified cap", unified, []Share{{"a", 0.5}}, 0.4, true},
		{"discrete fits to 1", discrete, []Share{{"a", 0.5}}, 0.5, false},
		{"discrete over", discrete, []Share{{"a", 0.6}}, 0.5, true},
		{"unknown share ignored", unified, []Share{{"a", 0}}, 0.85, false},
		{"unknown request", unified, []Share{{"a", 0.85}}, 0, false},
		{"cpu only", detect.Hardware{}, []Share{{"a", 0.9}}, 0.9, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := CheckFit(tc.hw, tc.running, "new", tc.want)
			if (err != nil) != tc.refuse {
				t.Fatalf("refuse=%v, got %v", tc.refuse, err)
			}
			if err != nil && (!strings.Contains(err.Error(), "a 0.") || !strings.Contains(err.Error(), "stop one")) {
				t.Errorf("message should name the running model and suggest stopping one: %v", err)
			}
		})
	}
}
