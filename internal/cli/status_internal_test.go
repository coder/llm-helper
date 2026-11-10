package cli

import "testing"

func TestCrashReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		now, baseline int
		want          string
	}{
		{"counts from the baseline", 5, 2, "the service is crash-looping (3 restarts since it started)"},
		{"no baseline to count from", 5, -1, "the service is crash-looping"},
		{"counter did not move", 2, 2, "the service is crash-looping"},
		{"counter unreadable", -1, 0, "the service is crash-looping"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := crashReason(tt.now, tt.baseline); got != tt.want {
				t.Errorf("crashReason(%d, %d) = %q, want %q", tt.now, tt.baseline, got, tt.want)
			}
		})
	}
}
