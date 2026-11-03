//go:build !linux && !darwin

package detect

import "runtime"

// totalMemBytes has no portable implementation; a 0 here just means the budget
// falls back to whatever MODEL_BUDGET_GB says in config.env.
func totalMemBytes() int64 { return 0 }

// AvailableMemMB has no portable implementation.
func AvailableMemMB() int { return 0 }

// platformFacts records only the platform: nothing else is read here.
func platformFacts(f *facts) { f.os, f.arch = runtime.GOOS, runtime.GOARCH }
