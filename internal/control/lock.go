package control

import (
	"os"
	"path/filepath"

	"github.com/coder/llm-helper/internal/config"
)

// LockInstances serializes fit checks, port picks and instance writes across processes.
func LockInstances(cfg *config.Config) (func(), error) {
	dir := cfg.Dirs.InstancesDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(dir, ".lock"), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	if err := lockFile(f); err != nil {
		f.Close()
		return nil, err
	}
	return func() {
		_ = unlockFile(f)
		f.Close()
	}, nil
}
