package engine

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
)

// LocalDir is the directory under the config directory that local engine
// definitions are read from.
const LocalDir = "engines.d"

var (
	localMu   sync.Mutex
	current   = shippedRegistry
	localErrs []error
)

// Default returns the engines in effect: the shipped ones, plus any local
// ones UseLocal has loaded.
func Default() *Registry {
	localMu.Lock()
	defer localMu.Unlock()
	return current
}

// UseLocal adds the engines in every *.json file in dir, in name order, to
// the ones Default returns. A missing dir is not an error. A file that does
// not parse, or that would redefine an engine, is skipped and reported, and
// the files after it still load. The problems are kept for LocalErrors, so
// doctor and 'engines list' can show them.
func UseLocal(dir string) []error {
	reg := shippedRegistry
	var errs []error
	entries, err := os.ReadDir(dir)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		errs = append(errs, fmt.Errorf("%s: %w", dir, err))
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
			names = append(names, e.Name())
		}
	}
	slices.Sort(names)
	for _, name := range names {
		path := filepath.Join(dir, name)
		data, err := os.ReadFile(path)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", path, err))
			continue
		}
		next, err := reg.Extend(data, path)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", path, err))
			continue
		}
		reg = next
	}
	localMu.Lock()
	defer localMu.Unlock()
	current, localErrs = reg, errs
	return errs
}

// LocalErrors reports the problems the last UseLocal found.
func LocalErrors() []error {
	localMu.Lock()
	defer localMu.Unlock()
	return slices.Clone(localErrs)
}
