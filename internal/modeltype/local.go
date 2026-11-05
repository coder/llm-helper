package modeltype

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

// LocalDir is the directory under the config directory that local model
// types are read from.
const LocalDir = "model-types.d"

var (
	localMu   sync.Mutex
	current   = shippedRegistry
	localErrs []error
)

// Default returns the model types in effect: the shipped ones, with any
// UseLocal loaded added or put in their place.
func Default() *Registry {
	localMu.Lock()
	defer localMu.Unlock()
	return current
}

// UseLocal loads every *.json file in dir, in name order, over the shipped
// model types. A file with a shipped id replaces that entry. A missing dir is
// not an error; a bad file is skipped and reported, and the files after it
// still load.
//
// Call it after engine.UseLocal, since a local file may name a local engine.
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
		p := filepath.Join(dir, name)
		data, err := os.ReadFile(p)
		var e Entry
		if err == nil {
			e, err = parse(name, data)
		}
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", p, err))
			continue
		}
		e.Source = p
		reg = reg.with(e)
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
