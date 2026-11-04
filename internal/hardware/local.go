package hardware

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

// LocalDir is the directory under the config directory that local
// definitions are read from.
const LocalDir = "hardware.d"

var (
	localMu   sync.Mutex
	current   = shippedRegistry
	localErrs []error
)

// Default returns the definitions in effect: the shipped ones, plus any
// local ones UseLocal has loaded.
func Default() *Registry {
	localMu.Lock()
	defer localMu.Unlock()
	return current
}

// UseLocal adds the entries under dir to the ones Default returns. dir is
// laid out like catalogue/hardware: one *.json file per entry in vendors/,
// architectures/ and machines/, each named after its id. Vendors load first,
// then architectures, then machines, each folder in name order. A missing
// dir or folder is not an error. A file that does not parse, or that would
// change something already defined, is skipped and reported, and the files
// after it still load: one bad file must not take a machine's other
// definitions with it. A *.json file directly in dir is reported rather than
// silently ignored. The problems are also kept for LocalErrors, so doctor can
// show them.
func UseLocal(dir string) []error {
	reg := shippedRegistry
	var errs []error
	strays, err := jsonFiles(dir)
	if err != nil {
		errs = append(errs, err)
	}
	for _, name := range strays {
		errs = append(errs, fmt.Errorf("%s: not loaded; put each entry in a file of its own in %s/, %s/ or %s/",
			filepath.Join(dir, name), KindVendor, KindArchitecture, KindMachine))
	}
	for _, k := range kinds {
		sub := filepath.Join(dir, string(k))
		names, err := jsonFiles(sub)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		for _, name := range names {
			path := filepath.Join(sub, name)
			data, err := os.ReadFile(path)
			if err == nil {
				var next *Registry
				if next, err = reg.Extend(k, path, data); err == nil {
					reg = next
				}
			}
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", path, err))
			}
		}
	}
	localMu.Lock()
	defer localMu.Unlock()
	current, localErrs = reg, errs
	return errs
}

// jsonFiles lists the *.json files in dir, in name order. A missing dir has
// none.
func jsonFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("%s: %w", dir, err)
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
			names = append(names, e.Name())
		}
	}
	slices.Sort(names)
	return names, nil
}

// LocalErrors reports the problems the last UseLocal found.
func LocalErrors() []error {
	localMu.Lock()
	defer localMu.Unlock()
	return slices.Clone(localErrs)
}
