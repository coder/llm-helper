package recipe

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

// LocalDir is the directory under the config directory that local recipes
// are read from.
const LocalDir = "recipes.d"

var (
	localMu      sync.Mutex
	localRecipes []TestedRecipe
	localErrs    []error
)

// UseLocal adds the recipes in every *.json file in dir, in name order, to
// the ones LoadTested returns. A local recipe is one measured on this
// deployment rather than shipped, and is listed as local. A file can only
// add recipes with new ids, so it cannot change what a shipped recipe
// launches. A missing dir is not an error; a bad file is skipped and
// reported, and the files after it still load.
//
// Call it after provider.UseLocal, since a local recipe may run on a local
// engine.
func UseLocal(dir string) []error {
	var errs []error
	shipped, err := parseShipped()
	if err != nil {
		errs = append(errs, err)
	}
	ids := map[string]bool{}
	for _, r := range shipped.Recipes {
		ids[r.ID] = true
	}
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
	var added []TestedRecipe
	for _, name := range names {
		path := filepath.Join(dir, name)
		got, err := readLocal(path, ids)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", path, err))
			continue
		}
		for _, r := range got {
			ids[r.ID] = true
		}
		added = append(added, got...)
	}
	localMu.Lock()
	defer localMu.Unlock()
	localRecipes, localErrs = added, errs
	return errs
}

func readLocal(path string, taken map[string]bool) ([]TestedRecipe, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	t, err := decode(data)
	if err != nil {
		return nil, err
	}
	var errs []error
	seen := map[string]bool{}
	for i := range t.Recipes {
		r := &t.Recipes[i]
		r.Local = path
		if taken[r.ID] || seen[r.ID] {
			errs = append(errs, fmt.Errorf("recipe %q is already defined; a local file can only add new ids", r.ID))
		}
		seen[r.ID] = true
		if err := r.validate(); err != nil {
			errs = append(errs, err)
		}
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return t.Recipes, nil
}

// LocalErrors reports the problems the last UseLocal found.
func LocalErrors() []error {
	localMu.Lock()
	defer localMu.Unlock()
	return slices.Clone(localErrs)
}
