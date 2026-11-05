package cli

import (
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/coder/llm-helper/docs"
)

// The committed reference must be exactly what the current command tree
// generates: a command added without regenerating, or a page left behind by
// a removed one, fails here instead of shipping stale docs.
func TestCLIReferenceIsUpToDate(t *testing.T) {
	generated, err := GenerateCLIReference()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join("..", "..", "docs")

	for rel, want := range generated {
		got, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Errorf("%s: %v -- run 'go run ./scripts/gendocs'", rel, err)
			continue
		}
		if string(got) != want {
			t.Errorf("%s is stale -- run 'go run ./scripts/gendocs'", rel)
		}
	}

	committed := map[string]bool{}
	err = filepath.WalkDir(filepath.Join(root, "reference"), func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(p, ".md") {
			return nil
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		committed[filepath.ToSlash(rel)] = true
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for rel := range committed {
		if _, ok := generated[rel]; !ok {
			t.Errorf("docs/%s is committed but no longer generated -- delete it", rel)
		}
	}

	// The sidebar lists the reference from manifest.json, so its CLI subtree
	// is generated too and has to match: a page the manifest does not route is
	// a page nobody can reach from the panel.
	m, err := docs.ReadManifest(os.DirFS(root))
	if err != nil {
		t.Fatal(err)
	}
	var got []docs.Route
	for _, r := range m.Routes {
		if r.Path == CLIReferencePath {
			got = r.Children
		}
	}
	if !reflect.DeepEqual(got, GenerateCLIRoutes()) {
		t.Errorf("manifest.json's %s children are stale -- run 'go run ./scripts/gendocs'", CLIReferencePath)
	}
}
