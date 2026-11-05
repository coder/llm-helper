// Command gendocs writes the CLI reference pages into the docs tree, and the
// matching children of the CLI reference route in manifest.json.
//
//	go run ./scripts/gendocs              # from the repo root
//	go generate ./...                     # same, via the hook in internal/cli
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"

	"github.com/coder/llm-helper/docs"
	"github.com/coder/llm-helper/internal/cli"
)

func main() {
	dir := flag.String("dir", "docs", "docs root to write reference pages into")
	flag.Parse()

	pages, err := cli.GenerateCLIReference()
	if err != nil {
		fail(err)
	}
	paths := make([]string, 0, len(pages))
	for p := range pages {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, rel := range paths {
		path := filepath.Join(*dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			fail(err)
		}
		if err := os.WriteFile(path, []byte(pages[rel]), 0o644); err != nil {
			fail(err)
		}
	}
	if err := writeManifest(*dir); err != nil {
		fail(err)
	}
	fmt.Printf("gendocs: wrote %d pages and their manifest routes to %s\n", len(pages), *dir)
}

// writeManifest replaces the CLI reference's children and leaves every other
// route as it was.
func writeManifest(dir string) error {
	m, err := docs.ReadManifest(os.DirFS(dir))
	if err != nil {
		return err
	}
	found := false
	for i := range m.Routes {
		if m.Routes[i].Path == cli.CLIReferencePath {
			m.Routes[i].Children = cli.GenerateCLIRoutes()
			found = true
		}
	}
	if !found {
		return fmt.Errorf("manifest.json has no %s route to hold the reference", cli.CLIReferencePath)
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), prettierArrays(b.Bytes()), 0o644)
}

// shortStringArray matches an array of string literals that the encoder put
// one per line. Prettier writes such an array on one line when it fits, so
// matching it here keeps the generated file passing the prettier check.
var shortStringArray = regexp.MustCompile(`\[\n\s+("[^"\n]*"(?:,\n\s+"[^"\n]*")*)\n\s*\]`)

func prettierArrays(b []byte) []byte {
	return shortStringArray.ReplaceAllFunc(b, func(m []byte) []byte {
		inner := shortStringArray.FindSubmatch(m)[1]
		inner = regexp.MustCompile(`,\n\s+`).ReplaceAll(inner, []byte(", "))
		if len(inner) > 60 {
			return m
		}
		return append(append([]byte("["), inner...), ']')
	})
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "gendocs:", err)
	os.Exit(1)
}
