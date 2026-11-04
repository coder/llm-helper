// Package catalogue holds the definitions llm-helper ships as data, one JSON
// file each, named by id: engines/ has the inference servers it can launch,
// and recipes/ the launch configurations that were run and measured on named
// hardware. Each folder's README says how to add one.
package catalogue

import (
	"embed"
	"io/fs"
)

//go:embed engines/*.json recipes/*.json
var files embed.FS

// Engines is every engine definition shipped in the binary.
var Engines = sub("engines")

// Recipes is every tested recipe shipped in the binary.
var Recipes = sub("recipes")

func sub(dir string) fs.FS {
	f, err := fs.Sub(files, dir)
	if err != nil {
		panic(err)
	}
	return f
}
