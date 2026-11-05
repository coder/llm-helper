// Package catalogue holds the definitions llm-helper ships as data, one JSON
// file each, named by id: engines/ has the inference servers it can launch,
// recipes/ the launch configurations that were run and measured on named
// hardware, model-types/ what every repo of a HuggingFace model_type needs on
// each engine, and hardware/ what is known about specific GPUs. Each folder's
// README says how to add one.
package catalogue

import (
	"embed"
	"io/fs"
)

// The hardware reports are test fixtures, read from disk by the tests, so
// they are not embedded.
//
//go:embed engines/*.json recipes/*.json model-types/*.json
//go:embed hardware/vendors/*.json hardware/architectures/*.json hardware/machines/*.json
var files embed.FS

// Engines is every engine definition shipped in the binary.
var Engines = sub("engines")

// Recipes is every tested recipe shipped in the binary.
var Recipes = sub("recipes")

// ModelTypes is every model-type definition shipped in the binary.
var ModelTypes = sub("model-types")

// Hardware is every vendor, architecture and machine shipped in the binary,
// each kind in a folder of its own.
var Hardware = sub("hardware")

func sub(dir string) fs.FS {
	f, err := fs.Sub(files, dir)
	if err != nil {
		panic(err)
	}
	return f
}
