// Package recipes holds the tested recipes: launch configurations that were
// run and measured on named hardware, one JSON file per recipe, named by the
// recipe's id. See README.md for how to add one.
package recipes

import "embed"

// FS is every recipe shipped in the binary.
//
//go:embed *.json
var FS embed.FS
