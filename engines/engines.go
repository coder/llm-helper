// Package engines holds the inference servers llm-helper can launch, one
// JSON definition per file, named by the engine's id. See README.md for how
// to add one.
package engines

import "embed"

// FS is every engine definition shipped in the binary.
//
//go:embed *.json
var FS embed.FS
