// llm-helper manages an OpenAI-compatible inference server on one machine
// or several: saved model profiles, one-command switching, HuggingFace
// search, recipes across machines, and an optional Coder model-picker sync.
// See README.md.
package main

import (
	"os"

	"github.com/coder/vllm-helper/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:]))
}
