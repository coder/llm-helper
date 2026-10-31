package provider

import "strings"

// llamaCPP is llama.cpp's OpenAI-compatible server, llama-server. It serves
// GGUF on Metal, CUDA or the CPU, and sizes its own context and GPU layers to
// fit the machine, so a profile needs only the model.
//
// Unlike mlx_lm.server it checks an API key, so with one set the command binds
// every interface the way vllm serve does. Without a key it binds 127.0.0.1,
// so an unset VLLM_API_KEY never leaves the endpoint open. It downloads a
// repo's GGUF itself, and its /health reports healthy only once the model is
// loaded.
type llamaCPP struct{}

func (llamaCPP) Name() string    { return LlamaCPP }
func (llamaCPP) BinKey() string  { return "LLAMA_SERVER_BIN" }
func (llamaCPP) BinName() string { return "llama-server" }

func (llamaCPP) Capabilities() Capabilities {
	return Capabilities{APIKey: true, GGUFQuant: true}
}

// Command passes MODEL as --alias too, so /v1/models reports exactly MODEL,
// as vLLM's does, whether the model comes from the Hub or a file. The API key
// is not on the command line, where ps shows it to every user: Env passes it
// as LLAMA_API_KEY.
func (llamaCPP) Command(s Spec) ([]string, error) {
	source := "-hf"
	if isLocalGGUF(s.Model) {
		source = "-m"
	}
	host := "127.0.0.1"
	if s.APIKey != "" {
		host = "0.0.0.0"
	}
	args := []string{s.Bin, source, s.Model, "--alias", s.Model, "--host", host, "--port", s.Port}
	return appendExtraArgs(args, s.ExtraArgs)
}

// Env sets LLAMA_API_KEY, the variable llama-server reads in place of
// --api-key, so the key never appears in the process list.
func (llamaCPP) Env(s Spec) []string {
	if s.APIKey == "" {
		return nil
	}
	return []string{"LLAMA_API_KEY=" + s.APIKey}
}

// isLocalGGUF reports whether model is a file on disk rather than a Hugging
// Face repo: a repo id is never an absolute path, and one ending in .gguf
// names a file.
func isLocalGGUF(model string) bool {
	return strings.HasPrefix(model, "/") || strings.HasSuffix(strings.ToLower(model), ".gguf")
}
