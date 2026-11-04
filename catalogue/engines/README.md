# Engines

Each file here is one inference server llm-helper can launch, named after the engine's id.
`vllm.json` is vLLM, `mlx.json` is MLX, `llamacpp.json` is llama.cpp and `tensorfold-python.json` is TensorFold.

To add an engine, copy the closest file to `<id>.json` and change it.
No Go code is needed: shared code acts on the engine's `capabilities`, never on its name.
The file is checked when it loads, and the tests in `internal/provider` render every file here for every rank.

The fields are described in "Contributing an engine" in [docs/providers.md](../../docs/providers.md).
To try an engine without rebuilding, put its file in `~/.config/vllm-helper/engines.d/` instead.
