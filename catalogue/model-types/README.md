# Model types

Each file here is one HuggingFace `model_type`, named after it: the value in a repo's `config.json`, such as `gpt_oss` in `openai/gpt-oss-120b`.
It says what every repo of that type needs on each engine, so the finetunes and re-uploads of a model get it too.

To add one, copy the closest file to `<model_type>.json`.
Under `engines`, keyed by an engine id from [`catalogue/engines/`](../engines), it can list files to fetch (`files`), flags to add when the card did not name them (`args`), and chat template fixes (`chat_template_fixes`).
Give each a `why`.
A type whose chat template has a switch for reasoning before the answer, such as Qwen3's `enable_thinking`, also gets a top-level `thinking` block; a file can have only that.
A configuration for one specific repo that was run and measured belongs in [`catalogue/recipes/`](../recipes) instead.

The fields are described in [docs/model-types.md](../../docs/model-types.md).
To try one without rebuilding, put its file in `~/.config/vllm-helper/model-types.d/`; a file there with a shipped id replaces the shipped one.
