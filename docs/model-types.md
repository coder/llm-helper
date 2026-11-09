---
title: Model types
description: What a model's architecture needs to run correctly, and how to add or change it.
---

Some architectures need more than their model card says.
gpt-oss needs vocab files on disk and parsers for its harmony format, and Qwen3.5's chat template raises an error on a system message after the first turn.
llm-helper keeps this as data, one file per HuggingFace `model_type` in [`catalogue/model-types/`](https://github.com/coder/llm-helper/tree/main/catalogue/model-types).
`new` and the panel read the repo's `config.json`, find the file for its `model_type`, and apply what it says for the engine the profile runs on.
A GGUF repo is matched by the `model_type` of the model it was quantized from.

A model type applies to every repo of that architecture, including finetunes and re-uploads.
A launch configuration for one specific repo, run and measured on named hardware, is a [recipe](./recipes.md).

`llm-helper model-types list` shows the model types in effect, and `llm-helper model-types show <id>` shows what one does.

## Fields

```json
{
  "schema_version": 1,
  "id": "gpt_oss",
  "display_name": "gpt-oss",
  "why": "Why this architecture needs anything at all.",
  "thinking": {
    "kwarg": "enable_thinking",
    "default": true,
    "why": "What the switch does to replies."
  },
  "engines": {
    "vllm": {
      "files": [
        {
          "dir": "tiktoken-encodings",
          "env": "TIKTOKEN_ENCODINGS_BASE",
          "base_url": "https://openaipublic.blob.core.windows.net/encodings/",
          "names": ["o200k_base.tiktoken", "cl100k_base.tiktoken"],
          "why": "What fails without them."
        }
      ],
      "args": [
        {
          "flag": "--tool-call-parser",
          "values": ["openai", "openai_gptoss"],
          "with": ["--enable-auto-tool-choice"],
          "why": "What fails without it."
        }
      ],
      "chat_template_fixes": [
        {
          "contains": "text that marks the broken template",
          "match": "a regular expression",
          "replace": "its replacement, with ${1} for groups",
          "why": "What fails without it."
        }
      ]
    }
  }
}
```

- `id` is the `model_type` value, and the file is named after it.
- `thinking` describes a chat template switch for reasoning before the answer. `kwarg` is the template variable, and `default` is whether the model thinks when nothing sets it. With `detect` left out or set to `template`, it applies only to repos whose chat template names `kwarg`, so a release of the same architecture without the switch is not flagged. `detect: always` applies it to every repo of the type, for types whose chat formatting lives in the engine rather than a template.
- `engines` is keyed by engine id. An engine with no entry gets nothing added. `files` apply on any engine. `args` and `chat_template_fixes` are in vLLM's flag syntax, so they are only accepted for engines with the `vllm_args` capability.
- `files` are fetched into `dir` under the data directory when a profile is created, unless already there. With `env` set, the profile gets that variable pointing at the directory.
- `args` are added to the profile's `EXTRA_ARGS` unless the card already named the flag. `values` are tried in order and the first the installed engine registers is used; if none is registered, the flag is left out. `with` flags are added alongside.
- `chat_template_fixes` apply when the repo's `chat_template.jinja` contains `contains`. The patched copy is saved under the data directory and passed with `--chat-template`. If the text is there but `match` does not match, nothing is patched and `new` says so.

A file needs `engines`, `thinking`, or both.
Each file is checked when it loads: an unknown field, an unknown engine, `args` or `chat_template_fixes` for an engine without `vllm_args`, a bad regular expression, a `dir` outside the data directory, a `base_url` that is not https, a `kwarg` that is not a template variable name or an unknown `detect` is refused.

## Thinking models

When a repo matches a `thinking` entry, `inspect` says whether the model thinks by default, `new` records it as `THINKING` in the profile, and `list` and `status` show it.
`new --thinking on|off` and the panel's Thinking control set it for the new profile.
They pass the kwarg through the engine's `chat_template_kwargs_flag` (`--default-chat-template-kwargs` on vLLM), so a client that sends its own `chat_template_kwargs` still overrides it per request.
An engine without that flag cannot switch thinking, and `new` says so.
Models that always think and have no switch are not flagged.

## Changing one locally

Put a file in `~/.config/vllm-helper/model-types.d/`, named after its `model_type`.
A new id adds a model type, and a shipped id replaces the shipped file, so you can change what llm-helper does for an architecture without rebuilding.
A file that fails to load is skipped, and `doctor` and `model-types list` say why.
