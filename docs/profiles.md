---
title: Profiles
description: Saved model settings, and the knobs inside them.
---

A profile is a small env file in `~/.config/vllm-helper/models/` holding the `vllm serve` settings that work for one model.
Switching models means switching profiles, and `llm-helper new` and `llm-helper search` write one for every model they pull.

```dotenv
# Status: verified 2026-07-29
MODEL=Qwen/Qwen3-Coder-Next-FP8
MAX_MODEL_LEN=131072
GPU_MEMORY_UTILIZATION=0.85
KV_CACHE_MEMORY_BYTES=17179869184
EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser qwen3_coder"
```

## Fields

| Key                      | Meaning                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `MODEL`                  | Required. The HuggingFace repo to serve. GGUF repos take a quant: `org/repo-GGUF:Q8_0`. |
| `MAX_MODEL_LEN`          | Context length in tokens. Profiles written by llm-helper default to `131072`.           |
| `GPU_MEMORY_UTILIZATION` | Fraction of memory vLLM may use. Default `0.85` on Linux, `0.3` on macOS.               |
| `KV_CACHE_MEMORY_BYTES`  | Size of the KV cache in bytes, computed when the profile is written.                    |
| `EXTRA_ARGS`             | Extra flags passed to `vllm serve` verbatim.                                            |
| `PROVIDER`               | The server that runs this profile, `vllm` when absent. See [Providers](./providers.md). |

Any other `KEY=VALUE` in the file is exported into the server's environment as-is.

### EXTRA_ARGS

`EXTRA_ARGS` is parsed like a shell line, so keep it quoted when it carries more than one flag.
A flag value with spaces in it — a JSON payload, for example — can be quoted inside.

### KV cache sizing

`KV_CACHE_MEMORY_BYTES` is sized when the profile is written, from the model's own attention shape and `KV_CONTEXT_TARGET` in `config.env` — four full contexts by default.
With it set, `GPU_MEMORY_UTILIZATION` is a ceiling on what vLLM may touch rather than the size of the cache.

Profiles written before this key existed keep working.
Re-create one to get a sized cache.

## Editing a profile

1. Run `llm-helper edit <profile>` to open it in `$EDITOR` — or `$VISUAL`, falling back to `nano` and then `vi`.
2. Save your changes.

The editor writes the profile file directly, so the result is checked after you save, not before: `MODEL` must be present, `EXTRA_ARGS` must parse, and `MAX_MODEL_LEN` must be a positive number of tokens.
A change that fails the check is reported with the reason, but it is already saved — run `edit` again to fix it, or the next restart will fail.

To change a setting without an editor:

```sh
llm-helper set <profile> MAX_MODEL_LEN=40960
```

`set` validates everything before it writes anything, so a bad pair partway through the list leaves the profile untouched.
Prefix a value that starts with a dash with `--`:

```sh
llm-helper set <profile> -- EXTRA_ARGS=--enable-prefix-caching
```

## Applying changes

A change to the active profile takes effect on `llm-helper restart`.
A change to any other profile applies the next time you switch to it.

## Next steps

- [Models and quantizations](./models.md) — how profiles get written in the first place.
- [Memory budget](./memory.md) — what decides whether a model fits.
