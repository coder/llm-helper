---
title: Models and quantizations
description: Finding a model that fits this machine, and choosing a quant.
---

llm-helper reads a model's HuggingFace card before anything is saved: it checks the weight size against what this machine can hold, and the card's flags against the vLLM you have installed.

## Finding a model

- `llm-helper search <query>` lists matches with their sizes and walks you through pulling one.
- `llm-helper new <org/model>` pulls a repo outright, saves it as a profile, and switches to it.
- `llm-helper inspect <org/model>` shows one repo's license, its weight size against this machine's budget, and the vLLM flags its card suggests — without pulling anything.

`search`, `inspect`, and `new` pre-fill known vLLM serving flags from the model card: tool call parsers, reasoning parsers, and context length.
`search` marks what is too big for this machine and what cannot load, and preselects the biggest quant that fits.
`new` refuses a quant that cannot load rather than serving it — activating a profile restarts the service, so a model that cannot load would take the running one down with it.

## Choosing a quantization

Reach for a quantized release when a model is large: roughly 1 byte per parameter for FP8/INT8 and 0.5 for INT4/AWQ/GPTQ, against the budget `doctor` reports.
Take the **largest** one that fits rather than the smallest.
Quantization trades quality for size, so there is nothing to gain from a 2-bit release on a machine with room for 8-bit.

Most repos hold a single precision, and choosing a quant means choosing a different repo — `Qwen/Qwen3-32B` versus `Qwen/Qwen3-32B-AWQ` — which is an ordinary `llm-helper new`.

GGUF repos are the exception: they ship a dozen quants side by side.
`inspect` lists them, `new` takes a `--quant`, and `search` asks interactively with the best fit preselected.
The panel offers the same choice as a dropdown.
Whichever you use, the quant is recorded in the profile, so what runs is what you picked.

```sh
llm-helper inspect bartowski/Qwen_Qwen3-32B-GGUF   # what's in there, and which fits
llm-helper new bartowski/Qwen_Qwen3-32B-GGUF --quant Q8_0
```

## GGUF needs a plugin

GGUF support left vLLM core in 0.27.1 for an out-of-tree package, `vllm-gguf-plugin`.
A stock vLLM cannot load a GGUF file at all — the failure is that `gguf` is not a registered quantization method.

1. Run `llm-helper install-gguf`.

It installs the plugin into the interpreter that owns the configured vLLM — not whatever `python` is on your PATH — and takes effect on the next switch.
Nothing running is disturbed.

The plugin builds a CUDA extension from source: expect several minutes and a compiler.
vLLM calls its own GGUF support experimental, and a native quant (AWQ, GPTQ, FP8, NVFP4) is faster where one exists.

Two things are reported rather than worked around:

- Quants split across files, and quants in a per-quant subfolder, cannot be loaded on any released version of the plugin — those are listed, but not offered.
- A mixture-of-experts model can only be served as GGUF when the plugin can map its fused expert weights, which today means OLMoE and MiniMax-M2.
  Any other MoE is refused — look for a safetensors quant instead.
  A safetensors MoE is served by vLLM directly.

## Next steps

- [Memory budget](./memory.md) — what "fits" means.
- [Profiles](./profiles.md) — what the pull saved.
