---
title: Running on macOS
description: vLLM's experimental CPU backend on Apple Silicon, and what it changes.
---

vLLM has no GPU backend on Apple Silicon, so a Mac runs models on the CPU through vLLM's experimental CPU backend.
Expect it to be useful for small models and development, not for serving anything large.
A 0.6B model in BF16 serves at roughly 30 tokens/sec on an M4 Pro, which is enough to develop against.

## Build vLLM

There are no Apple Silicon wheels, so vLLM has to be built from source:

```sh
brew install cmake ninja
uv venv --python 3.12 ~/venvs/vllm
git clone https://github.com/vllm-project/vllm.git ~/vllm-src && cd ~/vllm-src
VIRTUAL_ENV=~/venvs/vllm uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
VIRTUAL_ENV=~/venvs/vllm VLLM_TARGET_DEVICE=cpu uv pip install -e . --index-strategy unsafe-best-match
```

Then run `llm-helper setup` and point `VLLM_BIN` at `~/venvs/vllm/bin/vllm`.

## What is different

- **Quantized weights don't work.**
  The CPU backend handles FP32, FP16, and BF16, so the usual advice to reach for an FP8 or INT4 release doesn't apply — pick a model small enough to run unquantized.
- **`GPU_MEMORY_UTILIZATION` means system RAM.**
  Despite the name, on the CPU backend it reserves a fraction of total RAM, and vLLM refuses to start unless that much is free right then.
  Profiles default to a much lower fraction here, and `llm-helper start` lowers it further when the machine is busy, logging the change.
- **Logs are a file, not a journal.**
  The launchd job writes to `~/Library/Logs/vllm-helper/<label>.log` — `com.coder.vllm.log` by default.

The service runs as a launchd job in user scope by default, so controlling it needs no root.

## Next steps

- [Troubleshooting](./troubleshooting.md) — including the log command for macOS.
- [Models and quantizations](./models.md) — why an unquantized model is the right pick here.
