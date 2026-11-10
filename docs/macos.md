---
title: Running on macOS
description: Serving on the Apple Silicon GPU through vllm-metal, and what it changes.
---

On Apple Silicon, llm-helper serves on the GPU through [vllm-metal](https://github.com/vllm-project/vllm-metal), a vLLM plugin that runs models with MLX and Metal.
It needs macOS 15 or later on an Apple Silicon Mac.
When vllm-metal is not installed beside `VLLM_BIN`, vLLM's CPU backend still works as a fallback; see [CPU backend](#cpu-backend).
This page is about the vLLM provider: mlx-lm and llama.cpp also serve on the Apple Silicon GPU, see [Providers](./providers.md).

## Install vllm-metal

Install the stable release with Homebrew:

```sh
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

To try a development build instead, run the install script.
It creates a venv at `~/.venv-vllm-metal`:

```sh
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

`pip install vllm-metal` is not a supported way to install it.
The [vllm-metal docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/) cover what each release supports.

## Run setup

1. Run `llm-helper setup`.

Setup looks for `vllm` on your PATH first.
If it is not there, it tries where the two installers put it: `/opt/homebrew/opt/vllm-metal/libexec/bin/vllm`, then `~/.venv-vllm-metal/bin/vllm`.
A Homebrew install is recorded by its `opt` path rather than the versioned Cellar path, so `brew upgrade` does not leave `VLLM_BIN` pointing at nothing.

Setup then asks the Python beside `VLLM_BIN` whether vllm-metal is installed, and asks MLX how much memory the GPU may use.
When both answer, setup says so:

```text
Apple M5 Ultra (unified memory), serving on the GPU through vllm-metal (239 GB Metal working set)
```

`llm-helper doctor` shows the same on its `backend:` line.
On the CPU backend, that line says so and suggests installing vllm-metal.

## Pick an MLX model

vllm-metal loads MLX checkpoints, usually from [mlx-community](https://huggingface.co/mlx-community) on Hugging Face:

```sh
llm-helper new mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit
```

On an M5 Ultra under a system-scope LaunchDaemon, that model serves at about 85 tokens/sec for a single stream.
Its card has no `vllm serve` snippet, so `llm-helper new` adds no flags.
For tool calls, set the parser by hand:

```sh
llm-helper set qwen3-coder-30b-a3b-instruct-8bit -- EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser qwen3_coder"
```

Its chat template puts only the first system message at the top of the prompt.
A client that sends several, as Coder Agents does, gets the rest placed as separate turns just before the user's message, and the model can read them as things the user said.
To avoid that, join them into one before the request reaches the server, for example in a gateway in front of it.

Which models run follows vllm-metal's [supported models](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md) list, not vLLM's.
Quantizations made for CUDA, such as FP8 block or NVFP4, do not run.
The architecture has to be supported by vllm-metal as well.
For example, DeepSeek-V4-Flash does not run on vllm-metal 0.32: stock mlx-lm has no model for it, and vllm-metal's paged attention does not handle its compressed sparse attention.
Mainline llama.cpp does run it on the GPU, from Unsloth's GGUF.

## What is different

- **`GPU_MEMORY_UTILIZATION` is a share of the Metal working set.**
  The working set is how much unified memory macOS lets the GPU use, and macOS already keeps it below installed RAM.
  New profiles default to `0.9` here.
  vllm-metal allocates the KV cache as it goes and checks for memory pressure at startup, so `llm-helper start` leaves the value as it is.
- **The memory budget comes from the working set.**
  It is 90% of the working set, minus about 2 GiB for vllm-metal itself and room for a KV cache.
  A 256 GiB M5 Ultra gets a weights budget of about 194 GB.
  If MLX cannot be asked, the working set is estimated at 75% of RAM, and the budget's reason says it is an estimate.
- **`doctor` skips the build toolchain.**
  vllm-metal ships prebuilt wheels and Metal kernels, so there is nothing to compile.
- **Logs are a file, not a journal.**
  Each model's launchd job writes to a log file; see [Logs](#logs).

## Run on a headless Mac

Each running model is its own launchd job, labelled `com.coder.llm-helper.<profile>`.
By default each is a LaunchAgent in `~/Library/LaunchAgents`, so controlling it needs no root.
A LaunchAgent needs a GUI login session.
Over SSH on a Mac nobody is logged in to, loading it fails with launchd error 125, and llm-helper says why.

On a headless or server Mac, run it as a LaunchDaemon instead:

1. Stop any model that is running with `llm-helper down <profile>`.
2. Set `SERVICE_SCOPE=system` in `~/.config/vllm-helper/config.env`.
3. Run `llm-helper up <profile>` for each model.

`up` installs `/Library/LaunchDaemons/com.coder.llm-helper.<profile>.plist`.
It starts at boot and runs as the user who installed it, so it reads that user's config and model cache.
`llm-helper status` and `doctor` read its state without sudo.
Starting, stopping and restarting a model need sudo, as system scope does on Linux.

Earlier releases ran one job, `com.coder.vllm`, for whichever model was active.
`llm-helper migrate` moves that model to its own job and removes the old plist.
A LaunchDaemon installed by vllm-helper 0.5.0 or earlier runs as root; the job `migrate` installs runs as you.

## Give the GPU more memory

macOS caps how much unified memory the GPU may wire, and that cap is the working set.
`llm-helper metal-limit` shows the RAM, the current `iogpu.wired_limit_mb` (`0` is the macOS default), the working set that results, and a suggested limit.
The suggestion leaves 16 GiB for macOS on a Mac with 128 GiB or more, and 8 GiB on a smaller one.

```sh
llm-helper metal-limit              # what it is now, and a suggestion
llm-helper metal-limit --set auto   # apply the suggestion, or pass a size in MiB
llm-helper restart <profile>        # vllm-metal reads the working set at startup
llm-helper metal-limit --reset      # back to the macOS default
```

`--set` writes `METAL_WIRED_LIMIT_MB` to `config.env` and applies the limit now with sudo.
The setting does not survive a reboot, so it also installs a LaunchDaemon, `com.coder.vllm-helper.wired-limit`, that applies it again at every boot.
When the live limit is already at or above the suggestion, `--set auto` keeps it rather than raising it, and still installs the daemon and records the limit if either is missing.
Whenever the live limit differs from `METAL_WIRED_LIMIT_MB`, `llm-helper start` waits up to 60 seconds for it to match before it starts vLLM, then warns and starts anyway.
`--reset` removes that daemon and sets the limit back to `0`.
`doctor` shows the current limit and suggests raising it.
`metal-limit` needs macOS 15 or later on Apple Silicon, the same as vllm-metal.

After a raise, the ceiling vLLM may touch grows on the next restart, because `GPU_MEMORY_UTILIZATION` is a share of the working set.
What happens to the KV cache depends on the vllm-metal build:

- vllm-metal 0.32.0 and earlier ignore `KV_CACHE_MEMORY_BYTES` and give the cache whatever `GPU_MEMORY_UTILIZATION` leaves after the weights.
  `llm-helper new` leaves `KV_CACHE_MEMORY_BYTES` out of the profile there and says why, so `KV_CONTEXT_TARGET` has no effect.
  The cache grows with the working set on the next restart, with nothing to change.
- Development builds after 0.32.0 honor `KV_CACHE_MEMORY_BYTES`, and `llm-helper new` sizes it for `KV_CONTEXT_TARGET` full contexts.
  That size does not grow: each profile keeps the cache size it was planned with against the old working set.
  Planning again only changes a cache that was cut short to fit; one that already holds `KV_CONTEXT_TARGET` full contexts comes out the same.
  To use the extra room in the cache, either set the size yourself with `llm-helper set <profile> KV_CACHE_MEMORY_BYTES=<bytes>`, or plan the profile again.
  `llm-helper new <repo>` reuses a profile that already exists, so remove it first with `llm-helper remove <profile>`.
  A running profile cannot be removed, so stop it with `llm-helper down <profile>` first.

A higher limit gives the GPU more of the RAM and leaves less headroom for macOS.
That is a good trade on a dedicated server, and a worse one on a desktop you also work on.

## Logs

Each model's launchd job writes to `<label>.log`, which is `com.coder.llm-helper.<profile>.log`:

- User scope: `~/Library/Logs/vllm-helper/com.coder.llm-helper.<profile>.log`
- System scope: `/Library/Logs/vllm-helper/com.coder.llm-helper.<profile>.log`

`llm-helper logs <profile>` follows that model's log.

## CPU backend

Without vllm-metal beside `VLLM_BIN`, vLLM runs models on the CPU through its experimental CPU backend.
It is useful for small models and development, not for serving anything large.
A 0.6B model in BF16 serves at roughly 30 tokens/sec on an M4 Pro.

There are no Apple Silicon wheels for the CPU backend, so it has to be built from source:

```sh
brew install cmake ninja
uv venv --python 3.12 ~/venvs/vllm
git clone https://github.com/vllm-project/vllm.git ~/vllm-src && cd ~/vllm-src
VIRTUAL_ENV=~/venvs/vllm uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
VIRTUAL_ENV=~/venvs/vllm VLLM_TARGET_DEVICE=cpu uv pip install -e . --index-strategy unsafe-best-match
```

Then run `llm-helper setup` and point `VLLM_BIN` at `~/venvs/vllm/bin/vllm`.

On the CPU backend:

- **Quantized weights don't work.**
  It handles FP32, FP16, and BF16, so pick a model small enough to run unquantized.
- **`GPU_MEMORY_UTILIZATION` means system RAM.**
  It reserves a fraction of total RAM, and vLLM refuses to start unless that much is free right then.
  Profiles default to `0.3`, and `llm-helper start` lowers it further when the machine is busy, logging the change.
- **`doctor` checks the build toolchain.**
  It looks for cmake, ninja, and the Command Line Tools, and with `--yes` installs what Homebrew can.

## Next steps

- [Troubleshooting](./troubleshooting.md): reading the log, and when vLLM refuses to start.
- [Memory budget](./memory.md): what decides whether a model fits, and how to override it.
- [Models and quantizations](./models.md): finding a model that fits this machine.
