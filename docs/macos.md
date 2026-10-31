---
title: Running on macOS
description: Serving on the Apple Silicon GPU through vllm-metal, and what it changes.
---

On Apple Silicon, vllm-helper serves on the GPU through [vllm-metal](https://github.com/vllm-project/vllm-metal), a vLLM plugin that runs models with MLX and Metal.
It needs macOS 15 or later on an Apple Silicon Mac.
When vllm-metal is not installed beside `VLLM_BIN`, vLLM's CPU backend still works as a fallback; see [CPU backend](#cpu-backend).

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

1. Run `vllm-helper setup`.

Setup looks for `vllm` on your PATH first.
If it is not there, it tries where the two installers put it: `/opt/homebrew/opt/vllm-metal/libexec/bin/vllm`, then `~/.venv-vllm-metal/bin/vllm`.
A Homebrew install is recorded by its `opt` path rather than the versioned Cellar path, so `brew upgrade` does not leave `VLLM_BIN` pointing at nothing.

Setup then asks the Python beside `VLLM_BIN` whether vllm-metal is installed, and asks MLX how much memory the GPU may use.
When both answer, setup says so:

```text
Apple M5 Ultra (unified memory), serving on the GPU through vllm-metal (239 GB Metal working set)
```

`vllm-helper doctor` shows the same on its `backend:` line.
On the CPU backend, that line says so and suggests installing vllm-metal.

## Pick an MLX model

vllm-metal loads MLX checkpoints, usually from [mlx-community](https://huggingface.co/mlx-community) on Hugging Face:

```sh
vllm-helper new mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit
```

On an M5 Ultra under a system-scope LaunchDaemon, that model serves at about 85 tokens/sec for a single stream.
Its card has no `vllm serve` snippet, so `vllm-helper new` adds no flags.
For tool calls, set the parser by hand:

```sh
vllm-helper set qwen3-coder-30b-a3b-instruct-8bit -- EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser qwen3_coder"
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
  vllm-metal allocates the KV cache as it goes and checks for memory pressure at startup, so `vllm-helper start` leaves the value as it is.
- **The memory budget comes from the working set.**
  It is 90% of the working set, minus about 2 GiB for vllm-metal itself and room for a KV cache.
  A 256 GiB M5 Ultra gets a weights budget of about 194 GB.
  If MLX cannot be asked, the working set is estimated at 75% of RAM, and the budget's reason says it is an estimate.
- **`doctor` skips the build toolchain.**
  vllm-metal ships prebuilt wheels and Metal kernels, so there is nothing to compile.
- **Logs are a file, not a journal.**
  The launchd job writes to a log file; see [Logs](#logs).

## Run on a headless Mac

By default the service is a LaunchAgent in `~/Library/LaunchAgents`, so controlling it needs no root.
A LaunchAgent needs a GUI login session.
Over SSH on a Mac nobody is logged in to, loading it fails with launchd error 125, and vllm-helper says why.

On a headless or server Mac, run it as a LaunchDaemon instead:

1. Set `SERVICE_SCOPE=system` in `~/.config/vllm-helper/config.env`.
2. Run `vllm-helper setup` again.

That installs `/Library/LaunchDaemons/com.coder.vllm.plist`.
It starts at boot and runs as the user who installed it, so it reads that user's config and model cache.
`vllm-helper status` and `doctor` read its state without sudo.
Starting, stopping, restarting, and switching it need sudo, as system scope does on Linux.

A LaunchDaemon installed by vllm-helper 0.5.0 or earlier runs as root, and `setup` keeps a plist that already exists.
To replace it with one that runs as you, unload and remove it, then run setup again:

```sh
sudo launchctl bootout system/com.coder.vllm
sudo rm /Library/LaunchDaemons/com.coder.vllm.plist
vllm-helper setup
```

Use your `SERVICE_NAME` in place of `com.coder.vllm` if you changed it.
The server is down from the `bootout` until setup loads the new daemon.

## Give the GPU more memory

macOS caps how much unified memory the GPU may wire, and that cap is the working set.
`vllm-helper metal-limit` shows the RAM, the current `iogpu.wired_limit_mb` (`0` is the macOS default), the working set that results, and a suggested limit.
The suggestion leaves 16 GiB for macOS on a Mac with 128 GiB or more, and 8 GiB on a smaller one.

```sh
vllm-helper metal-limit              # what it is now, and a suggestion
vllm-helper metal-limit --set auto   # apply the suggestion, or pass a size in MiB
vllm-helper restart                  # vllm-metal reads the working set at startup
vllm-helper metal-limit --reset      # back to the macOS default
```

`--set` writes `METAL_WIRED_LIMIT_MB` to `config.env` and applies the limit now with sudo.
The setting does not survive a reboot, so it also installs a LaunchDaemon, `com.coder.vllm-helper.wired-limit`, that applies it again at every boot.
When the live limit is already at or above the suggestion, `--set auto` keeps it rather than raising it, and still installs the daemon and records the limit if either is missing.
Whenever the live limit differs from `METAL_WIRED_LIMIT_MB`, `vllm-helper start` waits up to 60 seconds for it to match before it starts vLLM, then warns and starts anyway.
`--reset` removes that daemon and sets the limit back to `0`.
`doctor` shows the current limit and suggests raising it.
`metal-limit` needs macOS 15 or later on Apple Silicon, the same as vllm-metal.

After a raise, the ceiling vLLM may touch grows on the next restart, because `GPU_MEMORY_UTILIZATION` is a share of the working set.
What happens to the KV cache depends on the vllm-metal build:

- vllm-metal 0.32.0 and earlier ignore `KV_CACHE_MEMORY_BYTES` and give the cache whatever `GPU_MEMORY_UTILIZATION` leaves after the weights.
  `vllm-helper new` leaves `KV_CACHE_MEMORY_BYTES` out of the profile there and says why, so `KV_CONTEXT_TARGET` has no effect.
  The cache grows with the working set on the next restart, with nothing to change.
- Development builds after 0.32.0 honor `KV_CACHE_MEMORY_BYTES`, and `vllm-helper new` sizes it for `KV_CONTEXT_TARGET` full contexts.
  That size does not grow: each profile keeps the cache size it was planned with against the old working set.
  Planning again only changes a cache that was cut short to fit; one that already holds `KV_CONTEXT_TARGET` full contexts comes out the same.
  To use the extra room in the cache, either set the size yourself with `vllm-helper set <profile> KV_CACHE_MEMORY_BYTES=<bytes>`, or plan the profile again.
  `vllm-helper new <repo>` reuses a profile that already exists, so remove it first with `vllm-helper remove <profile>`.
  The active profile cannot be removed, so switch to another one first.

A higher limit gives the GPU more of the RAM and leaves less headroom for macOS.
That is a good trade on a dedicated server, and a worse one on a desktop you also work on.

## Logs

The launchd job writes to `<label>.log`, which is `com.coder.vllm.log` by default:

- User scope: `~/Library/Logs/vllm-helper/com.coder.vllm.log`
- System scope: `/Library/Logs/vllm-helper/com.coder.vllm.log`

`vllm-helper logs` follows whichever one the service uses.

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

Then run `vllm-helper setup` and point `VLLM_BIN` at `~/venvs/vllm/bin/vllm`.

On the CPU backend:

- **Quantized weights don't work.**
  It handles FP32, FP16, and BF16, so pick a model small enough to run unquantized.
- **`GPU_MEMORY_UTILIZATION` means system RAM.**
  It reserves a fraction of total RAM, and vLLM refuses to start unless that much is free right then.
  Profiles default to `0.3`, and `vllm-helper start` lowers it further when the machine is busy, logging the change.
- **`doctor` checks the build toolchain.**
  It looks for cmake, ninja, and the Command Line Tools, and with `--yes` installs what Homebrew can.

## Next steps

- [Troubleshooting](./troubleshooting.md): reading the log, and when vLLM refuses to start.
- [Memory budget](./memory.md): what decides whether a model fits, and how to override it.
- [Models and quantizations](./models.md): finding a model that fits this machine.
