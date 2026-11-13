---
title: Providers
description: The inference servers a profile can run on, and what each one changes.
---

A provider is the inference server a profile runs on.
llm-helper does the same work for every provider: profiles, starting and stopping models, the service units, the health check, and Coder sync.
What changes is the command the service runs, and which of llm-helper's checks apply.

| Provider  | Server                         | Runs on                             | `PROVIDER` |
| --------- | ------------------------------ | ----------------------------------- | ---------- |
| vLLM      | `vllm serve`                   | Linux with an NVIDIA GPU, and macOS | `vllm`     |
| MLX       | `mlx_lm.server`, from `mlx-lm` | Apple Silicon                       | `mlx`      |
| llama.cpp | `llama-server`                 | Linux and macOS, GPU or CPU         | `llamacpp` |

vLLM is the default, and it is what every install from before providers existed runs.

Recipes can also run an engine that has only a container runtime.
The first is TensorFold (`tensorfold-python`), which serves GLM-5.3 Flash across two DGX Sparks; see [Recipes](./recipes.md).

Each provider is a definition in data, not code.
List them, and see how each one is launched, with:

```sh
llm-helper engines list
llm-helper engines show tensorfold-python
```

## How a profile picks one

Each profile records its own provider in a `PROVIDER=` line.
A profile without one is a vLLM profile.

`PROVIDER` in `config.env` only picks the provider for the profiles that `new`, `search` and the panel create from then on.
Changing it leaves your existing profiles on the server they were made for, so a machine can hold profiles for several providers side by side and run them together or one at a time.

Coder sync tells Coder each model's context limit from `MAX_MODEL_LEN`.
MLX and llama.cpp profiles have none, since those servers size their own context, so Coder is told 131072 tokens.
Set `MAX_MODEL_LEN` on the profile to tell Coder the real limit. Only vLLM passes it to the server.

```sh
llm-helper set <profile> MAX_MODEL_LEN=32768
```

## MLX

[mlx-lm](https://github.com/ml-explore/mlx-lm) serves models on the Apple Silicon GPU through MLX.
It reads a model's tool-call and reasoning formats from its tokenizer, so an MLX profile needs no parser flags.
Use a repo already converted to MLX, such as one from [mlx-community](https://huggingface.co/mlx-community), for quantized weights.

1. Install mlx-lm, usually in a venv:

   ```sh
   uv venv ~/venvs/mlx
   VIRTUAL_ENV=~/venvs/mlx uv pip install mlx-lm
   ```

2. On a new machine, run setup for MLX. On a machine that is already set up, add the two lines to `~/.config/vllm-helper/config.env` instead:

   ```sh
   llm-helper setup --provider mlx
   ```

   ```sh
   PROVIDER=mlx
   MLX_LM_BIN=/Users/you/venvs/mlx/bin/mlx_lm.server
   ```

3. Pull a model:

   ```sh
   llm-helper new mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit
   ```

An MLX profile holds only `PROVIDER`, `MODEL` and `EXTRA_ARGS`.
The service runs `mlx_lm.server --model <MODEL> --port <PORT>` followed by `EXTRA_ARGS`.
vLLM's settings do not apply, so llm-helper does not copy flags from the model card, size a KV cache, or fit a memory fraction for it.

Things to know about `mlx_lm.server`:

- **It checks no API key.** `VLLM_API_KEY` does nothing for an MLX profile, and any client that can reach the port can use it.
- **It listens on `127.0.0.1` by default.** To serve another machine, add `--host 0.0.0.0` to `EXTRA_ARGS`, and only do that on a network you trust.
- **A request can load another model.** The server loads whatever repo a request names in `model`, so an open port can be made to download and load any model.
- **Replies stop at 512 tokens** when a client sends no `max_tokens`. Add `--max-tokens <n>` to `EXTRA_ARGS` to raise that default.
- **Its health check answers while the model is still loading.** `status` and `up --wait` can report the endpoint up before the first request can be served.

```sh
llm-helper set <profile> -- EXTRA_ARGS='--host 0.0.0.0 --max-tokens 32768'
```

## llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` serves GGUF models on Metal, CUDA, Vulkan, ROCm or the CPU.
It sizes its context and GPU layers to fit the machine, so a llama.cpp profile needs only the model.
Use a repo of GGUF files, such as one from [bartowski](https://huggingface.co/bartowski) or [unsloth](https://huggingface.co/unsloth).

1. Install llama.cpp. Homebrew has it on macOS and Linux, and the [releases](https://github.com/ggml-org/llama.cpp/releases) have prebuilt binaries; for an AMD GPU on Linux, see [AMD GPUs on Linux](#amd-gpus-on-linux):

   ```sh
   brew install llama.cpp
   ```

2. On a new machine, run setup for llama.cpp. On a machine that is already set up, add the two lines to `~/.config/vllm-helper/config.env` instead:

   ```sh
   llm-helper setup --provider llamacpp
   ```

   ```sh
   PROVIDER=llamacpp
   LLAMA_SERVER_BIN=/opt/homebrew/bin/llama-server
   ```

3. Pull a model, and pick a quant if the repo holds several:

   ```sh
   llm-helper new bartowski/Qwen_Qwen3-32B-GGUF --quant Q4_K_M
   ```

A llama.cpp profile holds only `PROVIDER`, `MODEL` and `EXTRA_ARGS`, with the quant in `MODEL` as `<repo>:<quant>`.
The service runs `llama-server -hf <MODEL> --alias <MODEL> --host 0.0.0.0 --port <PORT>` followed by `EXTRA_ARGS`.
The API key goes in the server's environment as `LLAMA_API_KEY`, not on the command line, where `ps` would show it to every user on the machine.
A `MODEL` that is an absolute path or ends in `.gguf` is loaded from disk with `-m` instead.
Without `--quant`, llama.cpp picks `Q4_K_M`, or the first GGUF in the repo when there is no `Q4_K_M`.

Things to know about `llama-server`:

- **It checks the API key.** Clients send `VLLM_API_KEY`, as they do for vLLM, so it listens on every interface like vLLM does. Add `--host 127.0.0.1` to `EXTRA_ARGS` to keep it on this machine. With no `VLLM_API_KEY` set, it listens on `127.0.0.1` only.
- **It downloads the model when it starts.** The first start of a profile takes as long as the download, and the GGUF goes to the same Hugging Face cache vLLM uses.
- **Its health check waits for the model.** `status` and `up --wait` report the endpoint up only once the model is loaded and it can serve.
- **Context comes from the model,** then shrinks to fit memory. Set it with `-c <tokens>` in `EXTRA_ARGS`.
- **Its token counters stand still while it reads a prompt.** `llamacpp:prompt_tokens_total` and `llamacpp:tokens_predicted_total` in `/metrics` stayed flat for 181 seconds while b12364 read a 45,000-token prompt, so a flat counter during a long prompt is not a stuck server.

```sh
llm-helper set <profile> -- EXTRA_ARGS='-c 65536'
```

### AMD GPUs on Linux

llama.cpp serves an AMD GPU through Vulkan or ROCm, and every [release](https://github.com/ggml-org/llama.cpp/releases) has a prebuilt Ubuntu x64 build of each.
A distribution's own `llama-server` can be far older: Debian's `/usr/bin/llama-server` is a 0.2.0 build, and `setup` finds it on PATH unless told otherwise.

1. Download a build and unpack it.
   The Vulkan one, for example:

   ```sh
   curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b12364/llama-b12364-bin-ubuntu-vulkan-x64.tar.gz
   mkdir -p ~/llama.cpp
   tar xzf llama-b12364-bin-ubuntu-vulkan-x64.tar.gz -C ~/llama.cpp --strip-components=1
   ```

2. Let your user open the GPU.
   `/dev/kfd` and `/dev/dri/renderD*` belong to the `render` group, and without it the ROCm build fails with "no ROCm-capable device is detected".
   `doctor` warns when you cannot open them.

   ```sh
   sudo usermod -aG render,video $USER
   ```

   Log out and back in for the groups to apply.
   A user's systemd manager that lingers keeps the groups it started with, so reboot if your models run with `SERVICE_SCOPE=user`.

3. Run setup with the server named in `LLAMA_SERVER_BIN`.
   The user scope needs no sudo, and lingering starts the models at boot:

   ```sh
   LLAMA_SERVER_BIN=$HOME/llama.cpp/llama-server llm-helper setup --provider llamacpp --service-scope user
   sudo loginctl enable-linger $USER
   ```

On a Ryzen AI Max (Strix Halo), the GPU loads the model into GTT memory, which is capped well below RAM by default; raise the cap first, as "AMD Ryzen AI Max (Strix Halo)" in [Memory budget](./memory.md) describes.

### Example: DeepSeek V4 Flash Vision on a Ryzen AI Max

This ran on a Ryzen AI Max with the Radeon 8060S and 128 GB, with the GTT limit raised to 120 GB, and llama.cpp b12364's Vulkan and ROCm 10 builds from the releases.
The model is Unsloth's `UD-IQ3_XXS` GGUF (103 GB), served as a shared endpoint with 4 slots.
llama-server finds the vision file (`mmproj`) in the same repo by itself, with `--offline` too.

```sh
llm-helper new unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF --quant UD-IQ3_XXS
llm-helper set deepseek-v4-flash-vision-exp-gguf-ud-iq3_xxs -- \
  EXTRA_ARGS="-c 1048576 -np 4 -ngl all --fit off -fa on -b 4096 -ub 2048 --temp 1.0 --top-p 0.95 --min-p 0.0 --metrics --offline"
```

|                            | ROCm 10                  | Vulkan (RADV)            |
| -------------------------- | ------------------------ | ------------------------ |
| Load to healthy            | 22 s                     | 21 s                     |
| One request, short prompt  | 15.2 tokens/sec          | 16.9 tokens/sec          |
| Prompt read, 10,449 tokens | 154 tokens/sec           | 201 tokens/sec           |
| Prompt read, 38,968 tokens | 118 tokens/sec           | 128 tokens/sec           |
| Two requests at once       | 12.0 each                | 12.0 each                |
| Four requests at once      | about 8.3 each, 33 total | about 8.2 each, 33 total |
| GTT memory in use          | 113.7 GB                 | 114.5 GB                 |

- Vulkan writes and reads prompts faster than ROCm on one request, and the two are level with several at once.
- With the model loaded, about 15 GB of the 128 GB was still free.
  At the stock GTT limit of 101 GB, the model does not fit.
- A tool call came back as correct `tool_calls`, a needle in a long context was found, and a red image was answered "Red."

### Prompt cache

`llama-server` keeps the prompts of finished requests in host memory, up to `--cache-ram` MiB (8192 by default), and reuses the longest matching start of a new prompt instead of reading it again.
On a coding agent, every chat begins with the same long system prompt and tool list, so a hit saves most of the first turn.
The cache is empty after the server starts, and old prompts are dropped when it is full.

Hybrid models, which mix attention layers with recurrent or sliding-window ones (GLM-5.3-Flash, Qwen3-Next and others), can only go back to a saved checkpoint, not to any point in the prompt.
llama.cpp saves those checkpoints at the start of user messages, so two prompts share cached work only when everything before the first user message is identical.
Chats with different tools, skills or rules differ there and read the whole prompt again.

Set the size with `--cache-ram` in `EXTRA_ARGS`, or `LLAMA_ARG_CACHE_RAM` in the profile:

```sh
llm-helper set <profile> -- LLAMA_ARG_CACHE_RAM=16384
```

On a machine whose GPU shares memory with the CPU, such as a Mac or a DGX Spark, the cache comes out of the same memory as the model and its context.
A warm-up request (`WARMUP_FILE`, see [Profiles](./profiles.md)) fills the cache after every start, so the first chat does not pay for it.

### Example: DeepSeek V4 Flash Vision on a Mac Studio

This ran on an M5 Ultra with 256 GB, llama.cpp 0.8.0 from Homebrew, and the wired limit raised to 240 GiB.
The model is Unsloth's `UD-Q8_K_XL` GGUF (162 GB in 5 files).
llama-server finds the vision file (`mmproj-F16.gguf`) in the same repo by itself.

```sh
llm-helper new unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF --quant UD-Q8_K_XL
llm-helper set deepseek-v4-flash-vision-exp-gguf-ud-q8_k_xl -- \
  EXTRA_ARGS="-c 1048576 -np 4 -ngl all --fit off -fa on -b 4096 -ub 2048 --temp 1.0 --top-p 0.95 --min-p 0.0 --metrics --offline"
```

- `-ngl all --fit off` keeps every layer on the GPU. With `--fit` on, llama.cpp can move layers to the CPU when memory looks short.
- `--offline` starts from the Hugging Face cache, so a boot does not wait for the network.
- It writes about 33 tokens/sec, reads prompts at about 825 tokens/sec, and uses 161.8 of the 240 GiB of GPU memory.
- Images, tool calls and thinking work through the chat template in the GGUF.

### Example: GLM-5.3-Flash on a Mac Studio

This ran on the same Mac, with the same llama.cpp 0.8.0.
llama.cpp runs GLM-5.3-Flash since ggml-org/llama.cpp#27773, and its MTP head since #29928.
The model is Unsloth's `UD-Q4_K_XL` GGUF (186 GiB in 6 files), plus `mmproj-F16.gguf` for images.

```sh
llm-helper new unsloth/GLM-5.3-Flash-GGUF --quant UD-Q4_K_XL
llm-helper set glm-5.3-flash-gguf-ud-q4_k_xl -- \
  EXTRA_ARGS="-c 1048576 -np 4 -ngl all --fit off -fa on -b 4096 -ub 2048 --spec-type draft-mtp --temp 1.0 --top-p 0.95 --min-p 0.0 --metrics --offline"
```

- `--spec-type draft-mtp` drafts tokens with the model's own MTP head. No second model is needed. 46 to 68 percent of drafts were accepted.
- One request writes about 58 tokens/sec. Four at once write about 16 tokens/sec each, 63 in total. These numbers come from Huihui's abliterated copy of the same quant (`huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF`), which has the same shapes.
- It reads prompts at about 585 tokens/sec, slower than DeepSeek V4 Flash on this Mac.
- `-c 1048576 -np 4` gives 4 slots of 262,144 tokens. Most of its layers are linear attention, so the KV cache is small (`inspect` gives about 14 KB a token).
- Images and tool calls work through the chat template in the GGUF.
- It always thinks, and the default `reasoning_effort` is `max`. A short question can use thousands of tokens. A client can send `"reasoning_effort": "low"` or `"high"`.

A machine can hold profiles for different providers.
On the same Mac, stopping this profile and starting a vLLM (vllm-metal) Qwen3-Coder profile in its place took about 20 seconds, and the same back.

## What only vLLM does

These apply to vLLM profiles alone:

- Reading flags from the model card, and checking parsers against the installed vLLM.
- KV cache sizing and the `GPU_MEMORY_UTILIZATION` fit.
- Checking a GGUF quant against what vLLM's GGUF plugin can load. For llama.cpp, `new --quant`, `search` and the panel offer every GGUF quant in a repo, split and subfolder ones too. For MLX they offer no GGUF quant.
- `adopt`, which looks for running vLLM servers.
- `shard`, which refuses a profile for another provider.
- The hardware definitions' forbidden flags, which are vLLM flags. A recipe on another engine is checked against the GPU vendors its engine runs on instead.

## Contributing an engine

An engine is a JSON definition, so adding one needs no Go code.
Each shipped engine is one file in [`catalogue/engines/`](https://github.com/coder/llm-helper/tree/main/catalogue/engines), named after its id, such as `catalogue/engines/vllm.json`.
To add one, copy the closest file to `catalogue/engines/<id>.json` and change it.

A definition holds `schema_version` and the engine's fields, and says:

- `runtime`: `process`, for a server the service unit runs on the host, `container`, for one a recipe runs on each node, or both.
- `why`: what the server is and why it is launched this way. Every param has a `why` too.
- `gpu_vendors`: the hardware vendor ids it runs on, plus `cpu`. A recipe refuses a node of any other vendor.
- `install`: how to get the server, by OS or `any`. A process engine needs one.
- `api_key`: `flag:--name`, `env:NAME` or `none`. Only a flag puts the key on the command line, and only a flag may use the `{api_key}` placeholder. A container's key is `env:NAME` or `none`, so it never reaches `docker run`'s command line.
- `serve` and `optional_args` for a process, or `container.roles` for a container: the command, with placeholders such as `{model}`, `{port}` and the engine's own params. An optional group is left out when any of its placeholders is empty. A container's `launch_args` take only what every rank shares: `{model}`, `{port}`, `{nodes}`, the params and the pinned models' paths.
- `health`: the path that answers 200, and whether it does so only once the model is loaded (`ready`) or as soon as the process listens (`live`).
- `capabilities`: the few differences shared code acts on, such as `vllm_args` for vLLM's sizing and flag checks. Launching and health checks act on a capability, never on an engine's name.
  `chat_template_kwargs_flag` names the flag that sets default chat template kwargs, which is how a profile turns a model's thinking on or off; see [Model types](./model-types.md).
- `notes`: optional sentences `inspect` prints for some kinds of model.
  `hybrid` is shown for a model whose layers are not all attention, such as one with linear-attention or Mamba layers.

The definition is checked when it loads.
An unknown field, an unknown placeholder, a key that would land on the command line, or a container with no role that serves HTTP is refused with the reason.
A file whose name is not its id is refused too.
The tests in `internal/engine` render every shipped engine for every rank, so a new engine is covered once its file is in `catalogue/engines/`.

To try an engine without rebuilding, put its file under `~/.config/vllm-helper/engines.d/`, in the same shape and named the same way as a file in `catalogue/engines/`.
A local file can only add engines with new ids; it cannot change how a shipped engine is launched.
A file that does not load is skipped and named by `engines list` and `doctor`, and the other files still load.
