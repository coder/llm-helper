---
title: Install and upgrade
description: Install the binary, configure the machine, and keep it current.
---

llm-helper ships as a single binary for Linux amd64 and arm64 and for macOS arm64.
It manages an existing inference server install; it does not install one itself.
On Linux that means an NVIDIA GPU with a vLLM usually living in a venv, or llama.cpp on a GPU or the CPU; on macOS it means vllm-metal, which serves on the Apple Silicon GPU, see [Running on macOS](./macos.md), or mlx-lm or llama.cpp, see [Providers](./providers.md).

## Install

1. Download the latest [release](https://github.com/coder/llm-helper/releases) for your platform.
2. Unpack it and install the binary:

   ```sh
   tar xzf llm-helper_*.tar.gz
   sudo install llm-helper /usr/local/bin/
   sudo ln -sf llm-helper /usr/local/bin/vllm-helper
   ```

   The archive also carries the same binary as `vllm-helper`, its old name, for installs from before the rename. The link keeps scripts that call the old name working.

3. Run `llm-helper setup`, or `llm-helper setup --provider mlx` or `--provider llamacpp` to serve with mlx-lm or llama.cpp.

If `vllm` isn't on your PATH, run setup with `VLLM_BIN` set to its full path in the environment, or set `VLLM_BIN` in `~/.config/vllm-helper/config.env` afterwards.
The other providers' keys, such as `LLAMA_SERVER_BIN`, work the same way, and one set in the environment wins over a server on PATH.

## What setup writes

`setup` writes two files under `~/.config/vllm-helper/`:

- **config.env** — settings that apply to every profile, sourced into the server's environment whenever a model starts.
- **secrets.env** — credentials, generated or filled in by you. Never commit this file anywhere.

The same directory can also hold **hardware.d/**, which `setup` does not create: local hardware definitions for a machine llm-helper does not recognize yet.
See [Hardware detection](./hardware.md).

| Key                       | Where       | Meaning                                                                                                                                    |
| ------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `PORT`                    | config.env  | First port a model listens on. Each model running at the same time takes the next free port. Default `8001`.                               |
| `PROVIDER`                | config.env  | The inference server new profiles are made for: `vllm` (default), `mlx` or `llamacpp`. See [Providers](./providers.md).                    |
| `VLLM_BIN`                | config.env  | Full path to the `vllm` executable. Required for vLLM profiles.                                                                            |
| `MLX_LM_BIN`              | config.env  | Full path to the `mlx_lm.server` executable. Required for MLX profiles.                                                                    |
| `LLAMA_SERVER_BIN`        | config.env  | Full path to the `llama-server` executable. Required for llama.cpp profiles.                                                               |
| `SERVICE_NAME`            | config.env  | The single service unit earlier releases ran: `vllm.service` on Linux, `com.coder.vllm` on macOS. Read only by `migrate`.                  |
| `SERVICE_SCOPE`           | config.env  | `system` on Linux (needs sudo to control) or `user`. macOS defaults to `user`; use `system` on a headless Mac.                             |
| `DEFAULT_PROFILE`         | config.env  | The model Coder selects first. Written by `llm-helper default`. See [Several models at once](./multi-instance.md).                         |
| `ROUTER_PORT`             | config.env  | Port the model router listens on. Written by `llm-helper router install-service`.                                                          |
| `MODEL_BUDGET_GB`         | config.env  | Overrides the measured model budget. See [Memory budget](./memory.md).                                                                     |
| `KV_CONTEXT_TARGET`       | config.env  | How many full-length requests each profile's KV cache is sized to hold. Default `4`.                                                       |
| `METAL_WIRED_LIMIT_MB`    | config.env  | macOS only. GPU wired memory limit in MiB, applied at boot. Written by `llm-helper metal-limit --set`. See [Running on macOS](./macos.md). |
| `UPDATE_CHANNEL`          | config.env  | Where `upgrade` looks for builds: `stable` or `dev`. Written when you pass `--channel`.                                                    |
| `CODER_URL`               | config.env  | Coder deployment to sync with. See [Coder integration](./coder.md).                                                                        |
| `CODER_AI_PROVIDER_ID`    | config.env  | The `openai-compat` provider to hang model entries off.                                                                                    |
| `CODER_ORGANIZATION`      | config.env  | Organization the entries hang off. Default `default`.                                                                                      |
| `CODER_MAX_OUTPUT_TOKENS` | config.env  | Reply cap sent with each model entry. Unset leaves Coder's own default (32,000).                                                           |
| `VLLM_API_KEY`            | secrets.env | API key clients must send to the endpoint. Generated at setup.                                                                             |
| `CODER_API_TOKEN`         | secrets.env | Coder session token for sync.                                                                                                              |
| `HF_TOKEN`                | either      | HuggingFace token, needed only for gated or private models. A value in `secrets.env` wins.                                                 |
| `GITHUB_TOKEN`            | either      | Token for reading releases while the repository is private. A value in `secrets.env` wins; `upgrade` also accepts it in the environment.   |

## Upgrade

```sh
llm-helper upgrade --check          # what's available, without installing
llm-helper upgrade                  # replace this binary
llm-helper upgrade --channel dev    # follow main instead of releases
```

There are two channels.
`stable` is the tagged releases.
`dev` is rebuilt from `main` on every commit that passes CI, so it carries work that hasn't been released yet — useful for trying a fix before it ships.
The channel you pass is remembered, so later `llm-helper upgrade` runs stay on it; switch back with `--channel stable`.

Upgrading replaces the binary only.
A running server was started by the old one and keeps serving until `llm-helper restart`.
If the binary lives somewhere only root can write, run the upgrade with `sudo`.

While this repository is private, reading releases needs a token: set `GITHUB_TOKEN` in the environment or in `config.env`.

Run `llm-helper version` to see what is running.
On the dev channel every build shares one moving release tag, so the commit is what tells two builds apart.

## Upgrading from vllm-helper

llm-helper is vllm-helper under a new name, and an upgrade from vllm-helper changes nothing on disk.
`vllm-helper upgrade` replaces the binary where it is, and the old name keeps working: config and state stay in `~/.config/vllm-helper` and `~/.local/state/vllm-helper`, and the `VLLM_HELPER_*` overrides are read as before.
Commands now print the new name in their hints, so link it beside the old one:

```sh
sudo ln -s vllm-helper "$(dirname "$(command -v vllm-helper)")/llm-helper"
```

The single service unit from earlier releases, `vllm.service` or `com.coder.vllm`, keeps running until you run `llm-helper migrate`, which moves the model it serves to its own [instance](./multi-instance.md).

## Next steps

- Follow the [Quickstart](./quickstart.md) to serve a first model.
- [Memory budget](./memory.md) — what this machine can hold, and how to override it.
