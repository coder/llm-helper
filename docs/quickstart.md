---
title: Quickstart
description: Set up llm-helper, pull a model, and serve it.
---

This guide takes a machine from nothing to serving a model through vLLM's OpenAI-compatible endpoint.
It assumes vLLM is installed, usually in a venv, and that the [llm-helper release](./install.md) for your platform is on the machine.

## Run setup

1. Run `llm-helper setup`.

Setup detects your hardware and any existing vLLM install, generates an API key, writes config, and installs a service unit — a systemd unit on Linux, a launchd job on macOS.
It never overwrites existing config and never restarts a running server.

2. If `vllm` isn't on your PATH, set `VLLM_BIN` in `~/.config/vllm-helper/config.env` to its full path.

## Pull a model

1. Run `llm-helper new` with a HuggingFace repo:

   ```sh
   llm-helper new Qwen/Qwen3-32B
   ```

This pulls the model, saves its settings as a profile, and restarts the server onto it.
The weights are downloaded by vLLM as it starts, so the first switch takes as long as the download does.

To browse instead of naming a repo outright, run `llm-helper search <query>`.
It lists matches with each one's weight size checked against what this machine can hold, and walks you through pulling the one you pick.

## Check it

1. Run `llm-helper status`.

`status` reports the service state, the active model, and whether the endpoint responds.
It distinguishes a load that is still in progress from a service failing to start — a large model can take many minutes, and the two look identical from the outside.

## Point a client at it

Any OpenAI-compatible client can use the endpoint at `http://<machine>:8001/v1`, with the API key from `~/.config/vllm-helper/secrets.env`.

## Switch models

1. Run `llm-helper list` to see the saved profiles.
2. Run `llm-helper switch <profile>` to load a different one.

Switching restarts the server, which takes a few minutes when the weights are cached and longer on first download.

## Script it

- `status`, `list`, `search`, `inspect`, `doctor` and `recipes list` take `--json`.
- `switch`, `new` and `restart` take `--wait` to block until the endpoint is healthy, `--no-follow` to return immediately, and `--timeout`.
- When stdout is piped, commands don't tail logs or start pickers. When stdin isn't a terminal, prompts take their defaults and sudo runs with `-n`.

## Next steps

- See what a profile holds in [Profiles](./profiles.md).
- Manage the machine from a browser with the [Web panel](./panel.md).
