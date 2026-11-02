---
title: Introduction
description: What llm-helper is, and when to reach for it.
---

llm-helper serves large models with vLLM, mlx-lm or llama.cpp on one GPU machine or several, and switches models with one command.
It wraps an OpenAI-compatible inference server, [vLLM](https://github.com/vllm-project/vllm) by default, [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon or [llama.cpp](https://github.com/ggml-org/llama.cpp) for GGUF models, with saved model profiles, HuggingFace search, tested multi-machine recipes, and an optional [Coder](https://coder.com) Agents sync.
It was called vllm-helper until it served more than vLLM, and the old name still works.

It was built for unified-memory machines like the NVIDIA DGX Spark, where one machine holds one big model and a bigger model needs two or more working together.
It works on any Linux machine with an NVIDIA GPU, and experimentally on macOS through vLLM's CPU backend.

## How it fits together

- **A profile** is a small env file holding the `vllm serve` settings that work for one model.
- **A service unit** runs the active profile: a systemd unit on Linux, a launchd job on macOS.
- **The CLI** switches profiles, pulls models, and diagnoses the machine.
- **The web panel** is what `llm-helper serve` serves: the same controls in a browser.
- **The endpoint** is the inference server's own OpenAI-compatible API, on port 8001 by default.

Inference traffic goes straight to the inference server on its own port.
llm-helper is a control plane, so nothing here sits in the path of a request — a bug in it cannot break inference.
The one exception is a [pool](./multi-node.md), whose router sits in front of the replicas by design.

The binary is self-contained: one file to put on a machine, with the docs you are reading compiled in.

## Next steps

- Follow the [Quickstart](./quickstart.md) to set the machine up and serve a first model.
- See what a profile holds in [Profiles](./profiles.md).
- Serve with mlx-lm or llama.cpp instead of vLLM with [Providers](./providers.md).
- Serve a model measured across several machines with [Recipes](./recipes.md).
- Pool replicas or shard one model across machines with [More than one machine](./multi-node.md).
- Browse the [CLI reference](./reference/cli.md) for every command.
