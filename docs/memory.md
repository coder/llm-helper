---
title: Memory budget
description: How llm-helper decides what this machine can hold, and how to override it.
---

The budget that `search`, `inspect`, and the panel's **Add model** page compare against is measured from the machine on every run.
It is the largest model whose weights, runtime overhead, and a usable KV cache all fit in memory — not a percentage of RAM, which is wrong in both directions at once.

Measuring on every run means the budget follows the machine: add RAM, or turn off the desktop that was eating it, and the next check knows.
Set `MODEL_BUDGET_GB` in `config.env` to override the measurement.
`doctor` says when an override is in force.

With a profile's KV cache sized (`KV_CACHE_MEMORY_BYTES`), `GPU_MEMORY_UTILIZATION` becomes a ceiling on what vLLM may touch rather than the size of the cache — see [Profiles](./profiles.md).

## More than one GPU

On a machine with several discrete GPUs, the budget is one GPU's memory, because that is what a server without `--tensor-parallel-size` loads onto.
The GPU measured is the first one the server will see: `CUDA_VISIBLE_DEVICES` is read from the environment, `config.env`, `secrets.env` and the profile being served, in that order, the last one set winning.
A shard reads the profile it serves, not the ones the machine's own instances run.
Servers llm-helper starts get `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so an index in `CUDA_VISIBLE_DEVICES` names the GPU `nvidia-smi` gives that index.
`doctor` says how many GPUs are hidden from the server.
Hiding every GPU leaves vLLM nothing to load onto, so the budget is zero and `doctor` warns.
A model larger than one GPU needs `--tensor-parallel-size` in the profile's `EXTRA_ARGS`.

How llm-helper decides whether a GPU has memory of its own is described in [Hardware detection](./hardware.md).

## Next steps

- [Profiles](./profiles.md) — where `KV_CACHE_MEMORY_BYTES` comes from.
- [Models and quantizations](./models.md) — how the budget decides what gets recommended.
- [Hardware detection](./hardware.md) — what is read from the machine, and how to add a definition.
