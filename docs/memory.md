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

## Next steps

- [Profiles](./profiles.md) — where `KV_CACHE_MEMORY_BYTES` comes from.
- [Models and quantizations](./models.md) — how the budget decides what gets recommended.
