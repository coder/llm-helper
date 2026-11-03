---
title: Hardware detection
description: What llm-helper reads from the machine, how it recognizes hardware, and how to add a definition.
---

llm-helper reads the machine on every run, decides how much memory a model can use, and refuses flags known to crash on this GPU.
What it knows about specific hardware lives in data, as hardware definitions, so supporting a new machine means adding a definition and a test fixture, not changing the detection code.

## What is read

- **NVIDIA GPUs** — every GPU `nvidia-smi` lists, with its memory, compute capability and addressing mode.
  `nvidia-smi` gets 10 seconds; a wedged driver is treated as `nvidia-smi` failing, rather than hanging the command.
- **`CUDA_VISIBLE_DEVICES`** — from the environment and from `config.env`, `secrets.env` and the active profile, so the GPUs sized are the ones the server will get.
- **Jetson markers** — `/etc/nv_tegra_release`, `/sys/class/tegra-firmware` and the device-tree model.
- **The PCI bus** — every display controller from a known vendor, and every processing accelerator, so an AMD GPU or an NPU is named even though llm-helper does not size for it yet.
- **The cgroup memory limit** — in a container or a limited systemd unit, the budget is held to the limit rather than the host's RAM.
- **Apple Silicon** — the SoC's brand string, which names its GPU.

`llm-helper doctor` prints what was found on its `hardware:` line, along with anything on the PCI bus and any cgroup limit.

## How memory is decided

Whether the GPU has memory of its own decides where the budget comes from, so llm-helper records what decided it.
The first of these that applies wins:

1. **definition** — a hardware definition names the GPU and says how its memory is shared.
2. **tegra** — the Jetson markers are present.
3. **addressing** — the GPU reports no memory of its own and reaches system memory through ATS, as a GB10 does.
4. **size** — the GPU reports nearly all of system memory as its own, and its addressing mode does not say it is a discrete card.
5. **reported** — the GPU reports memory of its own and nothing says it is shared: a discrete card.

A definition's memory is one of three kinds:

- **unified** — one pool shared with the CPU, such as a DGX Spark, a Jetson or a Mac.
- **discrete** — memory of the GPU's own.
- **coherent** — memory of the GPU's own with the CPU's memory coherently attached, as on Grace Hopper and Grace Blackwell. It is sized as unified until that has been measured on the hardware.

## Flags that crash an architecture

Each architecture can list flags and values known to abort vLLM at startup on it, each with the reason.
llm-helper refuses them wherever it builds a vLLM command line: `start`, a shard's head and workers, and a recipe launch on each of its nodes.
A node's architecture is recorded by `node add` and `node probe`, so a node probed before this was recorded is only checked once it is probed again.

## Hardware definitions

The shipped definitions are in [`internal/hardware/definitions.json`](https://github.com/coder/llm-helper/blob/main/internal/hardware/definitions.json).
`llm-helper hardware list` shows the ones in effect, including any added locally.

A definition matches a vendor and a regular expression over the GPU's name, case-insensitively.
Definitions are tried in order, so a more specific one goes before a broader one.
Every definition needs a `why`, because an entry nobody can explain cannot be reviewed, and cannot be removed when the reason stops holding.

## Local definitions

A machine owner can add definitions without a rebuild, by putting `*.json` files in `~/.config/vllm-helper/hardware.d/`.
Files load in name order, before every command.
A local file can add vendors, architectures, definitions, and forbidden flags on an existing architecture.
It cannot change anything already defined, so a stale file can teach llm-helper about a new machine but cannot change what it says about a known one or drop a safety check.
Local definitions are tried after the shipped ones.
A file that does not parse, or tries to change something, is skipped and named by `doctor` and `hardware list`, and the other files still load.
`llm-helper serve` reads the directory when it starts, so restart it after editing.

Every file starts with the format version this llm-helper reads:

```json
{
  "schema_version": 1,
  "definitions": [
    {
      "id": "nvidia_rtx_pro_6000_blackwell",
      "display_name": "NVIDIA RTX PRO 6000 Blackwell",
      "match": { "vendor": "nvidia", "gpu_name": "RTX PRO 6000 Blackwell" },
      "memory": "discrete",
      "arch": "sm120",
      "why": "A workstation card with 96 GB of its own memory, which can equal a host's RAM and look shared by size alone."
    }
  ]
}
```

## Contributing a definition

1. On the machine, run `llm-helper hardware report`.
   It prints the raw readings and what llm-helper concluded from them, as JSON, without a hostname, serial number, GPU UUID or bus address.
2. Save the report as `internal/hardware/testdata/<id>.json`, and start its `provenance` with `measured: `, followed by the machine and driver.
   A fixture written by hand from a specification starts with `synthetic: ` instead, so nobody cites an invented machine as evidence.
3. Add the definition to `internal/hardware/definitions.json`, and set the fixture's `detected` to what llm-helper should now conclude.
4. Run `go test ./internal/hardware/ ./internal/detect/`.
   Every shipped definition needs a fixture that detects it, which also catches a definition an earlier one shadows.
5. Open a pull request with the report in its description.

Trying the definition in `hardware.d` first, with `hardware list` and `doctor`, shows whether it matches before anything is rebuilt.

## Next steps

- [Memory budget](./memory.md) — what the budget does with the memory kind.
- [Recipes](./recipes.md) — recipes record the hardware definition they were measured on.
