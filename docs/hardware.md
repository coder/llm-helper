---
title: Hardware detection
description: What llm-helper reads from the machine, how it recognizes hardware, and how to add a definition.
---

llm-helper reads the machine on every run, decides how much memory a model can use, and refuses flags known to crash on this GPU.
What it knows about specific hardware lives in data, as hardware definitions in [`catalogue/hardware/`](https://github.com/coder/llm-helper/tree/main/catalogue/hardware), so supporting a new machine means adding a file and a report, not changing the detection code.
An NVIDIA GPU with no definition still works: its memory comes from `nvidia-smi` and its architecture from the compute capability it reports.

## What is read

- **NVIDIA GPUs** — every GPU `nvidia-smi` lists, with its memory, compute capability and addressing mode.
  `nvidia-smi` gets 10 seconds; a wedged driver is treated as `nvidia-smi` failing, rather than hanging the command.
- **`CUDA_VISIBLE_DEVICES`** — from the environment and from `config.env`, `secrets.env` and the profile being served, so the GPUs sized are the ones the server will get.
- **Jetson markers** — `/etc/nv_tegra_release`, `/sys/class/tegra-firmware` and the device-tree model.
- **The PCI bus** — every display controller from a known vendor, and every processing accelerator, so an AMD GPU or an NPU is named even when llm-helper does not size for it.
  A GPU that a hardware definition recognises by its PCI ids, such as the AMD Ryzen AI Max (Strix Halo), is sized from what amdgpu lets it map: its GTT limit (`mem_info_gtt_total`) plus its firmware carve-out (`mem_info_vram_total`). See "AMD Ryzen AI Max (Strix Halo)" in [Memory budget](./memory.md).
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

The shipped definitions are in [`catalogue/hardware/`](https://github.com/coder/llm-helper/tree/main/catalogue/hardware), one file per entry, in a folder for each kind:

- **`machines/`** — a kind of machine, such as `dgx_spark_gb10.json`: how to recognise its GPU and how its memory is shared.
- **`architectures/`** — a GPU architecture, such as `sm121.json`: its compute capability and the flags that abort vLLM on it.
- **`vendors/`** — an accelerator maker, such as `nvidia.json`, and its PCI vendor ids.
- **`reports/`** — what `llm-helper hardware report` printed for a machine, used as a test fixture.

Each file is named after the id it defines, starts with the format version this llm-helper reads, and is parsed strictly.
`llm-helper hardware list` shows the machines in effect, including any added locally.

A machine matches a vendor and a regular expression over the GPU's name, case-insensitively.
A GPU that its driver gives no name, such as an AMD GPU under `amdgpu`, is matched by its vendor and its PCI device ids instead, with `"pci_devices": ["0x1586"]`; `llm-helper hardware report` prints them.
No two machines may match the same GPU, so keep the pattern or the list narrow enough to match only yours; the tests run every report against every machine and fail on an overlap.
Every definition needs a `why`, because an entry nobody can explain cannot be reviewed, and cannot be removed when the reason stops holding.

## Local definitions

A machine owner can add definitions without a rebuild, by putting files in `~/.config/vllm-helper/hardware.d/`, laid out like the catalogue: `hardware.d/machines/<id>.json`, `hardware.d/architectures/<arch>.json` and `hardware.d/vendors/<id>.json`.
They load before every command: vendors, then architectures, then machines, each folder in name order.
A local file can add a vendor, an architecture, a machine, or forbidden flags on an existing architecture.
It cannot change anything already defined, so a stale file can teach llm-helper about a new machine but cannot change what it says about a known one or drop a safety check.
Local machines are tried after the shipped ones, and `doctor` names a local machine that claims a GPU a shipped one already matches.
A file that does not parse, tries to change something, or sits directly in `hardware.d` is skipped and named by `doctor` and `hardware list`, and the other files still load.
`llm-helper serve` reads the directory when it starts, so restart it after editing.

A machine file, `hardware.d/machines/nvidia_rtx_pro_6000_blackwell.json`:

```json
{
  "schema_version": 1,
  "id": "nvidia_rtx_pro_6000_blackwell",
  "display_name": "NVIDIA RTX PRO 6000 Blackwell",
  "match": { "vendor": "nvidia", "gpu_name": "RTX PRO 6000 Blackwell" },
  "memory": "discrete",
  "arch": "sm120",
  "why": "A workstation card with 96 GB of its own memory, which can equal a host's RAM and look shared by size alone."
}
```

## Contributing a definition

1. On the machine, run `llm-helper hardware report`.
   It prints the raw readings and what llm-helper concluded from them, as JSON, without a hostname, serial number, GPU UUID or bus address.
2. Save the report as `catalogue/hardware/reports/<id>.json`, and start its `provenance` with `measured: `, followed by the machine and driver.
   A report written by hand from a specification starts with `synthetic: ` instead, so nobody cites an invented machine as evidence.
3. Add `catalogue/hardware/machines/<id>.json`, copying the closest machine, and set the report's `detected` to what llm-helper should now conclude.
   Add an architecture or vendor file too if the machine names one that is missing.
4. Run `go test ./internal/hardware/ ./internal/detect/`.
   Every shipped machine needs a report that detects it, and no GPU in any report may match two machines.
5. Open a pull request with the report in its description.

A flag that aborts vLLM on an architecture goes in that architecture's file under `forbidden_args`, with its `why`, rather than on a machine.

Trying the definition in `hardware.d` first, with `hardware list` and `doctor`, shows whether it matches before anything is rebuilt.

## Next steps

- [Memory budget](./memory.md) — what the budget does with the memory kind.
- [Recipes](./recipes.md) — recipes record the hardware definition they were measured on.
