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

## Grace Hopper and Grace Blackwell

A GH200, GB200 or GB300 has HBM of its own, and the Grace CPU's LPDDR5X attached over NVLink-C2C, which the GPU reads coherently.
The budget is the HBM alone, because in the driver's default memory mode CUDA allocates only from it, so that is where the weights and the KV cache go.
The CPU's memory is a second pool, and `doctor` and `setup` print how much of it a model can use on their `offload:` line: system memory, less what the OS is using and 8 GiB for the server's own process.

The second pool is offered only to mixture-of-experts models.
Each token reads only a few of their experts, where a dense model would read every offloaded weight for every token.
`inspect` and the panel recommend the largest copy of a model that fits on the GPU.
Only when none does do they recommend the largest copy that fits with part of its experts in CPU memory, and say so.
Every copy that fits either way stays selectable.

A profile `new` creates for such a copy keeps the shortfall, plus 4 GiB more for the KV cache, in CPU memory.
For vLLM that is `--cpu-offload-gb <GiB> --cpu-offload-params experts` in `EXTRA_ARGS`, and `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` in the profile, which are the settings NVIDIA's DGX Station recipes use.
Pinning is off because on these machines pinned memory is taken from the GPU's.
llama.cpp's `llama-server` moves experts to CPU memory by itself when the weights outgrow the GPU, so its profile gets nothing added.
Each engine says which it does in the `offload` field of its definition; see [Providers](./providers.md).

Offloaded experts are read at the CPU memory's bandwidth, 396 GB/s on a DGX Station GB300 against 7.1 TB/s for its HBM, so decode is slower than for a copy that fits on the GPU.
How much slower depends on the model and has not yet been measured on a GB300.

A DGX Station also has a small display GPU, an RTX PRO 2000.
DGX OS names the GB300 in `CUDA_VISIBLE_DEVICES` in `/etc/environment`, which login shells read and services started by systemd do not.
On a machine whose GPUs are different models, `setup` writes `CUDA_VISIBLE_DEVICES` to `config.env`, naming the GPUs with the most memory by UUID, unless `config.env` already sets it.
`doctor` warns, with the line to add, until it is set.

Older drivers default to a NUMA memory mode instead, in which the HBM is also system memory.
There the OS can fill the HBM with page cache and vLLM's free-memory check can fail at startup, so update the driver before serving.
Leave MIG off: llm-helper does not size MIG instances.

## AMD Ryzen AI Max (Strix Halo)

A Ryzen AI Max GPU (Radeon 8060S, 8050S or 8040S) has no memory of its own beyond a carve-out the firmware sets aside, 512 MB by default.
It maps the rest from system memory through GTT, up to a limit the `ttm` kernel module sets, which `amdgpu` reports in `/sys/class/drm/card0/device/mem_info_gtt_total`.
The budget is measured against that limit plus the carve-out, not against all of RAM, and `doctor` prints the limit on its `gtt:` line.
llama.cpp sees the same memory: its ROCm build reports the GTT limit as the device's memory, and its Vulkan build reports the limit plus the carve-out.

The default limit leaves memory idle on a machine that only serves models.
On a 128 GB machine it is 101 GB, and a 103 GB model does not fit.
To raise it, set `pages_limit` in 4 KiB pages, rebuild the initramfs so the module reads it at boot, and reboot:

```sh
echo 'options ttm pages_limit=29360128' | sudo tee /etc/modprobe.d/ttm.conf
sudo update-initramfs -u
sudo reboot
```

29360128 pages is 112 GiB.
On a 128 GB machine that raised the limit to 120 GB, and with a 103 GB model loaded about 15 GB of RAM was still free.
`doctor` suggests a value for its machine, 90% of RAM in whole GiB, until the limit is within 4 GiB of it.
`update-initramfs` is the Debian and Ubuntu command; on Fedora run `sudo dracut -f` instead.
The OS shares the same memory, so the budget never counts more than RAM less what the OS is using, however high the limit is set.

## Next steps

- [Profiles](./profiles.md) — where `KV_CACHE_MEMORY_BYTES` comes from.
- [Models and quantizations](./models.md) — how the budget decides what gets recommended.
- [Hardware detection](./hardware.md) — what is read from the machine, and how to add a definition.
