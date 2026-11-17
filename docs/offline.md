---
title: Offline machines and external drives
description: Move models between machines on a drive, and serve a machine that cannot reach the Hugging Face Hub.
---

A machine with no route to the Hugging Face Hub can still serve any model that is in its Hugging Face cache.
`llm-helper export` copies models from a machine that has them onto a drive, and `llm-helper import` copies them into the cache of the machine that does not.

## Move a model on a drive

On a machine that has the model, copy it onto the mounted drive:

```sh
llm-helper export Qwen/Qwen3-8B-FP8 /media/usb
```

On the offline machine, copy it into the cache the server reads, then create the profile:

```sh
llm-helper import /media/usb Qwen/Qwen3-8B-FP8
llm-helper new Qwen/Qwen3-8B-FP8
```

`import` lists the models it finds when you name none, and asks which to copy.
`--all` copies every complete model, and `--dry-run` shows what would be copied and stops.

Both commands:

- Refuse a model that is incomplete, because it would not load on the other machine. A missing file, a link to a blob that is not there, no weights, a shard an index names but the folder lacks, or a missing piece of a split GGUF all count.
- Check every file against its recorded hash as it is copied, and fail on a damaged one without keeping it.
- Resume after an interruption, skip files already there, and never overwrite what is in the cache.
- Check the free space first, and say how much is needed.

`export` writes a plain copy of each file under `hub/` on the drive, with no links, so any filesystem can hold it.
A plain copy keeps no hash of its own, so `export` records each file's hash in `llm-helper-hashes/` beside the snapshot, and `import` checks the copy against it.
FAT32 cannot hold a file over 4 GiB, so `export` refuses a model with one: format the drive as exFAT, or use ext4 or APFS.

## What `import` reads

`import` finds a Hugging Face cache at the path you give, or under it in `hub/`, `huggingface/hub/` or `.cache/huggingface/hub/`.
That covers a drive written by `export`, and a copy of `~/.cache/huggingface`.
A cache that shares one blob between several snapshots is read as it is.

Two other layouts record no repo id, so name it:

```sh
llm-helper import /media/usb/qwen --repo Qwen/Qwen3-8B-FP8 --revision <commit>
```

- A folder written by `hf download --local-dir` needs `--repo`. Its commit is read from the folder.
- A plain model folder needs `--repo` and a commit. A git clone supplies its own, and any other folder needs `--revision`, because llm-helper does not invent one.

## Which cache the server reads

`import` and `export` work on the cache the server reads, which is the one `llm-helper doctor` prints under `MODEL CACHE`.
The service sources `config.env` and `secrets.env`, not your login shell, so an `HF_HOME` exported in a shell is not the server's.
In order, the first of these that `config.env` sets wins:

1. `LLAMA_CACHE`, for llama.cpp only.
2. `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`.
3. `HF_HOME`, with `/hub` added.
4. `XDG_CACHE_HOME`, with `/huggingface/hub` added.
5. `~/.cache/huggingface/hub`.

The server's Python libraries expand `~` and `$HOME` in `HF_HOME` and `HF_HUB_CACHE`, and llm-helper reads them the same way.
`LLAMA_CACHE` is read as written, so a `~` in it is not expanded.
`import` and `export` refuse a server cache that is not an absolute path the server resolves the same way, and `doctor` warns about it.
Set an absolute path in `config.env`, or pass `--cache`.

`doctor` warns when your shell's `HF_HOME` or `HF_HUB_CACHE` points somewhere else.
`--cache <dir>` names a different cache for one command.

## Serve with no network

Set `HF_HUB_OFFLINE=1` in `config.env` on a machine that never reaches the Hub.
`llm-helper new`, `inspect` and `search` then read the cache instead of the Hub:

- `new` and `inspect` read the model's config, chat template and file list from the cached snapshot.
- `search` lists the cached models that match your terms.
- A model that is not in the cache is reported as not cached, instead of failing on a network error.

Without the setting, the same happens once a request cannot reach the Hub, and the Hub is tried again a minute later.

A profile made while offline tells its server to stay off the network, as the engine defines it:

| Engine    | Setting in the new profile                                             |
| --------- | ---------------------------------------------------------------------- |
| vLLM      | `HF_HUB_OFFLINE=1`                                                     |
| MLX       | `HF_HUB_OFFLINE=1`                                                     |
| llama.cpp | `--offline` in `EXTRA_ARGS`, only when the installed `--help` lists it |

For llama.cpp, set `LLAMA_CACHE` in `config.env` if the cache is not the default one, because `llama-server -hf` reads that first.
A build that does not list `--offline` is left without it, and `new` says so.
Existing profiles are not changed: add the setting to one with `llm-helper edit <profile>`.

## Tested recipes

A recipe launches containers that load a pinned revision from each node's cache and refuse to start when it is missing.
`llm-helper import <drive-path> --recipe <id>` copies exactly the models the recipe pins into the cache its engine reads, and fails when the drive has another revision.
Run it on each node.

See [Models and quantizations](./models.md) for choosing a model, and [Recipes](./recipes.md) for the launch configurations.
