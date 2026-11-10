---
title: Several models at once
description: Running more than one model on a machine, and offering them all through one port.
---

Every model llm-helper runs is an instance: a profile with its own port, its own service and its own log.
Starting one never stops another, so a machine with the memory for it can serve a coding model and a small fast model side by side.
This works the same on one GPU machine or several, and each machine keeps its own set of instances.

## Start and stop

```sh
llm-helper up qwen3-coder
llm-helper up gemma-small
llm-helper status
llm-helper down gemma-small
```

- `up <profile>` starts the profile as an instance next to anything already running, and follows its log until the endpoint answers.
  Pass `--wait` to block without the log, or `--no-follow` to return as soon as it is started.
- `down <profile>` stops it and removes its service.
  With exactly one instance running, the name can be left out.
  `stop` is another name for `down`.
- `status` lists every instance with its port and whether its endpoint responds.
- `logs <profile>` and `restart <profile>` act on one instance, and on the only one when exactly one runs.

A profile name becomes part of the service name, so it must use lowercase letters, digits, `.`, `_` and `-`, starting with a letter or digit.
A running profile can't be removed; stop it first.

On Linux each instance is a unit of the `llm-helper@.service` template, so `llm-helper@qwen3-coder.service` serves `qwen3-coder`.
On macOS each one is its own launchd job, `com.coder.llm-helper.<profile>`; see [Running on macOS](./macos.md).

The panel shows one row per instance, with Restart, Stop and Make default, and a Run button for saved profiles that are not running; see [Web panel](./panel.md).

## Ports

The first instance takes `PORT`, 8001 unless config says otherwise.
Each later one takes the next port up that no other instance holds and nothing else is listening on.
An instance keeps its port when it is restarted.
`up --port N` picks one yourself.

`status` prints every endpoint, and a client reaches each model at `http://<machine>:<port>/v1`.

## Memory

Instances share the GPU's memory; nothing is set aside for each one in advance.
Before it starts anything, `up` checks that the new model fits next to the instances already running on the same GPUs.
An instance pinned to other GPUs, or a recipe whose head runs on another machine, does not count against it.
When it does not, it refuses and says how much the running models already use:

```text
'gemma-small' wants 0.45 of GPU memory, but the running models already use 0.60 of the 0.90 available (qwen3-coder 0.60); stop one first, or give this one a smaller share (--mem-util or --kv-cache)
```

- `--mem-util F` gives this instance a fraction of GPU memory, for example `0.3`, instead of the profile's own.
- `--kv-cache SIZE` gives it a fixed KV cache, for example `20G`, and skips the check.
  It sets `KV_CACHE_MEMORY_BYTES`, which vLLM reads and other providers ignore.

The share is recorded with the instance, not the profile, so the next `up` without a flag uses the profile's own again.
See [Memory budget](./memory.md) for how a profile's share is worked out.

## Reboots

Instances come back after a reboot in the order they were first started, oldest first, each one waiting for the one before it to load.
A model that no longer fits next to the others exits with status 78 instead of loading, so the ones already up keep their memory.
On Linux its unit then stays stopped (`RestartPreventExitStatus=78`); on macOS launchd keeps retrying it, and it starts once there is room.

## Coder

Sync offers every running instance in Coder's model picker.
The model Coder selects first is `DEFAULT_PROFILE`, set with `llm-helper default <profile>` or Make default in the panel.
When that is unset, or the profile is not running, the instance started first is the default.
`llm-helper default` without a name prints the current one.

A Coder AI provider has one base URL, so on its own it reaches one port.
Without the router, only the instance on `PORT` is enabled in the picker, and sync names the others it left out.
See [Coder integration](./coder.md).

## The model router

The router answers on one port and sends each request to the instance serving the model the request names.
It is off unless you install it:

```sh
llm-helper router install-service
```

- It listens on port 8000, or `--port N`, and records the port as `ROUTER_PORT` in config.
- It binds to loopback unless `VLLM_API_KEY` is set, in which case it listens on every interface, as the instances do.
  With the key set, every request to the router needs `Authorization: Bearer <key>`, and the router sends the same key to the instances when it lists their models.
- A recipe instance is reached on the machine that runs its head.
- Point the Coder AI provider's base URL at `http://<machine>:8000/v1`, and every running instance is enabled in the picker.
- `llm-helper router serve` runs it in the foreground instead; `--bind` there sets the address and overrides `--port`.
- `llm-helper router uninstall-service` removes it and clears `ROUTER_PORT`.

The instances keep their own ports, so clients that already talk to one directly are unaffected.

## Upgrading from a single-model release

Earlier releases ran one model through one unit, `vllm.service` on Linux or `com.coder.vllm` on macOS, and switched it between profiles.
After an upgrade that unit keeps serving its model until you run:

```sh
llm-helper migrate
```

`migrate` stops the old unit and starts the model it served as an instance on the same port.
For a recipe profile it also rewrites the recipe's service and health watchdog units to the per-instance form.
`doctor` reports when it is still needed, and `up` refuses to start a model until it has run.

`switch <profile>` still works for one more release: it stops every other instance and runs that one.
Use `up` and `down` instead.

## Recipes and multi-node profiles

A profile generated from a recipe runs in a container and can span machines.
`up` runs it as an instance like any other, and `recipes install-service --profile <name>` keeps it serving across reboots; see [Recipes](./recipes.md) and [Multi-node](./multi-node.md).
Its row in the panel has no Restart or Stop, because the container is managed with the recipe commands.
