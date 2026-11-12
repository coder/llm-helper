---
title: More than one machine
description: Register machines once, then run a tested recipe, pool replicas, or shard one model across them.
---

A model can span machines in three ways, and all three start from the same registry of nodes.

| You want                                          | Use                                      |
| ------------------------------------------------- | ---------------------------------------- |
| A tested configuration, with state and Coder sync | [`recipes launch`](#run-a-tested-recipe) |
| Several copies of one model behind one endpoint   | [`pool`](#a-pool-is-replicas)            |
| One model too large for a single machine          | [`shard`](#a-shard-is-one-model-split)   |

## Register the machines

`node` records a machine once; `pool`, `shard` and `recipes launch` then refer to it by name.
Every name passed to `--nodes` must be registered, including the machine you run the command on. A node whose host is this machine is recognized as local and run without SSH.

```sh
llm-helper node add spark1 --host http://spark1:8001 --ssh coder@spark1
llm-helper node add spark2 --host http://spark2:8001 --ssh coder@spark2
llm-helper node ls
llm-helper node probe
llm-helper node rm spark2
```

- `--host` is where the node answers OpenAI requests.
- `--ssh` is how this machine orchestrates it. Without it the target is derived from `--host`, and `node ls` marks it `(from --host)`.
- `node add` also describes the node's hardware over SSH, reading only `/proc/meminfo` and `nvidia-smi` there, and `node ls` shows it in the `HARDWARE` column. Run `node probe` to describe the nodes again, for example after a node that was down comes back.

The registry lives in `nodes.json` beside your config.

## Run a tested recipe

A [recipe](./recipes.md) is a configuration that was run and measured on named hardware, with its image pinned by digest:

```sh
llm-helper recipes list                                  # what is tested, and on what
llm-helper recipes show <id>                             # image, argv, provenance, measured numbers
llm-helper recipes launch <id> --nodes spark1,spark2     # rank 0, the first node, serves the API
llm-helper recipes stop <id> --nodes spark1,spark2
```

`recipes launch` starts the containers over SSH on the nodes you name, writes a profile of the same name, records it as a running instance, and points Coder at it.
Like `up`, it then waits until the head node answers its engine's health path, following the log on a terminal; `--no-follow` returns as soon as the containers have started.
When the head is another machine, the check runs on the head over SSH, so an engine that listens only on loopback, as TensorFold does by default, is still seen to serve.
`recipes stop` reports, per node, whether there was a container to remove.
From then on `status`, `logs`, `restart <profile>` and `down <profile>` act on the recipe's containers instead of a native service unit.
`--dry-run` prints the commands, with the API key redacted, and changes nothing.

Editing that profile is allowed and reaches the command line, but the entry then reports itself as the recipe plus local overrides and stops claiming the verified numbers.
An override known to abort on the recipe's hardware is refused before any node is touched.

To keep a recipe serving across reboots:

```sh
llm-helper recipes install-service --profile <profile>
```

This installs `vllm-cluster.service`, which runs the recipe that profile names.
It also stands down the single-model `vllm.service` an earlier release may have left enabled, because that unit cannot serve a container.
To keep more than one recipe serving, install one unit per profile, each with its own `--name`.
A watchdog timer relaunches the cluster when the endpoint stops answering.
A deliberate stop (`llm-helper down`, `recipes stop`, or stopping the unit) is remembered, so the watchdog leaves it stopped until the next launch.

By default the units go to `/etc/systemd/system` and installing them needs sudo.
Without sudo, set `SERVICE_SCOPE=user` in `config.env` first.
The units then go to `~/.config/systemd/user`, run as you, and are controlled with `systemctl --user`.
The install enables lingering with `loginctl enable-linger` so they start at boot without a login; if that fails, it prints the command to run.

## A pool is replicas

Every node serves the same model, and a router in front of them spreads requests across the healthy ones:

```sh
llm-helper pool serve --nodes spark1,spark2 --bind :8011 --sticky auto
llm-helper pool install-service --nodes spark1,spark2   # the same thing, as a systemd unit
llm-helper pool status                                  # per-backend health
```

Stickiness keeps a conversation on the node that already has its prefix cached. A repeated long prompt is nearly free on a warm node and expensive on a cold one. `auto` keys on an `X-Session-ID` header when the client sends one and falls back to the `Authorization` header, so a plain OpenAI client is still sticky per API key. `authorization` always uses the API key, and `header:Name` uses a header of your choosing.

## A shard is one model split

For a model too large for any single node, `shard` runs one vLLM server across several:

```sh
llm-helper shard install-ray --nodes spark1,spark2       # once, for the default Ray backend
llm-helper shard start --profile big --nodes spark1,spark2 --tp 2 --pp 1
llm-helper shard status --nodes spark1,spark2
llm-helper shard stop --nodes spark1,spark2
```

`--tp` splits each layer across GPUs and `--pp` splits the model by layer. Their product is the number of GPUs the server expects, so on two single-GPU machines it must be 2. `--pp` defaults to 2 and `--tp` to 1. Which split is faster depends on the workload, so measure both.

`shard start` runs in the foreground against a native vLLM install and writes no state. That makes it the command for trying a split before there is a recipe for it.

- Run `shard start` on the head node: the first in `--nodes`, unless `--head` names another. The head's part runs locally and every other node is driven over SSH (pass `--ssh-key` if that takes a key).
- `--backend mp` uses vLLM's multiprocessing executor instead of Ray. It needs the head first in `--nodes`.
- `--stop-services` first stops the single-node unit (`--service-name`, `vllm.service` by default) on every node, and needs passwordless sudo there.
- `--dry-run` prints the commands without running them.

> [!TIP]
> A sharded configuration that works is worth turning into a recipe. `recipes launch` then runs it with state, supervision and Coder sync attached.

## DGX Spark prerequisites

On a DGX Spark, `llm-helper spark doctor` checks the packages vLLM's JIT builds need, and `llm-helper spark install-prereqs` installs them with `apt-get`.

## Next steps

- [Recipes](./recipes.md): the tested multi-machine configurations.
- [CLI reference](./reference/cli.md): every flag for `node`, `pool` and `shard`.
