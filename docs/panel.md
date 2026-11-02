---
title: Web panel
description: The control panel that 'llm-helper serve' puts on the machine.
---

`llm-helper serve` serves a small panel for this machine: what is loaded, whether it is serving, and what else could be loaded instead.
It listens on localhost only unless told otherwise:

```sh
llm-helper serve                          # localhost only
llm-helper serve --listen 0.0.0.0:8080    # reachable from another machine
```

## Authentication

Every request needs the token printed at startup.
Open the URL it gives you and the token becomes a cookie; treat that URL like a password — anyone who has it can switch models and stop the server.

With `--listen`, the panel is reachable from the network over plain HTTP, and the token is the only thing between it and a stop button.
Put it behind a TLS proxy or a private network rather than exposing it directly.

The panel is a control plane only.
Inference traffic goes straight to vLLM on its own port, so stopping the panel never stops the model.

## The Machine page

The landing page shows what is loaded and what that costs: the active profile, endpoint health, how much of the machine's memory is in use against what it can give a model, and the server's log underneath.
**Restart** restarts the server onto the active profile; **Stop** stops it.

Switching and stopping need permission to control the service.
On Linux the unit is system-scope by default, so either run the panel with `sudo`, or allow those three commands for your user:

```txt
youruser ALL=(root) NOPASSWD: /usr/bin/systemctl restart vllm.service, /usr/bin/systemctl stop vllm.service, /usr/bin/systemctl start vllm.service
```

Without it the panel still reports everything, and says why the buttons are disabled rather than failing when you press them.

> [!NOTE]
> While a [recipe](./recipes.md) serves, the Machine page reports the cluster's health and the head container's log instead of the native unit's.
> Switch, Restart, Stop and pulling a model are disabled there with the reason; drive a recipe from the CLI, which picks the right runner from the active profile.

## The Add model page

**Add model** (`/models`) searches HuggingFace the way `llm-helper search` does.
Picking a result checks its weight size against this machine's budget and its model card's flags against your installed vLLM before anything is saved.

**Pull and switch** saves a profile and switches to it.
The weights themselves are downloaded by vLLM as it starts, so a big model spends a while in the panel's loading state before it serves.
Saved models can be removed from the same list they are switched from — the active one can't, and the weights stay in your HuggingFace cache either way.

When the installed vLLM cannot load GGUF, the page offers to install the plugin; see [Models and quantizations](./models.md).

## Docs

The **Docs** tab serves this documentation, from the binary itself — no internet needed.

## Next steps

- [Coder integration](./coder.md) — keep a Coder deployment's model picker truthful.
- [Troubleshooting](./troubleshooting.md) — when the endpoint is down.
