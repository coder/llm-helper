---
title: Web panel
description: The control panel that 'llm-helper serve' puts on the machine.
---

`llm-helper serve` serves a small panel for this machine: which models are running, whether each one is serving, and what else could run beside them.
It listens on localhost only unless told otherwise:

```sh
llm-helper serve                          # localhost only
llm-helper serve --listen 0.0.0.0:8080    # reachable from another machine
```

## Authentication

Every request needs the token printed at startup.
Open the URL it gives you and the token becomes a cookie; treat that URL like a password — anyone who has it can run and stop models on the machine.

With `--listen`, the panel is reachable from the network over plain HTTP, and the token is the only thing between it and a stop button.
Put it behind a TLS proxy or a private network rather than exposing it directly.

The panel is a control plane only.
Inference traffic goes straight to the inference server on its own port, so stopping the panel never stops the model.

## The Machine page

The landing page shows what is running and what that costs: how much of the machine's memory is in use against what it can give a model, and one row per [instance](./multi-instance.md) with its port, endpoint health and log.
**Restart** restarts an instance, **Stop** stops it, and **Make default** makes it the model Coder selects first.
Below them, every saved profile that isn't running has a **Run** button, which starts it beside the others if it fits in memory.

Running, restarting and stopping models need permission to control their services.
On Linux the units are system-scope by default, so either run the panel with `sudo`, or allow these commands for your user:

```txt
youruser ALL=(root) NOPASSWD: /usr/bin/systemctl restart llm-helper@*.service, /usr/bin/systemctl enable llm-helper@*.service, /usr/bin/systemctl disable --now llm-helper@*.service, !/usr/bin/systemctl * llm-helper@* *
```

The last entry matters: in sudoers a `*` also matches spaces, so without it the rule would allow extra units or paths after the instance.

Without it the panel still reports everything, and says why the buttons are disabled rather than failing when you press them.

> [!NOTE]
> While a [recipe](./recipes.md) serves an instance, its row reports the cluster's health and the head container's log instead of the native unit's.
> Restart and Stop are disabled on that row with the reason; drive a recipe from the CLI, which picks the right runner from the instance's profile.

## The Add model page

**Add model** (`/models`) searches HuggingFace the way `llm-helper search` does.
Picking a result checks its weight size against this machine's budget and its model card's flags against your installed vLLM before anything is saved.

**Pull and run** saves a profile and runs it beside whatever already runs, if it fits.
The weights themselves are downloaded by vLLM as it starts, so a big model spends a while in the panel's loading state before it serves.
Saved models are listed on the Machine page, where they can be run or removed — a running one can't be removed until it is stopped, and the weights stay in your HuggingFace cache either way.

When the model is a thinking model, the page shows a **Thinking** control set to the model's default; turning it off saves a profile that answers without the reasoning pass.
When the installed vLLM cannot load GGUF, the page offers to install the plugin; see [Models and quantizations](./models.md).

## Docs

The **Docs** tab serves this documentation, from the binary itself — no internet needed.

## Next steps

- [Coder integration](./coder.md) — keep a Coder deployment's model picker truthful.
- [Troubleshooting](./troubleshooting.md) — when the endpoint is down.
