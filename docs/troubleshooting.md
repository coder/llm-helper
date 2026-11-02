---
title: Troubleshooting
description: When the endpoint is down, or the service is crash-looping.
---

Run `llm-helper doctor` first.
It checks the build toolchain vLLM needs, inventories running servers, verifies your Coder and HuggingFace credentials against their APIs, and warns when config has drifted from reality.
With `--yes` it installs missing toolchain packages without asking.

## The endpoint isn't responding

`llm-helper status` prints the right command for your platform when the endpoint isn't responding, and says which of two very different things is happening: the service is still loading, or it is failing to start.
A large model can take many minutes to load, so the two look identical from the outside.

For crash loops, read the actual traceback:

```sh
journalctl -u vllm.service -n 100                          # Linux
tail -n 100 ~/Library/Logs/vllm-helper/com.coder.vllm.log  # macOS
```

`llm-helper logs` follows the same log — Ctrl+C to stop watching, the service keeps running.
`--no-follow` prints the last lines and exits, and `-n` sets how many.

## The first switch takes forever

A first `switch` or `new` downloads the weights, which can keep the boot log on screen for a long time.
Ctrl+C there only stops watching; the server keeps loading in the background, and `llm-helper status` tells you when it's up.
Use `--no-follow` to skip the wait entirely.

## vLLM refuses to start

On a machine whose memory is shared with the OS, llm-helper lowers `--gpu-memory-utilization` to what the machine can actually spare and logs the change.
If even that does not fit — only a sliver of memory was free — it says so and suggests closing some apps and switching again.

## It started outside llm-helper

If `status` or `doctor` reports a server that config doesn't account for — a different model serving on the configured port, or a unit llm-helper doesn't manage — run `llm-helper adopt` to save its live command line as a profile and take over future switches.
It does not restart or interrupt anything.

## Next steps

- [CLI reference](./reference/cli.md) — every flag of `doctor`, `status`, and `logs`.
