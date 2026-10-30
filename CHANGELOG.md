# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries for 0.3.1 and earlier were reconstructed from the commit history after
the fact.

## [Unreleased]

### Added

- Providers: a profile can run on an inference server other than vLLM. The
  first is MLX (`mlx_lm.server` from mlx-lm) on Apple Silicon. A profile
  records its server in `PROVIDER=`, and one without it is vLLM, so every
  existing profile serves as before. `PROVIDER` in `config.env` picks the
  server for new profiles only, and `setup --provider mlx` writes it with
  `MLX_LM_BIN`. vLLM-only steps (model-card flags, parser checks, KV sizing,
  GGUF quants, shards) are skipped or refused for an MLX profile. See
  `docs/providers.md`.
- llama.cpp provider: `llama-server` serves GGUF models on Metal, CUDA or the
  CPU. `setup --provider llamacpp` writes `LLAMA_SERVER_BIN`, and
  `new <repo> --quant Q4_K_M` picks one GGUF file out of a repo. Unlike MLX it
  checks `VLLM_API_KEY`, so it listens on every interface as vLLM does; with
  no key set it stays on `127.0.0.1`.

### Changed

- The tool is now called llm-helper. Releases ship the binary as both
  `llm-helper` and `vllm-helper`, archives are named `llm-helper_*`, and the
  help, version and docs use the new name. Nothing on disk changes: the config
  and state directories, `VLLM_HELPER_*` overrides, `VLLM_BIN`, `VLLM_API_KEY`
  and the `vllm.service` / `com.coder.vllm` units keep their names, and
  `vllm-helper upgrade` from 0.5.0 and earlier finds the new build as before.

## [0.5.0] - 2026-10-30

### Added

- Documentation, served by the binary itself: `vllm-helper serve` now has a
  **Docs** tab with a quickstart, the profile and memory knobs, the Coder
  integration, and a CLI reference generated from the command tree. The same
  Markdown lives in `docs/` and stays readable on GitHub.
- A public docs website: `cmd/vllm-helper-docs` serves the same pages without
  the panel, with each page's Markdown source, `/llms.txt`, `/llms-full.txt`
  and a sitemap, and `Dockerfile.docs` deploys it on Railway. It is a separate
  binary, so `vllm-helper` itself gains no unauthenticated routes.
- Recipes: launch configurations that have been run and measured, rather than
  inferred. An entry pins the model, the image by digest, the exact argv and
  the numbers it produced on named hardware, and `recipes list` / `recipes show`
  report where every setting came from. `recipes launch` starts the containers
  over SSH on the nodes you name — a recipe may span machines — writes a profile
  of the same name, and points Coder at what is now serving, so the picker and
  the server cannot disagree, then watches the load like `switch` does.
  `recipes stop` clears them again.
- `status`, `logs`, `restart` and `stop` follow whatever is actually serving.
  They used to assume the init-system service and would report systemd as
  healthy while a container did the work, or print "Stopped vllm.service" about
  a two-node cluster.
- `recipes install-service` installs a unit that keeps the active recipe
  serving, plus a timer that relaunches it when the endpoint stops answering.
  It stands the native unit down, which cannot serve a container, and holds
  off while a launch is in flight: a model takes minutes to load, and a
  watchdog that reads that silence as an outage restarts the previous profile
  over the top of the new one. A deliberate `stop` or `systemctl stop` is left
  stopped. The web panel shows a recipe's health and log, but its buttons do
  not drive recipes yet; use the CLI.
- Switching between a recipe and a native profile stops the one being left,
  so neither keeps the GPUs and port the other needs.
- Declared deviations. Where an entry differs from the upstream recipe it came
  from, it says so as data — the setting, upstream's value, ours, and why.
  `tools/recipediff` expands the upstream recipe and reports any difference
  that has not been declared, comparing content rather than commit ids so a
  pin that merely trails upstream is not reported as drift.
- More than one machine. `node add` records a machine once, by the URL it
  serves on and the SSH target that drives it; `pool`, `shard` and
  `recipes launch` refer to it by name. `pool serve` puts a router in front of
  replicas serving the same model and keeps a conversation on the node that
  already has its prefix cached; `pool install-service` runs it as a unit.
  `shard start` splits one model across machines with Ray or vLLM's
  multiprocessing executor, for trying a split before there is a recipe for
  it, and `shard install-ray` prepares the nodes. `spark doctor` and
  `spark install-prereqs` cover the build packages a DGX Spark needs.

### Changed

- Coder entries are matched by model rather than by profile name. A renamed
  profile, or a second machine serving the same model, now adopts the existing
  entry instead of adding a second row the picker would offer twice.
- `doctor` warns when the Coder token is within two days of expiring. Coder
  caps token lifetime and an expired token simply stops the picker being
  updated, with no other signal.
- `CODER_MAX_OUTPUT_TOKENS` raises Coder's server-side reply cap for the models
  this machine serves. Unset sends nothing, so whatever is configured in Coder
  stays.

## [0.4.0] - 2026-09-07

### Added

- A web panel, `vllm-helper serve`: what's loaded and whether it's serving,
  memory in use against what this machine can give a model, the saved profiles
  with a switch button, and the server log. It can also search HuggingFace and
  add or remove models, measuring a result against this machine before saving.
  Control plane only — inference goes straight to vLLM. Localhost and a token
  unless `--listen` says otherwise.
- Choosing which quantization to serve, for GGUF repos that ship several:
  `new --quant Q8_0`, a prompt in `search`, a dropdown in the panel. Saved as
  `MODEL=repo:QUANT` with the `--tokenizer` such a model needs.
- `install-gguf` installs the plugin vLLM now needs to read GGUF at all, into
  the interpreter that owns the configured vLLM. Applies from the next switch.
- `upgrade` replaces the binary from a published release, checked against the
  release's checksum. `--check` reports without installing.
- A dev channel: `upgrade --channel dev` follows a build of `main` from every
  commit that passes CI; `--channel stable` returns to tagged releases.
- macOS (Apple Silicon): launchd, memory from `sysctl`, `doctor` checks the
  cmake/ninja/clang toolchain with `brew` hints.
- `stop`, `restart`, `logs`, `inspect`, `edit`, and `set KEY=VALUE`.
- `switch`, `new`, and `restart` watch the boot, report when the model is
  actually serving, and fail fast on a crash loop. `--wait`, `--no-follow`,
  `--timeout`.
- `--json` for `status`, `list`, `search`, and `doctor`.
- `coder-sync prune` clears Coder entries for models this machine can't serve.
  Reports by default, deletes with `--yes`.
- `CODER_ORGANIZATION`, for Coder deployments with more than one.
- `version` prints the commit, which is what tells two dev builds apart.
- `doctor` verifies Coder and HuggingFace credentials against their APIs.

### Changed

- **Long flags need two dashes.** The old parser also accepted `-json`; the new
  one reads that as a bundle of short flags. `logs -n` still works.
- **An argument too many is an error** instead of being dropped, so
  `vllm-helper status runing` fails rather than reporting on something else.
- The KV cache is sized for the model being served instead of taking whatever
  vLLM finds left over. Every profile carried `GPU_MEMORY_UTILIZATION=0.85`, and
  that fraction is what decided the cache: a 1.1 GB model took 101.77 GiB of it.
  New profiles get `KV_CACHE_MEMORY_BYTES`, computed from the model's attention
  shape and `KV_CONTEXT_TARGET` full contexts (four by default), and
  `--gpu-memory-utilization` becomes a ceiling rather than the cache size.
  Existing profiles are untouched. A profile written here names a flag older
  vLLM builds don't have.
- The model budget is measured from the machine rather than fixed at 70% of
  unified memory or 80% of VRAM, both of which were invented — on a 128 GB
  machine the first held back 37 GiB for an OS that measures under 6 GiB. It is
  re-measured every run, so `MODEL_BUDGET_GB` is now an override and `setup`
  writes it commented out. **If you ran `setup` on an earlier version, that
  number is still in your `config.env` and now pins the budget; comment it out.**
  `doctor` says when an override is in force.
- `inspect`, `search` and the panel list every quant in a repo and lead with the
  biggest that fits, rather than reporting only the smallest — a 32B repo was
  being shown as 9.0 GB, which was its 2-bit quant. They also report how much
  context the model can hold on this machine.
- Quants vLLM can't load are marked, not offered: it addresses GGUF by name and
  only for a single file at the repo root, so split-file and subfolder copies
  are listed as unselectable, and a machine without the plugin is told before
  the pull rather than after the restart.
- GGUF builds of MoE models are marked unsupported. The plugin maps fused expert
  weights for OLMoE and MiniMax-M2 only; every other MoE fails at load, after
  the restart has already taken down what was serving.
- Coder sync is now a plugin behind a small interface. Nothing changes for
  users — same verbs, same config, same state file; it keeps the churn of an
  undocumented experimental API in one package.
- Commands are built on [cobra](https://github.com/spf13/cobra): grouped
  `--help`, per-command flags, cobra's wording for unknown ones. Exit codes
  unchanged. `--help` now works on `set`, `edit`, `remove`, and `coder-sync`.
- The boot log banner says that Ctrl+C stops watching and leaves the server
  loading. People were interrupting a first-run download believing they had
  killed it.
- Non-interactive sessions never prompt, tail, or block on `sudo`, which makes
  the CLI usable from scripts and a UI backend.
- `EXTRA_ARGS` is split shell-style, so a quoted value with spaces survives.
- `coder-sync register-all` only backfills what's missing.
- `start` exits cleanly when no model is active, instead of sending
  `Restart=on-failure` into a crash loop with nothing to serve.

### Fixed

- gpt-oss models work. Their cards carry no `vllm serve` snippet, so profiles
  were written with no flags, and served that way the model answers chat but
  returns an empty message for anything involving tools. `new` now adds the
  parsers and fetches the vocab files, where the installed vLLM confirms them.
- Model-card flags are checked against the vLLM installed here before reaching a
  profile. Anything this build won't accept used to be copied verbatim and
  surface as a crash loop on the next switch. Nothing is substituted
  automatically: parsers one character apart are different parsers.
- More of a card's flags are adopted — only five were recognised, so settings
  like `--max-num-seqs` were dropped without a word.
- `--gpu-memory-utilization` is capped where the GPU shares memory with the host
  (GB10, Jetson, Thor). The old guard asked whether an NVIDIA GPU was present,
  which is right for discrete VRAM and wrong for unified, where overshooting can
  wedge the host below the CUDA runtime rather than OOM cleanly. Linux hosts
  with no GPU are clamped to what's free, and no longer charged for a CUDA
  context they never create — a 7 GB CPU-only box was told nothing would run.
- Fit checks were 7.4% pessimistic: HuggingFace reports decimal GB and memory
  was read as GiB, so models that fit were called too big.
- Weights are measured wherever they live. The listing only looked at the repo
  root, so a model keeping weights one level down measured as 0 and read as
  "unknown", and a sharded repo past one page was never counted in full.
- Coder sync follows the model-config collection to
  `/api/v2/organizations/{org}/chats/models`. Against Coder 2.37 every sync
  failed with a 404 that read like a bad URL or token — which is what `doctor`
  said. Both generations stay supported.
- `status` tells a service that is failing to start from one still loading.
  Reporting a crash-looping unit as "still loading" sent people off to wait for
  a server that was never coming up; it now shows the last log line.
- The capability probe had stopped working wherever a vLLM plugin was installed:
  plugin registration on stdout meant the probe's JSON no longer parsed, so flag
  validation silently stopped and an installed GGUF plugin read as missing.
- `inspect` and the panel's model detail ran `vllm --version` before consulting
  the cache, which imports vLLM: 10.1s per lookup on a DGX Spark, now 0.25s.
- Installing the GGUF plugin from the panel no longer holds a request open for
  the whole source build. It runs on the server and the page polls it, so a
  reload rejoins an install in progress.
- `new` clamps `MAX_MODEL_LEN` to the model's `max_position_embeddings` even
  when the card has no snippet — the clamp sat behind an early return taken on
  exactly those cards, so `facebook/opt-125m` got 131072 and vLLM refused.
- `set` and `edit` refuse a `MAX_MODEL_LEN` that isn't a positive number, and
  `set` validates the whole batch before writing anything.
- A bad `MAX_MODEL_LEN` is reported before it reaches Coder, instead of becoming
  a context limit of 0 and an HTTP 400 that names nothing.
- Bare JSON in `EXTRA_ARGS` was mangled by shell-style splitting, turning
  `{"thinking":true}` into `{thinking:true}`, which vLLM rejects.
- Profile values round-trip: unusual leading or trailing whitespace was written
  unquoted and read back empty, and a NUL byte came back as a backslash.
- `adopt` saved a lossy copy of the server it took over — seven flags were
  recognised and the rest dropped, and `--gpu-memory-utilization` was hardcoded
  to `0.85`, so a server tuned to `0.92` came back with less KV cache. It could
  also carry credential-bearing flags into a profile and into `doctor --json`.
- `upgrade --check` no longer claims an update when you're already on that
  channel, and no longer writes `UPDATE_CHANNEL` while reporting.
- `--version` prints the commit, matching `version`.
- Redirecting to `/dev/null` no longer looks like a terminal, so
  `vllm-helper logs > /dev/null` prints and exits instead of following forever.
- `switch` could wait forever: only a `failed` unit ended the wait, so a service
  that went inactive left it polling a health endpoint that never answered.
- `doctor --yes` hung on a `sudo` prompt with no terminal to answer it.
- `status` reports the host's service manager rather than always saying systemd.
- `--` is honored as an end-of-flags terminator.

## [0.3.1] - 2026-08-03

### Fixed

- Chat templates that raise when a system message appears anywhere but first.
  Long-running conversations insert one mid-thread, which trips it reliably, and
  no vLLM flag controls the behavior. `new` now fetches the model's
  `chat_template.jinja`, patches it to drop a misplaced system message instead
  of raising, and points the profile at the patched copy.

## [0.3.0] - 2026-07-30

### Added

- Thinking mode is detected from the model card: the `deepseek_v3` and
  `poolside_v1` reasoning parsers get
  `--default-chat-template-kwargs {"thinking":true}` automatically, which vLLM
  silently no-ops without.
- `doctor` checks for the build packages vLLM's JIT kernel compilation needs.

### Changed

- **Breaking:** the Go binary is the only implementation. The bash version,
  its installer, the service template, and the curated model profiles are gone.
  Users define their own profiles, and `setup` and `adopt` handle installation.

### Fixed

- Env-file quoting for JSON-valued flags, broken in 0.2.0. Values like
  `--default-chat-template-kwargs {"thinking":true}` now round-trip correctly.

## [0.2.0] - 2026-07-30

### Added

- A Go CLI with full parity to the bash implementation. Both read and write the
  same XDG config, profile, and state files, so they were interchangeable during
  the transition.
- `adopt` detects an already-running vLLM server, saves its live command line as
  a profile, and takes over managing it without a restart. It scans host
  processes with container awareness, enumerates Docker and Podman containers,
  and probes localhost endpoints.
- `doctor` reports a hardware summary, an inventory of running servers, and
  warnings when the config has drifted from what's actually running.
- `setup`, a first-run wizard for binary installs: a hardware-aware model budget
  that distinguishes unified memory from discrete VRAM, vllm detection, secrets
  generation, and systemd unit installation in system or user scope.
- Release pipeline via GoReleaser and GitHub Actions, building linux/amd64,
  linux/arm64, and darwin/arm64 on every tag, plus CI covering gofmt, vet,
  cross-compilation, tests, and shellcheck.

## [0.1.0] - 2026-07-30

### Added

- Initial public release. A CLI for running a single-model,
  OpenAI-compatible vLLM server and switching models through saved profiles,
  with HuggingFace search, flag detection from model cards, and optional Coder
  Agents sync.
- An installable layout: config, secrets, state, and data split across XDG
  directories, with nothing written into the checkout, no hardcoded paths or
  users, and an idempotent `install.sh` supporting system- and user-scope
  systemd units.

[Unreleased]: https://github.com/coder/vllm-helper/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/coder/vllm-helper/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/coder/vllm-helper/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/coder/vllm-helper/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/coder/vllm-helper/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/coder/vllm-helper/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/coder/vllm-helper/releases/tag/v0.1.0
