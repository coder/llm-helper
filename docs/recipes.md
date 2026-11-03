---
title: Recipes
description: Launch configurations that were run and measured on named hardware, and how local overrides layer on top.
---

## Goal

Running a model should start from a curated recipe we have actually run, not from a
HuggingFace search plus autodetection. Search stays, clearly labelled as the escape
hatch for models we have no recipe for.

## Source authority

Two upstreams, each authoritative for a different thing. Mixing this up is how we
shipped a flag the image rejects (`--attention-backend B12X_MLA_SPARSE`).

| source                   | authoritative for                                                                           | licence    |
| ------------------------ | ------------------------------------------------------------------------------------------- | ---------- |
| `vllm-project/recipes`   | model facts: repo ids, variants, `min_vllm_version`, the `hardware:` verified map           | Apache-2.0 |
| `eugr/spark-vllm-docker` | image facts: backend names, loader, `mods`, `build_args`, anything describing its own image | MIT        |

eugr is the higher-signal source for DGX Spark / GB10 because its recipes are written
for that hardware and shipped with the image they target. vllm-project is broader but
hardware-blind for SM121.

eugr is MIT (Copyright 2026 Eugene Rakhmatulin); derived data must carry attribution.

## Three roles, kept separate

Conflating these is what made ingestion look hard.

1. **Launch path — our tested recipes only.** Digest-pinned, measured on our own
   hardware, `internal/recipe/tested.json`. This is the product.
2. **Search path — HF search, annotated.** "eugr has a recipe for this", "vllm-project
   claims gb200-verified". Signal without a claim that it runs here.
3. **Dev path — a diff tool, not an importer.** Parses both upstreams and reports where
   our tested set disagrees. Dev-only: it never ships in the binary and cannot break a
   launch.

The critical consequence: **we never execute eugr's `command:` template, only compare
against it.** Comparison tolerates a partial parse; execution does not. That removes
every integration risk below.

### Why not ingest eugr directly

- `command:` is a templated shell string, not an arg list: line continuations,
  `{placeholder}` substitution, `{{...}}` escaping literal JSON.
- `container`, `build_args` and `mods` reference eugr's build pipeline. We consume the
  published image by digest, so for us those fields are advisory.
- The repo moves fast. Anything mechanically ingested becomes a moving dependency.

### Why not drop the upstreams entirely

Without a systematic diff there is no way to learn we have fallen behind. That is not
hypothetical: we ran DeepSeek for a week with no speculative decoding because the
recipe was transcribed from a log instead of from eugr's YAML.

## Upstream schemas

eugr, `recipes/*.yaml`, `recipe_version: "1"` (uniform across all recipes, documented in
their `docs/AGENT_DEVELOPMENT.md`, and used by `run-recipe.py` for feature checks):
`model` (HF repo, the join key), `container`, `build_args[]`, `cluster_only`, `mods[]`,
`defaults{...}`, `env{}`, `command: |`.

vllm-project, `models/<org>/<Model>.yaml`: `model.model_id` (same join key),
`docker_image`, `min_vllm_version`, `base_args`, `meta.hardware`, `variants`,
`compatible_strategies`. Only entries listed under `hardware:` are verified; everything
else is a fail-open default, and we surface that distinction rather than inventing tiers.

A TensorFold launcher such as MiaAI-Lab's, `scripts/config.sh` (format `env-sh`): one `KEY="${KEY:-default}"` line per setting.
The exported keys are the server's environment, and the rest configure the launcher, which turns them into `tensorfold serve` flags.

## Provenance

Every tested recipe is authored in this repo but derived from an upstream, so each entry
carries a structured `source`:

```json
"source": {
  "repo": "eugr/spark-vllm-docker",
  "path": "recipes/deepseek-v4-flash-0731.yaml",
  "commit": "1761c49ad71ca9c8ca50772941ee4d06dcc40dc1",
  "license": "MIT"
}
```

Structured rather than a comment for three reasons: JSON has no comments; `recipediff`
needs it to know which upstream file and commit to compare an entry against; and MIT and
Apache-2.0 both want attribution. `recipes show` renders it as a "derived from" line, and
`TestTestedRecipesCiteAPinnedSource` rejects an entry without a repo, a full 40-character
commit and a licence.

The pinned commit is also the staleness signal: when it trails upstream HEAD,
`recipediff` has something to report.

### Declared deviations

Where an entry departs from its source, `source.deviations` records it as data:

```json
{
  "kind": "flag",
  "setting": "max_num_seqs",
  "upstream": "8",
  "ours": "32",
  "why": "measured ..."
}
```

`kind` is `flag`, `env` or `mod`, and it says where to look for the claim: the args, the
entry's `env`, or upstream's `mods` list. It exists because the two guards used to
contradict each other. The entry-side test built `"--" + setting` for every declaration,
so an env key became the flag `--VLLM-MOE-SKIP-PADDING` and could never be found, while
`recipediff` keyed env findings on the raw key -- which made two of the open differences
impossible to declare at all. With the kind recorded, each guard checks the place the
setting actually lives.

A declaration has to be falsifiable, so `recipediff` reads the values and not just the
key: the direction must agree (a drop cannot be declared as a change), `ours` must match
the entry's own value exactly, and `upstream` must match exactly unless upstream's value
is longer than 32 characters -- its `--speculative-config` and `--reasoning-config` JSON
blobs cannot usefully be restated, so prose is allowed there. A declaration that no longer
fits reads as `MISDECLARED` and still counts against the exit code; keying on the name
alone let a stale claim go on hiding a difference that had moved underneath it. An entry
whose source has no parser is reported as `NOT CHECKED` and counted too, because an entry
nobody can compare is an entry whose differences nobody has seen.

An empty `ours` means we dropped the setting. `setting` must name the thing the way
upstream does — the flag without its dashes (`max_model_len`), the env key, or the mod
path — because that is the key `recipediff` matches on, and
`TestDeviationsMatchTheArgsTheyDescribe` checks the claim against the entry itself. A
dropped setting leaves no trace to check, so the test also holds the name to the shape its
kind implies: a mod path has a slash, an env key is upper case, a flag is neither. Prose
in `notes` cannot do any of this.

Running the tool against both entries found differences that were never decisions, only
transcription. Four of them turned out to be no decision at all once the pinned image was
read: `--max-cudagraph-capture-size 64` is exactly what vLLM computes when the flag is
absent at our `max_num_seqs`, `--reasoning-config` is built from `--reasoning-parser`
with the same empty delimiters upstream passes, and both
`--default-chat-template-kwargs` restate what the `deepseek_v4` renderer already does.
Those are declared with the code that proves it.

Two per entry remain open and are deliberately left undeclared so the report keeps raising
them, because both change behaviour rather than restating it:
`VLLM_USE_MEGA_AOT_ARTIFACT` 1 to -1 (anything but `"1"` disables the mega AOT artifact)
and `VLLM_MOE_SKIP_PADDING` dropped (the image defaults it to on, upstream turns it off
for this model). Each needs a measurement or a reason, not a sentence invented after the
fact.

## Local overrides

A tested recipe is the **base layer**; the profile of the same name is the **override
layer**. `recipes launch` merges them last-wins per flag, which is the precedence vLLM
itself documents for a command line over a config file.

Editing is allowed and honoured. The earlier design refused edits to a recipe profile, but
the profile is not inert: the Coder sync reads `MAX_MODEL_LEN` and `DISPLAY_NAME` from it, while
the launcher built argv from the recipe alone. An edit therefore did half of something — it
moved what Coder advertised without moving what the server served. Merging removes that
asymmetry by construction.

What changes when overrides exist is the **claim**, not the permission:

- `recipes list`, `recipes show`, `list` and the launch output report the entry as the recipe
  plus local overrides, and `recipes show` stops presenting the verified numbers as if they
  described the running server.
- A future `recipe verify` must refuse to re-stamp measurements taken from an overridden
  run, because "we ran this" would then be false.

What is still refused is a **wrong command**, not an edit: native `start` on a container
recipe, which would run vLLM on the local machine while the cluster serves. `switch` to a
recipe is allowed, but only once that recipe has been launched at least once, because the
launch is what records its node topology. That precondition is checked ahead of
`WriteActiveEnv`: activation notifies Coder before restarting, so a late failure would
leave Coder advertising a model that is not serving.

Overriding any flag is allowed, including `--attention-backend` and `--load-format`. The
forbidden-flag check still fails closed on combinations known to abort on the recipe's
architecture, or on any node's own (see [Hardware detection](./hardware.md)). A
profile that merely restates a recipe value is not an override, so a freshly generated
profile reports none. Changing `MODEL` is refused: that is a different model, which the
recipe's flags and measurements do not describe.

An override reaches the server as written.
Only the recipe's own args are filled in, such as `{drafter_path}`, so a literal such as a chat template's `{{bos_token}}` is passed unchanged.

### Worked example: speculative decoding is a single-user override

`dspark` speculative decoding is the clearest case for the override layer rather than the
recipe. Measured on the two-Spark pair at 262K context, single stream:

| workload              | recipe     | plus dspark |
| --------------------- | ---------- | ----------- |
| repetitive/structured | 28.8 tok/s | 80 tok/s    |
| code                  | 28.8 tok/s | 44-56 tok/s |
| prose                 | 28.8 tok/s | 40 tok/s    |

It is not in the recipe because the cost lands on everyone else: KV cache falls from
681,894 to 420,192 tokens, concurrency from 2.60x to 1.60x, and load time from 150s to
195s. Speculation also spends spare compute between tokens, and on a busy server that
compute is already carrying other requests, so the single-stream numbers above are an
upper bound rather than what a shared box delivers. Time-to-first-token gets slightly
worse, which every waiting request pays for.

Enable it per-deployment when the box has one user:

```
llm-helper set deepseek-v4-flash-2spark EXTRA_ARGS='--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X","draft_load_config":{"load_format":"safetensors","safetensors_load_strategy":"lazy"}} --max-cudagraph-capture-size 48'
llm-helper recipes launch deepseek-v4-flash-2spark
```

`num_speculative_tokens` must be a multiple of the checkpoint's `dspark_block_size` (5 for
DeepSeek V4 Flash 0731); 3 truncates the block and 6 is rejected at boot. The capture size
is `max_num_seqs x (k + 1)`. Confirm acceptance from the server's own `SpecDecoding
metrics` log lines rather than from client-side tokens per second, because under
speculation one streamed chunk carries several tokens.

## Shipped recipes

The DeepSeek entries run on the same pinned vLLM image, sharded TP=2 across two DGX Sparks, at 262144 context.
The numbers are each entry's `verified` block in `internal/recipe/tested.json`.

| id                                | source                            | KV tokens | concurrency | load  |
| --------------------------------- | --------------------------------- | --------- | ----------- | ----- |
| `deepseek-v4-flash-2spark`        | `eugr/spark-vllm-docker@1761c49a` | 664,817   | 2.53x       | 161 s |
| `deepseek-v4-flash-vision-2spark` | `eugr/spark-vllm-docker@798528a2` | 670,498   | 2.56x       | 165 s |

Every difference from upstream is declared in `source.deviations` except two per entry,
which stay open on purpose (see [Declared deviations](#declared-deviations)):
`VLLM_USE_MEGA_AOT_ARTIFACT` (`1` upstream, `-1` here) and `VLLM_MOE_SKIP_PADDING` (`0`
upstream, unset here). `recipediff` reports exactly those four findings until each is
measured or explained.

Deliberate divergences to keep: port 8001 (matches `PORT`, fronted by the router on 8011) and an explicit `--max-model-len` rather than `auto`, so the served window and
`verified.context` stay pinned together.

`glm-5.3-flash-tf-2spark` runs GLM-5.3 Flash on TensorFold (`tensorfold-python`), sharded TP=2 across two DGX Sparks at 1048576 context.
It is derived from MiaAI-Lab's launcher, `scripts/config.sh` at `bf5f9452`, and pins the image by digest and both models, the weights and the DFlash2 drafter, by revision.
TensorFold reads only the Hugging Face cache mounted from each node, so download both revisions on every node first; `recipes launch` checks they are there and prints the `hf download` command for any that is not.
TensorFold checks no API key, so rank 0 listens on `127.0.0.1:8889`, and a proxy that checks one serves the network on 8888.
The drafter, `incoai/GLM-5.3-Flash-DFlash2`, is CC BY-NC-ND 4.0: non-commercial use only.

On an engine other than vLLM, a profile's `MAX_MODEL_LEN` is the only local override, passed as the engine's own context flag.
The other knobs are vLLM's flags.

## Invariants

- Images pinned by `@sha256:`, never a moving tag. eugr's `:latest` advanced under us
  mid-session, and a moved image is how a valid backend name becomes invalid.
- Tested means tested: strict list, no aspirational entries.
- Separate ids per topology rather than one entry with a topology switch.
- `--max-model-len`, or the engine's own context flag, must equal `verified.context`, enforced by
  `TestVerifiedContextMatchesMaxModelLen`.
- Extend upstream, do not fork it.

## Contributing a source

A recipe is data: llm-helper reads its upstream's settings into a launch, and never runs the upstream's scripts.
An entry in `tested.json` says:

- `provider`: the engine that runs it, from `llm-helper engines list`. It must have a container runtime. Empty is vLLM.
- `image`: pinned by `@sha256:` digest.
- `requires`: the hardware definition it needs, from `llm-helper hardware list`.
- `models`: each Hugging Face repo the engine reads from the cache, pinned to a full commit sha, with its role (`weights`, `drafter` or `adapter`). The args name a model's path as `{weights_path}`, `{drafter_path}` or `{adapter_path}`.
- `params`: settings of the engine's own, by the names `engines show` lists.
- `args` and `env`: what the server runs with.
- `source`: the upstream repo, path, commit and licence, the `format` its file is in, and every `deviations` entry where the recipe departs from it.
- `verified`: what was measured, on which hardware, on which date. Record only what was measured.

`source.format` is one of:

| format        | upstream file                                   | checked by `recipediff`                                       |
| ------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| `eugr-recipe` | eugr/spark-vllm-docker's `recipes/*.yaml`       | flags, env and mods                                           |
| `env-sh`      | a shell config of `KEY="${KEY:-default}"` lines | exported env, and each `config` deviation against the default |
| `manual`      | nothing machine-readable                        | not checked, and counted as such                              |

A deviation's `kind` is `flag`, `env`, `mod`, or `config` for a launcher key in an `env-sh` file, named as that file names it.
An upstream in a new layout needs a reader in `tools/recipediff` and a copy of the upstream file as its fixture; until then, cite it as `manual`.

A recipe measured on your own hardware can stay local: put it in a file under `~/.config/vllm-helper/recipes.d/`, in the same shape as `tested.json`.
`recipes list` marks it local, and it is held to the same checks as a shipped one.
A local file can only add recipes with new ids, so it cannot change what a shipped recipe launches.
A file that does not load is skipped and named by `recipes list` and `doctor`.
