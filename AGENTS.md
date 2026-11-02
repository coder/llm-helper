# vllm-helper Development Guidelines

Make the smallest correct change, follow existing patterns, and verify the result. Carry the requested task through implementation, verification, and necessary follow-up until it is complete or blocked by information or access you cannot obtain.

Prioritize correctness over agreement. State uncertainty instead of guessing, and push back on technically unsound requests with evidence.

vllm-helper is one Go binary that manages vLLM on one machine or several: model profiles, a systemd or launchd service, HuggingFace search, tested multi-machine recipes, a web panel, and a Coder model-picker sync. It drives real GPU machines that other people may be using, so read [Safety](#safety) before you run it.

## Autonomy and clarification

- Resolve routine ambiguity by inspecting the code, tests, docs, and history. Make reasonable, reversible assumptions that fit existing patterns; state consequential ones and keep working.
- Ask only when essential information cannot be recovered and would materially change the result, or when an action on real hardware, a shared branch, or a release requires authorization you do not have.
- When a change can only be verified on real hardware (recipes, shards, pools, service units, macOS), say what you verified and what you could not. Do not claim it works.

## Workflow

- Inspect the working tree before editing. For an existing PR, check out its branch first.
- Read `THISBOX.md` when present. It holds machine-local notes (endpoints, addresses, the models on this box), is gitignored, and is not imported automatically.
- Prefer targeted tests while iterating. Run every check CI runs for the affected area before handoff.
- Work on a branch and open a PR against `main`. Do not push to `main`, tag, cut a release, or force-push unless explicitly requested.
- Commit subjects use `area: what changed`, lowercase, where `area` is a package or topic (`cli`, `docs`, `recipes`, `web`, `readme`, `release`). The body explains why.
- Add an entry under `## [Unreleased]` in `CHANGELOG.md` in the same PR as any user-visible change. [RELEASING.md](RELEASING.md) has the rules; internal work (refactors, tests, CI) needs none.

## Essential commands

| Task                         | Command                                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| Test                         | `go test -race ./...`                                                                           |
| Test one                     | `go test ./internal/cli -run TestName`                                                          |
| Vet                          | `go vet ./...`                                                                                  |
| Format Go                    | `gofmt -w .`                                                                                    |
| Build a release target       | `GOOS=darwin GOARCH=arm64 go build ./...`                                                       |
| Regenerate the CLI reference | `go run ./scripts/gendocs`                                                                      |
| Test recipediff              | `cd tools/recipediff && go vet ./... && go test ./...`                                          |
| Format Markdown, YAML, JSON  | `bun install --frozen-lockfile && bun run fmt`                                                  |
| Check formatting             | `bun run fmt:ci`, or without bun `npx prettier@3.9.6 --check .` (the version in `package.json`) |
| Preview the docs site        | `go run ./cmd/vllm-helper-docs` (serves `localhost:8080`)                                       |

CI runs gofmt, vet, and the race tests on Linux and macOS, builds `linux/amd64`, `linux/arm64`, and `darwin/arm64`, tests `tools/recipediff`, and checks Prettier. The root `./...` skips `tools/recipediff` because it is a separate module.

## Repository guardrails

- **Control layer:** operations that change or report what is serving belong in `internal/control`, which the CLI and the web panel both call. Do not implement one in a single front end.
- **New commands:** set a `GroupID` from `internal/cli/root.go`, or `TestStartIsHiddenAndTheRestAreGrouped` fails. Then regenerate the CLI reference.
- **Help text:** command `Short`, `Long`, and flag usage strings are the published CLI reference. After changing any of them, run `go run ./scripts/gendocs`; `TestCLIReferenceIsUpToDate` fails on a stale or orphaned page.
- **Platform code:** platform-specific code lives in `_linux.go`, `_darwin.go`, and `_other.go` files. A function called from shared code needs a version in each, or the other release targets stop building.
- **On-disk compatibility:** config, profiles, and state outlive the binary. Changes to profile parsing, `EXTRA_ARGS` handling, or the service unit must still read files written by the previous release. Follow "Upgrade safety" in [RELEASING.md](RELEASING.md).
- **Recipes:** entries in `internal/recipe/tested.json` are measured on real hardware and pinned by image digest. Do not invent or generate one. Record each deviation from upstream as data in the entry, not as a comment.
- **Secrets:** keep API keys and tokens off command lines, out of dry-run output, and out of logs. Never commit `secrets.env`, `.env`, or a token.
- **Dependencies:** prefer the standard library. Ask before adding a module. `tools/recipediff` is a separate module so its dependencies stay out of the CLI binary.
- **Fixtures:** the paths in `.prettierignore` hold recorded HuggingFace responses, a fuzz corpus, and copies of upstream recipes. Keep them byte-identical to their sources; do not reformat or hand-edit them.
- **GitHub Actions:** keep top-level `permissions` minimal and grant more only to the job that needs it.

## Tests

- `foo_internal_test.go` is `package foo`; `foo_test.go` is `package foo_test`. Follow the file you are adding to.
- Tests must not touch the real machine. Point `VLLM_HELPER_CONFIG_DIR`, `VLLM_HELPER_STATE_DIR`, and `VLLM_HELPER_DATA_DIR` at `t.TempDir()`, use `httptest` for HTTP, and do not call real `systemctl`, `launchctl`, `ssh`, or `docker`.
- `TestMain` in `internal/cli` routes outbound HTTP to a dead port, so a test that reaches the network fails instead of flaking.
- `runCLI` in `internal/cli/run_test.go` swaps `os.Stdout` and `os.Stderr` for the whole process. Tests that use it must not call `t.Parallel()`.
- Fix a bug with a regression test that fails without the fix.

## Docs

`docs/docs.go` embeds `manifest.json`, `*.md`, and `reference/`, so every Markdown file in `docs/` ships in the binary and must be routed. Do not add notes, drafts, or a nested `AGENTS.md` there; docs guidance lives in this section.

- Every page needs a route in `docs/manifest.json`, or `TestEveryMarkdownPageIsReachable` fails.
- Hand-written pages carry `title` and `description` frontmatter identical to their manifest entry. GitHub shows the frontmatter; the panel and the public site show the manifest.
- Write one sentence per source line, and do not wrap to a fixed width. `docs/recipes.md` predates this; do not reflow it in an unrelated change.
- Link between pages with relative `.md` paths such as `./install.md`, so links resolve on GitHub, in the panel, and on the public site. `TestRelativeLinksResolve` checks them.
- Do not edit `docs/reference/cli/`. It is generated from the command tree; change the help text and run `go run ./scripts/gendocs`.
- Keep `README.md` a short overview that links to the docs. Put detail in `docs/`.
- Everything in `docs/` is published on the public docs site. Do not include internal hostnames, addresses, or notes.

## Code and writing style

- Name code for what it does, not how it is implemented or its history. Wrap errors with `%w` and the operation that failed.
- Give every package a package comment and every exported symbol a doc comment.
- Comments explain behavior, constraints, or rationale. Many existing comments record the bug or measurement that shaped the code; preserve them and keep them accurate when the code under them changes.
- Avoid unrelated edits. Ensure files end with a newline.

## Safety

- Do not run commands that change a machine to test a change: `setup`, `new`, `switch`, `restart`, `stop`, `adopt`, `upgrade`, `recipes launch`, `recipes stop`, `shard start`, `shard stop`, any `install-service`, `coder-sync`, or `doctor --yes`. They restart services, load models onto shared GPUs, install packages, or write to a live Coder deployment.
- To exercise the CLI, use scratch `VLLM_HELPER_*_DIR` directories and a closed port, as the tests do. Ask before touching real hardware or a real Coder deployment.
