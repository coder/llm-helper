<!-- markdownlint-disable MD041 -->
<div align="center">

<h1>llm-helper</h1>

Serve large models with vLLM, mlx-lm or llama.cpp on one GPU machine or several, and run several models at once.

[Quickstart](#quickstart) | [Docs][docs] | [CLI reference][docs-cli-reference] | [Changelog](CHANGELOG.md)

</div>

llm-helper manages an OpenAI-compatible inference server on one machine or across several: [vLLM](https://github.com/vllm-project/vllm) by default, [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon, or [llama.cpp](https://github.com/ggml-org/llama.cpp) for GGUF models. Each model's working settings are saved as a profile, and every profile you start runs as its own server on its own port, whether that model runs on this machine alone or spans a group of them. It was built for unified-memory machines like the NVIDIA DGX Spark, where one machine holds one big model and a bigger model needs two or more working together. It works on any Linux machine with an NVIDIA GPU, and on Apple Silicon Macs on the GPU through [vllm-metal](https://github.com/vllm-project/vllm-metal), mlx-lm or llama.cpp.

llm-helper was called vllm-helper until it served more than vLLM. The old command name, config directory and service names all keep working; see [Providers][docs-providers] for what changes per server.

- Save a model's serving settings as a profile, and run as many profiles at once as the memory holds, each with one command
- Search HuggingFace with the picked model's weight size checked against the machine's memory, and vLLM flags read from its model card
- Adopt a vLLM server that is already running, without restarting it
- Get machine details, like how a GPU's memory is shared, from a hardware catalogue, and record a new one with `hardware report`
- Register other machines once, then launch tested recipes across them: configurations run and measured on named hardware, pinned by image digest
- Pool replicas of one model behind a session-sticky router, or shard a model too big for one machine across several
- Control a machine from a web panel that also serves these docs
- Keep a [Coder](https://coder.com) deployment's model picker in step with the models running

> [!NOTE]
> llm-helper is experimental. It is a useful way to launch local LLMs, but use it on dedicated hardware that runs only your models, on a safe network.
>
> A bug in llm-helper can affect what it controls: the service units it installs, the server restarts when you start and stop models, the web panel, and the model list that it syncs to Coder. A bug cannot break a request in flight, because llm-helper is a control plane and inference traffic goes straight to the inference server on its own port. The exceptions are a pool and the optional router that offers every running model through one port, because each sits in the request path.

## Quickstart

> See [Install and upgrade][docs-install] for what setup writes and how to stay current, and [Running on macOS][docs-macos] for Apple Silicon.

llm-helper manages an existing vLLM, mlx-lm or llama.cpp install; it does not install any of them. Download the [latest release](https://github.com/coder/llm-helper/releases) for Linux amd64/arm64 or macOS arm64, then install it and run setup:

```shell
tar xzf llm-helper_*.tar.gz
sudo install llm-helper /usr/local/bin/
sudo ln -sf llm-helper /usr/local/bin/vllm-helper   # the old name, for scripts that still call it
llm-helper setup                                    # or: --provider mlx, --provider llamacpp
```

Setup detects the hardware and the server, generates an API key where the server takes one, and writes config. Each model you start runs as its own service (systemd on Linux, launchd on macOS). It never overwrites existing config or restarts a running server. If `vllm` isn't on your `PATH`, set `VLLM_BIN` in `~/.config/vllm-helper/config.env`.

Pull a model and serve it, or take over a server that is already running:

```shell
llm-helper new Qwen/Qwen3-32B   # save a profile and start serving it
llm-helper adopt                # or: manage the vLLM server already running here
llm-helper status               # running models, their ports, endpoint health
```

Point any OpenAI-compatible client at `http://<machine>:8001/v1` with the `VLLM_API_KEY` from `~/.config/vllm-helper/secrets.env`. A second model started with `llm-helper up <profile>` takes the next free port; [Several models at once][docs-multi-instance] covers ports, memory and the router. A first start downloads the weights, so it takes as long as the download does.

To serve across machines, register every machine a recipe will use on the one that will serve the API, that one included, then launch. llm-helper drives the other nodes over SSH and runs the recipe's container on each with Docker:

```shell
llm-helper node add spark1 --host http://spark1:8001 --ssh coder@spark1   # this machine
llm-helper node add spark2 --host http://spark2:8001 --ssh coder@spark2
llm-helper recipes list                              # what is tested, and on what hardware
llm-helper recipes launch <id> --nodes spark1,spark2
```

[More than one machine][docs-multi-node] covers recipes, pools and shards.

## How it works

| Piece         | What it is                                                                    | Where it lives                                                                            |
| ------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Config        | Settings for every profile: first port, `VLLM_BIN`, service scope, Coder sync | `~/.config/vllm-helper/config.env`                                                        |
| Secrets       | The endpoint's API key, and tokens for Coder and HuggingFace                  | `~/.config/vllm-helper/secrets.env`                                                       |
| Profile       | One model's serving settings, as a small env file                             | `~/.config/vllm-helper/models/<name>.env`                                                 |
| Instance      | A running profile, served by its own service unit or by recipe containers     | `llm-helper@<profile>.service` / `com.coder.llm-helper.<profile>`, `vllm-cluster.service` |
| Node registry | Other machines, by name, for recipes, pools and shards                        | `~/.config/vllm-helper/nodes.json`                                                        |
| Endpoint      | The engine's OpenAI-compatible API, one per instance                          | ports from `8001` up by default                                                           |

`up`, `down`, `restart`, `status` and `logs` act on whichever runner an instance belongs to, so they work the same for a native model and for a multi-machine recipe.

Everyday commands:

```shell
llm-helper list                  # saved profiles
llm-helper up <profile>          # serve another model beside the running ones
llm-helper down <profile>        # stop one
llm-helper search <query>        # find a model that fits this machine
llm-helper set <profile> K=V     # change a profile setting
llm-helper logs <profile>        # follow a model's server log
llm-helper doctor                # hardware, running servers, integrations, drift
llm-helper serve                 # web panel, with these docs under Docs
llm-helper recipes list          # tested configurations, and what verified them
```

`status`, `list`, `search`, `inspect`, `doctor` and `recipes list` take `--json`, and `up`, `new` and `restart` take `--wait`, so scripts and agents can drive llm-helper without parsing prose.

## Documentation

The docs are compiled into the binary: run `llm-helper serve` and open the **Docs** tab. The same pages are on the [docs site][docs], and their Markdown source is in [`docs/`](docs/README.md):

- [**Quickstart**][docs-quickstart]: From nothing to serving a model
- [**Profiles**][docs-profiles]: What a profile holds, and how to edit one
- [**Several models at once**][docs-multi-instance]: Ports, memory, and one port for every model
- [**Providers**][docs-providers]: vLLM, mlx-lm, llama.cpp, and what each one changes
- [**Models and quantizations**][docs-models]: Finding a model that fits, and choosing a quant
- [**Memory budget**][docs-memory]: How llm-helper decides what this machine can hold
- [**Web panel**][docs-panel]: The browser control plane, and the permissions it needs
- [**Coder integration**][docs-coder]: Keep a Coder deployment's model picker truthful
- [**Recipes**][docs-recipes]: Tested launch configurations, and how local overrides layer on top
- [**More than one machine**][docs-multi-node]: Run recipes, pools and shards across machines
- [**Troubleshooting**][docs-troubleshooting]: When the endpoint is down or the service crash-loops
- [**CLI reference**][docs-cli-reference]: Every command and flag, generated from the CLI

## Support

[Open an issue](https://github.com/coder/llm-helper/issues/new) for bugs and feature requests. Include the output of `llm-helper doctor`, which covers the hardware, the configured engine, running servers and config drift.

## Contributing

`go test ./...` needs no fixtures or network. CI also runs `go test -race ./...` on Linux and macOS, `go vet`, and formatting checks:

```shell
gofmt -w .                   # Go
bun install && bun run fmt   # Markdown and YAML
go run ./scripts/gendocs     # regenerate docs/reference/cli/ after changing a command or flag
```

Notable changes go in [CHANGELOG.md](CHANGELOG.md). Maintainers cutting a release follow [RELEASING.md](RELEASING.md).

## License

[Apache-2.0](LICENSE)

<!--
  TEMPORARY DOMAIN: vllm-helper-docs.up.railway.app is the Railway-generated
  domain for the public docs site (Dockerfile.docs), in use until it has a
  permanent hostname. When that changes, update these links and the BASE_URL
  variable on the Railway docs service.
-->

[docs]: https://vllm-helper-docs.up.railway.app/docs
[docs-cli-reference]: https://vllm-helper-docs.up.railway.app/docs/reference/cli
[docs-install]: https://vllm-helper-docs.up.railway.app/docs/install
[docs-macos]: https://vllm-helper-docs.up.railway.app/docs/macos
[docs-multi-node]: https://vllm-helper-docs.up.railway.app/docs/multi-node
[docs-quickstart]: https://vllm-helper-docs.up.railway.app/docs/quickstart
[docs-profiles]: https://vllm-helper-docs.up.railway.app/docs/profiles
[docs-multi-instance]: https://vllm-helper-docs.up.railway.app/docs/multi-instance
[docs-providers]: https://vllm-helper-docs.up.railway.app/docs/providers
[docs-models]: https://vllm-helper-docs.up.railway.app/docs/models
[docs-memory]: https://vllm-helper-docs.up.railway.app/docs/memory
[docs-panel]: https://vllm-helper-docs.up.railway.app/docs/panel
[docs-coder]: https://vllm-helper-docs.up.railway.app/docs/coder
[docs-recipes]: https://vllm-helper-docs.up.railway.app/docs/recipes
[docs-troubleshooting]: https://vllm-helper-docs.up.railway.app/docs/troubleshooting
