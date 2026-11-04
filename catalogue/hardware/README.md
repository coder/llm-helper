# Hardware

An NVIDIA GPU with no file here still works.
llm-helper reads its memory from `nvidia-smi` and its architecture from the compute capability it reports.
The files here hold what that leaves out: a machine whose GPU shares the CPU's memory, a flag that aborts vLLM on an architecture, or an accelerator maker whose PCI vendor id is unknown.

| Folder                            | One file per                                    | Add one when                                                     |
| --------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| [`machines/`](machines)           | kind of machine, named after its id             | llm-helper gets a machine wrong without it                       |
| [`architectures/`](architectures) | GPU architecture, named after it, such as sm121 | an architecture is missing, or a flag aborts vLLM on one         |
| [`vendors/`](vendors)             | accelerator maker, named after its id           | a maker's PCI vendor id is missing                               |
| [`reports/`](reports)             | `llm-helper hardware report` output             | you add a machine, since every machine needs a report to test it |

Each file has a `schema_version` and is parsed strictly, and a file whose name is not its id is refused.
No two machines may match the same GPU, and the tests check every report against every machine.

The steps are in "Contributing a definition" in [docs/hardware.md](../../docs/hardware.md).
To try an entry without rebuilding, put its file in the same folder under `~/.config/vllm-helper/hardware.d/`.
