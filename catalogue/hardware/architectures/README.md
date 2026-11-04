# Architectures

Each file here is one GPU architecture, named after it, such as `sm121.json`.
`compute_cap` is the CUDA compute capability `nvidia-smi` reports for it, so a GPU no machine names still gets its architecture's rules.
`forbidden_args` lists flags and values known to abort vLLM at startup on it, each with a `why`.
llm-helper refuses them wherever it builds a vLLM command line.

Add a file for a missing architecture, or a forbidden flag to an existing one.
A flag belongs here and not on a machine, because a kernel that aborts on sm121 aborts on every machine built on sm121.

The fields are described in [docs/hardware.md](../../../docs/hardware.md).
