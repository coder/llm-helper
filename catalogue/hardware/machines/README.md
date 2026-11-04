# Machines

Each file here is one kind of machine, named after its id, such as `dgx_spark_gb10.json`.
A machine matches a GPU by vendor and by a regular expression over the name the GPU reports, case-insensitively.
It says how the GPU's memory relates to the CPU's: `unified`, `discrete` or `coherent`.
Recipes record a machine's id as the hardware they were measured on, so an id never changes once shipped.

Add one when llm-helper gets a GPU wrong without it, for example a GPU that shares system memory but reports none of its own.
Keep `gpu_name` narrow enough to match only your GPU, because no two machines may match the same one.
Describe one chip family, not a catch-all: a GPU no machine names already gets its architecture from its compute capability and its memory from what it reports.
Give every claim a `why`, and add a report to [`reports/`](../reports) that detects the machine.

The fields are described in "Contributing a definition" in [docs/hardware.md](../../../docs/hardware.md).
