# Vendors

Each file here is one accelerator maker, named after its id, such as `nvidia.json`, with the PCI vendor ids its devices report.
llm-helper uses them to name a GPU on the PCI bus that no driver describes, and a machine's `match.vendor` names one.

Add one when a maker's PCI vendor id is missing.

The fields are described in [docs/hardware.md](../../../docs/hardware.md).
