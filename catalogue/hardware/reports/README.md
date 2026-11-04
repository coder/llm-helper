# Reports

Each file here is what `llm-helper hardware report` printed for one machine: the raw readings and what llm-helper should conclude from them.
A report names no host: no hostname, serial number, GPU UUID or bus address.
Reports are test fixtures and are not built into the binary.
The tests in `internal/detect` run every report's readings through detection and compare the result with its `detected`, so a change that alters what a machine is detected as fails, naming that machine.

`provenance` starts with `measured: ` for a report captured on real hardware, followed by the machine and driver.
A report written by hand from a specification starts with `synthetic: ` instead, so nobody cites an invented machine as evidence.
A report that detects a machine is named after the machine's id.
One that detects none, and shows a GPU working without a machine file, is named after the setup, such as `discrete-h100-x8.json`.

Keep each report exactly as it was printed; prettier skips the reports in this folder.
