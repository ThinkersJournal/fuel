# fuel-dispatch

Dispatch infrastructure for fuel: KernelBindingTable, per-backend registration wrappers, CompiledNode + compile_node, ExecutionPlan + compile_plan + resolve_kernel, FusedKernelRegistry, PipelinedExecutor. The layer that takes a graph and runs it against registered backends. Extracted from fuel-memory 2026-05-31.

Part of the [Fuel](https://github.com/ThinkersJournal/fuel) workspace: a DAG-first, lazy-only ML framework for Rust. Licensed under MIT OR Apache-2.0.
