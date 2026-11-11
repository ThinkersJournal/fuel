# fuel-cuda-backend

CUDA GPU backend for the fuel ML framework: dtype-tagged storage, kernel dispatch (matmul/conv/softmax/rope/rms_norm/quantized), and lazy-graph executor integration. Built on the baracuda-* FFI crate family — parallel structure to fuel-vulkan-backend on vulkane.

Part of the [Fuel](https://github.com/ThinkersJournal/fuel) workspace: a DAG-first, lazy-only ML framework for Rust. Licensed under MIT OR Apache-2.0.
