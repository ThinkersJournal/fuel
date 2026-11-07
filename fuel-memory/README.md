# fuel-memory

Unified storage abstraction for fuel: byte-shaped Storage owning bytes + dtype + backend memory region. Closed enum over CPU/CUDA/Vulkan/Metal variants. Backends provide kernels against these types; backends do not own their own storage type.

Part of the [Fuel](https://github.com/ciresnave/fuel) workspace: a DAG-first, lazy-only ML framework for Rust. Licensed under MIT OR Apache-2.0.
