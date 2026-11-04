// SPDX-License-Identifier: MIT OR Apache-2.0
//! Maintenance tools for the CUDA half of the verified ledger. Not a test, not part of the supported API.
//! Run: `cargo run -p fuel-cuda-dispatch --features cuda,dev-tools --bin fuel-cuda-dev-tools -- <tool>`.
//!
//! Tools (each needs a live CUDA device): `seed-cuda`, `seed-cuda-accuracy`, `check-cuda-accuracy`.
//! They used to be subcommands of `fuel-dev-tools` in `fuel-dispatch`; they moved with the CUDA kernel
//! bindings (ROADMAP row 137 step 6.2) because `fuel-dispatch` cannot name the registrar they need.

use fuel_cuda_dispatch::register_baracuda_cuda_kernels;
use fuel_dispatch::dev_tools;

fn main() {
    let tool = std::env::args().nth(1).unwrap_or_default();
    match tool.as_str() {
        "seed-cuda" => dev_tools::seed_cuda_ledger(register_baracuda_cuda_kernels),
        "seed-cuda-accuracy" => dev_tools::seed_cuda_accuracy(register_baracuda_cuda_kernels),
        "check-cuda-accuracy" => dev_tools::check_cuda_accuracy(register_baracuda_cuda_kernels),
        other => {
            eprintln!(
                "unknown tool {other:?}; expected seed-cuda | seed-cuda-accuracy | check-cuda-accuracy"
            );
            std::process::exit(2);
        }
    }
}
