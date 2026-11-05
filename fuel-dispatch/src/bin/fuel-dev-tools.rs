// SPDX-License-Identifier: MIT OR Apache-2.0
//! Maintenance tools that rewrite committed files. Not a test, not part of the
//! supported API. Run: `cargo run -p fuel-dispatch --features dev-tools --bin
//! fuel-dev-tools -- <tool>`.
//!
//! Tools: `seed-cpu`, `gap-228-flip`, `gap-228-report`. The CUDA tools (`seed-cuda`, `seed-cuda-accuracy`,
//! `check-cuda-accuracy`, live device) moved with the CUDA kernel bindings: `cargo run -p fuel-cuda-dispatch
//! --features cuda,dev-tools --bin fuel-cuda-dev-tools -- <tool>`. The Vulkan
//! tools (`seed-vulkan`, `seed-vulkan-accuracy`, `check-vulkan-accuracy`) moved with the Vulkan kernel
//! bindings: `cargo run -p fuel-vulkan-dispatch --features dev-tools --bin fuel-vulkan-dev-tools -- <tool>`.

use fuel_dispatch::dev_tools;

fn main() {
    let tool = std::env::args().nth(1).unwrap_or_default();
    match tool.as_str() {
        "seed-cpu" => dev_tools::seed_cpu_ledger(),
        "gap-228-flip" => dev_tools::gap_228_flip(),
        "gap-228-report" => dev_tools::gap_228_sections_report(),
        other => {
            eprintln!("unknown tool {other:?}; expected seed-cpu | gap-228-flip | gap-228-report");
            std::process::exit(2);
        }
    }
}
