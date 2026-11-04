// SPDX-License-Identifier: MIT OR Apache-2.0
//! Maintenance tools for the Vulkan half of the verified ledger. Not a test, not part of the supported API.
//! Run: `cargo run -p fuel-vulkan-dispatch --features dev-tools --bin fuel-vulkan-dev-tools -- <tool>`.
//!
//! Tools (each needs a live Vulkan device): `seed-vulkan`, `seed-vulkan-accuracy`, `check-vulkan-accuracy`.
//! They used to be subcommands of `fuel-dev-tools` in `fuel-dispatch`; they moved with the Vulkan kernel
//! bindings (ROADMAP row 137 step 6.1) because `fuel-dispatch` cannot name the registrar they need.

use fuel_dispatch::dev_tools;
use fuel_vulkan_dispatch::register_vulkan_kernels;

fn main() {
    let tool = std::env::args().nth(1).unwrap_or_default();
    match tool.as_str() {
        "seed-vulkan" => dev_tools::seed_vulkan_ledger(register_vulkan_kernels),
        "seed-vulkan-accuracy" => dev_tools::seed_vulkan_matmul_accuracy(register_vulkan_kernels),
        "check-vulkan-accuracy" => dev_tools::check_vulkan_accuracy(register_vulkan_kernels),
        other => {
            eprintln!(
                "unknown tool {other:?}; expected seed-vulkan | seed-vulkan-accuracy | check-vulkan-accuracy"
            );
            std::process::exit(2);
        }
    }
}
