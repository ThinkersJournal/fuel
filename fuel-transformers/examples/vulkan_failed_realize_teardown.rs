// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-398 repro: does a process that has had failed Vulkan realizes exit cleanly?
//!
//! A one-op `QMatMul` with a type `qmatmul_vk` refuses (Q2_K, still unwired, no kernel) is placed on Vulkan
//! and fails at realize with "not wired on Vulkan", the same failure that preceded the access violation
//! (exit 0xc0000005) seen after the row 124 probe measured the five types before row 119 wired them. This
//! example does `FAILS` such failed realizes (default 20), then one successful Q4_K realize (so the device
//! still works after the errors), prints the counts and returns from `main` normally: the exit code is the
//! observation. Run it N times through `gpu-run` and histogram the exit codes (see GAP-398).
//!
//! `FAILS=<n>` sets the failure count; `CHAIN=1` puts a successful wired `QMatMul` in front of the refused one; with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` the loader
//! enables the validation layer and its messages go to stderr.
//!
//! Build: `cargo build -p fuel-transformers --release --features dev-tools,vulkan --example vulkan_failed_realize_teardown`.

#[cfg(feature = "vulkan")]
fn main() {
    use fuel_graph::QuantType;
    use fuel_ir::Shape;
    use fuel_quantized::{BlockQ2K, BlockQ4K, GgmlType};
    use fuel_tensor::Device;
    use fuel_tensor::lazy::Tensor;

    fn words<B: GgmlType>(w: &[f32]) -> Vec<u32> {
        let mut blocks = vec![B::zeros(); w.len() / B::BLCK_SIZE];
        B::from_float(w, &mut blocks);
        // SAFETY: a block is a `#[repr(C)]` plain-old-data struct; this reads its bytes.
        let mut bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                std::mem::size_of_val(blocks.as_slice()),
            )
        }
        .to_vec();
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }
    /// `chain`: put a SUCCESSFUL wired Q4_K `QMatMul` (its output feeds the refused one) in front of the
    /// refused op, so a dispatch has been recorded and not yet flushed when the error is returned.
    fn run(qt: QuantType, w_u32: Vec<u32>, dev: &Device, chain: bool) -> Result<f32, String> {
        let (m, k, n) = (1usize, 256usize, 256usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let x = Tensor::from_f32(a, Shape::from_dims(&[m, k]), &Device::cpu())
            .map_err(|e| e.to_string())?;
        let nw = w_u32.len();
        let w = x
            .const_u32_like(w_u32, Shape::from_dims(&[nw]))
            .map_err(|e| e.to_string())?;
        let x = if chain {
            let w0: Vec<f32> = (0..k * k)
                .map(|i| ((i as f32) * 0.017).sin() * 0.5)
                .collect();
            let w0 = x
                .const_u32_like(
                    words::<BlockQ4K>(&w0),
                    Shape::from_dims(&[k * k / 256 * 36]),
                )
                .map_err(|e| e.to_string())?;
            x.qmatmul(&w0, QuantType::Q4_K_M, k, k)
                .map_err(|e| e.to_string())?
        } else {
            x
        };
        let y = x.qmatmul(&w, qt, k, n).map_err(|e| e.to_string())?;
        let g = y.graph_tensor();
        let graph = g.graph().clone();
        fuel_tensor::pipelined_bridge::realize_one_as::<f32>(&graph, g.id(), dev)
            .map(|v| v[0])
            .map_err(|e| e.to_string())
    }

    let fails: usize = std::env::var("FAILS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let dev = fuel_tensor::vulkan_backend::new_device_from_env().expect("a Vulkan device");
    let w: Vec<f32> = (0..256 * 256)
        .map(|i| ((i as f32) * 0.021).sin() * 0.7)
        .collect();
    let chain = std::env::var_os("CHAIN").is_some();
    println!("chain (a successful Q4_K op before the refused one): {chain}");
    let (mut failed, mut ok_unexpected) = (0, 0);
    let mut last = String::new();
    for _ in 0..fails {
        match run(QuantType::Q2K, words::<BlockQ2K>(&w), &dev, chain) {
            Err(e) => {
                failed += 1;
                last = e;
            }
            Ok(_) => ok_unexpected += 1,
        }
    }
    println!(
        "failed realizes: {failed} of {fails} (unexpected successes {ok_unexpected}); last: {last}"
    );
    match run(QuantType::Q4_K_M, words::<BlockQ4K>(&w), &dev, false) {
        Ok(v) => println!("control Q4_K realize after the failures: ok, out[0]={v:.4}"),
        Err(e) => println!("control Q4_K realize after the failures: FAILED: {e}"),
    }
    println!("returning from main");
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("build with --features dev-tools,vulkan");
    std::process::exit(2);
}
