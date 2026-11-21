// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 124 step 3: does a GGUF weight that stays quantized (Q8_0, Q4_K, Q6_K) reach the
//! Vulkan backend's `QMatMul` through a LOADED model, or does the planner fall back to the CPU?
//!
//! A program, not an `#[ignore]` test (the `#[ignore]` total is a ratchet that only goes down;
//! this needs a live GPU). It builds the synthetic one-layer llama of
//! `tests/gguf_resident_quantized.rs` in each format, runs the same forward on the CPU and on a
//! Vulkan device, and prints which backend the executor launched every `QMatMul` on
//! (`op_profile`), the resident bytes, and the logit difference against the CPU run.
//!
//! Run through the GPU mutex, alone:
//!
//! ```text
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -p fuel-transformers \
//!     --features dev-tools,vulkan --example gguf_resident_reach
//! ```
//!
//! `FUEL_VULKAN_DEVICE` (UUID or PCI address) picks the adapter, else the default discrete one.
//! The exit code is nonzero if a format's logits leave the tolerance or no `QMatMul` launch was
//! seen (a probe that saw nothing must not report success).

#[path = "../tests/common/gguf_synth.rs"]
mod synth;

use fuel_dispatch::compiled::op_profile;
use fuel_graph::QuantType;
use fuel_quantized::{BlockQ4_0, BlockQ4K, BlockQ6K, BlockQ8_0, GgmlType};
use synth::{files, load, projection_weights};

/// Same bound as `tests/gguf_resident_quantized.rs::LOGIT_TOL`: the CPU kernel quantizes
/// activations to 8 bits and the Vulkan route dequantizes the weight to f32, so they differ a
/// little; a wrong dequantization differs by far more.
const LOGIT_TOL: f32 = 0.04;

struct Outcome {
    qmatmul_vulkan: u64,
    qmatmul_cpu: u64,
    rel: f32,
}

fn reach<B: GgmlType>(
    tag: &str,
    code: u32,
    qtype: QuantType,
    dev: &fuel_tensor::Device,
) -> Outcome {
    let (qfile, _plain, qbytes) = files::<B>(code, 1.0);
    let model = load(&qfile, tag);
    let resident: usize = projection_weights(&model)
        .iter()
        .map(|w| w.resident_bytes())
        .sum();
    assert_eq!(
        resident, qbytes,
        "{qtype:?}: weights are not resident as blocks"
    );
    let toks = [1_u32, 5, 9];
    let vk = fuel_tensor::vulkan_backend::as_device(dev).expect("a Vulkan device");

    let cpu = model.forward(&toks, 0).expect("cpu forward").realize_f32();

    op_profile::set_enabled(true);
    let _ = op_profile::take();
    let on_vk = model
        .forward(&toks, 0)
        .expect("vulkan forward")
        .realize_f32_vulkan(&vk);
    let rows = op_profile::take();
    op_profile::set_enabled(false);

    // Control: the launches of ALL ops by backend. A realize on a Vulkan device that launched nothing
    // on Vulkan would make "QMatMul ran on the CPU" a statement about the placement of the whole graph.
    let all_vulkan: u64 = rows
        .iter()
        .filter(|r| r.1.contains("Vulkan"))
        .map(|r| r.2)
        .sum();
    let all_cpu: u64 = rows
        .iter()
        .filter(|r| r.1.contains("Cpu"))
        .map(|r| r.2)
        .sum();
    println!("  all ops: launches vulkan={all_vulkan} cpu={all_cpu}");
    if std::env::var_os("REACH_ALL_OPS").is_some() {
        for (op, backend, n, _) in &rows {
            println!("      {op} on {backend}: {n}");
        }
    }
    let mut qmatmul_vulkan = 0;
    let mut qmatmul_cpu = 0;
    for (op, backend, n, _) in &rows {
        if op.contains("QMatMul") {
            println!("    {op} on {backend}: {n} launches");
            if backend.contains("Vulkan") {
                qmatmul_vulkan += n;
            } else if backend.contains("Cpu") {
                qmatmul_cpu += n;
            }
        }
    }
    let (mut worst, mut scale) = (0.0_f32, 0.0_f32);
    for (a, b) in on_vk.iter().zip(&cpu) {
        assert!(
            a.is_finite() && b.is_finite(),
            "{qtype:?}: non-finite logit"
        );
        worst = worst.max((a - b).abs());
        scale = scale.max(b.abs());
    }
    assert_eq!(on_vk.len(), cpu.len());
    let rel = worst / scale.max(1e-3);
    println!(
        "  {qtype:?}: resident {resident} B; QMatMul launches vulkan={qmatmul_vulkan} cpu={qmatmul_cpu}; \
         worst |dlogit| {worst:e} vs max |logit| {scale:e} = {:.3}%",
        rel * 100.0
    );
    Outcome {
        qmatmul_vulkan,
        qmatmul_cpu,
        rel,
    }
}

/// Is a Vulkan QMatMul binding registered in the table the realize path uses (CPU is the control:
/// the CPU one must be present, or this lookup asks the wrong table)? Asked before and after the
/// realizes, because optional backends may register on first use.
fn report_bindings(when: &str) {
    use fuel_ir::DType::{F32, U32};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::probe::BackendId;
    let table = fuel_dispatch::dispatch::global_bindings();
    for be in [BackendId::Cpu, BackendId::Vulkan] {
        let n = table
            .lookup_alternatives(OpKind::QMatMul, &[F32, U32, F32], be)
            .len();
        println!("binding table ({when}): (QMatMul, [F32, U32, F32], {be:?}) alternatives = {n}");
    }
    // The capability registry the placement DP prices candidates with: does the Vulkan entry advertise
    // QMatMul (output dtype F32)? MatMul is the control. A backend with no caps, or caps missing the op,
    // loses its candidate before ranking.
    for caps in fuel_dispatch::dispatch::global_registry().backends() {
        println!(
            "  caps {:?} at {:?} ({when}): QMatMul/F32 = {}, MatMul/F32 = {}, {} (op,dtype) pairs",
            caps.backend_id,
            caps.device_location,
            caps.op_dtype_support.contains(&(OpKind::QMatMul, F32)),
            caps.op_dtype_support.contains(&(OpKind::MatMul, F32)),
            caps.op_dtype_support.len()
        );
    }
    // What each registered QMatMul / MatMul binding claims to cost for the same shape (a zero
    // estimate is the `unknown_cost` sentinel: "no claim").
    {
        use fuel_dispatch::OpParams;
        use fuel_ir::DeviceLocation;
        use fuel_ir::backend::{BackendCapabilities, SubstrateClass};
        // The two cost functions read shapes and params, not the capabilities; any caps will do.
        let caps = BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: Default::default(),
            required_alignment: 64,
            access_granularity_bits: 8,
            transfer_paths: Vec::new(),
            storage_substrate: SubstrateClass::HostBytes,
            compute_throughput_flops_per_ns: 1.0,
            mem_bandwidth_bytes_per_ns: 4.0,
        };
        let qp = OpParams::QMatMul {
            quant_type: QuantType::Q8_0,
            batch_count: 1,
            m: 64,
            n: 4096,
            k: 4096,
        };
        let q_shapes = [
            fuel_ir::Shape::from_dims(&[64, 4096]),
            fuel_ir::Shape::from_dims(&[4096 * 4096 / 32 * 34 / 4]),
            fuel_ir::Shape::from_dims(&[64, 4096]),
        ];
        for be in [BackendId::Cpu, BackendId::Vulkan] {
            for alt in table.lookup_alternatives(OpKind::QMatMul, &[F32, U32, F32], be) {
                let c = (alt.cost)(&q_shapes, &[F32, U32, F32], &qp, &caps);
                println!(
                    "  QMatMul {be:?} [{}] cost for m=64 n=k=4096 Q8_0: {c:?}",
                    alt.kernel_source
                );
                println!(
                    "  QMatMul {be:?} [{}] precision: {:?}; caps: {:?}",
                    alt.kernel_source, alt.precision, alt.caps
                );
            }
        }
        // Control: the Vulkan MatMul binding for the same GEMM.
        let mp = OpParams::Matmul {
            lhs_batch_dims: Vec::new(),
            rhs_batch_dims: Vec::new(),
            m: 64,
            n: 4096,
            k: 4096,
            m_compute: fuel_dispatch::kernel::MatmulM::All,
        };
        let m_shapes = [
            fuel_ir::Shape::from_dims(&[64, 4096]),
            fuel_ir::Shape::from_dims(&[4096, 4096]),
            fuel_ir::Shape::from_dims(&[64, 4096]),
        ];
        for alt in table.lookup_alternatives(OpKind::MatMul, &[F32, F32, F32], BackendId::Vulkan) {
            let c = (alt.cost)(&m_shapes, &[F32, F32, F32], &mp, &caps);
            println!(
                "  MatMul Vulkan [{}] cost for 64x4096x4096: {c:?}",
                alt.kernel_source
            );
            println!(
                "  MatMul Vulkan [{}] precision: {:?}; caps: {:?}",
                alt.kernel_source, alt.precision, alt.caps
            );
        }
    }
}

/// A bare graph, no model: activation `[m, k]` and one resident weight `[n, k]`, one `qmatmul`, realized
/// on the Vulkan device. Separates "QMatMul itself stays on the CPU" from "QMatMul inside the loaded
/// model's graph stays on the CPU".
fn isolated<B: GgmlType>(
    tag: &str,
    qtype: QuantType,
    m: usize,
    k: usize,
    n: usize,
    dev: &fuel_tensor::Device,
) {
    use fuel_ir::Shape;
    use fuel_tensor::lazy::Tensor;
    let vk = fuel_tensor::vulkan_backend::as_device(dev).expect("a Vulkan device");
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.021).sin() * 0.7)
        .collect();
    let mut blocks = vec![B::zeros(); n * k / B::BLCK_SIZE];
    B::from_float(&w_f32, &mut blocks);
    let bytes = fuel_quantized::blocks_as_bytes(&blocks).to_vec();
    let w_u32: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let x = Tensor::from_f32(a, Shape::from_dims(&[m, k]), &fuel_tensor::Device::cpu()).unwrap();
    let w = x
        .const_u32_like(w_u32, Shape::from_dims(&[bytes.len() / 4]))
        .unwrap();
    let y = x.qmatmul(&w, qtype, k, n).expect("qmatmul");
    op_profile::set_enabled(true);
    let _ = op_profile::take();
    let out = y.realize_f32_vulkan(&vk);
    let rows = op_profile::take();
    op_profile::set_enabled(false);
    let launches: Vec<String> = rows
        .iter()
        .map(|(op, be, n, _)| format!("{op}@{be} x{n}"))
        .collect();
    println!(
        "  isolated {tag} m={m} k={k} n={n}: out[0]={:.4}; launches: {}",
        out[0],
        launches.join(", ")
    );
}

fn main() {
    let dev = fuel_tensor::vulkan_backend::new_device_from_env().expect("a Vulkan device");
    let vk = fuel_tensor::vulkan_backend::as_device(&dev).expect("a Vulkan device");
    println!("adapter: [{}] {}", vk.gpu_id, vk.device_name);

    report_bindings("before any realize");

    println!("isolated QMatMul (bare graph, weight resident as blocks):");
    for (m, k, n) in [(1, 256, 256), (1, 4096, 4096), (64, 4096, 4096)] {
        isolated::<BlockQ4_0>("Q4_0", QuantType::Q4_0, m, k, n, &dev);
        isolated::<BlockQ8_0>("Q8_0", QuantType::Q8_0, m, k, n, &dev);
        isolated::<BlockQ4K>("Q4_K", QuantType::Q4_K_M, m, k, n, &dev);
    }

    let mut failed = false;
    let runs = [
        (
            "Q4_0 (control: Vulkan fuses it)",
            reach::<BlockQ4_0>("q40", 2, QuantType::Q4_0, &dev),
        ),
        ("Q8_0", reach::<BlockQ8_0>("q8", 8, QuantType::Q8_0, &dev)),
        (
            "Q4_K",
            reach::<BlockQ4K>("q4k", 12, QuantType::Q4_K_M, &dev),
        ),
        ("Q6_K", reach::<BlockQ6K>("q6k", 14, QuantType::Q6K, &dev)),
    ];
    report_bindings("after the realizes");
    for (name, o) in &runs {
        let seen = o.qmatmul_vulkan + o.qmatmul_cpu;
        if seen == 0 {
            println!("FAIL {name}: no QMatMul launch was recorded; the probe saw nothing");
            failed = true;
        }
        if o.rel > LOGIT_TOL {
            println!(
                "FAIL {name}: logits differ by {:.3}% (> {:.1}%)",
                o.rel * 100.0,
                LOGIT_TOL * 100.0
            );
            failed = true;
        }
        let route = match (o.qmatmul_vulkan > 0, o.qmatmul_cpu > 0) {
            (true, false) => "Vulkan only",
            (false, true) => "CPU fallback only",
            (true, true) => "mixed",
            (false, false) => "none",
        };
        println!("{name}: QMatMul ran on: {route}");
    }
    if failed {
        std::process::exit(1);
    }
}
