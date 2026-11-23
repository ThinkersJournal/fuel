// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 124 step 3: does a GGUF weight that stays quantized (Q4_0, Q8_0, Q4_K, Q6_K) reach the
//! GPU backend's `QMatMul` through a LOADED model, or does the planner fall back to the CPU?
//!
//! A program, not an `#[ignore]` test (the `#[ignore]` total is a ratchet that only goes down;
//! this needs a live GPU). It builds the synthetic one-layer llama of
//! `tests/gguf_resident_quantized.rs` in each format, runs the same forward on the CPU and on a GPU
//! device (Vulkan or CUDA), and prints which backend the executor launched every `QMatMul` on
//! (`op_profile`), the resident bytes, and the logit difference against the CPU run. It also runs bare
//! one-op graphs and prints the binding table, capability registry and the registered bindings'
//! cost and precision claims, the evidence GAP-393 was diagnosed from.
//!
//! Run through the GPU mutex, alone (the first argument picks the backend; default Vulkan when it is
//! compiled in, else CUDA):
//!
//! ```text
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -p fuel-transformers \
//!     --features dev-tools,vulkan --example gguf_resident_reach -- vulkan
//! ```
//!
//! `FUEL_VULKAN_DEVICE` (UUID or PCI address) picks the Vulkan adapter, else the default discrete one;
//! CUDA uses device 0. The exit code is nonzero if a format's logits leave the tolerance or no
//! `QMatMul` launch was seen (a probe that saw nothing must not report success).

#[cfg(any(feature = "vulkan", feature = "cuda"))]
#[path = "../tests/common/gguf_synth.rs"]
mod synth;

#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod probe {
    use super::synth::{files, load, projection_weights};
    use fuel_dispatch::compiled::op_profile;
    use fuel_graph::QuantType;
    use fuel_ir::probe::BackendId;
    use fuel_quantized::{BlockQ4_0, BlockQ4K, BlockQ6K, BlockQ8_0, GgmlType};
    use fuel_tensor::Device;
    use fuel_tensor::lazy::Tensor;

    /// Same bound as `tests/gguf_resident_quantized.rs::LOGIT_TOL`: the CPU kernel quantizes
    /// activations to 8 bits and the GPU routes dequantize the weight to f32, so they differ a
    /// little; a wrong dequantization differs by far more.
    const LOGIT_TOL: f32 = 0.04;

    #[derive(Clone, Copy)]
    enum Gpu {
        #[cfg(feature = "vulkan")]
        Vulkan,
        #[cfg(feature = "cuda")]
        Cuda,
    }

    impl Gpu {
        fn backend(self) -> BackendId {
            match self {
                #[cfg(feature = "vulkan")]
                Gpu::Vulkan => BackendId::Vulkan,
                #[cfg(feature = "cuda")]
                Gpu::Cuda => BackendId::Cuda,
            }
        }

        /// The backend's name as `op_profile` renders it (`Debug` of the `BackendId`).
        fn tag(self) -> String {
            format!("{:?}", self.backend())
        }
    }

    fn pick_gpu() -> Gpu {
        let want = std::env::args().nth(1);
        match want.as_deref() {
            #[cfg(feature = "vulkan")]
            Some("vulkan") | None => Gpu::Vulkan,
            #[cfg(all(feature = "cuda", not(feature = "vulkan")))]
            None => Gpu::Cuda,
            #[cfg(feature = "cuda")]
            Some("cuda") => Gpu::Cuda,
            other => {
                eprintln!("unknown or not-compiled-in backend {other:?}; try `vulkan` or `cuda`");
                std::process::exit(2);
            }
        }
    }

    fn new_device(gpu: Gpu) -> Device {
        match gpu {
            #[cfg(feature = "vulkan")]
            Gpu::Vulkan => {
                let dev =
                    fuel_tensor::vulkan_backend::new_device_from_env().expect("a Vulkan device");
                let vk = fuel_tensor::vulkan_backend::as_device(&dev).expect("a Vulkan device");
                println!("adapter: [{}] {}", vk.gpu_id, vk.device_name);
                dev
            }
            #[cfg(feature = "cuda")]
            Gpu::Cuda => {
                let dev = fuel_tensor::cuda_backend::new_device(0).expect("a CUDA device");
                println!("adapter: CUDA device 0 at {:?}", dev.location());
                dev
            }
        }
    }

    /// Realize a lazy tensor on `dev` through the production bridge (the same path
    /// `realize_f32_vulkan` / `realize_f32_cuda` use).
    fn realize(t: &Tensor, dev: &Device) -> Vec<f32> {
        let g = t.graph_tensor();
        let graph = g.graph().clone();
        fuel_tensor::pipelined_bridge::realize_one_as::<f32>(&graph, g.id(), dev)
            .expect("realize on the GPU device")
    }

    struct Outcome {
        qmatmul_gpu: u64,
        qmatmul_cpu: u64,
        rel: f32,
    }

    type Rows = [(String, String, u64, std::time::Duration)];

    fn launches(rows: &Rows, backend_tag: &str, only_qmatmul: bool) -> u64 {
        rows.iter()
            .filter(|r| r.1.contains(backend_tag) && (!only_qmatmul || r.0.contains("QMatMul")))
            .map(|r| r.2)
            .sum()
    }

    fn reach<B: GgmlType>(
        tag: &str,
        code: u32,
        qtype: QuantType,
        gpu: Gpu,
        dev: &Device,
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

        let cpu = model.forward(&toks, 0).expect("cpu forward").realize_f32();

        op_profile::set_enabled(true);
        let _ = op_profile::take();
        let trace_before = fuel_dispatch::optimizer_trace::snapshot();
        let fwd = model.forward(&toks, 0).expect("gpu forward");
        let on_gpu = realize(&fwd, dev);
        let trace = fuel_dispatch::optimizer_trace::snapshot().since(&trace_before);
        let rows = op_profile::take();
        op_profile::set_enabled(false);
        // The executor's own arm counts and backlog feed for this realize: a `Branch` resolved to arm 1
        // is a node the load-aware picker moved off its first-choice placement.
        println!(
            "  executor trace: arm_chosen {:?} of {} branches; backlog added {} ns, retired {} ns",
            trace.arm_chosen,
            trace.branches_resolved(),
            trace.backlog_added_ns,
            trace.backlog_retired_ns
        );

        let gtag = gpu.tag();
        // Control: the launches of ALL ops by backend. A realize on a GPU device that launched nothing
        // on the GPU would make "QMatMul ran on the CPU" a statement about the whole graph.
        println!(
            "  all ops: launches {gtag}={} cpu={}",
            launches(&rows, &gtag, false),
            launches(&rows, "Cpu", false)
        );
        if std::env::var_os("REACH_ALL_OPS").is_some() {
            for (op, backend, n, _) in &rows {
                println!("      {op} on {backend}: {n}");
            }
        }
        let qmatmul_gpu = launches(&rows, &gtag, true);
        let qmatmul_cpu = launches(&rows, "Cpu", true);
        let (mut worst, mut scale) = (0.0_f32, 0.0_f32);
        for (a, b) in on_gpu.iter().zip(&cpu) {
            assert!(
                a.is_finite() && b.is_finite(),
                "{qtype:?}: non-finite logit"
            );
            worst = worst.max((a - b).abs());
            scale = scale.max(b.abs());
        }
        assert_eq!(on_gpu.len(), cpu.len());
        let rel = worst / scale.max(1e-3);
        println!(
            "  {qtype:?}: resident {resident} B; QMatMul launches {gtag}={qmatmul_gpu} cpu={qmatmul_cpu}; \
             worst |dlogit| {worst:e} vs max |logit| {scale:e} = {:.3}%",
            rel * 100.0
        );
        Outcome {
            qmatmul_gpu,
            qmatmul_cpu,
            rel,
        }
    }

    /// The bindings, capability entries and cost/precision claims the placement rule reads, for the CPU and
    /// the chosen GPU backend (the CPU is the control). Asked before and after the realizes, because
    /// optional backends may register on first use.
    fn report_bindings(when: &str, gpu: Gpu) {
        use fuel_dispatch::OpParams;
        use fuel_ir::DType::{F32, U32};
        use fuel_ir::DeviceLocation;
        use fuel_ir::backend::{BackendCapabilities, SubstrateClass};
        use fuel_ir::dispatch::OpKind;
        let table = fuel_dispatch::dispatch::global_bindings();
        let bes = [BackendId::Cpu, gpu.backend()];
        for be in bes {
            let n = table
                .lookup_alternatives(OpKind::QMatMul, &[F32, U32, F32], be)
                .len();
            println!(
                "binding table ({when}): (QMatMul, [F32, U32, F32], {be:?}) alternatives = {n}"
            );
        }
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
        for be in bes {
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
        // Control: the GPU MatMul binding for the same GEMM.
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
        let be = gpu.backend();
        for alt in table.lookup_alternatives(OpKind::MatMul, &[F32, F32, F32], be) {
            let c = (alt.cost)(&m_shapes, &[F32, F32, F32], &mp, &caps);
            println!(
                "  MatMul {be:?} [{}] cost for 64x4096x4096: {c:?}",
                alt.kernel_source
            );
            println!(
                "  MatMul {be:?} [{}] precision: {:?}; caps: {:?}",
                alt.kernel_source, alt.precision, alt.caps
            );
        }
    }

    /// A bare graph, no model: activation `[m, k]` and one resident weight `[n, k]`, one `qmatmul`, realized
    /// on the GPU device. Separates "QMatMul itself stays on the CPU" from "QMatMul inside the loaded
    /// model's graph stays on the CPU".
    fn isolated<B: GgmlType>(
        tag: &str,
        qtype: QuantType,
        (m, k, n): (usize, usize, usize),
        dev: &Device,
    ) {
        use fuel_ir::Shape;
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
        let x = Tensor::from_f32(a, Shape::from_dims(&[m, k]), &Device::cpu()).unwrap();
        let w = x
            .const_u32_like(w_u32, Shape::from_dims(&[bytes.len() / 4]))
            .unwrap();
        let y = x.qmatmul(&w, qtype, k, n).expect("qmatmul");
        op_profile::set_enabled(true);
        let _ = op_profile::take();
        let out = realize(&y, dev);
        let rows = op_profile::take();
        op_profile::set_enabled(false);
        let l: Vec<String> = rows
            .iter()
            .map(|(op, be, n, _)| format!("{op}@{be} x{n}"))
            .collect();
        println!(
            "  isolated {tag} m={m} k={k} n={n}: out[0]={:.4}; launches: {}",
            out[0],
            l.join(", ")
        );
    }

    /// ROADMAP row 169, the kernel read: a Q8_0 weight whose block bytes are not a multiple of 4 (an odd `n` of
    /// 34-byte blocks) is stored rounded UP to whole `u32` elements, zero-padded at the tail. `isolated` cannot
    /// show this (it drops the tail bytes with `as_chunks`). This builds the padded buffer the way the loader
    /// does, realizes the one-op graph on the GPU device and compares every output with the exact dequantized
    /// product, so a kernel or a size check that misreads the padded buffer shows as an error or a panic.
    /// Returns `true` when the realize succeeded, a QMatMul launch was seen, and the error is within `LOGIT_TOL`.
    fn odd_n<B: GgmlType>(
        qtype: QuantType,
        (m, k, n): (usize, usize, usize),
        dev: &Device,
        gtag: &str,
    ) -> bool {
        use fuel_ir::Shape;
        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.021).sin() * 0.7)
            .collect();
        let mut blocks = vec![B::zeros(); n * k / B::BLCK_SIZE];
        B::from_float(&w_f32, &mut blocks);
        let mut bytes = fuel_quantized::blocks_as_bytes(&blocks).to_vec();
        let exact = bytes.len();
        bytes.resize(exact.next_multiple_of(4), 0); // the loader's rounding up
        let w_u32: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let mut deq = vec![0f32; n * k];
        B::to_float(&blocks, &mut deq);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let x = Tensor::from_f32(a.clone(), Shape::from_dims(&[m, k]), &Device::cpu()).unwrap();
        let w = x
            .const_u32_like(w_u32.clone(), Shape::from_dims(&[w_u32.len()]))
            .unwrap();
        let y = match x.qmatmul(&w, qtype, k, n) {
            Ok(y) => y,
            Err(e) => {
                println!("  odd-N {qtype:?} m={m} k={k} n={n}: qmatmul REFUSED: {e:?}");
                return false;
            }
        };
        let g = y.graph_tensor();
        let graph = g.graph().clone();
        op_profile::set_enabled(true);
        let _ = op_profile::take();
        let out = fuel_tensor::pipelined_bridge::realize_one_as::<f32>(&graph, g.id(), dev);
        let rows = op_profile::take();
        op_profile::set_enabled(false);
        let out = match out {
            Ok(o) => o,
            Err(e) => {
                println!("  odd-N {qtype:?} m={m} k={k} n={n}: realize FAILED: {e:?}");
                return false;
            }
        };
        let (on_gpu, on_cpu) = (launches(&rows, gtag, true), launches(&rows, "Cpu", true));
        let mut scale = 0f32;
        let mut worst = 0f32;
        let mut worst_at = (0usize, 0usize);
        for mi in 0..m {
            for ni in 0..n {
                let want: f32 = (0..k).map(|kk| a[mi * k + kk] * deq[ni * k + kk]).sum();
                scale = scale.max(want.abs());
                let err = (out[mi * n + ni] - want).abs();
                if err > worst {
                    worst = err;
                    worst_at = (mi, ni);
                }
            }
        }
        let rel = worst / scale.max(f32::MIN_POSITIVE);
        let ok = out.len() == m * n && (on_gpu + on_cpu) > 0 && rel <= LOGIT_TOL;
        println!(
            "  odd-N {qtype:?} m={m} k={k} n={n}: buffer {exact} B -> {} u32; QMatMul launches {gtag}={on_gpu} cpu={on_cpu}; \
             max |err|/max|want| = {rel:.2e} at (row {}, col {}) {}",
            w_u32.len(),
            worst_at.0,
            worst_at.1,
            if ok { "OK" } else { "FAIL" }
        );
        ok
    }

    /// Wall time of the same one-op `qmatmul` graph realized on the CPU device and on the GPU device
    /// (warm-up, then the median of `REPS`). The GPU figure includes everything the realize does: the
    /// weight and activation uploads and the result download, because the executor re-realizes the
    /// graph from host constants each call. This is the evidence for "placement follows the static
    /// prior": it says whether the GPU `QMatMul` is actually faster for the shape, copies included.
    fn timing<B: GgmlType>(
        tag: &str,
        qtype: QuantType,
        (m, k, n): (usize, usize, usize),
        dev: &Device,
    ) {
        use fuel_ir::Shape;
        const REPS: usize = 7;
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
        let x = Tensor::from_f32(a, Shape::from_dims(&[m, k]), &Device::cpu()).unwrap();
        let w = x
            .const_u32_like(w_u32, Shape::from_dims(&[bytes.len() / 4]))
            .unwrap();
        let y = x.qmatmul(&w, qtype, k, n).expect("qmatmul");
        let median = |d: &Device| -> f64 {
            let _ = realize(&y, d);
            let mut t: Vec<f64> = (0..REPS)
                .map(|_| {
                    let t0 = std::time::Instant::now();
                    let out = realize(&y, d);
                    std::hint::black_box(&out);
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[REPS / 2]
        };
        let cpu = median(&Device::cpu());
        let gpu = median(dev);
        println!(
            "  timing {tag} m={m} k={k} n={n}: cpu {cpu:.3} ms, gpu {gpu:.3} ms (median of {REPS}, copies included), gpu/cpu = {:.2}",
            gpu / cpu
        );
    }

    pub fn main() {
        // The executor backlog feed is OFF by default; this probe reports its effect, so it is ON here
        // unless `REACH_FEED=0` (then the trace shows what the default does).
        let feed_on = std::env::var("REACH_FEED").as_deref() != Ok("0");
        fuel_dispatch::dispatch::set_backlog_feed_enabled(feed_on);
        println!(
            "executor backlog feed: {}",
            if feed_on { "ON" } else { "OFF" }
        );
        let gpu = pick_gpu();
        let dev = new_device(gpu);
        let gtag = gpu.tag();

        report_bindings("before any realize", gpu);

        println!("isolated QMatMul (bare graph, weight resident as blocks):");
        for shape in [(1, 256, 256), (1, 4096, 4096), (64, 4096, 4096)] {
            isolated::<BlockQ4_0>("Q4_0", QuantType::Q4_0, shape, &dev);
            isolated::<BlockQ8_0>("Q8_0", QuantType::Q8_0, shape, &dev);
            isolated::<BlockQ4K>("Q4_K", QuantType::Q4_K_M, shape, &dev);
        }

        println!("timing (same one-op graph on the CPU device and the GPU device):");
        for shape in [(1, 4096, 4096), (64, 4096, 4096)] {
            timing::<BlockQ4_0>("Q4_0", QuantType::Q4_0, shape, &dev);
            timing::<BlockQ8_0>("Q8_0", QuantType::Q8_0, shape, &dev);
            timing::<BlockQ4K>("Q4_K", QuantType::Q4_K_M, shape, &dev);
        }

        let mut failed = false;
        println!(
            "odd-N padded weight (row 169; K=32 Q8_0, buffer rounded up to whole u32; n=98 is the even control):"
        );
        for n in [97usize, 98] {
            for m in [1usize, 4] {
                if !odd_n::<BlockQ8_0>(QuantType::Q8_0, (m, 32, n), &dev, &gtag) {
                    failed = true;
                }
            }
        }
        let runs = [
            (
                "Q4_0",
                reach::<BlockQ4_0>("q40", 2, QuantType::Q4_0, gpu, &dev),
            ),
            (
                "Q8_0",
                reach::<BlockQ8_0>("q8", 8, QuantType::Q8_0, gpu, &dev),
            ),
            (
                "Q4_K",
                reach::<BlockQ4K>("q4k", 12, QuantType::Q4_K_M, gpu, &dev),
            ),
            (
                "Q6_K",
                reach::<BlockQ6K>("q6k", 14, QuantType::Q6K, gpu, &dev),
            ),
        ];
        report_bindings("after the realizes", gpu);
        for (name, o) in &runs {
            let seen = o.qmatmul_gpu + o.qmatmul_cpu;
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
            let route = match (o.qmatmul_gpu > 0, o.qmatmul_cpu > 0) {
                (true, false) => format!("{gtag} only"),
                (false, true) => "CPU fallback only".to_string(),
                (true, true) => "mixed".to_string(),
                (false, false) => "none".to_string(),
            };
            println!("{name}: QMatMul ran on: {route}");
        }
        if failed {
            std::process::exit(1);
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn main() {
    probe::main();
}

#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
fn main() {
    eprintln!("build with --features dev-tools,vulkan or dev-tools,cuda");
    std::process::exit(2);
}
