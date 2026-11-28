// SPDX-License-Identifier: MIT OR Apache-2.0
//! JIT ingestion (verify / ingest / adopt / reject) tests on a live CUDA device, moved here from
//! `fuel-dispatch/src/jit_ingest.rs`'s unit tests (GAP-418): since ROADMAP row 137 step 6.2 the CUDA kernels reach
//! the global binding table only through `fuel_cuda_dispatch::install_global()`, which `fuel-dispatch` cannot call,
//! and the test-only `cuda_test_kernels` wrappers they used are replaced by the production
//! `baracuda_dispatch::binary::{add_f32, mul_f32}` (the same drivers). The private ingest functions come through the
//! hidden, semver-exempt `fuel_dispatch::jit_test_exports` module (feature `dev-tools`).
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-cuda-dispatch --features jit
//! --test jit_ingest_live -- --include-ignored --test-threads=1`.

#![cfg(feature = "jit")]

use std::sync::Arc;
use std::time::Duration;

use fuel_dispatch::jit_test_exports::{VerifyVerdict, ingest_one, verify_candidate};
use fuel_dispatch::{
    CandidateKernel, IngestOutcome, IngestionConfig, IngestionService, ProviderFeedback,
};
use fuel_dispatch::{FlagReport, RejectionReport};

/// Spec-B Task-5 acceptance (live GPU): a candidate whose kernel is the
/// CUDA `add_f32` wrapper, carrying a 2-input `Add` decompose and matching
/// F32 operands, verifies `Pass` — its output is byte-identical to the
/// reference realized from the decompose (elementwise sum), so it's
/// bit-stable AND meets every `PrecisionGuarantee::REFERENCE` numeric bound
/// (0 ULP / 0 relative / 0 absolute). Earns one `pass` `LedgerRecord` per
/// declared claim (bit_stable + the 3 numeric bounds = 4) PLUS the
/// `kiss_ref_advisory` record (f32 single-primitive Add is on the kiss-ref
/// floor, so the advisory cross-check fires — see
/// `kiss_ref_advisory_records_for_add_f32`). `#[ignore]`'d (needs a live
/// CUDA device). Candidate/probe construction mirrors
/// `invoker_cuda.rs`'s `cuda_invoker_runs_add_elementwise_f32_end_to_end`.
/// A [`ProviderFeedback`] that fires a one-shot notify on EITHER
/// callback (unlike `RecordingFeedback` above, which is Task-7-local
/// and mock-verify-only) — the deterministic sync point Task 8's e2e
/// tests wait on instead of sleeping, mirroring `RecordingFeedback::
/// with_notify`'s pattern but against the real `ingest_one` path.
#[derive(Default)]
struct E2eFeedback {
    adopted: std::sync::Mutex<Vec<(String, fuel_graph::registry::FusedOpId)>>,
    rejected: std::sync::Mutex<Vec<RejectionReportSnapshot>>,
    notify: std::sync::Mutex<Option<std::sync::mpsc::Sender<()>>>,
}

/// An owned snapshot of a [`RejectionReport`] — the report itself
/// borrows nothing but `ledger_record` isn't `Clone`-needed here, so
/// this just lifts the two fields the assertions below check into an
/// owned, `'static` value the callback can push into a `Mutex` from
/// inside the borrowed `&RejectionReport`.
#[derive(Debug)]
struct RejectionReportSnapshot {
    failed_claim: &'static str,
    detail: String,
}

impl E2eFeedback {
    fn with_notify(tx: std::sync::mpsc::Sender<()>) -> Self {
        Self {
            notify: std::sync::Mutex::new(Some(tx)),
            ..Default::default()
        }
    }

    fn fire_notify(&self) {
        if let Some(tx) = self.notify.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }
}

impl ProviderFeedback for E2eFeedback {
    fn on_rejected(&self, report: &RejectionReport) {
        self.rejected.lock().unwrap().push(RejectionReportSnapshot {
            failed_claim: report.failed_claim,
            detail: report.detail.clone(),
        });
        self.fire_notify();
    }

    fn on_adopted(&self, entry_point: &str, id: fuel_graph::registry::FusedOpId) {
        self.adopted
            .lock()
            .unwrap()
            .push((entry_point.to_string(), id));
        self.fire_notify();
    }

    fn on_flagged(&self, _report: &FlagReport) {
        self.fire_notify();
    }
}

/// TEST-ONLY [`fuel_dispatch::kernel::KernelRef`]: wraps the STAGED interleaved
/// `rope_apply` driver (`fuel_cuda_backend::baracuda::attention::
/// rope_apply_fused_f32_into`) as a candidate kernel. Takes the 3
/// rope-shaped inputs `(x, cos, sin)` + 1 output and `OpParams::Rope` —
/// exactly the arity Fuel's rotate-half ROPE reference uses. The driver
/// narrows Fuel's FULL-width cos/sin to baracuda's half-width tables
/// internally, then applies the INTERLEAVED (2k, 2k+1) rotation. This is
/// the faithful interleaved-vs-rotate-half bug (the plan's PREFER path):
/// the candidate invokes cleanly and deterministically, and its output
/// differs from Fuel's rotate-half recipe on the SAME probe → a numeric
/// ("max_*") rejection, never an adopt.
fn interleaved_rope_apply_candidate_kernel(
    inputs: &[Arc<std::sync::RwLock<fuel_memory::Storage>>],
    outputs: &mut [Arc<std::sync::RwLock<fuel_memory::Storage>>],
    _layouts: &[fuel_ir::Layout],
    params: &fuel_dispatch::kernel::OpParams,
) -> fuel_ir::Result<()> {
    use fuel_dispatch::dispatch::{cuda_input, cuda_output, read_storage, write_storage};
    use fuel_dispatch::kernel::OpParams;

    if inputs.len() != 3 || outputs.len() != 1 {
        return Err(fuel_ir::Error::Msg(format!(
            "interleaved_rope_apply_candidate_kernel: expected 3 inputs (x, cos, sin) + 1 \
             output, got {} + {}",
            inputs.len(),
            outputs.len()
        )));
    }
    let (outer_count, seq, head_dim) = match params {
        OpParams::Rope {
            outer_count,
            seq,
            head_dim,
        } => (*outer_count, *seq, *head_dim),
        other => {
            return Err(fuel_ir::Error::Msg(format!(
                "interleaved_rope_apply_candidate_kernel: expected OpParams::Rope, got {other:?}"
            )));
        }
    };
    let x_guard = read_storage(&inputs[0])?;
    let cos_guard = read_storage(&inputs[1])?;
    let sin_guard = read_storage(&inputs[2])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let x_cuda = cuda_input(&x_guard)?;
    let cos_cuda = cuda_input(&cos_guard)?;
    let sin_cuda = cuda_input(&sin_guard)?;
    let out_cuda = cuda_output(&mut out_guard)?;
    fuel_cuda_backend::baracuda::attention::rope_apply_fused_f32_into(
        x_cuda,
        cos_cuda,
        sin_cuda,
        outer_count,
        seq,
        head_dim,
        out_cuda,
    )
}

#[test]
#[ignore = "requires a live CUDA device"]
fn verify_candidate_add_f32_passes_against_its_decompose() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_cuda_backend::CudaDevice;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_ir::DType;
    use fuel_ir::probe::BackendId;
    use unpopped_vocab::{ElementKind, OperandDesc};

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let decompose = PatternNode::Op {
        op: OpTag::Add,
        attrs: OpAttrs::default(),
        operands: vec![
            PatternNode::Bind { index: 0 },
            PatternNode::Bind { index: 1 },
        ],
    };
    let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
    let cand = CandidateKernel {
        entry_point: "add_f32".to_string(),
        kernel: fuel_cuda_dispatch::baracuda_dispatch::binary::add_f32,
        op_params: fuel_dispatch::kernel::OpParams::None,
        decompose: Some(decompose),
        operands: vec![od, od],
        dtypes: vec![DType::F32, DType::F32],
        kernel_revision_hash: 0x00AD_DF32,
        declared: fuel_dispatch::fused::PrecisionGuarantee::REFERENCE,
        backend: BackendId::Cuda,
        claimed_op: None,
    };

    let (verdict, records) = verify_candidate(
        &cand,
        &fuel_dispatch::probe_device::CudaProbeDevice::new(dev),
    );
    assert!(
        matches!(verdict, VerifyVerdict::Pass),
        "expected Pass, got {verdict:?}"
    );
    // REFERENCE declares 4 machine-checkable claims → 4 pass records,
    // plus the kiss-ref advisory record (f32 floor Add) = 5.
    assert_eq!(
        records.len(),
        5,
        "one pass record per declared claim + the kiss-ref advisory: {records:?}"
    );
    assert!(
        records.iter().all(|r| r.result == "pass"),
        "all pass: {records:?}"
    );
    assert!(records.iter().all(|r| r.backend == "Cuda"));
    assert!(
        records
            .iter()
            .all(|r| r.kernel_revision_hash == 0x00AD_DF32)
    );
    assert!(records.iter().any(|r| r.claim == "kiss_ref_advisory"));
    assert!(
        records
            .iter()
            .any(|r| r.claim == "bit_stable_on_same_hardware")
    );
    assert!(records.iter().any(|r| r.claim == "max_ulp"));
    assert!(records.iter().any(|r| r.claim == "max_relative"));
    assert!(records.iter().any(|r| r.claim == "max_absolute"));
}

/// Spec-B Task-6 acceptance (live GPU), REJECTION leg.
///
/// The plan's original Step-1 test (interleaved `rope_apply_f32` candidate
/// vs a rotate-half `decompose`) is infeasible here: rotate-half rope
/// isn't expressible as a `PatternNode` (elementwise-only — `emit`/
/// `register_runtime_fused` reject non-representable/shape-changing ops),
/// and `rope_apply` is a reverted registration, not a wired `KernelRef`
/// (see `~/.claude/projects/.../rope-convention-mismatch-baracuda-fuel.md`).
/// This test exercises the SAME essential property — `ingest_one` must
/// reject a candidate that computes a DIFFERENT function than the region
/// it claims to replace — with an elementwise substitute: `kernel` is
/// CUDA `mul_f32` (elementwise product) offered for a 2-input `Add`
/// region. The candidate declares `PrecisionGuarantee::REFERENCE` (a
/// NUMERIC claim, not just bit-stability) so `verify_candidate` actually
/// compares candidate-vs-reference output — mul is just as deterministic
/// as add, so a bit-stable-only claim would wrongly pass without ever
/// comparing values. `1+2=3 != 1*2=2` on the synthetic probe, so the
/// numeric bound fails and `ingest_one` must return `Rejected`, never
/// `Adopted`.
#[test]
#[ignore = "requires a live CUDA device"]
fn ingest_rejects_mul_candidate_for_the_add_region() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_cuda_backend::CudaDevice;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_ir::DType;
    use fuel_ir::probe::BackendId;
    use unpopped_vocab::{ElementKind, OperandDesc};

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let decompose = PatternNode::Op {
        op: OpTag::Add,
        attrs: OpAttrs::default(),
        operands: vec![
            PatternNode::Bind { index: 0 },
            PatternNode::Bind { index: 1 },
        ],
    };
    let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
    let cand = CandidateKernel {
        entry_point: "test::ingest::mul_for_add_region".to_string(),
        kernel: fuel_cuda_dispatch::baracuda_dispatch::binary::mul_f32,
        op_params: fuel_dispatch::kernel::OpParams::None,
        decompose: Some(decompose),
        operands: vec![od, od],
        dtypes: vec![DType::F32, DType::F32],
        kernel_revision_hash: 0x1_9E57_0001,
        declared: fuel_dispatch::fused::PrecisionGuarantee::REFERENCE,
        backend: BackendId::Cuda,
        claimed_op: None,
    };

    match ingest_one(
        &cand,
        &fuel_dispatch::probe_device::CudaProbeDevice::new(dev),
    ) {
        IngestOutcome::Rejected(r) => {
            assert!(
                r.failed_claim.contains("max")
                    || r.failed_claim == "vs_decompose"
                    || (r.failed_claim == "probe_set" && r.detail.contains("max_")),
                "expected a precision claim naming the mismatch, got: {} / {}",
                r.failed_claim,
                r.detail
            );
        }
        IngestOutcome::Adopted(_) => {
            panic!(
                "mul_f32 must NOT be adopted for an Add region — it computes a different function"
            )
        }
        IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
    }
}

/// Spec-B Task-6 acceptance (live GPU), ADOPTION leg — the counterpart to
/// [`ingest_rejects_mul_candidate_for_the_add_region`]: a candidate whose
/// kernel genuinely matches its `decompose` (CUDA `add_f32` for a 2-input
/// `Add` region) verifies `Pass` and is `Adopted`. After adoption, the
/// adopted id's kernel must be visible to the capability gate on Cuda via
/// [`fuel_dispatch::runtime_fused_kernels::fused_kernel_available_in`] — the same
/// predicate `offer_runtime_fused_arm` gates on before offering the fused
/// arm. `entry_point` carries a distinctive per-test-run suffix (unlike
/// the shared `add_f32`/`mul_f32` entry points above) because
/// `adopt_runtime_fused` registers into the PROCESS-GLOBAL runtime-fused
/// registry + binding table, which other `#[ignore]` tests in this binary
/// share — a colliding name across runs/tests would risk resolving a
/// stale/different registration rather than this call's own.
#[test]
#[ignore = "requires a live CUDA device"]
fn ingest_adopts_add_candidate_for_the_add_region() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_cuda_backend::CudaDevice;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_ir::DType;
    use fuel_ir::probe::BackendId;
    use unpopped_vocab::{ElementKind, OperandDesc};

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let decompose = PatternNode::Op {
        op: OpTag::Add,
        attrs: OpAttrs::default(),
        operands: vec![
            PatternNode::Bind { index: 0 },
            PatternNode::Bind { index: 1 },
        ],
    };
    let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
    let cand = CandidateKernel {
        entry_point: "test::ingest::add_for_add_region::run_1nge_5702".to_string(),
        kernel: fuel_cuda_dispatch::baracuda_dispatch::binary::add_f32,
        op_params: fuel_dispatch::kernel::OpParams::None,
        decompose: Some(decompose),
        operands: vec![od, od],
        dtypes: vec![DType::F32, DType::F32],
        kernel_revision_hash: 0x1_9E57_0002,
        declared: fuel_dispatch::fused::PrecisionGuarantee::REFERENCE,
        backend: BackendId::Cuda,
        claimed_op: None,
    };

    match ingest_one(
        &cand,
        &fuel_dispatch::probe_device::CudaProbeDevice::new(dev),
    ) {
        IngestOutcome::Adopted(id) => {
            let table = fuel_dispatch::dispatch::global_bindings();
            assert!(
                fuel_dispatch::runtime_fused_kernels::fused_kernel_available_in(
                    &table,
                    id,
                    BackendId::Cuda
                ),
                "the adopted candidate's kernel must be visible to the capability gate on Cuda",
            );
        }
        IngestOutcome::Rejected(r) => panic!(
            "add_f32 for an Add region must be Adopted, got Rejected: {} / {}",
            r.failed_claim, r.detail
        ),
        IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
    }
}

/// Spec-B Task-6 acceptance (live GPU), the ROPE ORACLE headline —
/// REJECTION leg. A candidate whose kernel is the STAGED INTERLEAVED
/// `rope_apply` driver (see [`interleaved_rope_apply_candidate_kernel`]),
/// claiming `FusedOps::ROPE` with rope-shaped F32 operands (x
/// `[1, seq, head_dim]`, cos/sin FULL-width `[seq, head_dim]`) and
/// `PrecisionGuarantee::REFERENCE`, must be REJECTED: `ingest_one` realizes
/// FUEL's registered ROTATE-HALF ROPE recipe on the SAME probe
/// (`reference_from_registered_recipe`), and the interleaved candidate's
/// output differs → the numeric bound fails (`failed_claim` contains
/// "max"). Never `Adopted` — a wrong-ROPE claimant must not enter the
/// binding table.
///
/// APPROACH — the plan's PREFER path (faithful interleaved rope), NOT the
/// elementwise fallback: the candidate wraps the real
/// `rope_apply_fused_f32_into` driver (the genuine interleaved-vs-rotate-
/// half convention bug), which wraps cleanly here because its
/// `(x, cos, sin)` arity matches Fuel's rope reference and it tolerates
/// Fuel's full-width cos/sin (narrows to half-width internally). No
/// submitted `decompose` (rope isn't a `PatternNode`), so the structural
/// recipe-identity pre-check is correctly skipped and the NUMERIC
/// registered-recipe reference is what rejects it.
#[test]
#[ignore = "requires a live CUDA device"]
fn rope_oracle_rejects_interleaved_rope_claiming_rope() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_cuda_backend::CudaDevice;
    use fuel_graph::registry::FusedOps;
    use fuel_ir::DType;
    use fuel_ir::probe::BackendId;
    use unpopped_vocab::{ElementKind, OperandDesc};

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let (seq, head_dim) = (4usize, 8usize); // head_dim even (baracuda rope)
    let x_od = OperandDesc::new(
        3,
        &[1, seq as i64, head_dim as i64],
        &[(seq * head_dim) as i64, head_dim as i64, 1],
        ElementKind::F32,
        16,
    );
    // cos/sin FULL-width [seq, head_dim] — Fuel's rotate-half reference
    // convention; the interleaved candidate narrows them internally.
    let trig_od = OperandDesc::new(
        2,
        &[seq as i64, head_dim as i64],
        &[head_dim as i64, 1],
        ElementKind::F32,
        16,
    );
    let cand = CandidateKernel {
        entry_point: "test::rope_oracle::interleaved_rope_apply_claims_rope".to_string(),
        kernel: interleaved_rope_apply_candidate_kernel,
        op_params: fuel_dispatch::kernel::OpParams::Rope {
            outer_count: 1,
            seq,
            head_dim,
        },
        decompose: None,
        operands: vec![x_od, trig_od, trig_od],
        dtypes: vec![DType::F32, DType::F32, DType::F32],
        kernel_revision_hash: 0x1_9E57_0003,
        declared: fuel_dispatch::fused::PrecisionGuarantee::REFERENCE,
        backend: BackendId::Cuda,
        claimed_op: Some(FusedOps::ROPE),
    };

    match ingest_one(
        &cand,
        &fuel_dispatch::probe_device::CudaProbeDevice::new(dev),
    ) {
        IngestOutcome::Rejected(r) => {
            assert!(
                r.failed_claim.contains("max")
                    || r.failed_claim == "recipe_identity"
                    || (r.failed_claim == "probe_set"
                        && (r.detail.contains("max_") || r.detail.contains("recipe_identity"))),
                "expected a precision/identity rejection (interleaved != rotate-half), \
                 got: {} / {}",
                r.failed_claim,
                r.detail
            );
            // Pin the REASON, not just the claim id: `first_numeric_claim`
            // also names a `max_*` claim when the REFERENCE realize itself
            // errors (`verify_candidate_impl`'s
            // "reference realize from registered recipe failed: {e:?}"
            // detail), which would ALSO satisfy `contains("max")` above for
            // the WRONG reason (a broken reference, not a genuine
            // interleaved-vs-rotate-half numeric mismatch). Rule that out
            // explicitly so a future regression that breaks rotate-half
            // reference realization fails LOUDLY here instead of being
            // mistaken for this oracle working.
            assert!(
                !r.detail.contains("reference realize"),
                "rejection must be a genuine numeric mismatch vs the realized rotate-half \
                 reference, not a reference-realize failure: {}",
                r.detail
            );
        }
        IngestOutcome::Adopted(_) => panic!(
            "interleaved rope claiming ROPE must NOT be adopted — it computes a different \
             function than Fuel's rotate-half rope"
        ),
        IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
    }
}

/// Spec-B Task-6 acceptance (live GPU), the ROPE ORACLE — ADOPTION leg
/// (positive counterpart). No CUDA ROTATE-HALF rope kernel exists as a
/// single callable `KernelRef` (Fuel's rotate-half rope runs via the
/// primitive decompose; every callable CUDA rope kernel is interleaved), so
/// — per the plan's documented FALLBACK — this adopts a MATCHING
/// ELEMENTWISE claimant, exercising the SAME claimed-op path the rope
/// rejection does (structural recipe-identity + the numeric
/// registered-recipe reference), just on an op that HAS a matching CUDA
/// kernel. A runtime-registered `Add` region is the claimed op; the
/// candidate's kernel is CUDA `add_f32` and its submitted decompose IS the
/// `Add` region → recipe-identity matches AND `add_f32`'s output equals the
/// registered recipe realized on the same probe (0 ULP) → `Adopted`, and
/// the adopted kernel is visible to the capability gate on Cuda. A
/// per-run-distinct id suffix avoids process-global registry collisions
/// with other `#[ignore]` tests in this binary.
#[test]
#[ignore = "requires a live CUDA device"]
fn rope_oracle_adopts_matching_elementwise_claimant() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_cuda_backend::CudaDevice;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_graph::runtime_fused::register_runtime_fused;
    use fuel_ir::DType;
    use fuel_ir::probe::BackendId;
    use unpopped_vocab::{ElementKind, OperandDesc};

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    // A runtime-registered Add op is the CLAIMED op; the candidate submits
    // the very same Add region as its decompose (recipe-identity matches).
    let add_region = PatternNode::Op {
        op: OpTag::Add,
        attrs: OpAttrs::default(),
        operands: vec![
            PatternNode::Bind { index: 0 },
            PatternNode::Bind { index: 1 },
        ],
    };
    let claimed_id = register_runtime_fused(
        "test::rope_oracle::adopt::claimed_add::run_r0pe_0004",
        add_region.clone(),
    )
    .expect("register the runtime Add op the candidate claims");

    let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
    let cand = CandidateKernel {
        entry_point: "test::rope_oracle::adopt::add_f32::run_r0pe_0004".to_string(),
        kernel: fuel_cuda_dispatch::baracuda_dispatch::binary::add_f32,
        op_params: fuel_dispatch::kernel::OpParams::None,
        decompose: Some(add_region),
        operands: vec![od, od],
        dtypes: vec![DType::F32, DType::F32],
        kernel_revision_hash: 0x1_9E57_0004,
        declared: fuel_dispatch::fused::PrecisionGuarantee::REFERENCE,
        backend: BackendId::Cuda,
        claimed_op: Some(claimed_id),
    };

    match ingest_one(
        &cand,
        &fuel_dispatch::probe_device::CudaProbeDevice::new(dev),
    ) {
        IngestOutcome::Adopted(id) => {
            let table = fuel_dispatch::dispatch::global_bindings();
            assert!(
                fuel_dispatch::runtime_fused_kernels::fused_kernel_available_in(
                    &table,
                    id,
                    BackendId::Cuda
                ),
                "the adopted claimant's kernel must be visible to the capability gate on Cuda"
            );
        }
        IngestOutcome::Rejected(r) => panic!(
            "a matching add_f32 claimant (recipe-identity + 0-ULP numeric) must be Adopted, \
             got Rejected: {} / {}",
            r.failed_claim, r.detail
        ),
        IngestOutcome::Flagged(r) => panic!("unexpected Flagged: {} / {}", r.claim, r.detail),
    }
}

/// Spec-B Task-8 acceptance (live GPU): drive the WHOLE ingestion
/// service end-to-end through its PRODUCTION constructor
/// (`IngestionService::start`, not the `start_with_verify` test seam
/// Task 7's tests use) — enqueue → worker thread → `ingest_one` →
/// `verify_candidate` → `reference_output` → `adopt_verified` →
/// `ProviderFeedback::on_adopted`. Asserts BOTH that the callback fired
/// AND that the adopted op is genuinely visible to the capability gate
/// (`fused_kernel_available_in`) — the same check
/// `ingest_adopts_add_candidate_for_the_add_region` makes directly
/// against `ingest_one`, now exercised through the async service.
#[test]
#[ignore = "requires a live CUDA device"]
fn ingestion_service_adopts_a_matching_add_candidate_e2e() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_ir::probe::BackendId;

    use fuel_cuda_backend::CudaDevice;

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let svc = IngestionService::start(
        std::sync::Arc::new(fuel_dispatch::probe_device::CudaProbeDevice::new(dev)),
        IngestionConfig::default(),
    );

    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let fb = Arc::new(E2eFeedback::with_notify(tx));
    svc.enqueue(e2e_add_candidate(), Some(fb.clone()))
        .expect("enqueue accepted (queue starts empty)");

    rx.recv_timeout(Duration::from_secs(30))
        .expect("worker should report the adopted outcome within 30s");

    let adopted = fb.adopted.lock().unwrap();
    assert_eq!(
        adopted.len(),
        1,
        "exactly one on_adopted callback: {adopted:?}"
    );
    let (entry_point, id) = &adopted[0];
    assert_eq!(entry_point, "test::e2e::add_for_add_region::run_e2e_8801");
    assert!(
        fb.rejected.lock().unwrap().is_empty(),
        "the matching add_f32 candidate must not be rejected"
    );

    let table = fuel_dispatch::dispatch::global_bindings();
    assert!(
        fuel_dispatch::runtime_fused_kernels::fused_kernel_available_in(
            &table,
            *id,
            BackendId::Cuda
        ),
        "the adopted candidate's kernel must be visible to the capability gate on Cuda"
    );
    drop(table);

    svc.shutdown();
}

/// Spec-B Task-8 acceptance (live GPU), rejection leg — the counterpart
/// to [`ingestion_service_adopts_a_matching_add_candidate_e2e`]: the SAME
/// production service, driven end-to-end, must reject a candidate whose
/// kernel computes a different function than the region it claims to
/// replace (`mul_f32` offered for an `Add` region), reporting a
/// PRECISION claim (`failed_claim` contains "max") via `on_rejected` —
/// never `on_adopted`.
#[test]
#[ignore = "requires a live CUDA device"]
fn ingestion_service_rejects_a_mismatched_mul_candidate_e2e() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_cuda_backend::CudaDevice;

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let svc = IngestionService::start(
        std::sync::Arc::new(fuel_dispatch::probe_device::CudaProbeDevice::new(dev)),
        IngestionConfig::default(),
    );

    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let fb = Arc::new(E2eFeedback::with_notify(tx));
    svc.enqueue(e2e_mul_candidate(), Some(fb.clone()))
        .expect("enqueue accepted (queue starts empty)");

    rx.recv_timeout(Duration::from_secs(30))
        .expect("worker should report the rejected outcome within 30s");

    let rejected = fb.rejected.lock().unwrap();
    assert_eq!(
        rejected.len(),
        1,
        "exactly one on_rejected callback: {rejected:?}"
    );
    assert!(
        rejected[0].failed_claim.contains("max")
            || (rejected[0].failed_claim == "probe_set" && rejected[0].detail.contains("max_")),
        "expected a precision claim naming the mismatch, got: {} / {}",
        rejected[0].failed_claim,
        rejected[0].detail
    );
    assert!(
        fb.adopted.lock().unwrap().is_empty(),
        "the mismatched mul_f32 candidate must not be adopted"
    );

    svc.shutdown();
}

/// Build the `Add`-region `add_f32` candidate (matching function) with a
/// DISTINCTIVE `entry_point` so it doesn't collide with any other
/// `#[ignore]` test's process-global registration in this binary (same
/// discipline as `ingest_adopts_add_candidate_for_the_add_region`).
fn e2e_add_candidate() -> CandidateKernel {
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_ir::DType;
    use fuel_ir::probe::BackendId;
    use unpopped_vocab::{ElementKind, OperandDesc};

    let decompose = PatternNode::Op {
        op: OpTag::Add,
        attrs: OpAttrs::default(),
        operands: vec![
            PatternNode::Bind { index: 0 },
            PatternNode::Bind { index: 1 },
        ],
    };
    let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
    CandidateKernel {
        entry_point: "test::e2e::add_for_add_region::run_e2e_8801".to_string(),
        kernel: fuel_cuda_dispatch::baracuda_dispatch::binary::add_f32,
        op_params: fuel_dispatch::kernel::OpParams::None,
        decompose: Some(decompose),
        operands: vec![od, od],
        dtypes: vec![DType::F32, DType::F32],
        kernel_revision_hash: 0x1_9E57_8801,
        declared: fuel_dispatch::fused::PrecisionGuarantee::REFERENCE,
        backend: BackendId::Cuda,
        claimed_op: None,
    }
}

/// Build the `mul_f32`-vs-`Add`-region MISMATCHED candidate — the plan's
/// original Step-1 interleaved-rope rejection leg is infeasible (rotate-
/// half rope isn't a `PatternNode`; `rope_apply` is a reverted
/// registration, not a wired `KernelRef` — see
/// `ingest_rejects_mul_candidate_for_the_add_region`'s doc comment above
/// for the full rationale). `declared: REFERENCE` is load-bearing: it
/// declares a NUMERIC claim so the mul-vs-add mismatch is actually
/// compared and caught, not just skipped as an unclaimed property.
fn e2e_mul_candidate() -> CandidateKernel {
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_ir::DType;
    use fuel_ir::probe::BackendId;
    use unpopped_vocab::{ElementKind, OperandDesc};

    let decompose = PatternNode::Op {
        op: OpTag::Add,
        attrs: OpAttrs::default(),
        operands: vec![
            PatternNode::Bind { index: 0 },
            PatternNode::Bind { index: 1 },
        ],
    };
    let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
    CandidateKernel {
        entry_point: "test::e2e::mul_for_add_region::run_e2e_8802".to_string(),
        kernel: fuel_cuda_dispatch::baracuda_dispatch::binary::mul_f32,
        op_params: fuel_dispatch::kernel::OpParams::None,
        decompose: Some(decompose),
        operands: vec![od, od],
        dtypes: vec![DType::F32, DType::F32],
        kernel_revision_hash: 0x1_9E57_8802,
        declared: fuel_dispatch::fused::PrecisionGuarantee::REFERENCE,
        backend: BackendId::Cuda,
        claimed_op: None,
    }
}
