// SPDX-License-Identifier: MIT OR Apache-2.0
//! What `fuel_cuda_dispatch::install_global` leaves in the PROCESS-WIDE state. Three tests moved here from
//! `fuel-dispatch`'s unit tests (ROADMAP row 137 step 6.2): they asserted that `global_bindings()` carried the CUDA
//! kernels on its own, which stopped being true when the registration moved above `fuel-dispatch`. Each now
//! installs first, exactly as `fuel-tensor`'s `cpu_accel::ensure_installed` does before a plan is built.
//! Hermetic: no device is used (the topology test is explicit that it does not assume a CUDA-capable GPU).

#![cfg(feature = "cuda")]

use fuel_dispatch::dispatch::global_bindings;
use fuel_dispatch::runtime_fused_kernels::fused_kernel_available;
use fuel_dispatch::topology::SystemTopology;
use fuel_graph::registry::FusedOps;
use fuel_ir::DType;
use fuel_ir::DeviceLocation;
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;

/// `register_cuda_kernels` (the cuBLAS F32 MatMul that stays in `fuel-dispatch`) AND
/// `register_baracuda_cuda_kernels` both land in the global table. The PTX-duplicate unary/binary registrations
/// were stripped in the fuel-cuda-kernels retirement (commit d9898fec), so elementwise keys carry exactly ONE
/// alternative: baracuda's.
#[test]
fn global_bindings_auto_registers_cuda_paths() {
    fuel_cuda_dispatch::install_global();
    let b = global_bindings();
    // register_cuda_kernels residue: F32 MatMul via cuBLAS.
    let cublas = b.lookup(
        OpKind::MatMul,
        &[DType::F32, DType::F32, DType::F32],
        BackendId::Cuda,
    );
    assert!(
        cublas.is_ok(),
        "cuBLAS F32 MatMul should be registered on Cuda"
    );

    // Baracuda path: int8 MatMul exists only via baracuda. If this resolves, both paths fired.
    let baracuda_int8 = b.lookup(
        OpKind::MatMul,
        &[DType::I8, DType::I8, DType::I8],
        BackendId::Cuda,
    );
    assert!(
        baracuda_int8.is_ok(),
        "baracuda int8 MatMul should be registered on Cuda",
    );

    // Post-strip invariant: F32 unary Neg resolves with exactly one alternative (baracuda is the single source
    // of truth for CUDA elementwise; a second entry would mean a duplicate registration crept back in).
    let alts = b.lookup_alternatives(
        OpKind::NegElementwise,
        &[DType::F32, DType::F32],
        BackendId::Cuda,
    );
    assert!(
        alts.len() == 1,
        "expected exactly 1 CUDA Neg F32 alternative (baracuda); got {}",
        alts.len(),
    );
}

/// The CUDA `flash_decoding` kernel registers only into `KernelBindingTable`; the static-id bridge must see it
/// once the CUDA kernels are installed.
#[test]
fn fused_kernel_available_bridges_static_flash_attn_from_binding_table_on_cuda() {
    fuel_cuda_dispatch::install_global();
    drop(global_bindings());
    assert!(
        fused_kernel_available(FusedOps::FLASH_ATTN, BackendId::Cuda),
        "the CUDA flash_decoding binding registers only into KernelBindingTable; \
         fused_kernel_available must bridge to it for the static FLASH_ATTN id",
    );
}

/// With the CUDA kernels installed the topology must report Cuda as a backend and CUDA's default device must be in
/// `backends_for(Cuda { gpu_id: 0 })`. No CUDA-capable GPU is assumed: the binding-table walk is what surfaces the
/// backend. `install_global` bumps the topology generation, so `current()` rebuilds after it.
#[test]
fn cuda_backend_discovered_when_feature_enabled() {
    fuel_cuda_dispatch::install_global();
    let topology = SystemTopology::current();
    assert!(
        topology.backends().contains(&BackendId::Cuda),
        "cuda kernels are installed; topology should report Cuda. Got: {:?}",
        topology.backends(),
    );
    let cuda_dev = DeviceLocation::Cuda { gpu_id: 0 };
    let here = topology.backends_for(cuda_dev);
    assert!(
        here.contains(&BackendId::Cuda),
        "Cuda should be among the backends targeting {:?}, got {:?}",
        cuda_dev,
        here,
    );
}
