// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 173 (d): `SharedHostStorage` aliases a model's weight `Arc` and must never write through it.
//!
//! One test per in-place entry point of `DynBackendStorage`: the shared `Arc` must come out
//! byte-identical AND the storage must have let go of it (`is_shared() == false`), while the
//! written storage holds the written values. The detector is two-armed on purpose: a writer that
//! mutated the `Arc` in place would fail the bytes check, one that "forgot" to copy first and
//! wrote nowhere would fail the values check.

use fuel_backend_contract::dyn_backend::{DynBackendDevice, DynBackendStorage};
use fuel_cpu_backend::SharedHostStorage;
use fuel_cpu_backend::dyn_impl::{CpuBackendDevice, CpuStorage};
use fuel_ir::op::BinaryOp;
use fuel_ir::{HostBuffer, Layout, Scalar, Shape, SharedHostData};
use std::any::Any;
use std::sync::Arc;

const N: usize = 8;

fn src_arc() -> Arc<[f32]> {
    (0..N)
        .map(|i| 1.0 + i as f32 * 0.25)
        .collect::<Vec<_>>()
        .into()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn shared(a: &Arc<[f32]>) -> SharedHostStorage {
    SharedHostStorage::new(SharedHostData::F32(Arc::clone(a)))
}

fn read(s: &dyn DynBackendStorage) -> Vec<f32> {
    match s.to_host_buffer_dyn().unwrap() {
        HostBuffer::F32(v) => v,
        other => panic!("expected F32, got {:?}", other.dtype()),
    }
}

fn lay() -> Layout {
    Layout::contiguous(Shape::from_dims(&[N]))
}

/// The two-armed detector.
fn assert_cow(a: &Arc<[f32]>, before: &[u32], s: &SharedHostStorage, expect: &[f32], what: &str) {
    assert_eq!(
        bits(a),
        before,
        "{what}: the SHARED Arc was mutated in place"
    );
    assert!(
        !s.is_shared(),
        "{what}: the storage still references the shared Arc after a write"
    );
    assert_eq!(
        bits(&read(s)),
        bits(expect),
        "{what}: the write did not land in the storage"
    );
    assert_eq!(
        Arc::strong_count(a),
        1,
        "{what}: the Arc was not released after the copy"
    );
}

#[test]
fn const_set_does_not_write_through_the_arc() {
    let a = src_arc();
    let before = bits(&a);
    let mut s = shared(&a);
    s.const_set_dyn(Scalar::F32(9.0), &lay()).unwrap();
    assert_cow(&a, &before, &s, &[9.0; N], "const_set_dyn");
}

#[test]
fn copy_strided_src_into_a_shared_destination_does_not_write_through_the_arc() {
    let a = src_arc();
    let before = bits(&a);
    let mut dst = shared(&a);
    let src = CpuStorage(HostBuffer::F32(vec![7.0; N]));
    src.copy_strided_src_dyn(&mut dst, 0, &lay()).unwrap();
    assert_cow(&a, &before, &dst, &[7.0; N], "copy_strided_src_dyn");
}

#[test]
fn copy2d_into_a_shared_destination_does_not_write_through_the_arc() {
    let a = src_arc();
    let before = bits(&a);
    let mut dst = shared(&a);
    let src = CpuStorage(HostBuffer::F32(vec![5.0; N]));
    // d1 rows x d2 cols, both strides contiguous: a full overwrite.
    src.copy2d_dyn(&mut dst, 2, 4, 4, 4, 0, 0).unwrap();
    assert_cow(&a, &before, &dst, &[5.0; N], "copy2d_dyn");
}

#[test]
fn scatter_set_and_scatter_add_set_do_not_write_through_the_arc() {
    // These two `&mut self` writers are NOT in the design note's list of four; found by reading the trait.
    let ids = CpuStorage(HostBuffer::U32(vec![1, 3]));
    let src = CpuStorage(HostBuffer::F32(vec![50.0, 60.0]));
    let il = Layout::contiguous(Shape::from_dims(&[2]));

    let a = src_arc();
    let before = bits(&a);
    let mut s = shared(&a);
    s.scatter_set_dyn(&lay(), &src, &il, &ids, &il, 0).unwrap();
    let mut want = a.to_vec();
    want[1] = 50.0;
    want[3] = 60.0;
    assert_cow(&a, &before, &s, &want, "scatter_set_dyn");

    let a = src_arc();
    let before = bits(&a);
    let mut s = shared(&a);
    s.scatter_add_set_dyn(&lay(), &src, &il, &ids, &il, 0)
        .unwrap();
    let mut want = a.to_vec();
    want[1] += 50.0;
    want[3] += 60.0;
    assert_cow(&a, &before, &s, &want, "scatter_add_set_dyn");
}

#[test]
fn the_as_any_mut_escape_hatch_does_not_write_through_the_arc() {
    let a = src_arc();
    let before = bits(&a);
    let mut s = shared(&a);
    {
        let any: &mut dyn Any = s.as_any_mut();
        let cpu = any
            .downcast_mut::<CpuStorage>()
            .expect("a CPU storage downcasts to CpuStorage");
        match &mut cpu.0 {
            HostBuffer::F32(v) => v[0] = -1.0,
            _ => unreachable!(),
        }
    }
    let mut want = a.to_vec();
    want[0] = -1.0;
    assert_cow(&a, &before, &s, &want, "as_any_mut");
}

#[test]
fn reads_do_not_copy_or_unshare_and_compute_matches_an_owned_buffer_bit_for_bit() {
    let a = src_arc();
    let before = bits(&a);
    let s = shared(&a);

    // The residency-OFF / cold-upload read: a copy OUT, nothing cached, still shared.
    assert_eq!(bits(&read(&s)), before);
    assert!(
        s.is_shared() && !s.is_materialised(),
        "a host read must not materialise or unshare"
    );
    assert_eq!(
        s.shared().unwrap().data_ptr(),
        a.as_ptr() as *const u8,
        "the storage aliases the Arc"
    );

    // A compute op materialises one view, once, and the storage is STILL shared.
    let owned = CpuStorage(HostBuffer::F32(a.to_vec()));
    let rhs = CpuStorage(HostBuffer::F32((0..N).map(|i| 0.1 * i as f32).collect()));
    for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Sub, BinaryOp::Div] {
        let x = s.binary_op_dyn(&rhs, &lay(), &lay(), op).unwrap();
        let y = owned.binary_op_dyn(&rhs, &lay(), &lay(), op).unwrap();
        assert_eq!(
            bits(&read(&*x)),
            bits(&read(&*y)),
            "{op:?}: shared vs owned differ"
        );
    }
    // And as the RIGHT operand (the rhs downcast goes through `as_any`).
    let x = rhs
        .binary_op_dyn(&s, &lay(), &lay(), BinaryOp::Add)
        .unwrap();
    let y = rhs
        .binary_op_dyn(&owned, &lay(), &lay(), BinaryOp::Add)
        .unwrap();
    assert_eq!(bits(&read(&*x)), bits(&read(&*y)));
    assert!(s.is_shared() && s.is_materialised());
    assert_eq!(bits(&a), before);
}

#[test]
fn cloning_a_shared_storage_shares_and_a_write_to_the_clone_leaves_the_original() {
    let a = src_arc();
    let s = shared(&a);
    let mut c = s.clone();
    assert_eq!(Arc::strong_count(&a), 3, "source + two storages");
    c.const_set_dyn(Scalar::F32(0.0), &lay()).unwrap();
    assert!(s.is_shared() && !c.is_shared());
    assert_eq!(
        bits(&read(&s)),
        bits(&a),
        "the original storage changed through its clone"
    );
    assert_eq!(read(&c), vec![0.0; N]);
}

#[test]
fn the_cpu_device_hands_out_shared_storage_and_the_trait_default_copies() {
    let a = src_arc();
    let st = CpuBackendDevice
        .storage_from_shared_host_dyn(SharedHostData::F32(Arc::clone(&a)))
        .unwrap();
    assert_eq!(
        Arc::strong_count(&a),
        2,
        "the CPU device must keep the Arc, not copy it"
    );
    drop(st);
    assert_eq!(Arc::strong_count(&a), 1);
    // The default (what CUDA/Metal/Vulkan inherit) is the deep copy.
    let d = SharedHostData::F32(Arc::clone(&a)).to_host_buffer();
    assert_eq!(
        Arc::strong_count(&a),
        1,
        "to_host_buffer must hold no Arc once the temporary is gone"
    );
    assert!(matches!(d, HostBuffer::F32(v) if v.as_ptr() != a.as_ptr()));
}
