// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 173 (d): building a graph `Const` from a model's weight `Arc` must not copy the weights.
//!
//! Measured with a counting global allocator: the bytes of every allocation >= 64 KiB made
//! while the graph is built. Before the change every `NodeHandle::from_f32(arc, ..)` did
//! `Arc<[f32]>::to_vec` (1,508 MB for the Qwen3-0.6B prefill, 315 weight arrays); this test
//! uses 12 synthetic 1 MiB tensors so the arithmetic is exact: a copying build allocates
//! >= 12 MiB, a sharing build allocates ~0.

use fuel_graph::NodeHandle;
use fuel_ir::Shape;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

static COUNTING: AtomicBool = AtomicBool::new(false);
static BIG_BYTES: AtomicUsize = AtomicUsize::new(0);
const BIG: usize = 64 * 1024;

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.size() >= BIG && COUNTING.load(Ordering::Relaxed) {
            BIG_BYTES.fetch_add(l.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if n >= BIG && COUNTING.load(Ordering::Relaxed) {
            BIG_BYTES.fetch_add(n, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static A: Counting = Counting;

/// One counted section at a time: the counter is process-global.
static SERIAL: Mutex<()> = Mutex::new(());

fn cpu() -> Arc<dyn fuel_backend_contract::DynBackendDevice> {
    Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice)
}

const TENSORS: usize = 12;
const ELEMS: usize = 256 * 1024; // 1 MiB of f32

fn weights() -> Vec<Arc<[f32]>> {
    (0..TENSORS)
        .map(|t| {
            (0..ELEMS)
                .map(|i| {
                    f32::from_bits(
                        0x3f80_0000 ^ ((i as u32).wrapping_mul(2654435761) >> 9) ^ t as u32,
                    )
                })
                .collect::<Vec<f32>>()
                .into()
        })
        .collect()
}

#[test]
fn building_arc_fed_consts_copies_no_weight_bytes() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let w = weights();
    let dev = cpu();
    let payload = TENSORS * ELEMS * 4;

    BIG_BYTES.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let handles: Vec<NodeHandle> = w
        .iter()
        .map(|a| NodeHandle::from_f32(Arc::clone(a), Shape::from_dims(&[ELEMS]), &dev).unwrap())
        .collect();
    COUNTING.store(false, Ordering::SeqCst);
    let copied = BIG_BYTES.load(Ordering::SeqCst);

    println!(
        "COPY BYTES: {copied} of {payload} payload bytes ({} MiB payload)",
        payload >> 20
    );
    assert!(
        copied < payload / 100,
        "building {TENSORS} Arc-fed Consts allocated {copied} bytes >= 64 KiB; the payload is {payload}: \
         the weights were copied"
    );

    // The graph still holds the data: each slot reads back bit-equal.
    for (h, a) in handles.iter().zip(&w) {
        let slot = h.storage_for().expect("a Const has a slot");
        let host = slot.read().unwrap().as_dyn().to_host_buffer_dyn().unwrap();
        match host {
            fuel_ir::HostBuffer::F32(v) => assert!(
                v.iter()
                    .zip(a.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "slot bytes differ from the source"
            ),
            other => panic!("expected F32, got {:?}", other.dtype()),
        }
    }
}

/// The control: the copying path is measurable by this very instrument (an owned `Vec` goes in,
/// so the storage holds a private copy only when the caller made one). Without it, "0 bytes"
/// could mean the counter is dead.
#[test]
fn the_counter_sees_a_real_copy() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let w = weights();
    BIG_BYTES.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let copies: Vec<Vec<f32>> = w.iter().map(|a| a.to_vec()).collect();
    COUNTING.store(false, Ordering::SeqCst);
    let seen = BIG_BYTES.load(Ordering::SeqCst);
    assert!(
        seen >= TENSORS * ELEMS * 4,
        "counter saw only {seen} bytes of a {} byte copy",
        TENSORS * ELEMS * 4
    );
    drop(copies);
}

/// The graph holds one strong reference per Const and releases it with the graph: the
/// observable half of "what keeps the host copy alive after the device upload".
#[test]
fn a_const_holds_exactly_one_strong_reference_until_the_graph_drops() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = cpu();
    let a: Arc<[f32]> = vec![1.0f32; 1024].into();
    assert_eq!(Arc::strong_count(&a), 1);
    let h = NodeHandle::from_f32(Arc::clone(&a), Shape::from_dims(&[1024]), &dev).unwrap();
    assert_eq!(
        Arc::strong_count(&a),
        2,
        "the shared Const must hold the source Arc (1 clone)"
    );
    drop(h);
    assert_eq!(
        Arc::strong_count(&a),
        1,
        "dropping the graph must release the source Arc"
    );
}
