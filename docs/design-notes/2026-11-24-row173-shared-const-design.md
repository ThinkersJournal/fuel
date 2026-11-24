# Row 173 (d): share the source `Arc` in a graph `Const` instead of copying it (DESIGN ONLY, 2026-11-24)

Lane: fuel. Docs only: nothing here is built, and no hot-path code was changed. Inputs: the measurements in
`2026-11-23-qwen3-prefill-attribution.md` (a warm Qwen3-0.6B Q4_K_M prefill spends 312 ms of `d1.build_graph` copying
315 weight arrays, 1,508 MB, and ~93 ms freeing the copies; 68% of a 590 ms prefill) and a read of the code at
`origin/main` `58debefec` (every count below is `git grep` over tracked `*.rs` files at that ref, with a control).

## 1. What happens today

`fuel-graph` `NodeHandle::from_f32` (and `from_f64`, `from_bf16`, `from_f16`, `from_u32`, plus the `*_on` siblings:
15 call sites, lines ~3990 to ~4180 of `fuel-graph/src/lib.rs`) takes an `Arc<[T]>`, records a weak `ConstOrigin` (row 172)
and then does `HostBuffer::F32(v.to_vec())`. The `Vec` goes to `from_host_buffer_on`, which calls
`device.storage_from_host_buffer_owned_dyn(buf)`; on the CPU device that is `Box::new(CpuStorage(buf))` (`fuel-cpu-backend/src/dyn_impl.rs`
`pub struct CpuStorage(pub HostBuffer)`), put in an `Arc<RwLock<Storage>>` slot of the graph. So the graph owns a private
host copy of every weight, and the source `Arc` (the model's own copy) is untouched.

Who reads that private copy: (a) residency OFF, or the first (cold) upload with residency ON: the realize reads the
slot through `to_host_buffer_dyn()`, which is `Ok(self.0.clone())`, i.e. ANOTHER full copy, and uploads it;
(b) a CPU realize that actually computes on the Const reads the `CpuStorage` in place; (c) nothing else, in particular **not a warm call with
residency ON**: the realize cache is seeded with the resident device storage, so the slot is never read and the
copy is built, held and freed for nothing.

## 2. Shape of an `Arc`-backed variant (three options)

**A. A shared arm per dtype in `HostBuffer`** (`F32Shared(Arc<[f32]>)`, and the same for F64, BF16, F16, U32). Population
that sees the enum: the definition is `fuel-ir/src/cpu_storage.rs` (19 `HostBuffer::` mentions in that file, the control that the
grep finds the definition); `HostBuffer::` appears on **492 lines in 22 files** (`fuel-cpu-backend/src/dyn_impl.rs` 249,
`fuel-tensor/src/pipelined_bridge.rs` 32, `fuel-cuda-backend/src/device.rs` 32, `fuel-metal-backend/src/storage.rs` 31,
`fuel-cpu-backend/src/ops.rs` 30, `fuel-graph/src/lib.rs` 29, ...), of which 65 lines in 18 files name `HostBuffer::F32(`;
the borrowed twin `HostBufferRef::` is on 105 lines in 7 files. Every exhaustive `match` over the enum needs 5 more arms or a
shared-to-owned normaliser: wide, compile-driven, touches every backend.

**B. Change what the existing arms hold:** `F32(Buf<f32>)` with `enum Buf<T> { Owned(Vec<T>), Shared(Arc<[T]>) }`,
`Deref<Target=[T]>`, `to_mut()` that clones on first write (what `Arc::make_mut` does for `CpuStorageBytes`). Constructors
(`HostBuffer::F32(vec)`) become `HostBuffer::F32(vec.into())`, readers that use `&Vec<f32>` as a slice keep working through
`Deref`, anything that uses `Vec`-only API (`push`, `into_iter`, moving the `Vec` out) fails to compile and is the work list.
Same 22 files, but a smaller number of lines than A (no new arms), at the price of a type change in a public enum.

**C. A new storage type, enum untouched:** `SharedHostStorage` (holds `Arc<[T]>` per dtype, an `enum` inside the type), created by
a NEW defaulted method on `DynBackendDevice` (`storage_from_shared_host_dyn(&self, Arc<..>)`) whose default is today's
behaviour (`to_vec` then `storage_from_host_buffer_owned_dyn`), overridden only by the CPU device. CUDA, Metal and Vulkan
(their `storage_from_host_buffer_owned_dyn` impls: 4 files besides the contract trait) keep the default and do not change.
`HostBuffer` and its 492 uses are not touched. Cost moves to the storage's trait methods: it must implement
`DynBackendStorage` (read paths trivially; `to_host_buffer_dyn` returns an owned `HostBuffer` = a copy, which is exactly the
residency-OFF/cold read in section 1(a), so that path is no worse than today).

## 3. What must not mutate a shared buffer

A shared `Arc<[T]>` is read-only memory (it is the model's weights). The in-place writers on `CpuStorage(HostBuffer)` are
the `&mut self` methods of `DynBackendStorage` in `fuel-cpu-backend/src/dyn_impl.rs`: `copy_strided_src_dyn`
(`cpu_copy_strided_src(src, dst: &mut HostBuffer, ..)`), `copy2d_dyn` (`cpu_copy2d`), `const_set_dyn` (`cpu_const_set`)
and `as_any_mut` (an escape hatch: a caller can downcast and mutate; 5 `downcast_ref/mut::<CpuStorage>` uses in 3 files,
of 43 `downcast_ref::<` uses in the tree, all but 2 of those 5 in `fuel-cpu-backend`). These are destinations: a weight `Const` is
never a destination in the model code read here, but KV-cache `Const`s are written (WriteSlice), and nothing in the type
system says a given `Const` is a weight. So the rule is copy-on-write at those four sites (take the owned `Vec` on first write, exactly
the contract `CpuStorageBytes` documents: "if shared, the bytes clone before mutation"), with a test that writes through each
and checks the source `Arc` is unchanged. The `ConstOrigin` pin (row 172) already stops `Arc::get_mut` on the source.
Two further constraints: `CpuStorageBytes` (`Arc<AlignedBytes>`, 64-byte aligned) cannot ADOPT a source `Arc<[f32]>`
(4-byte aligned) without a copy, so the shared storage must stay a separate type; and `HostBuffer: Clone` stays a deep copy.

## 4. Residency OFF, memory

With residency OFF the upload reads the host copy, so sharing saves the `to_vec` at build (312 ms) but the read at upload
still goes `to_host_buffer_dyn` (a clone) unless the upload path is taught to read the shared slice directly
(`upload_host_buffer_to_device` takes a `HostBuffer` today). So the lever is **ON-only as designed above**; making OFF
benefit needs the upload to accept a slice, one more signature change in `fuel-tensor/src/pipelined_bridge.rs` (32
`HostBuffer::` lines there). Memory: removes a transient 1.5 GB (1,508 MB measured) host allocation per prefill call;
nothing new is retained (the storage holds an `Arc` clone of memory the model already holds). The 622 MB f32 embedding table
is the largest single item, and also the one a CPU realize would read in place.

## 5. The red-first test

On the Vulkan box: `qwen3_vulkan_profile` with `QWEN3_PROFILE_PREFILL=1` on the attribution tree. RED today:
`PREFILL#1 const to_vec copies: 315 calls, 1508 MB`, `d1.build_graph` ~315 ms, `d1.drop_graph` ~93 ms. GREEN means: the same
print reads ~0 MB for the weight arrays (the counter in `timed_const_copy` only counts what still goes through `to_vec`),
`d1.build_graph` falls to the 2.8 ms of non-copy work, `d1.drop_graph` to the small-free remainder (~6 ms), AND the logits FNV
hash is identical to the unmodified build (3 processes, cold + warm, as in the residency PR), AND a unit test: a `Const`
built from an `Arc<[f32]>` shares its pointer (`Arc::ptr_eq` on the storage's slice), a `copy_strided_src_dyn` / `copy2d_dyn` / `const_set_dyn`
on it leaves the source `Arc` bytes unchanged and gives the destination the written values (the COW sabotage: remove the COW and the source changes).
A CPU-device realize of a graph with a shared Const must match the old result bit for bit.

## 6. Recommendation and cost

Option C, ON-only first. Reason: it is the only option whose blast radius is one defaulted trait method plus one CPU storage
type, leaves the 492 `HostBuffer::` uses and the CUDA, Metal and Vulkan backends untouched, and falls back to today's
behaviour by default (revertable by deleting the override). Cost **M**: about 300 to 450 lines (the storage type with its read
trait methods, 4 COW sites, the constructor change in `fuel-graph` from `to_vec` to the new method at the 15 sites, the tests above),
one PR, plus a decision on whether the constructors' `ConstOrigin` weak record and the storage's `Arc` clone interplay with the
residency pin (a strong clone in the storage would make an "evicted" model's weights stay alive through the graph: the storage must
hold the SAME pin semantics, which needs a short design check before code). Option B is **L** (public enum type change across 22
files) and option A **M-L** with every exhaustive match touched. Making residency OFF benefit adds **S** (the upload takes a slice).
Expected gain on the measured predicate: up to ~400 ms of a 590 ms warm prefill (312 build + ~90 drop), i.e. about 0.59 s to
about 0.2 s, NOT yet shown: the 0.2 s is arithmetic from the attribution tables, to be replaced by the red/green run in section 5.

Not decided here (PM): whether this is worth M before the transfer-calibration probe (cold 1 s, row 173 (c)) and `run_lockstep`
(49 ms every call), which are cheaper levers on other terms.
