# Seam S3: executor (fuel-tensor, fuel-dispatch) and the backends

ROADMAP row 137, first seam. Method from `2026-11-01-seams-and-cleanup-program.md`; numbers from
`scripts/structure_report.py` (definitions in `2026-11-01-structure-measurement.md`) at `origin/main`
`4569a530b` (0.35.14, script) and `5602a2c53` (0.35.15, the greps named below). Written 2026-11-01. Nothing here has been changed
yet; section 6 is the proposed order of small PRs. Leads are marked as such where they are not
measured.

## 1. What crosses the seam (method step 1)

LEFT = `fuel-tensor` (tier 100) and `fuel-dispatch` (tier 80); RIGHT = the seven backend crates
(tiers 40 to 90). Measured: LEFT uses RIGHT at 874 sites over 551 distinct items in 84 files
(top: `CudaDevice` 91, `CudaStorageBytes` 50, `VulkanStorageBytes` 46, `VulkanBackend` 29,
`CpuStorageBytes` 26); RIGHT uses LEFT at 33 sites over 11 items (`fuel_dispatch::kernel::KernelCaps`,
`OpParams`, `KernelBindingTable`: the vendor backends at tier 90 sit above dispatch at tier 80, so
this is the permitted direction). Up-tier sites: 0. Co-change: 19 of 102 executor-side commits also
touch a backend (19%), so the two sides already change mostly independently.

The seam has THREE parts, and they are different kinds of coupling:

1. **Storage.** `fuel_memory::BackendStorage` is a closed enum, one cfg-gated variant per backend
   (`Cpu`, `Cuda`, `Vulkan`, `Metal`), wrapped by `Storage` (bytes plus dtype tag). The executor
   matches on it directly: `BackendStorage::` appears 184 times in `fuel-dispatch/src/pipelined.rs`
   alone and 139, 63 and 37 times (Cpu, Cuda, Vulkan) across `fuel-dispatch/src`, 19, 3, 3 and 1
   across `fuel-tensor/src`.
2. **Devices and handles.** Finding the device handle a kernel needs: `find_cuda_device_in_cache`
   and `find_vulkan_backend_in_cache` (`pipelined.rs`, 9 call sites) search the executor's
   storage cache for any storage on the target backend and take its handle;
   `seed_extra_device_handles`, `seed_placed_device_handles` and `device_seed_storage`
   (`pipelined_bridge.rs`) exist to put such a storage in the cache by pushing a dummy `Op::Const`
   anchor node into the user's graph.
3. **Identity of a backend and a place.** `BackendId` (kind of backend) and `DeviceLocation`
   (kind plus ordinal) are both in `fuel-ir`, and the conversion between them is written out by hand
   in many places (section 3).

## 2. Two abstractions for the same thing (the main finding)

`fuel-backend-contract` defines trait objects for exactly this seam: `DynBackendStorage`,
`DynBackendDevice`, `BackendRuntime`, `BackendStreams`. They are implemented: `DynBackendStorage` for
the CPU, CUDA and Metal storages, `DynBackendDevice` for all four devices, `BackendRuntime` for the
CUDA and Vulkan devices and (in `fuel-tensor`) a `DeviceRuntimeHandle` wrapper. The Vulkan device's own source
says it "does not return DynBackendStorage objects". And the hot path does not use them:
`DynBackendStorage` appears once in `fuel-dispatch/src`, zero times in `fuel-tensor/src`.

The storage design (`docs/storage-unification.md`) chose the OTHER shape: a closed enum plus a
`BackendStorageOps` trait with convenience methods on the enum, so callers ask `storage.len_bytes()`
or `storage.copy_from(src)` and do not match. `BackendStorageOps` does not exist in the code (no
hit in any `.rs` file). So the enum shipped without the methods that were meant to keep callers
from matching on it, and a second, trait-object abstraction sits beside it, implemented but unused
by the executor.

This is the root of the measured backend-name counts: the executor has nothing to call except the
variants. Either abstraction would remove most of them; having both removes none. Which one is
intended is an architecture decision (section 7).

## 3. Leaks found by reading (method step 3)

- **Copied mapping tables.** `DeviceLocation -> BackendId` is written four times with the same four
  arms (`fuel-dispatch/src/optimize.rs`, `pipelined.rs` inline, `plan.rs` as `backend_for_device`,
  `fuel-tensor/src/pipelined_bridge.rs`). `BackendId -> DeviceLocation { gpu_id: 0 }` is written
  four times (`dispatch.rs`, `plan.rs` `default_device_for`, `topology.rs`, `fuel-graph/src/opt.rs`),
  all with the same arms and the same `_ => DeviceLocation::Cpu` fallback. Both types live in
  `fuel-ir`, so one definition there serves every crate. The copies agree, which is why this is
  behaviour-preserving; the `gpu_id: 0` default is also the place that has to change when devices
  are named by identity (row 125).
- **Backend-specific control flow in generic code.** `pipelined_bridge.rs` has a Vulkan branch in
  the realize path (`matches!(target_loc, DeviceLocation::Vulkan { .. })`), a Vulkan special case in
  `device_for_location` that returns an error telling the caller to use a different entry point, and
  per-backend arms in `device_seed_storage`. `fuel-dispatch/src/jit_ingest.rs` carries 70
  `cfg(feature = ...)` lines (the JIT path is CUDA-only today; not read in detail).
- **Handle discovery by searching a cache.** A handle is found by scanning the storage cache for
  any storage whose backend matches (and, since the row 114 fix, whose device matches). That makes
  "a device exists" a property of "some tensor happens to live on it", which is why the seeding
  anchors exist.
- **Index as identity.** `gpu_id` is a within-process ordinal and its order differs between builds
  on the development box (row 125). The `gpu_id: 0` defaults above assume otherwise.
- **Public internals.** `VulkanBackend` exposes its physical device as a public field; the Vulkan
  backend's `lib.rs` is about 14,000 lines.

## 4. What each side needs from the other (method step 4)

The executor needs, per device: allocate, upload, download, copy to another device, report free and
total memory, and a way to run a kernel it already holds a `KernelRef` for. It does not need to know
which backend it is talking to for any of those. The backends need, from the executor: the kernel
call signature (`KernelRef`, `OpParams`, layouts), capabilities and cost reporting. That is already
in `fuel-backend-contract` and `fuel-dispatch::kernel`; the second half of the seam is in order.
The first half (storage and device handles) is the part with no single interface in use.

## 5. The gap (method step 5)

| need | designed | exists | used by the executor |
| --- | --- | --- | --- |
| storage queries and copies | `BackendStorageOps` methods on the enum | no | no (matches) |
| per-device handle | `DynBackendDevice` | yes, 4 devices | 2 lines in `fuel-dispatch`, 16 in `fuel-tensor` mention it |
| storage as an object | `DynBackendStorage` | yes, 3 backends | 1 use |
| a registry of live devices | none designed | no | cache search plus anchor nodes |
| kind-of-backend to place and back | none | 8 copies | 8 copies |

## 6. Proposed order of PRs (method step 6)

Each is small, behaviour-preserving unless stated, red-first where behaviour could change, and
re-measured with `structure_report.py` (cfg lines and backend-path uses for the touched crate, and
the S3 item count).

1. **One mapping in `fuel-ir`:** `DeviceLocation::backend()` and `BackendId::default_location()`
   (gpu 0, CPU for a backend it does not know), replacing the eight copies. Round-trip and
   every-variant tests. Smallest and safest; also the one place the `gpu_id: 0` default will later
   change.
2. **Give `BackendStorage` the designed methods** (`len_bytes`, `location`, and the copy
   operation) and replace the `pipelined.rs` matches that only ask a question, not the ones that
   dispatch a kernel. Measure the drop in `BackendStorage::` uses.
3. **A device registry for one realize,** keyed by `DeviceLocation`, handed to the executor, in place
   of cache search, `seed_extra_device_handles`, `seed_placed_device_handles` and the anchor nodes.
   This changes behaviour at the seeding boundary and needs the row 114 two-GPU probe as its
   oracle (it failed on exactly this code before the fix).
4. **Identity in place of index** for the places that choose a device (row 125's selector).
5. **Park the trait-object alternative** (section 7): leave `DynBackendStorage` and the Vulkan
   exception as they are and decide deletion later, with reachability output.
6. **Move backend-gated code out of `fuel-dispatch`** (the JIT ingest path first) behind the
   backend crates' trait implementations.

Steps 1 and 2 need no decision; the storage decision (section 7) is made.

### 6b. Step 3 broken into parts (written 2026-11-02, measured at `origin/main` 79b2a83c2, 0.36.7)

Where the code stands: `find_cuda_device_in_cache` and `find_vulkan_backend_in_cache` (`fuel-dispatch/src/pipelined.rs`)
each scan the whole storage cache for a storage on the wanted GPU and take its handle; four call sites use them (the
`Op::Alloc` path and one other path, for CUDA and for Vulkan). The cache is filled for devices that hold no tensor by
`device_seed_storage` (`fuel-tensor/src/pipelined_bridge.rs`), called from `seed_extra_device_handles`,
`seed_placed_device_handles`, `inference_context.rs` (3 places) and `kv_block_pool_device.rs`.

One fact that shapes the order: a lookup runs while the realize is in progress, and the cache gains storages during it,
so a device can become findable part-way through (a copy to a GPU creates a storage on it). A registry built once from
the cache at the start would change behaviour for that case. So the first part must keep the answer identical to a scan.

- **3a. The registry type, no behaviour change.** `DeviceRegistry` in `fuel-dispatch`: a map from `DeviceLocation` to a
  handle (`Cuda(CudaDevice)` or `Vulkan(Arc<VulkanBackend>)`), filled lazily: a lookup tries the map, then scans the
  cache exactly as today and remembers the first match. The four call sites ask the registry. Test: for a cache with
  a storage on each of two locations the registry returns the same handle as the old scan, and returns `None` where the
  scan did (CPU-only unit test with fake locations is not possible for the handle types, so the live two-GPU probe
  from row 114 is the oracle, run through `gpu-run`).
- **3b. Hand the registry in.** The realize entry points take an optional registry; the bridge fills it from the
  `Device` handles it already holds; with a registry supplied the anchor nodes are no longer pushed. The row 114 probe
  decides.
- **3c. Delete the seeding.** `seed_extra_device_handles`, `seed_placed_device_handles`, `device_seed_storage` and the
  "caller must seed the cache" error text.

Each part is its own PR with its own version. 3a changes no output; 3b and 3c change the seeding boundary.

### 6c. Step 4, identity in place of index (written 2026-11-02 at `origin/main` 05e4a234, 0.37.0; reading only, no code)

What is already there (ROADMAP row 125, step one, 0.35.17): `fuel_ir::DeviceIdentity { uuid, pci }` with `same_physical_device`
(UUIDs decide when both sides have a usable one; an all-zero UUID counts as absent; the PCI address is the fallback; unknown is
never "same"), `VulkanBackend::identity()`, `fuel_vulkan_backend::probe::device_identities()`,
`fuel_cuda_backend::probe::device_identities()`, and `DeviceMemory.identity` in the device inventory. Measured on the
development box in that step: the CUDA device and the Vulkan RTX 4070 report the same UUID and PCI address, and the Vulkan
enumeration order differs between builds (index 0 was the AMD iGPU in Vulkan-only builds and the RTX 4070 in CUDA-plus-Vulkan
builds). What is not there: anything that CHOOSES a device by identity, and anything that keys state by identity.

Where an ordinal is the key or the selector today (counted with `grep` over tracked `fuel-*/src` files, so a size, not a design):
the token `gpu_id` appears on 138 lines in 18 files of `fuel-dispatch`, 50 in 3 files of `fuel-graph`, 17 in 3 of `fuel-ir`, 15 in 2 of
`fuel-tensor`, 11 in 3 of `fuel-vulkan-backend`, 10 in 2 of `fuel-memory`, 9 in 3 of `fuel-hardware`, 6 in 3 of `fuel-cuda-backend`
and 3 in 2 of `fuel-metal-backend`. Constructors that take an ordinal: `fuel_tensor::cuda_backend::new_device(ordinal)`,
`DeviceSelection::Index` in `fuel-hardware/src/inventory.rs`, `transfer_cost.rs` and `fuel-tensor/src/factories.rs`, and the
two Vulkan example binaries. In-process maps keyed by `DeviceLocation` (which carries the ordinal): `DeviceRegistry` (this
step's subject), the transfer-path table `HashMap<(DeviceLocation, DeviceLocation), TransferPath>` and the in-flight counters
`HashMap<DeviceLocation, AtomicU32>` in `dispatch.rs`, and the optimizer's `placements: HashMap<NodeId, DeviceLocation>` in
`optimize.rs`. All of these live for one process. The one cross-run store I checked is the judge cache, and it is NOT keyed by an ordinal: `fuel_ir::probe::EquivalenceKey` is `(backend, vendor_id, device_id, compute_capability, driver_version)`, a hardware class plus the driver release (its own doc says the driver is in the key so a driver upgrade invalidates the profile), and two identical cards share one class by design while `DeviceDescriptor.device_index` stays an ordinal on the descriptor, outside the key. So 4c is not needed for the judge cache. Any other persisted state keyed by an ordinal I did not look for beyond the judge cache and the maps above (**UNREAD**).

What "identity in place of index" can mean, from least to most invasive:
- **4a. Choose by identity.** A selector (`DeviceSelection::Identity` for Vulkan, a CUDA counterpart, and an inventory lookup
  `location_of(identity)`) resolves an identity to the ordinal of THIS process, once, at device creation; everything downstream keeps
  using `DeviceLocation { gpu_id }` as the in-process handle. Fixes "name the MI50" and the order flip between builds. Oracle: the
  live inventory test (every CUDA device matches exactly one Vulkan device), a synthetic-enumeration test of the resolver in both
  orders, and a live test that creating by identity returns the same physical device in a CUDA-plus-Vulkan and a Vulkan-only build.
- **4b. Register by identity.** `DeviceRegistry::with_*` also records the handle's `DeviceIdentity`, and `registered_*` can be asked by
  identity, so a caller holding an identity does not need the ordinal. No change to `DeviceLocation`. Oracle: the row 114 probe
  with the two GPUs registered in both orders.
- **4c. Key persisted state by identity.** Only if the profile or judge cache is keyed by ordinal (open, see above). Changing
  `DeviceLocation` itself to carry an identity would touch the ~250 lines above and the published `fuel-ir` type, a breaking
  change; not proposed.

Recommendation: 4a then 4b, each its own PR with the probe plus a sabotage as oracle and a hand CUDA compile (no runner, board
item 366); 4c is not needed for the judge cache (above); look for other persisted ordinal keys before dropping it entirely.

### 6d. Step 5, the trait-object alternative: evidence and decision (written 2026-11-03 at `origin/main` 043de464, 0.37.18)

Section 7 rules that `DynBackendStorage`, `DynBackendDevice` and the Vulkan exception stay as they are, and
that deletion is decided later with reachability output in hand. The PM's instruction of 2026-11-03: count
callers across every workspace crate (with a positive control) and the reverse dependencies of
`fuel-backend-contract`; if there are no callers and no outside dependents, delete as a Breaking change,
else keep and record why. This is that evidence. **Decision: keep.**

**1. The name, code only (comments stripped; a positive control: the same scan finds the impls in the
backend crates).** `DynBackendStorage` appears in `fuel-backend-contract` (127), `fuel-cpu-backend` (62),
`fuel-cuda-backend` (62), `fuel-metal-backend` (61) and `fuel-vulkan-backend` (8), and in no other crate; the
five other mentions are comments (`fuel-dispatch/src/pipelined.rs`, `fuel-graph/src/lib.rs`,
`fuel-ir/src/{lib,storage,backend}.rs`). Section 2's "1 use in `fuel-dispatch`" was a doc comment.

**2. But the trait is reached without its name, and that decides it.** `DynBackendDevice` is live (code
uses: `fuel-graph` 23, `fuel-tensor` 8, `fuel-dispatch` 2, `fuel-inference` 2, `fuel-lazy-examples` 2), and
three of its methods return `Box<dyn DynBackendStorage>`: `zeros_impl_dyn`, `storage_from_host_buffer_dyn`
and `storage_from_host_buffer_owned_dyn`. `fuel-graph` calls `storage_from_host_buffer_owned_dyn` at three
production sites (`NodeHandle::from_*` at lib.rs:3980, `const_like` at 4226, `build_filled_const` at 12149;
a fourth call, at 14921, is in the test module), wraps each result with `Storage::from_dyn` and stores it as a
constant node's data; `allocate_bundled_storage` has 7 more uses in `fuel-graph` (production and tests not
separated), and `fuel-tensor` and `fuel` import
the `fuel_backend_contract::Storage` handle. So building a tensor from host data goes through
`DynBackendStorage`. The executor does not (it matches the closed enum); the graph's host-data boundary does.
`DynQuantizedStorage` and `QuantizedDeviceKernels` are also live (`fuel-loaders`, `fuel-quantized`).

**3. Dependants.** crates.io lists 12 reverse dependencies of `fuel-backend-contract` (latest 0.36.7): all are
Fuel workspace members. On this machine the only other project that depends on it is Lightbulb, which uses
`fuel_backend_contract::backend::BackendRuntime` and nothing else (same scan over every sibling project under
`C:\Projects`; the positive control finds the Fuel uses). Third parties off crates.io and off this machine
were not looked for (the crate has 2263 downloads in total).

**4. CodeRipper 0.20.1 `check reachability`** on `fuel-backend-contract` alone reports 4 items with zero
callers inside the crate: `BackendCapabilityProvider`, `InplaceOp1`, `InplaceOp2`, `InplaceOp3`;
`DynBackendStorage` is not reported. The check is per crate and reads the committed tree. It fails on the
repository root (a virtual workspace); the CI portfolio log holds no reachability findings for these types.

**Why keep.** Deleting `DynBackendStorage` means re-plumbing `NodeHandle::from_*` and the graph's constant
nodes (and `allocate_bundled_storage`, the `Storage` handle, and the quantized traits that take it), in a published crate, while the executor's own path already avoids it. That is a design change with no
consumer asking for it, not a cleanup; it belongs to a future decision on the graph's storage type, not to
row 137.

**The four unreached items (decision of the PM, 2026-11-03).** `InplaceOp1/2/3`, `BackendCapabilityProvider`:
zero code uses outside the contract crate in the workspace (same code-only scan), no backend implements them,
and no outside project names them. **Deleted: `InplaceOp1/2/3` and the dead `Storage::inplace_op1/2/3`
delegates** (0.38.0, Breaking; `inplace_op.rs` mentioned `DynBackendStorage` 15 times, comments included).
Checked by compiling the 10 workspace crates that depend on the contract crate (`cargo check --all-targets`,
exit 0); the CUDA and Metal backends were not compiled, the text census covers them. **Kept:
`BackendCapabilityProvider`.** It has no implementation and no caller, and that is known and deliberate: the
architecture document (`docs/architecture/05-backend-contract.md`, Tier 1) specifies it, the dlpack extension
spec depends on it, and `fuel-ir/tests/doc_declared_traits.rs` uses it as its positive control. Deleting it
would need a constitution amendment, which is CireSnave's call and was not asked. I had reported it as
"unreached" before checking the architecture documents; the code census was right and was not the whole picture.

## 7. Decision (2026-11-01, CireSnave, relayed by the PM)

The question was whether the storage abstraction is (a) the closed enum plus ops methods, or (b)
trait objects. **Ruled: (a).** Keep the closed `BackendStorage` enum and write the
`BackendStorageOps` methods the design specifies; revisit trait objects only if backends are ever
to live outside Fuel. Row 138 may proceed on that basis.

What follows for this seam:

- Step 2 (the designed methods on `BackendStorage`) is now the main storage cleanup and needs no
  further decision. It is how the 184-site match surface shrinks.
- Step 5 changes: `DynBackendStorage` and the Vulkan exception to it are not adopted. They stay as
  they are and are recorded as the parked alternative; whether they are deleted is a separate
  decision, to be made with CodeRipper's reachability output in hand (it already reports several of
  these traits as having no callers).
- The seam contract for row 138 is therefore the enum plus the ops methods plus the kernel call
  signature, which means the backend crates stay in the Fuel workspace. A split of the backends
  into separate projects is off the table under this ruling; the kernel-runner seams that already
  exist (the `fuel-kernel-seam` crates) are unaffected.
- The leak list in section 3 still stands. The device registry (step 3) and identity in place of
  index (step 4) are independent of this ruling.

## 8. Limits

Read-through of the files named, not a full audit; `jit_ingest.rs`, `plan.rs` beyond the mapping
tables, and `scheduling.rs` were counted but not read; counts include `#[cfg(test)]` code in `src`;
the 184-site figure counts any line containing `BackendStorage::` (patterns, constructors and
returns alike) without classifying them.
