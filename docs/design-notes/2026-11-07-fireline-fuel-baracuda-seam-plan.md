# Fuel half of the FireLine seam plan: the Fuel <-> baracuda (kernel-synthesizer) interface

**Status:** one-page input to FireLine's seam inventory (FireLine HANDOFF task 1/2; board 295). Measured at `origin/main@28e0aef6` (0.43.1) on 2026-11-07, plus baracuda's working tree read-only (`crates/baracuda-seam`, `crates/baracuda-cuda-seam`). The baracuda half is the baracuda lane's to write; every baracuda fact below is read from its tree and is marked for that lane to confirm. Nothing here is code.
**Asked by:** the PM for CireSnave, 2026-11-07. Supersedes the "own version line" proposal in `2026-10-24-seam-crate-versioning.md` (option B): FireLine's per-crate version line is that idea at project scale.

## 1. The interface today (three Fuel crates, 4,593 lines)

| Fuel crate | What it is | KISS sub-standard it carries | Fuel users outside the three |
|---|---|---|---|
| `fuel-kernel-seam-announce` | frozen `#[repr(C)]` `SeamHello` envelope, `SEAM_MAGIC`, `SEAM_CAP_*` bits, `Role`, `negotiate_with_role` (dependency-free) | Announce | **none** (0 `.rs` files, 0 Cargo dependents) |
| `fuel-kernel-seam-types` | `OpTag`, `OpAttrs` (+ canonical bytes, `UnresolvedAttr`), `RegionKey`, `PatternNode`, `shape_expr` (typed shape AST + wire codec + evaluator) | Grammar (+ Ops §6.20 for `shape_expr`) | 31 `.rs` files: `fuel-graph` 25, `fuel-dispatch` 6, `fuel-kiss-ref-backend` 4 |
| `fuel-kernel-seam` | `Synthesizer` trait, `JitRequest`/`JitBudget`/`JitResponse`, `SynthArtifact`, `ArtifactKind`, `LinkEntry` (live-call envelope; deliberately has no graph/dispatch dependency) | Synth/Provision | 6 `.rs` files (`fuel-dispatch`, `fuel-graph`) |

Consumer on the other side: `baracuda-cuda-seam` requires `fuel-kernel-seam{,-types}` `^0.43` and `unpopped-vocab`; `baracuda-seam` carries **its own copy** of `SeamHello`, `SEAM_MAGIC`, `SEAM_ENVELOPE_VERSION`, `SEAM_MAX_PROFILES` and the capability bits (names differ: `SEAM_CAP_DLPACK_EXT_*` vs Fuel's `SEAM_CAP_FDX_*`; `SEAM_CAP_JIT_ON_REQUEST` vs `SEAM_CAP_PROVISION_ON_REQUEST`, both `1 << 32`). **That is two sources of truth today**, held together only by each side's own size/offset asserts. Baracuda lane: confirm.

## 2. What moves into FireLine (one crate per interface)

Recommendation: keep the three-way split, because it already mirrors KISS's sub-standards and the announce crate is deliberately dependency-free so a provider that only negotiates never pulls the grammar.

| FireLine crate (names are FireLine's call) | Takes | Retires |
|---|---|---|
| announce | whole of `fuel-kernel-seam-announce` **and** baracuda-seam's `SeamHello`/magic/caps copy, one name per capability bit | the duplicate in `baracuda-seam`; Fuel's unused announce crate |
| grammar (types) | `OpTag`, `OpAttrs`, canonical bytes, `RegionKey`, `PatternNode`, `UnresolvedAttr` | `fuel-kernel-seam-types` |
| synth (envelope) | `Synthesizer`, `Jit*`, `SynthArtifact`, `ArtifactKind`, `LinkEntry` | `fuel-kernel-seam` |
| conformance (shared suite) | see section 5 | Fuel's `kiss_ops_619_divergence` test moves with it |

`shape_expr` (1,092 lines) is a decision for FireLine plus the KISS lane: it is KISS-Ops §6.20 and byte-matches the KISS reference through vendored golden vectors, so it is a third copy of one codec (Fuel, KISS ref `conformance/src/shape_expr.rs`, and whatever baracuda decodes with). Proposal: it moves with the grammar crate for now, and the golden vectors stay KISS's (FireLine's suite consumes them, never owns a copy).

## 3. What stays

- Fuel: `fuel_graph::jit::op_to_tag` (Op -> OpTag projection), `match_region`, all adoption (`fuel-dispatch` `jit_adopt`, `jit_carrier`, `jit_ingest`, `jit_cuda_load`, `runtime_fused_kernels`), the `[patch.crates-io]` unification hack (deleted by the move), `fuel-kiss-ref-backend` (KISS reference adapter, consumes `kiss-ref-core`).
- baracuda: `BaracudaSynthesizer`, `baracuda-cuda-emit`, kernel internals, the `baracuda_seam_hello()` FFI entry (the provider's thin adapter: it fills FireLine's `SeamHello`).
- Rule applied: anything naming Fuel tensors/graphs/devices, or baracuda kernel internals, stays.

## 4. What each side changes on the first release (one breaking move)

- Fuel: workspace `Cargo.toml` swaps three `path` members/pins for crates.io FireLine crates (`~`/exact); delete the three crates and the `[patch.crates-io]` block; rewrite `use fuel_kernel_seam*::` in 37 files (mechanical rename; no logic); packaged-notice, provenance and publish-plan entries drop 3 crates; `jit-synth` coherence check gets a CI test that only ONE copy of each FireLine crate is in `cargo tree`. Breaking: yes (public re-exports of seam types, if any, change path); Fuel's second number moves once.
- baracuda: see section 8 (its lane's half, merged here).
- Order (FireLine HANDOFF): the Fuel-to-producer seam moves first. Sequence: FireLine publishes (patent sweep first) -> Fuel PR and baracuda PR each build against the published crates -> both release; until both land the old crates stay published, so nothing orphans.
- Other consumers: none known outside Fuel and baracuda for announce/grammar/synth (crates.io reverse dependencies not queried; FireLine inventory should). `lightbulb`, `unpopped` and `vulkane` were not checked here.

## 5. The conformance test that fails if either side drifts

One suite, in FireLine, that both repos run in CI against their own adapter:
1. **Envelope bytes:** golden bytes + size/offset of `SeamHello` and every `SEAM_CAP_*` value; both sides link the same crate, so a renamed or renumbered bit is a compile error, not a test.
2. **Negotiation matrix:** every (role x profile-set x capability-mask) case of `negotiate_with_role`, run through Fuel's caller and through `baracuda_seam_hello()`'s output.
3. **Grammar round trip:** canonical bytes of `OpAttrs` for every `OpTag`, and `RegionKey`, against checked-in vectors; Fuel's `PatternNode` encode must decode in baracuda's synthesizer and the reverse.
4. **Live call:** a mock `Synthesizer` driven through `synthesize` -> `take_kernel`, then the real `BaracudaSynthesizer` through the same script (baracuda side only, no GPU).
5. **Single-copy check** in each consumer: `cargo tree -d` shows one version of each FireLine crate.
Drift in any of the five turns a named test red in the repo that drifted.

## 6. Relation to KISS and versioning

KISS owns the standards and golden corpora; FireLine owns the Rust crates implementing the Fuel<->provider interfaces; `unpopped-vocab`/`kiss-ops-vocab` stay the shared vocabulary and are depended on, never copied. Versions follow the agreed rule: each FireLine crate's version follows its interface (pre-1.0 a break moves that crate's second number alone), consumers use `~` or exact (never `^` -- baracuda's current `^0.43` is the pattern the tilde CI check will refuse), all crates meet at one number at a major release. This removes the coupling measured in `2026-10-24-seam-crate-versioning.md` (13 seam publishes in 16 weeks, 5 different second numbers in 10 days, almost all with no seam change).

## 7. Open questions for the other lanes

1. FireLine/KISS: does `shape_expr` belong in the grammar crate or in a KISS-owned crate? (3 copies today.)
2. baracuda: which capability names win (`FDX` vs `DLPACK_EXT`; `PROVISION_ON_REQUEST` vs `JIT_ON_REQUEST`), and does anything outside `baracuda-seam` link its copy?
3. FireLine: reverse dependencies of the three published crates on crates.io.

## 8. Baracuda's half (from the baracuda lane, read at baracuda `c911e9ff`, alpha.98; Fuel confirmations marked)

- **Baracuda has two crates on this seam:** `baracuda-seam` (470 lines: its own `SeamHello`, `baracuda_hello()`, the C entry `baracuda_seam_hello`, capability bits; no Fuel dependency) and `baracuda-cuda-seam` (1,318 lines: `BaracudaSynthesizer`, the `Synthesizer` Fuel calls).
- **A fourth thing moves that section 2 did not list: region conversion.** `to_internal`, `to_internal_at`, `optag_name`, `MAX_REGION_DEPTH` (Fuel `PatternNode`/`OpTag` to `unpopped::PatternNode`) and `hash_region`/`region_op_id` live in `baracuda-cuda-seam` and name nothing CUDA-specific (any provider, Vulkane included, needs them). FireLine owns them so exactly ONE crate converts and the graph holds one grammar crate (Unpopped#18 removed the other copy for that reason). Fuel has no copy of this conversion, so Fuel's side of that move is nothing.
- **Stays in baracuda:** `BaracudaSynthesizer` (wires `baracuda_cuda_emit::Cuda`, NVRTC, the artifact registry, the compile budget; becomes a thin implementor of FireLine's synthesizer trait), the symbol `baracuda_seam_hello`, `BARACUDA_CAPABILITIES`/`BARACUDA_PROFILE`.
- **First-release changes:** `baracuda-cuda-seam` swaps `^0.43` for `~` FireLine requirements, and its `tests/seam_pin.rs` (which asserts caret-only, exactly the opposite) is rewritten to require `~`; `baracuda-seam` deletes the duplicate envelope and keeps only the C entry and capability set (public names `SEAM_CAP_DLPACK_EXT_*`/`SEAM_CAP_JIT_ON_REQUEST` change: breaking for baracuda's second number); keep one release where `baracuda-seam` re-exports FireLine's `SeamHello` so a graph never holds two. FireLine's crates carry the interface's version, not `alpha.N`.
- **Answers to baracuda's two questions (Fuel, measured at `origin/main@28e0aef6`):** (1) Nothing in Fuel imports `baracuda-seam`: the only hits are comments in `Cargo.toml` and `fuel-kernel-seam-announce/src/lib.rs`; `gap315_seam_probe.rs` is an announce-only example and does not link it. (2) **Bits to freeze:** the ones both sides already agree on, bits 0-5 (`FDX_*`/`DLPACK_EXT_*`, one name each), 32 (`PROVISION_ON_REQUEST` = `JIT_ON_REQUEST`) and 33 (`CONTRACT_QUERY`, Fuel's). Bits 34 (`KISC_FRAMING`) and 35 (`RECIPE_IMPORT`) stay provisional and un-frozen until the KISS Announce lane co-allocates them with a consumer on each side; Fuel has no use for either today and intends no values. Name choice (FDX vs DLPACK_EXT) is for FireLine and KISS Announce.
- **Added conformance items from baracuda (fold into section 5):** `SeamHello` is 56 bytes with the frozen offsets and wire magic `53 45 41 4D`, round trip `parse_hello(to_wire_bytes(h)) == h`, truncated/wrong-magic/wrong-version/over-long input are typed errors; the C entry returns 0 on success and 2 on null, never allocates or panics; a covered `OpTag` converts and an uncovered one is a typed decline (depth over 64 declines, `#[non_exhaustive]` additions decline rather than break the build); region ids are stable under scalar values and change under attr, structure or operand layout/dtype (6 tests move from `baracuda-cuda-seam`); a stub-compiler build returns a typed decline; a with-device PTX-loads test runs on a self-hosted runner only.
- **Not verified by either lane:** Vulkane's seam use, the bit values Fuel might intend later, crates.io reverse dependencies.
