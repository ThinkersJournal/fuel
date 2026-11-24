# Row 174 (provisional number): fireline 0.5 and unpopped-vocab 0.56 (2026-11-24)

Lane: fuel. Inputs: dependabot PRs #699 (fireline-grammar 0.4.0 -> 0.5.0), #701 (fireline-synth 0.4.0 -> 0.5.0), #700
(unpopped-vocab 0.47.0 -> 0.56.2; crates.io today is **0.56.5**), their `Check (ubuntu-latest)` logs, and a local
reproduction on `origin/main` `5a2c85060` (Cargo.lock edits only in a throwaway worktree; commands below).

## What breaks

Not Fuel's own API use. Each PR bumps ONE crate of a family whose members are pinned to each other, so the lock ends up
with two copies of the same crate and the same-named types are two types (E0308 `expected X, found a different X`).

| PR | Errors (CI, `fuel-dispatch` lib 2, lib test 4 to 8) | Cause |
|---|---|---|
| #699 grammar 0.5.0 | `fuel-dispatch/src/jit_adopt.rs:281,486`, `jit_carrier.rs:118,224`: `PatternNode` is `fireline_grammar` 0.5.0 on one side and 0.4.5 on the other | `fireline-synth` 0.4.x requires grammar `~0.4` |
| #701 synth 0.5.0 | same class | `baracuda-cuda-seam` 0.47.0 requires synth and grammar `~0.4` (the lock keeps 0.4.5 of both) |
| #700 vocab 0.56.2 | `jit_adopt.rs:275,488,491`, `jit_carrier.rs:118,210,213,219,222`: `ElementKind`, `OperandDesc`, `ArchSku` from 0.47.0 vs 0.56.x | `fireline-synth` 0.5.0 (latest), `baracuda-types` / `baracuda-kernels-types` alpha.148 (latest) and `baracuda-cuda-seam` 0.47.0 (latest) all require `unpopped-vocab ~0.47.0` |

Reproduced locally (crates.io dependency requirements read from the registry API for the exact versions):

- Bumping `fireline-grammar` AND `fireline-synth` to `~0.5` together: `cargo check -p fuel-dispatch -p fuel-graph -p fuel-kiss-ref-backend --all-targets`
  and `cargo check -p fuel-dispatch --features jit --all-targets` both finish clean, **no source edit**. But the lock then holds
  grammar 0.4.5 + 0.5.0 and synth 0.4.5 + 0.5.0 because `baracuda-cuda-seam` 0.47.0 still wants 0.4:
  `fuel-cuda-dispatch --features jit-synth` would mix the two (NOT compiled here: it needs the CUDA forge).
- Bumping `unpopped-vocab` to `0.56.5`: `fuel-dispatch --features jit` fails with the #700 errors (a second copy at 0.47.0 via `fireline-synth` 0.5.0 and baracuda alpha.148).

## Why it waits

Both bumps are **blocked on publishers, not on Fuel**: a `baracuda-cuda-seam` (and `baracuda-types` / `baracuda-kernels-types`) release
on `fireline-* ~0.5` and `unpopped-vocab ~0.56`, and a `fireline-synth` release on `unpopped-vocab ~0.56`. Fuel's pins
(`unpopped-vocab = "0.47.0"`, `fireline-grammar = "~0.4"`, `fireline-synth = "~0.4"` in the root `Cargo.toml`) exist to hold ONE copy of each
(see the comments there from rows 41a and the 0.38 to 0.40 bumps); moving ahead of the publishers re-creates the two-types problem.

## Where Fuel depends on unpopped-vocab

`cargo tree -i unpopped-vocab` at `5a2c85060` (default features): only through `baracuda-kernels-types` and `baracuda-types` alpha.148
(`fuel-cuda-backend`, `fuel-dispatch`). Direct dependency edges: `fuel-dispatch` (optional, the `jit` feature) and `fuel-cuda-dispatch` (optional, `jit`).
The direct pin is **necessary only to keep one version in the graph** (the JIT glue builds `OperandDesc` / `ArchSku` values it hands to `fireline-synth`);
it carries no Fuel code that needs 0.47 specifically.

## Size, day, dependencies

- Fuel side: **S** once unblocked: manifest bumps (grammar, synth, vocab) + one lock refresh, expected zero source edits for grammar and synth (shown above); vocab 0.47 -> 0.56 is nine minors, so budget M until compiled against the published synth.
- PR day: not 11-25 to 11-28 (calendar full); **11-29 or later**, and only after the publishers below exist.
- Depends on: the baracuda lane's release on fireline 0.5 / vocab 0.56, and the fireline lane's `fireline-synth` on vocab 0.56. Check: `cargo info baracuda-cuda-seam` (requirement of `fireline-grammar`, `unpopped-vocab`), `cargo info fireline-synth`.
- Dependabot will reopen at the newer version; that is expected.

## Update 2026-11-24 (fireline 0.6.0 published)

pm-kernels read the registry: `fireline-grammar`, `-synth`, `-announce`, `-contract`, `-classify` are 0.6.0; `fireline-synth` 0.6.0 requires `fireline-grammar ~0.6` and `unpopped-vocab ~0.56.0`. Fuel can move the fireline pair to `~0.6` alone, but `baracuda-types` / `baracuda-kernels-types` alpha.148 still require `unpopped-vocab ~0.47.0`, so the graph would hold two vocab copies again (the #700 E0308 class). Order: baracuda republishes types/kernels-types (and cuda-seam) on vocab `~0.56` first; then one Fuel change moves fireline `~0.6` and vocab 0.56.5 together, measured with registry versions only. The 0.5.0 / 0.56.2 targets in the sections above are superseded.
