# GAP-398: a process exits with an access violation after a failed Vulkan realize (evidence, 2026-11-23)

Lane: fuel. Status: REPRODUCED on demand, CAUSE FOUND AT THE LEVEL OF "an open recorder batch at device drop", FIX 21 lines (see the end). Section "The crash signature" and before are the evidence. Box: RTX 4070
Laptop (Vulkan adapter 1) plus an AMD iGPU, every run through `gpu-run`; exit 139 is the MSYS reading of
`0xC0000005` (a PowerShell run reads `-1073741819`).

## The reproducer

The probe example `gguf_unwired_quant_probe` as committed on branch `row124-audit` (`18a2c897c`, tree =
`7ecc79f6e` plus the example), built `--release --features dev-tools,vulkan -j 4`, which measured the five
then-unwired GGUF types: 7 types, the five subjects fail to realize with "QuantType X not wired on Vulkan".
It prints every verdict, then **exits 139 every time: 50 of 50 plain runs and 20 of 20 runs with
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`** (the loader's own log shows the layer inserted). **Validation output: none, UNVERIFIED:
there is no positive control that the layer's messages would reach stderr, so this says nothing about whether
the layer saw a problem**. Exit-code histogram: `{139: 70}`.

The probe as committed on the row 119 branch (all seven types wired, nothing fails) exits 0 (`{0}`).

## What does NOT crash (and what that rules out)

All counts are runs; exit codes are all `0` unless shown.

- A bare process that does `FAILS` failed single-op realizes (a `QMatMul` of Q2_K, still unwired), then one good
  Q4_K realize, then returns from `main`: 20 failures/run, 60 plain runs + 30 validation-layer runs + 1 run of 200
  failures: `{0: 91}`. With one successful Q4_K op feeding the refused one (`CHAIN`): 20 runs of 20 failures and
  10 of 1 failure, `{0: 30}`. So a failed realize alone, and a recorded-but-unflushed dispatch before it, do not
  crash.
- The probe's model path is what crashes: with the isolated bare-op part of the probe skipped
  (`SKIP_ISO`), `{139: 5}`; with only the isolated part (`SKIP_MODEL`), `{0: 5}`.
- Not a race with background work: sleeping 1.5 s after each type, or 5 s before returning, `{139: 5}` each.
- Not the main thread's thread-local destructors: running the model path on a spawned thread that is joined
  before `main` returns, `{139: 5}`.
- The crash needs the device to be DROPPED: `std::mem::forget(dev)` at the end of the model-only probe, `{0: 5}`,
  and of the full probe, `{0: 5}`. (Variants that disabled the op profiler exited 1 through the probe's own
  `process::exit(1)` for a different reason, which also skips the drop; they say nothing.)

## The crash signature (one `cdb` run, `-o -g -G`, through `gpu-run`)

```
Access violation - code c0000005 (first chance)   ExceptionAddress 0000000000000000
Attempt to execute non-executable address 0000000000000000        (Parameter[0] = 8: an instruction fetch)
 0  0x0                                                       <- a call through a NULL function pointer
 1  nvoglv64!vkGetInstanceProcAddr+0xb4e50   \  NVIDIA driver frames; the module exports only, so the
 2  nvoglv64!vkGetInstanceProcAddr+0xb7825    } offsets are from the nearest export, not function names
 3  nvoglv64!vkEnumerateInstanceVersion+0xbb  /
 4  gguf_unwired_quant_probe!...fuel_memory...Storage::with_bundle+0x6cf   (a merged drop-glue symbol)
 5  gguf_unwired_quant_probe!alloc::sync::Arc<fuel_vulkan_backend::VulkanBackend>::drop_slow+0xfd
 6  gguf_unwired_quant_probe!alloc::sync::Arc<dyn Fn(fuel_loaders::model_progress::ProgressEvent)...>::drop_slow
 7  gguf_unwired_quant_probe!probe::main+0x1c7a
```

Reading: during `probe::main`, dropping the last `Arc<VulkanBackend>` runs the backend's field destructors,
one of which calls into the NVIDIA driver, and the driver calls a null pointer. That agrees with the
`forget(dev)` result. It does not say which Vulkan object's destruction it was, why the driver's table has a
null entry after a failed model realize and not after a failed single-op realize, or whether the null is the
driver's or ours (an object destroyed twice, or after its device).

## The cause and the fix (same day)

`VulkanBackend::synchronize_pending()` (= `force_flush`) called on the device right before `main` returns, in the
old probe: `{139: 5}` without it, `{0: 5}` model-only and `{0: 5}` full with it. So the crash needs a recorder
batch left OPEN (dispatches recorded, never submitted) when the backend is destroyed: a realize that errors
part-way (the model path: ops before the refused `QMatMul` were recorded) returns without flushing, and nothing
flushed it at drop. A single refused op (the bare process) records nothing first, which is why it never crashed.
`VulkanBackend` had no `Drop`; its fields destroy in declaration order (device first) with the batch still
recording.

Fix: `impl Drop for VulkanBackend` calling `force_flush()` first (never panics: a poisoned recorder or failed
flush is logged). Red/green on the old probe (`row124-audit` `18a2c897c` + this patch, `-j 4`, `gpu-run`): model-only
`{139: 5}` before, `{0: 10}` after; full probe `{139: 70}` before, `{0: 30}` after. The backend code is identical
at `7ecc79f6e` and `58debefec` (empty diff), so the patch applies to main as is.

**No automated test is committed for it:** with every GGUF type wired the failing model realize that produced the
open batch can no longer be built from a GGUF, and a direct kernel call that leaves a batch open was not tried as a
crash test. The regression instrument is the old probe at `18a2c897c`.

## Not done / next

- Which `vkDestroy*` call the driver dies in (frames unnamed, no PDB; not needed once the open batch is flushed).
- A crash test that leaves a batch open without a failing realize (call a kernel wrapper directly, drop the backend).
- Another driver (the AMD iGPU, adapter 0) and Lavapipe were not tried.
- The reproducer is the OLD probe only; to rerun it: `git -C <worktree> checkout 18a2c897c`, build, run through
  `gpu-run`, expect exit 139.
