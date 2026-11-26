// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 173 lever 2: the transfer-calibration probe must run once per PROCESS, not once per topology
//! snapshot. Every `bump_topology_generation()` (a backend or kernel registration) rebuilds the
//! snapshot, and the per-snapshot `OnceLock` used to probe the same GPUs again: the cold Qwen3
//! prefill paid ~1 s for it twice (startup estimates, then the profiled prefill).
//!
//! One `#[test]` on purpose: `calibrate_runs()` is process-global, so a sibling test calling
//! `transfer_calibration()` would make the count mean something else.

use fuel_dispatch::topology::SystemTopology;
use fuel_hardware::transfer_cost::calibrate_runs;

#[test]
fn a_rebuilt_topology_snapshot_reuses_the_probe_it_already_ran() {
    let before = calibrate_runs();

    let first = SystemTopology::current();
    let cal_first = first.transfer_calibration().clone();
    assert_eq!(
        calibrate_runs() - before,
        1,
        "the first snapshot probes once"
    );

    // What registering a backend or kernel does: advance the generation, rebuild the snapshot.
    SystemTopology::refresh();
    let second = SystemTopology::current();
    assert!(
        second.generation() > first.generation(),
        "control: the snapshot really was rebuilt (generation {} -> {})",
        first.generation(),
        second.generation()
    );
    assert!(
        !std::ptr::eq(&*first, &*second),
        "control: a different snapshot object"
    );
    let cal_second = second.transfer_calibration().clone();

    let runs = calibrate_runs() - before;
    println!(
        "CALIBRATE RUNS across two snapshots: {runs} (devices {:?})",
        second.devices()
    );
    assert_eq!(
        runs, 1,
        "the second snapshot probed AGAIN: {runs} probes in one process"
    );
    assert_eq!(
        cal_first, cal_second,
        "reusing the probe must not change the estimates"
    );
}
