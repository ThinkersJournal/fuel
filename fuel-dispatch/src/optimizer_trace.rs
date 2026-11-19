// SPDX-License-Identifier: MIT OR Apache-2.0
//! One place to read what the optimizer dropped and what the executor chose (ROADMAP rows 157, 158).
//!
//! Two kinds of event, both cheap and always on:
//!
//! - **The executor's arm choice.** Every `Op::Branch` the streaming walk resolves bumps one atomic per
//!   arm index (`0`, `1`, `2`, and `3` for "3 or more"). No allocation, no lock.
//! - **What the region pathfinder did with its candidates.** Candidates dropped because they overlap a
//!   larger chosen region, branches recorded, branches collapsed to one arm, and branches
//!   `finalize_branches` REJECTED, with the reason. Rejections used to be thrown away (`.ok().flatten()`);
//!   the reasons are kept here, newest [`MAX_KEPT_REJECTIONS`], behind a lock that only a rejection takes.
//!
//! Read it with [`snapshot`]; compare two snapshots with [`TraceSnapshot::since`]. The counters are
//! process-wide and monotonic, so a test that wants "this realize only" takes a snapshot before and after.
//! `FUEL_PRINT_BRANCH_COUNT=1` still prints the same events as they happen.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use fuel_ir::probe::BackendId;

/// How many rejection reasons are kept (the newest).
pub const MAX_KEPT_REJECTIONS: usize = 16;

/// Number of arm-index slots: arms `0`, `1`, `2`, and slot `3` for arm 3 and above.
pub const ARM_SLOTS: usize = 4;

/// Backend slots for [`Trace::record_lookup_none`]: `Cpu`, `Cuda`, `Vulkan`, `Metal`, and slot 4 for
/// any other backend.
pub const BACKEND_SLOTS: usize = 5;

fn backend_slot(backend: BackendId) -> usize {
    match backend {
        BackendId::Cpu => 0,
        BackendId::Cuda => 1,
        BackendId::Vulkan => 2,
        BackendId::Metal => 3,
        _ => 4,
    }
}

/// The counters. The process has one ([`global`]); tests build their own so they cannot race with
/// the rest of the test binary.
pub struct Trace {
    arm_chosen: [AtomicU64; ARM_SLOTS],
    lookup_none: [AtomicU64; BACKEND_SLOTS],
    overlap_dropped: AtomicU64,
    recorded: AtomicU64,
    collapsed: AtomicU64,
    rejected: AtomicU64,
    backlog_added_ns: AtomicU64,
    backlog_retired_ns: AtomicU64,
    rejections: Mutex<VecDeque<String>>,
}

/// A copy of the counters at one moment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TraceSnapshot {
    /// Branches resolved to arm `i` (slot 3 = arm 3 and above).
    pub arm_chosen: [u64; ARM_SLOTS],
    /// Per backend slot (`Cpu`, `Cuda`, `Vulkan`, `Metal`, other): how often the selector's runtime
    /// lookup returned NO handle for a candidate, so the picker had no VRAM or load signal for it.
    pub lookup_none: [u64; BACKEND_SLOTS],
    /// Region candidates dropped because they overlap a larger chosen region.
    pub overlap_dropped: u64,
    /// Branches recorded by the pathfinder.
    pub recorded: u64,
    /// Branches that collapsed to a single arm (no decision point).
    pub collapsed: u64,
    /// Branches `finalize_branches` rejected.
    pub rejected: u64,
    /// Estimated nanoseconds the executor added to a device's backlog (row 158: every kernel it submitted
    /// to a GPU, at the node's compile-time estimate).
    pub backlog_added_ns: u64,
    /// Estimated nanoseconds taken back out of a device's backlog when the realize that added them ended.
    /// Equal to `backlog_added_ns` once every realize has returned; a gap is a leak.
    pub backlog_retired_ns: u64,
    /// The newest rejection reasons, oldest first (at most [`MAX_KEPT_REJECTIONS`]).
    pub rejections: Vec<String>,
}

impl TraceSnapshot {
    /// Counter growth since `earlier` (saturating). `rejections` is the reasons added since: the
    /// newest `rejected` growth entries of this snapshot's kept reasons.
    pub fn since(&self, earlier: &TraceSnapshot) -> TraceSnapshot {
        let mut arm_chosen = [0u64; ARM_SLOTS];
        for (i, slot) in arm_chosen.iter_mut().enumerate() {
            *slot = self.arm_chosen[i].saturating_sub(earlier.arm_chosen[i]);
        }
        let mut lookup_none = [0u64; BACKEND_SLOTS];
        for (i, slot) in lookup_none.iter_mut().enumerate() {
            *slot = self.lookup_none[i].saturating_sub(earlier.lookup_none[i]);
        }
        let rejected = self.rejected.saturating_sub(earlier.rejected);
        let keep = (rejected as usize).min(self.rejections.len());
        TraceSnapshot {
            arm_chosen,
            lookup_none,
            overlap_dropped: self.overlap_dropped.saturating_sub(earlier.overlap_dropped),
            recorded: self.recorded.saturating_sub(earlier.recorded),
            collapsed: self.collapsed.saturating_sub(earlier.collapsed),
            rejected,
            backlog_added_ns: self
                .backlog_added_ns
                .saturating_sub(earlier.backlog_added_ns),
            backlog_retired_ns: self
                .backlog_retired_ns
                .saturating_sub(earlier.backlog_retired_ns),
            rejections: self.rejections[self.rejections.len() - keep..].to_vec(),
        }
    }

    /// Total branches resolved, over every arm slot.
    pub fn branches_resolved(&self) -> u64 {
        self.arm_chosen.iter().sum()
    }
}

impl Trace {
    /// An empty trace.
    pub const fn new() -> Self {
        Self {
            arm_chosen: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            lookup_none: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            overlap_dropped: AtomicU64::new(0),
            recorded: AtomicU64::new(0),
            collapsed: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            backlog_added_ns: AtomicU64::new(0),
            backlog_retired_ns: AtomicU64::new(0),
            rejections: Mutex::new(VecDeque::new()),
        }
    }

    /// The executor resolved a branch to arm `arm`.
    pub fn record_arm_chosen(&self, arm: usize) {
        self.arm_chosen[arm.min(ARM_SLOTS - 1)].fetch_add(1, Ordering::Relaxed);
    }

    /// The selector's runtime lookup returned no handle for a candidate on `backend` (it had a lookup
    /// configured), so the picker had no VRAM or load signal for that candidate.
    pub fn record_lookup_none(&self, backend: BackendId) {
        self.lookup_none[backend_slot(backend)].fetch_add(1, Ordering::Relaxed);
    }

    /// The pathfinder's outcome for one `propose` pass.
    pub fn record_regions(
        &self,
        overlap_dropped: usize,
        recorded: usize,
        collapsed: usize,
        rejection_reasons: &[String],
    ) {
        self.overlap_dropped
            .fetch_add(overlap_dropped as u64, Ordering::Relaxed);
        self.recorded.fetch_add(recorded as u64, Ordering::Relaxed);
        self.collapsed
            .fetch_add(collapsed as u64, Ordering::Relaxed);
        self.rejected
            .fetch_add(rejection_reasons.len() as u64, Ordering::Relaxed);
        if rejection_reasons.is_empty() {
            return;
        }
        // A poisoned lock only means a reason was lost; counting must never panic.
        if let Ok(mut kept) = self.rejections.lock() {
            for reason in rejection_reasons {
                if kept.len() == MAX_KEPT_REJECTIONS {
                    kept.pop_front();
                }
                kept.push_back(reason.clone());
            }
        }
    }

    /// The executor added `ns` of estimated work to a device's backlog.
    pub fn record_backlog_added(&self, ns: u64) {
        let _ = self
            .backlog_added_ns
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_add(ns))
            });
    }

    /// The executor took `ns` back out of a device's backlog (its realize ended or unwound).
    pub fn record_backlog_retired(&self, ns: u64) {
        let _ = self
            .backlog_retired_ns
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_add(ns))
            });
    }

    /// Copy the counters.
    pub fn snapshot(&self) -> TraceSnapshot {
        let mut arm_chosen = [0u64; ARM_SLOTS];
        for (i, slot) in arm_chosen.iter_mut().enumerate() {
            *slot = self.arm_chosen[i].load(Ordering::Relaxed);
        }
        let mut lookup_none = [0u64; BACKEND_SLOTS];
        for (i, slot) in lookup_none.iter_mut().enumerate() {
            *slot = self.lookup_none[i].load(Ordering::Relaxed);
        }
        TraceSnapshot {
            arm_chosen,
            lookup_none,
            overlap_dropped: self.overlap_dropped.load(Ordering::Relaxed),
            recorded: self.recorded.load(Ordering::Relaxed),
            collapsed: self.collapsed.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            backlog_added_ns: self.backlog_added_ns.load(Ordering::Relaxed),
            backlog_retired_ns: self.backlog_retired_ns.load(Ordering::Relaxed),
            rejections: self
                .rejections
                .lock()
                .map(|kept| kept.iter().cloned().collect())
                .unwrap_or_default(),
        }
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

static GLOBAL: Trace = Trace::new();

/// The process-wide trace the executor and the pathfinder write to.
pub fn global() -> &'static Trace {
    &GLOBAL
}

/// A copy of the process-wide counters.
pub fn snapshot() -> TraceSnapshot {
    GLOBAL.snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_choices_count_per_arm_and_clamp_the_tail() {
        let t = Trace::new();
        t.record_arm_chosen(0);
        t.record_arm_chosen(0);
        t.record_arm_chosen(1);
        t.record_arm_chosen(7); // arm 3 and above share slot 3
        let s = t.snapshot();
        assert_eq!(s.arm_chosen, [2, 1, 0, 1]);
        assert_eq!(s.branches_resolved(), 4);
    }

    #[test]
    fn region_outcomes_accumulate_and_rejections_keep_their_reason() {
        let t = Trace::new();
        t.record_regions(5, 2, 1, &["InvalidBranch: shape".to_string()]);
        t.record_regions(0, 1, 0, &[]);
        let s = t.snapshot();
        assert_eq!(
            (s.overlap_dropped, s.recorded, s.collapsed, s.rejected),
            (5, 3, 1, 1)
        );
        assert_eq!(s.rejections, vec!["InvalidBranch: shape".to_string()]);
    }

    #[test]
    fn only_the_newest_rejection_reasons_are_kept_but_all_are_counted() {
        let t = Trace::new();
        let reasons: Vec<String> = (0..MAX_KEPT_REJECTIONS + 4)
            .map(|i| format!("reason {i}"))
            .collect();
        t.record_regions(0, 0, 0, &reasons);
        let s = t.snapshot();
        assert_eq!(s.rejected as usize, MAX_KEPT_REJECTIONS + 4);
        assert_eq!(s.rejections.len(), MAX_KEPT_REJECTIONS);
        assert_eq!(s.rejections.first().map(String::as_str), Some("reason 4"));
        assert_eq!(
            s.rejections.last().map(String::as_str),
            Some("reason 19"),
            "the newest is kept"
        );
    }

    #[test]
    fn since_reports_only_the_growth_and_the_reasons_added() {
        let t = Trace::new();
        t.record_arm_chosen(0);
        t.record_regions(1, 1, 0, &["old".to_string()]);
        let before = t.snapshot();
        t.record_arm_chosen(1);
        t.record_arm_chosen(1);
        t.record_regions(2, 0, 0, &["new a".to_string(), "new b".to_string()]);
        let delta = t.snapshot().since(&before);
        assert_eq!(delta.arm_chosen, [0, 2, 0, 0]);
        assert_eq!(
            (delta.overlap_dropped, delta.recorded, delta.rejected),
            (2, 0, 2)
        );
        assert_eq!(
            delta.rejections,
            vec!["new a".to_string(), "new b".to_string()],
            "the reasons added since, not the old one"
        );
    }

    #[test]
    fn lookup_misses_count_per_backend_and_show_in_since() {
        let t = Trace::new();
        t.record_lookup_none(BackendId::Vulkan);
        let before = t.snapshot();
        t.record_lookup_none(BackendId::Vulkan);
        t.record_lookup_none(BackendId::Cpu);
        let delta = t.snapshot().since(&before);
        assert_eq!(delta.lookup_none, [1, 0, 1, 0, 0], "Cpu 1, Vulkan 1 since");
        assert_eq!(t.snapshot().lookup_none, [1, 0, 2, 0, 0], "totals");
    }

    #[test]
    fn a_fresh_trace_is_all_zero() {
        assert_eq!(Trace::new().snapshot(), TraceSnapshot::default());
    }
}
