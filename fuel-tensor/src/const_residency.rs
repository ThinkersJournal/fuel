// SPDX-License-Identifier: MIT OR Apache-2.0
//! Device-resident weight storages kept by the identity of the host array they were built from
//! (ROADMAP row 172).
//!
//! A prefill (`forward_with_kv_context`) builds a fresh graph on every call, so every weight `Const`
//! is a new node with a new `NodeId` and a `NodeId`-keyed cache can never hit. What does not change is
//! the shared host `Arc<[T]>` the weight was built from; `fuel_graph::ConstOrigin` records its identity
//! and this registry keys the uploaded device storage by it. One registry belongs to one
//! [`Device`](crate::Device) (every clone of it shares the registry), so a consumer that makes a new
//! `InferenceContext` per sequence still finds the model already resident.
//!
//! Soundness: an entry keeps a strong pin on the source array, so the array's address cannot be
//! reused while the entry exists, the array cannot be mutated in place (`Arc::get_mut` fails while the
//! pin is held), and an immutable `Arc<[T]>` with the same address and length is the same bytes. A
//! `Const` built from a per-call array has no live source ([`fuel_graph::ConstOrigin::pin`] returns
//! `None`) and is never registered, so per-token data cannot accumulate here.
//!
//! Budget: the registry holds at most a byte cap (set by the device before every seeding: a stated
//! share of the device's memory, see `inference_context`); past it the least recently used entries
//! are evicted, and a weight larger than the whole cap is served to the one realize that uploaded it
//! but never retained. A device that cannot report its memory gets no residency at all.
//!
//! Lifetime: an entry whose source array nothing else holds (the model was dropped) is evicted by
//! [`WeightResidency::sweep_orphans`], which runs at the start of every seeding, so a model swap does
//! not grow the registry. [`Device::clear_weight_residency`](crate::Device::clear_weight_residency)
//! drops everything at once, and dropping the last clone of the `Device` drops the registry.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use fuel_graph::{ConstOrigin, ConstOriginKey};
use fuel_ir::Result;
use fuel_memory::Storage;

/// Counters of a device's weight residency (see [`Device::weight_residency_stats`](crate::Device::weight_residency_stats)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct WeightResidencyStats {
    /// Weight `Const`s served from an earlier upload.
    pub hits: u64,
    /// Weight `Const`s uploaded (each one a miss that became resident).
    pub uploads: u64,
    /// Entries dropped because their source array was no longer held by anything else.
    pub evicted: u64,
    /// Entries resident now.
    pub resident: usize,
    /// Bytes resident now (the sum of the entries' source-array sizes).
    pub resident_bytes: u64,
    /// Entries evicted to stay under the byte cap (a subset of `evicted`).
    pub evicted_for_budget: u64,
    /// Weights uploaded but not retained because one weight alone exceeded the whole cap.
    pub rejected: u64,
    /// The byte cap in force at the last seeding (`0` before any seeding).
    pub cap_bytes: u64,
}

struct Resident<S> {
    bytes: u64,
    last_used: u64,
    origin: ConstOrigin,
    /// Keeps the source array alive for as long as its key is trusted.
    _pin: Box<dyn std::any::Any + Send + Sync>,
    storage: S,
}

/// Registry of uploaded weights, generic over the storage handle so the logic is testable without a GPU.
pub(crate) struct WeightResidency<S> {
    entries: HashMap<ConstOriginKey, Resident<S>>,
    hits: u64,
    uploads: u64,
    evicted: u64,
    evicted_for_budget: u64,
    rejected: u64,
    resident_bytes: u64,
    /// Monotonic use counter: an entry's `last_used` is the tick of its latest hit or insert.
    tick: u64,
    /// Byte cap for what stays resident; `None` means no cap (only tests build that).
    cap: Option<u64>,
}

/// Bytes of the host array a weight was built from.
fn origin_bytes(origin: &ConstOrigin) -> u64 {
    let k = origin.key();
    (k.len as u64).saturating_mul(k.dtype.size_in_bytes() as u64)
}

impl<S: Clone> WeightResidency<S> {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            hits: 0,
            uploads: 0,
            evicted: 0,
            evicted_for_budget: 0,
            rejected: 0,
            resident_bytes: 0,
            tick: 0,
            cap: None,
        }
    }

    /// Set the byte cap; a lowered cap takes effect at the next insert.
    pub(crate) fn set_cap(&mut self, cap: Option<u64>) {
        self.cap = cap;
    }

    /// Evict least recently used entries until `incoming` more bytes fit under the cap.
    fn make_room(&mut self, incoming: u64) {
        let Some(cap) = self.cap else { return };
        while self.resident_bytes.saturating_add(incoming) > cap {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, r)| r.last_used)
                .map(|(k, _)| *k)
            else {
                break;
            };
            if let Some(r) = self.entries.remove(&oldest) {
                self.resident_bytes -= r.bytes;
                self.evicted += 1;
                self.evicted_for_budget += 1;
            }
        }
    }

    /// Drop every entry whose source array nothing but this registry's pin holds any more.
    pub(crate) fn sweep_orphans(&mut self) -> usize {
        let before = self.entries.len();
        let mut freed = 0u64;
        self.entries.retain(|_, r| {
            let keep = r.origin.strong_count() > 1;
            if !keep {
                freed += r.bytes;
            }
            keep
        });
        self.resident_bytes -= freed;
        let gone = before - self.entries.len();
        self.evicted += gone as u64;
        gone
    }

    /// Drop every entry.
    pub(crate) fn clear(&mut self) {
        self.evicted += self.entries.len() as u64;
        self.entries.clear();
        self.resident_bytes = 0;
    }

    /// The resident storage for `origin`, calling `upload` exactly once per live source array.
    /// `Ok(None)` means the source is gone (a per-call array): the caller must not treat it as a weight.
    pub(crate) fn get_or_upload(
        &mut self,
        origin: &ConstOrigin,
        upload: impl FnOnce() -> Result<S>,
    ) -> Result<Option<S>> {
        self.tick += 1;
        if let Some(r) = self.entries.get_mut(&origin.key()) {
            self.hits += 1;
            r.last_used = self.tick;
            return Ok(Some(r.storage.clone()));
        }
        let Some(pin) = origin.pin() else {
            return Ok(None);
        };
        let storage = upload()?;
        self.uploads += 1;
        let bytes = origin_bytes(origin);
        if self.cap.is_some_and(|cap| bytes > cap) {
            // One weight alone is bigger than the whole budget: this realize uses it, nobody keeps it.
            self.rejected += 1;
            return Ok(Some(storage));
        }
        self.make_room(bytes);
        self.resident_bytes += bytes;
        self.entries.insert(
            origin.key(),
            Resident {
                bytes,
                last_used: self.tick,
                origin: origin.clone(),
                _pin: pin,
                storage: storage.clone(),
            },
        );
        Ok(Some(storage))
    }

    pub(crate) fn stats(&self) -> WeightResidencyStats {
        WeightResidencyStats {
            hits: self.hits,
            uploads: self.uploads,
            evicted: self.evicted,
            resident: self.entries.len(),
            resident_bytes: self.resident_bytes,
            evicted_for_budget: self.evicted_for_budget,
            rejected: self.rejected,
            cap_bytes: self.cap.unwrap_or(0),
        }
    }
}

/// The registry a [`Device`](crate::Device) carries: uploaded weight storages, shared by every clone.
pub(crate) struct DeviceWeights(pub(crate) Mutex<WeightResidency<Arc<RwLock<Storage>>>>);

impl DeviceWeights {
    pub(crate) fn new() -> Self {
        Self(Mutex::new(WeightResidency::new()))
    }
}

impl std::fmt::Debug for DeviceWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.0.lock().unwrap_or_else(|p| p.into_inner()).stats();
        f.debug_tuple("DeviceWeights").field(&stats).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_graph::NodeHandle;
    use std::sync::Arc;

    /// The origin a `Const` built from `data` records, read back through the graph.
    fn origin_of(data: &Arc<[u32]>) -> ConstOrigin {
        let dev = crate::Device::cpu();
        let t = NodeHandle::from_u32(
            Arc::clone(data),
            fuel_ir::Shape::from_dims(&[data.len()]),
            dev.as_dyn(),
        )
        .unwrap();
        let g = t.graph().read().unwrap();
        g.const_origins()
            .next()
            .map(|(_, o)| o.clone())
            .expect("a Const built from an Arc records its origin")
    }

    #[test]
    fn the_same_source_uploads_once_across_graphs() {
        let w: Arc<[u32]> = vec![1, 2, 3, 4].into();
        let mut r = WeightResidency::<u32>::new();
        let mut calls = 0;
        // Two DIFFERENT graphs (two `from_u32` calls), one source array.
        for _ in 0..3 {
            let o = origin_of(&w);
            let got = r
                .get_or_upload(&o, || {
                    calls += 1;
                    Ok(7)
                })
                .unwrap();
            assert_eq!(got, Some(7));
        }
        assert_eq!(
            calls, 1,
            "one upload for one live source, however many graphs"
        );
        assert_eq!((r.stats().hits, r.stats().uploads), (2, 1));
    }

    /// The sabotage sibling: two different live arrays are two entries, never one (a key that ignored the
    /// address would serve one weight's storage for the other).
    #[test]
    fn different_sources_do_not_share_a_storage() {
        let a: Arc<[u32]> = vec![1, 2, 3, 4].into();
        let b: Arc<[u32]> = vec![1, 2, 3, 4].into(); // equal bytes, different allocation
        let mut r = WeightResidency::<u32>::new();
        let sa = r.get_or_upload(&origin_of(&a), || Ok(1)).unwrap();
        let sb = r.get_or_upload(&origin_of(&b), || Ok(2)).unwrap();
        assert_eq!((sa, sb), (Some(1), Some(2)));
        assert_eq!(r.stats().resident, 2);
    }

    /// A per-call array (its source already dropped) is never registered: per-token data cannot pile up.
    #[test]
    fn a_dropped_source_is_not_registered() {
        let w: Arc<[u32]> = vec![9, 9].into();
        let o = origin_of(&w);
        drop(w);
        let mut r = WeightResidency::<u32>::new();
        let mut called = false;
        let got = r
            .get_or_upload(&o, || {
                called = true;
                Ok(0)
            })
            .unwrap();
        assert_eq!(got, None);
        assert!(!called && r.stats().resident == 0);
    }

    /// An entry pins its source: while it exists the array stays alive, so its address cannot be reused.
    #[test]
    fn an_entry_keeps_its_source_alive() {
        let w: Arc<[u32]> = vec![5, 6].into();
        let weak = Arc::downgrade(&w);
        let mut r = WeightResidency::<u32>::new();
        r.get_or_upload(&origin_of(&w), || Ok(3)).unwrap();
        drop(w);
        assert!(
            weak.upgrade().is_some(),
            "the registry pin must keep the array alive"
        );
        drop(r);
        assert!(
            weak.upgrade().is_none(),
            "dropping the registry releases it"
        );
    }

    // ---- LIFETIME ----

    /// Dropping the model (the last holder of the source array) lets the sweep evict its entry, which
    /// releases both the storage and the pin on the source.
    #[test]
    fn a_dropped_model_is_swept_and_its_storage_released() {
        let w: Arc<[u32]> = vec![1, 2, 3].into();
        let weak_src = Arc::downgrade(&w);
        let storage = Arc::new(7u32);
        let mut r = WeightResidency::<Arc<u32>>::new();
        r.get_or_upload(&origin_of(&w), || Ok(Arc::clone(&storage)))
            .unwrap();
        assert_eq!(Arc::strong_count(&storage), 2, "registry holds one handle");
        // The model still holds `w`: nothing to sweep.
        assert_eq!(r.sweep_orphans(), 0);
        assert_eq!(r.stats().resident, 1);
        drop(w); // the model is unloaded
        assert_eq!(r.sweep_orphans(), 1);
        let st = r.stats();
        assert_eq!(
            (
                st.hits,
                st.uploads,
                st.evicted,
                st.resident,
                st.resident_bytes
            ),
            (0, 1, 1, 0, 0)
        );
        assert_eq!(
            Arc::strong_count(&storage),
            1,
            "the storage handle is released"
        );
        assert!(
            weak_src.upgrade().is_none(),
            "the source array is released too"
        );
    }

    /// `clear` releases everything at once, models still loaded or not.
    #[test]
    fn clear_releases_every_storage() {
        let ws: Vec<Arc<[u32]>> = (0..4u32).map(|i| vec![i, i + 1].into()).collect();
        let storage = Arc::new(0u32);
        let mut r = WeightResidency::<Arc<u32>>::new();
        for w in &ws {
            r.get_or_upload(&origin_of(w), || Ok(Arc::clone(&storage)))
                .unwrap();
        }
        assert_eq!(Arc::strong_count(&storage), 5);
        r.clear();
        assert_eq!(Arc::strong_count(&storage), 1);
        assert_eq!(r.stats().resident, 0);
        assert_eq!(r.stats().evicted, 4);
    }

    /// A model swap loop (load, serve, drop, repeat) keeps the registry at one model's worth, never growing.
    #[test]
    fn a_model_swap_loop_does_not_grow_the_registry() {
        let mut r = WeightResidency::<u32>::new();
        let mut high_water = 0;
        for model in 0..50u32 {
            let weights: Vec<Arc<[u32]>> = (0..6u32)
                .map(|i| vec![model, i, model ^ i, 1].into())
                .collect();
            r.sweep_orphans(); // what the device does at the start of every seeding
            for w in &weights {
                r.get_or_upload(&origin_of(w), || Ok(model)).unwrap();
            }
            high_water = high_water.max(r.stats().resident);
            // the model is dropped here, at the end of the iteration
        }
        r.sweep_orphans();
        assert_eq!(r.stats().resident, 0, "the last model dropped too");
        assert!(
            high_water <= 6,
            "one model's worth (6) at a time, saw {high_water}"
        );
        assert_eq!(r.stats().uploads, 300);
        assert_eq!(r.stats().evicted, 300);
    }

    // ---- IDENTITY ----

    /// The address of a registered weight cannot be reused by a new array while the entry lives, so a
    /// new array with different content can never alias a stale entry. Several thousand same-size
    /// allocations are tried to make the allocator hand the old address back; a hit on any of them
    /// would serve the OLD storage for the NEW bytes. (The mutant that drops the pin is run against this
    /// in the row 172 notes: it goes red.)
    #[test]
    fn a_reused_address_is_never_a_hit() {
        let a: Arc<[u32]> = vec![1, 1, 1, 1].into();
        let old_ptr = a.as_ptr() as usize;
        let mut r = WeightResidency::<u32>::new();
        r.get_or_upload(&origin_of(&a), || Ok(100)).unwrap();
        drop(a); // the model drops its handle; the registry's pin is all that is left
        let mut keep = Vec::new();
        let mut landed_on_old_address = false;
        for i in 0..4000u32 {
            let b: Arc<[u32]> = vec![i, 2, 2, 2].into(); // different bytes, same length
            if b.as_ptr() as usize == old_ptr {
                landed_on_old_address = true;
            }
            let got = r.get_or_upload(&origin_of(&b), || Ok(200 + i)).unwrap();
            assert_eq!(
                got,
                Some(200 + i),
                "iteration {i}: a new array must be uploaded, never served the old storage"
            );
            keep.push(b);
        }
        assert!(
            !landed_on_old_address,
            "the pin must keep the old address from being reused"
        );
    }

    /// A registered weight cannot be mutated in place: the registry's pin makes the array shared, so
    /// `Arc::get_mut` (the only in-place route for an `Arc<[T]>`) refuses. A replacement array with
    /// equal content but its own allocation is a different weight and is uploaded afresh.
    #[test]
    fn a_pinned_weight_cannot_be_mutated_in_place() {
        let mut w: Arc<[u32]> = vec![4, 5, 6].into();
        assert!(Arc::get_mut(&mut w).is_some(), "unique before registration");
        let mut r = WeightResidency::<u32>::new();
        r.get_or_upload(&origin_of(&w), || Ok(1)).unwrap();
        assert!(
            Arc::get_mut(&mut w).is_none(),
            "the registry pin must make in-place mutation impossible"
        );
        let copy: Arc<[u32]> = w.iter().copied().collect::<Vec<_>>().into();
        let got = r.get_or_upload(&origin_of(&copy), || Ok(2)).unwrap();
        assert_eq!(got, Some(2), "an equal-content copy is a distinct weight");
    }

    // ---- BYTE BUDGET ----

    /// A 4-element `u32` array is 16 bytes.
    fn arr(tag: u32) -> Arc<[u32]> {
        vec![tag, tag + 1, tag + 2, tag + 3].into()
    }

    /// The registry never holds more than the cap and evicts the least recently used entry first; a hit
    /// refreshes an entry. This test FAILS when the cap enforcement is removed (see the row 172 notes).
    #[test]
    fn the_byte_cap_holds_and_evicts_least_recently_used() {
        let (a, b, c, d) = (arr(0), arr(10), arr(20), arr(30));
        let mut r = WeightResidency::<u32>::new();
        r.set_cap(Some(40)); // room for two 16-byte weights
        r.get_or_upload(&origin_of(&a), || Ok(1)).unwrap();
        r.get_or_upload(&origin_of(&b), || Ok(2)).unwrap();
        assert_eq!((r.stats().resident, r.stats().resident_bytes), (2, 32));
        r.get_or_upload(&origin_of(&c), || Ok(3)).unwrap(); // a is the least recently used
        let st = r.stats();
        assert!(
            st.resident_bytes <= 40,
            "over the cap: {} bytes",
            st.resident_bytes
        );
        assert_eq!((st.resident, st.evicted_for_budget), (2, 1));
        // `a` went; `b` is still a hit (and is now the most recently used).
        assert_eq!(r.get_or_upload(&origin_of(&b), || Ok(99)).unwrap(), Some(2));
        r.get_or_upload(&origin_of(&d), || Ok(4)).unwrap(); // evicts `c`, the least recently used now
        assert_eq!(
            r.get_or_upload(&origin_of(&b), || Ok(99)).unwrap(),
            Some(2),
            "b was refreshed, so it stays"
        );
        assert_eq!(
            r.get_or_upload(&origin_of(&c), || Ok(33)).unwrap(),
            Some(33),
            "c was the least recently used, so it is a miss again"
        );
        assert!(r.stats().resident_bytes <= 40);
    }

    /// One weight bigger than the whole cap is handed to the realize that uploaded it and never kept.
    #[test]
    fn a_weight_larger_than_the_cap_is_served_but_not_kept() {
        let big = arr(0); // 16 bytes
        let mut r = WeightResidency::<u32>::new();
        r.set_cap(Some(8));
        let o = origin_of(&big);
        assert_eq!(r.get_or_upload(&o, || Ok(5)).unwrap(), Some(5));
        assert_eq!(
            r.get_or_upload(&o, || Ok(6)).unwrap(),
            Some(6),
            "uploaded again: it was not kept"
        );
        let st = r.stats();
        assert_eq!(
            (st.resident, st.resident_bytes, st.rejected, st.uploads),
            (0, 0, 2, 2)
        );
    }

    /// `resident_bytes` follows sweeps and clears, so a freed model gives its budget back.
    #[test]
    fn freed_models_give_their_budget_back() {
        let a = arr(0);
        let mut r = WeightResidency::<u32>::new();
        r.set_cap(Some(100));
        r.get_or_upload(&origin_of(&a), || Ok(1)).unwrap();
        assert_eq!(r.stats().resident_bytes, 16);
        drop(a);
        r.sweep_orphans();
        assert_eq!(r.stats().resident_bytes, 0);
        let b = arr(5);
        r.get_or_upload(&origin_of(&b), || Ok(2)).unwrap();
        r.clear();
        assert_eq!(r.stats().resident_bytes, 0);
    }
}
