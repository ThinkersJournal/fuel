// SPDX-License-Identifier: MIT OR Apache-2.0
//! Device-resident weight storages kept by the identity of the host array they were built from
//! (ROADMAP row 172).
//!
//! A prefill (`forward_with_kv_context`) builds a fresh graph on every call, so every weight `Const`
//! is a new node with a new `NodeId` and a `NodeId`-keyed cache can never hit. What does not change is
//! the shared host `Arc<[T]>` the weight was built from; `fuel_graph::ConstOrigin` records its identity
//! and this registry keys the uploaded device storage by it. A registry belongs to one
//! `InferenceContext`, so it lives and dies with that context's device.
//!
//! Soundness: an entry keeps a strong pin on the source array, so the array's address cannot be
//! reused while the entry exists, and an immutable `Arc<[T]>` with the same address and length is the
//! same bytes. A `Const` built from a per-call array has no live source ([`fuel_graph::ConstOrigin::pin`]
//! returns `None`) and is never registered, so per-token data cannot accumulate here.

use std::collections::HashMap;

use fuel_graph::{ConstOrigin, ConstOriginKey};
use fuel_ir::Result;

struct Resident<S> {
    /// Keeps the source array alive for as long as its key is trusted.
    _pin: Box<dyn std::any::Any + Send + Sync>,
    storage: S,
}

/// Registry of uploaded weights, generic over the storage handle so the logic is testable without a GPU.
pub(crate) struct WeightResidency<S> {
    entries: HashMap<ConstOriginKey, Resident<S>>,
    hits: u64,
    uploads: u64,
}

impl<S: Clone> WeightResidency<S> {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            hits: 0,
            uploads: 0,
        }
    }

    /// The resident storage for `origin`, calling `upload` exactly once per live source array.
    /// `Ok(None)` means the source is gone (a per-call array): the caller must not treat it as a weight.
    pub(crate) fn get_or_upload(
        &mut self,
        origin: &ConstOrigin,
        upload: impl FnOnce() -> Result<S>,
    ) -> Result<Option<S>> {
        if let Some(r) = self.entries.get(&origin.key()) {
            self.hits += 1;
            return Ok(Some(r.storage.clone()));
        }
        let Some(pin) = origin.pin() else {
            return Ok(None);
        };
        let storage = upload()?;
        self.uploads += 1;
        self.entries.insert(
            origin.key(),
            Resident {
                _pin: pin,
                storage: storage.clone(),
            },
        );
        Ok(Some(storage))
    }

    /// `(hits, uploads)` since creation.
    pub(crate) fn counters(&self) -> (u64, u64) {
        (self.hits, self.uploads)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
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
        assert_eq!(r.counters(), (2, 1));
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
        assert_eq!(r.len(), 2);
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
        assert!(!called && r.len() == 0);
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
}
