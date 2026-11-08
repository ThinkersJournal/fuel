// SPDX-License-Identifier: MIT OR Apache-2.0
//! The device side of JIT-candidate verification, with no backend named.
//!
//! Verifying a provider's candidate kernel needs a *reference*: Fuel builds a
//! small graph (the candidate's own `decompose`, or Fuel's registered recipe
//! lowered to primitives), runs it on a device, and compares bytes. Only the
//! last step touches a device, and it needs three things: which backend the
//! device is (so the graph's nodes can be stamped), a way to run a graph whose
//! leaves are host bytes, and the result as host bytes. [`ProbeDevice`] is
//! exactly that, so [`crate::jit_ingest_probe`] no longer names CUDA.
//!
//! Design: `docs/design-notes/2026-11-07-row137-step63b-probe-seam.md`
//! (PM ruling 2026-11-07: a minimal public [`ProbeTensor`]; the private
//! `fkc::verify` API is NOT published for this, GAP-236 stays its own item).
//! This is step A1 of that note: the two reference builders. Running the
//! candidate kernel itself (`invoke_candidate`) joins the trait in A2.

use std::sync::{Arc, RwLock};

use fuel_graph::{Graph, NodeId};
use fuel_ir::DType;
use fuel_ir::Result;
use fuel_ir::probe::BackendId;

/// One host-resident tensor crossing the probe seam: logical dtype, shape and
/// the raw little-endian element bytes. No backend type appears in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeTensor {
    /// Logical element type.
    pub dtype: DType,
    /// Dimension extents, outermost first.
    pub shape: Vec<usize>,
    /// Element bytes in row-major order.
    pub bytes: Vec<u8>,
}

/// A device that can realize a reference graph for candidate verification.
///
/// Implementors upload `inputs` (each bound to its `Const` leaf id), realize
/// `sink` on their device, and return the output as host bytes. They never
/// panic: every failure is an `Err`.
pub trait ProbeDevice: Send + Sync {
    /// The backend whose kernels realize the graph; the caller stamps it on
    /// every non-leaf node before calling [`ProbeDevice::realize`].
    fn backend(&self) -> BackendId;

    /// Realize `sink` of `graph` with each `(leaf id, tensor)` in `inputs`
    /// bound to its leaf, and return the sink's value as host bytes.
    fn realize(
        &self,
        graph: Arc<RwLock<Graph>>,
        sink: NodeId,
        inputs: &[(NodeId, ProbeTensor)],
    ) -> Result<ProbeTensor>;
}

/// The CUDA [`ProbeDevice`]: uploads each input to fresh CUDA storage, realizes
/// the graph on a one-device registry, and reads the output back. This is the
/// code `jit_ingest_probe` carried inline before; it moves to
/// `fuel-cuda-dispatch` in step A3 of the design note.
#[cfg(feature = "cuda")]
pub struct CudaProbeDevice {
    device: fuel_cuda_backend::CudaDevice,
}

#[cfg(feature = "cuda")]
impl CudaProbeDevice {
    /// Wrap a live CUDA device.
    pub fn new(device: fuel_cuda_backend::CudaDevice) -> Self {
        Self { device }
    }
}

#[cfg(feature = "cuda")]
impl ProbeDevice for CudaProbeDevice {
    fn backend(&self) -> BackendId {
        BackendId::Cuda
    }

    fn realize(
        &self,
        graph: Arc<RwLock<Graph>>,
        sink: NodeId,
        inputs: &[(NodeId, ProbeTensor)],
    ) -> Result<ProbeTensor> {
        use crate::pipelined::{DeviceRegistry, PipelinedExecutor, StorageCache};
        use fuel_cuda_backend::CudaStorageBytes;
        use fuel_ir::Error;

        // H2D: upload every probe into fresh CUDA-resident storage and bind it
        // to its Const leaf id.
        let mut cache = StorageCache::new();
        for (id, t) in inputs {
            let cb = CudaStorageBytes::from_cpu_bytes(&self.device, &t.bytes)?;
            cache.insert(
                *id,
                Arc::new(RwLock::new(fuel_memory::Storage::new(
                    fuel_memory::BackendStorage::Cuda(cb),
                    t.dtype,
                ))),
            );
        }

        // Realize the sink on the device.
        let (out_arc, layout) = PipelinedExecutor::realize_with_devices(
            graph,
            sink,
            cache,
            DeviceRegistry::new().with_cuda(self.device.clone()),
        )?;

        // D2H: read the CUDA output storage back to host bytes.
        let guard = out_arc.read().map_err(|_| {
            Error::Msg("CudaProbeDevice::realize: output storage RwLock poisoned".to_string())
        })?;
        let bytes = match &guard.inner {
            fuel_memory::BackendStorage::Cuda(c) => c.to_cpu_bytes()?,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(Error::Msg(
                    "CudaProbeDevice::realize: realized output storage is not CUDA".to_string(),
                ));
            }
        };
        Ok(ProbeTensor {
            dtype: guard.dtype,
            shape: layout.shape().dims().to_vec(),
            bytes,
        })
    }
}
