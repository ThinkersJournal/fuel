// SPDX-License-Identifier: MIT OR Apache-2.0
//! The device handles found or handed in during one realize (ROADMAP row 137, seam S3 step 3).
//!
//! Moved out of `pipelined.rs` (step 3b-3), where it sat in the largest and most-depended-on file of the
//! crate. `crate::pipelined::DeviceRegistry` still names it.

#[cfg(any(feature = "cuda", feature = "vulkan"))]
use crate::pipelined::StorageCache;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
use fuel_ir::DeviceLocation;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
use std::collections::HashMap;

/// Device handles found during one realize, keyed by `DeviceLocation` (ROADMAP row 137, seam S3 step 3a).
///
/// A handle is still found the way it always was: the first lookup for a place scans the storage cache for
/// a storage that lives there (`find_cuda_device_in_cache`, `find_vulkan_backend_in_cache`), and the
/// registry remembers the answer so the next lookup for that place does not scan. The scan runs against
/// the cache as it is AT THAT MOMENT, so a device that becomes findable part-way through a realize (a copy
/// creates the first storage on it) is found exactly when it was before. Nothing is pre-filled and nothing
/// is invalidated: a handle stays valid for the whole realize. Parts 3b and 3c replace the scan with handles
/// handed in by the caller and delete the seeding.
#[derive(Default)]
pub struct DeviceRegistry {
    #[cfg(feature = "cuda")]
    cuda: HashMap<DeviceLocation, fuel_cuda_backend::CudaDevice>,
    #[cfg(feature = "vulkan")]
    vulkan: HashMap<DeviceLocation, std::sync::Arc<fuel_vulkan_backend::VulkanBackend>>,
}

impl DeviceRegistry {
    /// An empty registry (the state at the start of every realize).
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the CUDA device a caller already holds, so a realize does not have to find it by scanning the
    /// storage cache for a tensor that lives there (ROADMAP row 137, seam S3 step 3b). The place is the one
    /// the device reports for itself.
    #[cfg(feature = "cuda")]
    pub fn with_cuda(mut self, device: fuel_cuda_backend::CudaDevice) -> Self {
        self.cuda.insert(device.location(), device);
        self
    }

    /// Record the Vulkan backend a caller already holds (see [`Self::with_cuda`]); the place is the backend's
    /// own `gpu_id`.
    #[cfg(feature = "vulkan")]
    pub fn with_vulkan(
        mut self,
        backend: std::sync::Arc<fuel_vulkan_backend::VulkanBackend>,
    ) -> Self {
        self.vulkan.insert(
            DeviceLocation::Vulkan {
                gpu_id: backend.gpu_id,
            },
            backend,
        );
        self
    }

    /// The CUDA device a caller handed in for `gpu_id`, without looking in any cache. `None` when none was
    /// registered for that place.
    #[cfg(feature = "cuda")]
    pub fn registered_cuda(&self, gpu_id: usize) -> Option<&fuel_cuda_backend::CudaDevice> {
        self.cuda.get(&DeviceLocation::Cuda { gpu_id })
    }

    /// The Vulkan backend a caller handed in for `gpu_id`, without looking in any cache. `None` when none
    /// was registered for that place.
    #[cfg(feature = "vulkan")]
    pub fn registered_vulkan(
        &self,
        gpu_id: usize,
    ) -> Option<&std::sync::Arc<fuel_vulkan_backend::VulkanBackend>> {
        self.vulkan.get(&DeviceLocation::Vulkan { gpu_id })
    }

    /// The CUDA device at `gpu_id`: remembered, else found in `cache` as before.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda(
        &mut self,
        cache: &StorageCache,
        gpu_id: usize,
    ) -> Option<fuel_cuda_backend::CudaDevice> {
        let at = DeviceLocation::Cuda { gpu_id };
        if let Some(dev) = self.cuda.get(&at) {
            return Some(dev.clone());
        }
        let dev = find_cuda_device_in_cache(cache, gpu_id)?;
        self.cuda.insert(at, dev.clone());
        Some(dev)
    }

    /// The Vulkan backend at `gpu_id`: remembered, else found in `cache` as before.
    #[cfg(feature = "vulkan")]
    pub(crate) fn vulkan(
        &mut self,
        cache: &StorageCache,
        gpu_id: usize,
    ) -> Option<std::sync::Arc<fuel_vulkan_backend::VulkanBackend>> {
        let at = DeviceLocation::Vulkan { gpu_id };
        if let Some(dev) = self.vulkan.get(&at) {
            return Some(dev.clone());
        }
        let dev = find_vulkan_backend_in_cache(cache, gpu_id)?;
        self.vulkan.insert(at, dev.clone());
        Some(dev)
    }
}

/// Search the input cache for any `BackendStorage::Cuda(s)` whose
/// device matches `gpu_id`. Returns a clone of the `CudaDevice` Arc
/// the storage carries — the caller uses it to call
/// `CudaStorageBytes::alloc` for `Op::Alloc` on Cuda. Returns `None`
/// if no matching storage exists.
///
/// Phase 3a of bridge-retirement: callers seed the cache with a 0-byte
/// CUDA storage (`fuel-tensor::pipelined_bridge::device_seed_storage`)
/// so this lookup succeeds for the first Op::Alloc; subsequent
/// Op::Allocs see prior Op::Alloc outputs in the cache.
#[cfg(feature = "cuda")]
fn find_cuda_device_in_cache(
    cache: &StorageCache,
    gpu_id: usize,
) -> Option<fuel_cuda_backend::CudaDevice> {
    for arc in cache.values() {
        let guard = arc.read().ok()?;
        if let fuel_memory::BackendStorage::Cuda(c) = &guard.inner {
            // CudaDevice carries its DeviceLocation (gpu_id ordinal)
            // through `location()`; match against the target gpu_id.
            // Multi-GPU future work can refine the match; today's
            // single-GPU setups always match.
            if matches!(c.device().location(), DeviceLocation::Cuda { gpu_id: g } if g == gpu_id) {
                return Some(c.device().clone());
            }
        }
    }
    None
}

/// Vulkan counterpart of [`find_cuda_device_in_cache`]. Returns an
/// `Arc<VulkanBackend>` clone derived from any cached Vulkan storage
/// whose backend's `gpu_id` matches.
#[cfg(feature = "vulkan")]
fn find_vulkan_backend_in_cache(
    cache: &StorageCache,
    gpu_id: usize,
) -> Option<std::sync::Arc<fuel_vulkan_backend::VulkanBackend>> {
    for arc in cache.values() {
        let guard = arc.read().ok()?;
        if let fuel_memory::BackendStorage::Vulkan(v) = &guard.inner
            && let Some(backend) = v.backend()
            && backend.gpu_id == gpu_id
        {
            return Some(backend.clone());
        }
    }
    None
}
