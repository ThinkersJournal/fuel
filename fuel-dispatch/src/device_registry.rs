// SPDX-License-Identifier: MIT OR Apache-2.0
//! The device handles found or handed in during one realize (ROADMAP row 137, seam S3 step 3).
//!
//! Moved out of `pipelined.rs` (step 3b-3), where it sat in the largest and most-depended-on file of the
//! crate. `crate::pipelined::DeviceRegistry` still names it.

#[cfg(any(feature = "cuda", feature = "vulkan"))]
use fuel_ir::DeviceLocation;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
use std::collections::HashMap;

/// The device handles one realize works with, keyed by `DeviceLocation` (ROADMAP row 137, seam S3).
///
/// The caller hands the executor the devices it holds (`with_cuda`, `with_vulkan`); the executor asks for the
/// handle of a place when it allocates on, or copies to, that place. There is no other source: it used to find a
/// handle by scanning the storage cache for a tensor that lived on the place, which made "a device exists" a
/// property of "some tensor happens to be on it" and needed 0-byte anchor storages (step 3c removed both). A
/// place nobody registered is a typed error at the lookup, naming the place.
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

    /// The CUDA device registered for `gpu_id`, if any.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda(&self, gpu_id: usize) -> Option<fuel_cuda_backend::CudaDevice> {
        self.registered_cuda(gpu_id).cloned()
    }

    /// The Vulkan backend registered for `gpu_id`, if any.
    #[cfg(feature = "vulkan")]
    pub(crate) fn vulkan(
        &self,
        gpu_id: usize,
    ) -> Option<std::sync::Arc<fuel_vulkan_backend::VulkanBackend>> {
        self.registered_vulkan(gpu_id).cloned()
    }
}
