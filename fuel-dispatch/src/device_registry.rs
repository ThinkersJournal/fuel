// SPDX-License-Identifier: MIT OR Apache-2.0
//! The device handles found or handed in during one realize (ROADMAP row 137, seam S3 step 3).
//!
//! Moved out of `pipelined.rs` (step 3b-3), where it sat in the largest and most-depended-on file of the
//! crate. `crate::pipelined::DeviceRegistry` still names it.

#[cfg(any(feature = "cuda", feature = "vulkan"))]
use fuel_ir::DeviceLocation;
#[cfg(any(feature = "cuda", feature = "vulkan", test))]
use fuel_ir::probe::DeviceIdentity;
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

/// The one entry whose identity is the same physical GPU as `wanted` (ROADMAP row 137 step 4b), or
/// `None` when no entry, or more than one, matches. An entry whose identity is unknown never matches
/// (unknown is not "same"). Never takes the first of several: that would reintroduce the order
/// dependence this exists to remove.
#[cfg(any(feature = "cuda", feature = "vulkan", test))]
fn unique_by_identity<'a, T>(
    entries: impl IntoIterator<Item = (Option<DeviceIdentity>, &'a T)>,
    wanted: &DeviceIdentity,
) -> Option<&'a T> {
    let mut hits = entries
        .into_iter()
        .filter(|(id, _)| id.is_some_and(|i| i.same_physical_device(wanted)))
        .map(|(_, handle)| handle);
    let first = hits.next()?;
    hits.next().is_none().then_some(first)
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

    /// The registered Vulkan backend that drives the physical GPU `wanted` names (UUID, else PCI address),
    /// whatever ordinal this process enumerated it at (ROADMAP row 137 step 4b). `None` when no, or more
    /// than one, registered backend is that GPU. Nothing is stored: each lookup asks the registered
    /// backends for their identity.
    #[cfg(feature = "vulkan")]
    pub fn registered_vulkan_by_identity(
        &self,
        wanted: &DeviceIdentity,
    ) -> Option<&std::sync::Arc<fuel_vulkan_backend::VulkanBackend>> {
        unique_by_identity(self.vulkan.values().map(|b| (b.identity(), b)), wanted)
    }

    /// The registered CUDA device that is the physical GPU `wanted` names (see
    /// [`Self::registered_vulkan_by_identity`]); the identity may come from another API. `None` also when
    /// the CUDA identity query itself fails.
    #[cfg(feature = "cuda")]
    pub fn registered_cuda_by_identity(
        &self,
        wanted: &DeviceIdentity,
    ) -> Option<&fuel_cuda_backend::CudaDevice> {
        let ids = fuel_cuda_backend::probe::device_identities().ok()?;
        unique_by_identity(
            self.cuda.iter().map(|(place, device)| {
                let id = match place {
                    DeviceLocation::Cuda { gpu_id } => ids.get(*gpu_id).copied().flatten(),
                    _ => None,
                };
                (id, device)
            }),
            wanted,
        )
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

#[cfg(test)]
mod tests {
    use super::unique_by_identity;
    use fuel_ir::probe::DeviceIdentity;

    fn id(b: u8) -> Option<DeviceIdentity> {
        Some(DeviceIdentity {
            uuid: Some([b; 16]),
            pci: None,
        })
    }

    #[test]
    fn finds_the_entry_in_either_registration_order() {
        let wanted = id(1).unwrap();
        assert_eq!(
            unique_by_identity([(id(1), &"a"), (id(2), &"b")], &wanted),
            Some(&"a")
        );
        assert_eq!(
            unique_by_identity([(id(2), &"b"), (id(1), &"a")], &wanted),
            Some(&"a")
        );
    }

    #[test]
    fn unknown_entries_never_match_and_no_match_is_none() {
        let wanted = id(1).unwrap();
        assert_eq!(unique_by_identity([(None, &"x")], &wanted), None);
        assert_eq!(unique_by_identity([(id(2), &"b")], &wanted), None);
        let nobody = DeviceIdentity {
            uuid: None,
            pci: None,
        };
        assert_eq!(unique_by_identity([(None, &"x")], &nobody), None);
    }

    #[test]
    fn two_matches_decline_instead_of_taking_the_first() {
        let wanted = id(1).unwrap();
        assert_eq!(
            unique_by_identity([(id(1), &"a"), (id(2), &"b"), (id(1), &"c")], &wanted),
            None
        );
    }
}
