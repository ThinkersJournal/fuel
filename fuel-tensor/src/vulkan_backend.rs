// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bridge between [`fuel_vulkan_backend::VulkanBackend`] and
//! [`crate::Device`].
//!
//! Mirrors the [`crate::cuda_backend`] shape: a `From<VulkanBackend>
//! for Device` impl plus an `as_device(&Device) -> &VulkanBackend`
//! downcast helper. The Vulkan storage path runs on the byte-shape
//! [`fuel_vulkan_backend::VulkanStorageBytes`] substrate, so the
//! downcast returns an `Arc<VulkanBackend>` callers thread into
//! [`fuel_vulkan_backend::VulkanBackend::alloc_bytes_handle`] /
//! [`fuel_vulkan_backend::VulkanBackend::upload_bytes_handle`] /
//! [`fuel_vulkan_backend::VulkanBackend::download_bytes`].
//!
//! The full `DynBackendDevice` trait body lives on
//! [`fuel_vulkan_backend::VulkanBackendDevice`] (a wrapper around
//! `Arc<VulkanBackend>`); fuel-core knows about it only via the
//! `From` impl below.

use std::sync::Arc;

use fuel_vulkan_backend::{VulkanBackend, VulkanBackendDevice};

use crate::{Device, Error, Result};

impl From<VulkanBackend> for Device {
    fn from(backend: VulkanBackend) -> Self {
        Device::custom(Arc::new(VulkanBackendDevice::new(Arc::new(backend))))
    }
}

impl From<Arc<VulkanBackend>> for Device {
    fn from(backend: Arc<VulkanBackend>) -> Self {
        Device::custom(Arc::new(VulkanBackendDevice::new(backend)))
    }
}

/// Convenience constructor: pick the discrete GPU if available, fall
/// back to the first compute-capable device. Mirrors the CPU/CUDA
/// `Device::cpu()` / `cuda_backend::new_device` ergonomics.
pub fn new_device() -> Result<Device> {
    Ok(VulkanBackend::new()?.into())
}

/// Creates a Vulkan device for the physical GPU with this stable identity (UUID or PCI address),
/// resolved to an adapter index once, here. Unlike an index, the identity survives enumeration
/// reordering (ROADMAP row 125 part 2).
pub fn new_device_by_identity(wanted: &fuel_ir::probe::DeviceIdentity) -> Result<Device> {
    Ok(
        VulkanBackend::with_selection(fuel_vulkan_backend::DeviceSelection::Identity(*wanted))?
            .into(),
    )
}

/// The environment variable [`new_device_from_env`] reads.
pub const DEVICE_ENV_VAR: &str = "FUEL_VULKAN_DEVICE";

/// [`new_device_by_identity`] when `FUEL_VULKAN_DEVICE` holds a selector, else [`new_device`]. A
/// set-but-unparsable value is an error naming the variable, never a silent fall-back to the
/// default adapter.
pub fn new_device_from_env() -> Result<Device> {
    match fuel_ir::probe::identity_from_env(DEVICE_ENV_VAR) {
        Ok(Some(wanted)) => new_device_by_identity(&wanted),
        Ok(None) => new_device(),
        Err(msg) => Err(Error::Msg(msg).bt()),
    }
}

/// Returns the underlying [`Arc<VulkanBackend>`] handle, or an error
/// if `device` is not a Vulkan device.
///
/// The returned `Arc` is a fresh clone — callers feed it into
/// [`VulkanBackend::alloc_bytes_handle`] /
/// [`VulkanBackend::upload_bytes_handle`] for byte-storage allocation
/// + binding-table-dispatch reachability.
pub fn as_device(device: &Device) -> Result<Arc<VulkanBackend>> {
    device
        .inner
        .as_any()
        .downcast_ref::<VulkanBackendDevice>()
        .map(|wrapper| Arc::clone(wrapper.inner()))
        .ok_or_else(|| Error::Msg("expected a Vulkan device".into()).bt())
}
