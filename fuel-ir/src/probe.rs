// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cross-backend device enumeration — the foundation for Phase 6b's
//! `probe → judge → dispatch` lifecycle.
//!
//! # Why
//!
//! Fuel dispatches ops across multiple backends (CPU, reference, CUDA,
//! Vulkan, later Metal / MKL / AOCL). Each backend may see zero, one,
//! or many physical devices. A single physical GPU may be reachable
//! through more than one backend (an NVIDIA card is visible to both
//! CUDA and Vulkan) and will produce meaningfully different
//! performance profiles depending on which backend routes it. The
//! profiling unit — and therefore the dispatch-table key — is not
//! "backend" but `(backend, device)`.
//!
//! This module defines the data types that represent that pair in a
//! persistence-friendly shape. The actual enumeration logic lives in
//! each backend crate (a free function `enumerate_devices() ->
//! Result<Vec<DeviceDescriptor>>`, not a trait method — see "Why a
//! convention, not a trait" below). A top-level collector in
//! `fuel-graph-router` (or a future `fuel-judge`) calls every
//! backend's enumerator and assembles the combined device table.
//!
//! # Equivalence classes
//!
//! Four identical RTX 4090s through CUDA produce four essentially-
//! identical performance profiles — the Judge doesn't need to run
//! four times. Two [`DeviceDescriptor`]s that share an
//! [`EquivalenceKey`] can share a profile: profile once per key,
//! apply to every device in the class. Cross-backend equivalence
//! never holds (CUDA and Vulkan on the same silicon still key
//! differently because they have different driver paths and
//! submission overhead).
//!
//! # Why a convention, not a trait
//!
//! `BackendProbe` *could* be a trait implemented by each backend's
//! device or storage type. But enumeration is a **static** operation —
//! it doesn't need a live backend instance, and requiring one creates
//! a circular bootstrap problem (you can't know which backends exist
//! until you've enumerated them, but trait impls need types, and the
//! types may not be loadable if their runtime isn't present). The
//! cleaner shape is a plain free function per backend crate that can
//! be called via the crate's public API without constructing any
//! state. The [`BackendProbe`] trait below is a *documentation anchor*
//! — a marker trait each backend-crate-level probe function satisfies
//! in spirit, so the contract is in one place and the collector has
//! a single import to follow.

use crate::DeviceLocation;
use crate::error::Result;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Identifier for a Fuel backend implementation. Stable across runs
/// so the Judge's persisted profile tables remain valid when the
/// process restarts. Adding a new variant is a non-breaking change;
/// the enum is `#[non_exhaustive]` so downstream pattern matches stay
/// forward-compatible.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BackendId {
    /// `fuel-cpu-backend` — optimized CPU executor (gemm + rayon).
    /// One descriptor per physical CPU in the current scope; NUMA
    /// splits are Phase 7b territory.
    Cpu,
    /// `fuel-cuda-backend` via baracuda.
    Cuda,
    /// `fuel-vulkan-backend` via vulkane.
    Vulkan,
    /// `fuel-metal-backend` (future — not yet probe-wired).
    Metal,
    // `Aocl` and `Mkl` variants retired 2026-06-08 (backend-extensions
    // Phase 2). AOCL and oneMKL are now kernel-source extensions of
    // `BackendId::Cpu` — their kernels register as siblings at the
    // same binding-table key, distinguished by `BindingEntry::kernel_source`.
    // See `docs/architecture/05-backend-contract.md` §Trait surface.
}

impl BackendId {
    /// A short lowercase identifier suitable for log lines, directory
    /// paths, and JSON keys. Stable across Fuel versions.
    pub fn as_str(self) -> &'static str {
        match self {
            BackendId::Cpu => "cpu",
            BackendId::Cuda => "cuda",
            BackendId::Vulkan => "vulkan",
            BackendId::Metal => "metal",
        }
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything a Judge / dispatch table needs to know about a single
/// `(backend, device)` pair. Constructed by each backend's
/// `enumerate_devices()` and consumed by the collector + the Judge.
///
/// Field layout is stable: adding fields is non-breaking so long as
/// they're additive (`#[serde(default)]` on new optional fields keeps
/// older profile files loading).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeviceDescriptor {
    /// Which backend this descriptor belongs to.
    pub backend: BackendId,
    /// Zero-based index within the backend. For CPU this is always 0
    /// until NUMA splits arrive; for CUDA/Vulkan it's the device
    /// ordinal the backend's own API uses.
    pub device_index: u32,
    /// Human-readable device name as reported by the driver.
    /// Examples: `"NVIDIA GeForce RTX 4070"`, `"AMD Radeon RX 7900 XT"`,
    /// `"Intel(R) Core(TM) i9-14900K"`. Used primarily for logs.
    pub hardware_sku: String,
    /// PCI-SIG vendor ID (NVIDIA=0x10DE, AMD=0x1002, Intel=0x8086).
    /// `0` for CPU entries that don't map to a PCI vendor.
    pub vendor_id: u32,
    /// PCI-SIG device ID for discrete hardware; zero for CPU.
    pub device_id: u32,
    /// CUDA compute capability, when applicable. `None` for non-CUDA
    /// entries. Stored as `(major, minor)` — e.g. `(8, 9)` for sm_89.
    pub compute_capability: Option<(u32, u32)>,
    /// The **default** lanes per subgroup / wave / warp — the primary
    /// kernel-specialization axis on every SIMT backend. 32 on NVIDIA,
    /// 64 on AMD GCN/CDNA, 32 *or* 64 on RDNA, 8/16/32 on Intel.
    /// `None` when the backend cannot report it (CPU, or a Vulkan
    /// instance created below 1.1).
    ///
    /// **This is the default width a shader sees, NOT the set the
    /// device supports.** A device may *admit several*: an AMD 610M
    /// reports a default of 64 while admitting dynamic, 32 and 64.
    /// Vulkan exposes the pinnable range through
    /// `SubgroupProperties::size_control`, which is 1.3 core / the EXT
    /// — unreachable from Fuel's V1_2 instances today, so only the
    /// default is captured here.
    ///
    /// **Consequence if Fuel ever specializes kernels by wave width:
    /// `Option<u32>` is then the wrong shape**, not merely
    /// under-documented — a wave32-pinned kernel and a wave64 one are
    /// different cells, so the descriptor would need *the admissible
    /// set plus the default*. Mirror vulkane's split
    /// (`admissible_subgroups()` alongside `default_subgroup`) rather
    /// than widening this field in place.
    ///
    /// Deliberately **not** part of [`EquivalenceKey`]: it is a
    /// function of `(vendor_id, device_id)`, which the key already
    /// carries, so including it would add no discriminating power and
    /// would churn the Judge's cache for nothing.
    pub subgroup_width: Option<u32>,
    /// Driver version string as reported by the backend. Cache
    /// invalidation key: if the driver rev changes, Judge re-runs.
    pub driver_version: String,
    /// Total device-visible memory in bytes. Zero if unknown.
    pub total_memory_bytes: u64,
    /// Fuel's placement enum for this descriptor. Populated by the
    /// backend so the router can build `DeviceLocation` handles
    /// without re-querying.
    pub location: DeviceLocation,
}

impl DeviceDescriptor {
    /// The key the Judge uses to share a profile across identical
    /// devices. Two descriptors that compare equal here can share a
    /// single profile entry; the table's runtime dispatch just looks
    /// up the equivalence key rather than the full descriptor.
    pub fn equivalence_key(&self) -> EquivalenceKey {
        EquivalenceKey {
            backend: self.backend,
            vendor_id: self.vendor_id,
            device_id: self.device_id,
            compute_capability: self.compute_capability,
            driver_version: self.driver_version.clone(),
        }
    }
}

/// Hash key that groups [`DeviceDescriptor`]s likely to profile
/// identically. Explicitly **scoped by backend** — a CUDA RTX 4090
/// and a Vulkan RTX 4090 are the same silicon but different
/// submission paths, so they never share an equivalence class.
///
/// The driver version is part of the key because the same
/// `(vendor, device, capability)` can perform measurably
/// differently across driver releases; invalidating on driver
/// upgrade is cheap insurance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EquivalenceKey {
    pub backend: BackendId,
    pub vendor_id: u32,
    pub device_id: u32,
    pub compute_capability: Option<(u32, u32)>,
    pub driver_version: String,
}

/// Marker trait documenting the enumeration contract each backend
/// crate satisfies. **Not dispatch-through** — each backend provides
/// a plain `pub fn enumerate_devices()` at crate root, and the
/// collector calls them by name. This trait is a single place to
/// reference the expected signature.
///
/// See the module docs for why this is a convention rather than a
/// dispatch trait.
pub trait BackendProbe {
    /// Enumerate every `(backend, device)` pair this backend can
    /// currently reach. Empty vec = backend loaded but no usable
    /// hardware visible (e.g. CUDA runtime present but zero GPUs).
    /// `Err` = backend could not be probed at all (e.g. driver
    /// dynamic-load failed).
    fn enumerate_devices() -> Result<Vec<DeviceDescriptor>>;
}

impl DeviceLocation {
    /// The kind of backend that drives this place: the one definition of the mapping
    /// (`Cpu` to [`BackendId::Cpu`], each GPU variant to its backend, whatever the ordinal).
    pub fn backend(self) -> BackendId {
        match self {
            DeviceLocation::Cpu => BackendId::Cpu,
            DeviceLocation::Cuda { .. } => BackendId::Cuda,
            DeviceLocation::Vulkan { .. } => BackendId::Vulkan,
            DeviceLocation::Metal { .. } => BackendId::Metal,
        }
    }
}

impl BackendId {
    /// The place a backend's work goes when nothing names a device: GPU ordinal 0, or the CPU.
    /// The match is exhaustive inside this crate, so adding a `BackendId` variant fails to compile
    /// here until it is given a default place. The ordinal is a within-process enumeration index,
    /// not an identity (ROADMAP row 125); this is the one place that default lives.
    pub fn default_location(self) -> DeviceLocation {
        match self {
            BackendId::Cpu => DeviceLocation::Cpu,
            BackendId::Cuda => DeviceLocation::Cuda { gpu_id: 0 },
            BackendId::Vulkan => DeviceLocation::Vulkan { gpu_id: 0 },
            BackendId::Metal => DeviceLocation::Metal { gpu_id: 0 },
        }
    }
}

/// A PCI bus address as the OS and the GPU APIs report it. `function` is `None` when the source
/// does not say (CUDA reports domain, bus and device number only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PciAddress {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: Option<u32>,
}

/// What identifies ONE physical GPU across processes and across APIs (ROADMAP row 125). The
/// enumeration index (`DeviceLocation::{Cuda,Vulkan}.gpu_id`) is only a within-process handle: on
/// this box the Vulkan order flipped between two runs. Key placement on this, not on the index.
///
/// `uuid` is the 16-byte device UUID that Vulkan (`VkPhysicalDeviceIDProperties`), CUDA
/// (`cuDeviceGetUuid`), NVML and OpenGL agree on. It is `None` when the source could not report one
/// (and an all-zero value is treated as absent by [`Self::same_physical_device`]: a zeroed join key
/// reads as a valid UUID that every device shares). `pci` is the bus address, the fallback join key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeviceIdentity {
    pub uuid: Option<[u8; 16]>,
    pub pci: Option<PciAddress>,
}

impl DeviceIdentity {
    /// The UUID, if present and not all zero.
    pub fn usable_uuid(&self) -> Option<[u8; 16]> {
        self.uuid.filter(|u| u.iter().any(|b| *b != 0))
    }

    /// Do these two identities name the same physical GPU? UUIDs decide when both sides have a
    /// usable one (equal means same, different means different). Otherwise the PCI address decides
    /// (domain, bus and device, plus function when both sides report it). With neither, the answer
    /// is `false`: unknown is not "same".
    pub fn same_physical_device(&self, other: &DeviceIdentity) -> bool {
        if let (Some(a), Some(b)) = (self.usable_uuid(), other.usable_uuid()) {
            return a == b;
        }
        match (self.pci, other.pci) {
            (Some(a), Some(b)) => {
                a.domain == b.domain
                    && a.bus == b.bus
                    && a.device == b.device
                    && match (a.function, b.function) {
                        (Some(x), Some(y)) => x == y,
                        _ => true,
                    }
            }
            _ => false,
        }
    }
}

/// Why [`resolve_identity_ordinal`] found no single device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentityResolveError {
    /// No enumerated device is the wanted physical GPU (including: the wanted identity carries
    /// neither a usable UUID nor a PCI address, since unknown is never "same").
    NoMatch,
    /// More than one enumerated device claims to be the wanted GPU (their ordinals, ascending).
    /// Guessing the first would reintroduce the order dependence this exists to remove.
    Ambiguous(Vec<usize>),
}

/// Resolve a [`DeviceIdentity`] to the enumeration ordinal of THIS process (ROADMAP row 137,
/// step 4a). `enumerated[i]` is the identity of the device the process enumerates at ordinal `i`.
/// The answer is a within-process handle and must not be stored: the next process or build may
/// enumerate in a different order, which is the reason to ask by identity at all.
pub fn resolve_identity_ordinal(
    wanted: &DeviceIdentity,
    enumerated: &[DeviceIdentity],
) -> std::result::Result<usize, IdentityResolveError> {
    let hits: Vec<usize> = enumerated
        .iter()
        .enumerate()
        .filter(|(_, id)| wanted.same_physical_device(id))
        .map(|(i, _)| i)
        .collect();
    match hits.as_slice() {
        [] => Err(IdentityResolveError::NoMatch),
        [one] => Ok(*one),
        _ => Err(IdentityResolveError::Ambiguous(hits)),
    }
}

/// Why a device-selector string is not a [`DeviceIdentity`] (ROADMAP row 125, part 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentityParseError {
    /// The string is empty or only whitespace.
    Empty,
    /// It looks like a UUID (32 hex digits, dashes optional) but is not one.
    BadUuid,
    /// An all-zero UUID names no device (it reads as absent), so it is refused here rather than
    /// resolving to "no match" later.
    ZeroUuid,
    /// It looks like a PCI address but is not `[domain:]bus:device[.function]` in hex.
    BadPci,
}

impl std::fmt::Display for IdentityParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Empty => "empty device selector",
            Self::BadUuid => {
                "not a device UUID (want 8-4-4-4-12 hex digits, optional `GPU-` prefix)"
            }
            Self::ZeroUuid => "an all-zero UUID names no device",
            Self::BadPci => "not a PCI address (want [domain:]bus:device[.function], hex)",
        })
    }
}

impl std::error::Error for IdentityParseError {}

/// Parse a device selector a person can type: a UUID (`05c56c06-f951-3d45-5c8a-c76bbcb1deaa`, the
/// `GPU-` form `nvidia-smi -L` prints, or 32 bare hex digits) or a PCI address
/// (`01:00.0`, `0000:01:00.0`, `01:00`; hex, the function optional). The result is a
/// [`DeviceIdentity`] for the `*_by_identity` constructors; a UUID sets only `uuid`, a PCI address
/// only `pci`, so the match is made on exactly what the caller named.
impl std::str::FromStr for DeviceIdentity {
    type Err = IdentityParseError;

    fn from_str(text: &str) -> std::result::Result<Self, Self::Err> {
        let t = text.trim();
        if t.is_empty() {
            return Err(IdentityParseError::Empty);
        }
        let bare = t
            .strip_prefix("GPU-")
            .or_else(|| t.strip_prefix("gpu-"))
            .or_else(|| t.strip_prefix("MIG-"))
            .unwrap_or(t);
        // A PCI address has a `:`; a UUID never does.
        if bare.contains(':') {
            return parse_pci(bare).map(|pci| DeviceIdentity {
                uuid: None,
                pci: Some(pci),
            });
        }
        let hex: String = bare.chars().filter(|c| *c != '-').collect();
        if hex.len() != 32 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(IdentityParseError::BadUuid);
        }
        let mut uuid = [0u8; 16];
        for (i, byte) in uuid.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
                .map_err(|_| IdentityParseError::BadUuid)?;
        }
        if uuid.iter().all(|b| *b == 0) {
            return Err(IdentityParseError::ZeroUuid);
        }
        Ok(DeviceIdentity {
            uuid: Some(uuid),
            pci: None,
        })
    }
}

fn parse_pci(text: &str) -> std::result::Result<PciAddress, IdentityParseError> {
    let hex = |s: &str, max_digits: usize| -> std::result::Result<u32, IdentityParseError> {
        if s.is_empty() || s.len() > max_digits || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(IdentityParseError::BadPci);
        }
        u32::from_str_radix(s, 16).map_err(|_| IdentityParseError::BadPci)
    };
    let parts: Vec<&str> = text.split(':').collect();
    let (domain, bus, tail) = match parts.as_slice() {
        [bus, tail] => (0, hex(bus, 2)?, *tail),
        [domain, bus, tail] => (hex(domain, 4)?, hex(bus, 2)?, *tail),
        _ => return Err(IdentityParseError::BadPci),
    };
    let (device, function) = match tail.split_once('.') {
        Some((d, f)) => (hex(d, 2)?, Some(hex(f, 1)?)),
        None => (hex(tail, 2)?, None),
    };
    // PCI device numbers are 5 bits, functions 3 bits.
    if device > 0x1f || function.is_some_and(|f| f > 7) {
        return Err(IdentityParseError::BadPci);
    }
    Ok(PciAddress {
        domain,
        bus,
        device,
        function,
    })
}

/// Read a device selector from the environment variable `var` (for example `FUEL_CUDA_DEVICE`).
/// `Ok(None)` when it is unset or empty; an unparsable value is an error naming the variable, never
/// a silent fall-back to the default device (that would run on the wrong GPU without saying so).
pub fn identity_from_env(var: &str) -> std::result::Result<Option<DeviceIdentity>, String> {
    identity_from_value(var, std::env::var(var).ok().as_deref())
}

/// The pure half of [`identity_from_env`]: `value` is what the variable held, `None` if unset.
pub fn identity_from_value(
    var: &str,
    value: Option<&str>,
) -> std::result::Result<Option<DeviceIdentity>, String> {
    match value {
        Some(v) if !v.trim().is_empty() => v
            .parse::<DeviceIdentity>()
            .map(Some)
            .map_err(|e| format!("{var}={v:?}: {e}")),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res_ident(uuid_byte: u8, bus: u32) -> DeviceIdentity {
        DeviceIdentity {
            uuid: Some([uuid_byte; 16]),
            pci: Some(PciAddress {
                domain: 0,
                bus,
                device: 0,
                function: None,
            }),
        }
    }

    #[test]
    fn resolver_follows_the_device_across_both_enumeration_orders() {
        let nv = res_ident(0x11, 1);
        let amd = res_ident(0x22, 6);
        // Vulkan-only build order vs CUDA-plus-Vulkan build order (row 125 measurement).
        let amd_first = [amd, nv];
        let nv_first = [nv, amd];
        assert_eq!(resolve_identity_ordinal(&nv, &amd_first), Ok(1));
        assert_eq!(resolve_identity_ordinal(&nv, &nv_first), Ok(0));
        assert_eq!(resolve_identity_ordinal(&amd, &amd_first), Ok(0));
        assert_eq!(resolve_identity_ordinal(&amd, &nv_first), Ok(1));
    }

    #[test]
    fn resolver_declines_unknown_absent_and_ambiguous() {
        let nv = res_ident(0x11, 1);
        let amd = res_ident(0x22, 6);
        assert_eq!(
            resolve_identity_ordinal(&nv, &[amd]),
            Err(IdentityResolveError::NoMatch)
        );
        assert_eq!(
            resolve_identity_ordinal(&nv, &[]),
            Err(IdentityResolveError::NoMatch)
        );
        // Neither UUID nor PCI: unknown is never "same", even against another unknown.
        let none = DeviceIdentity {
            uuid: None,
            pci: None,
        };
        assert_eq!(
            resolve_identity_ordinal(&none, &[none]),
            Err(IdentityResolveError::NoMatch)
        );
        // An all-zero UUID is absent, so two zeroed devices on different buses stay distinct.
        let z = |bus| DeviceIdentity {
            uuid: Some([0; 16]),
            pci: Some(PciAddress {
                domain: 0,
                bus,
                device: 0,
                function: None,
            }),
        };
        assert_eq!(resolve_identity_ordinal(&z(1), &[z(6), z(1)]), Ok(1));
        // Two enumerated entries for the same GPU: refuse rather than take the first.
        assert_eq!(
            resolve_identity_ordinal(&nv, &[nv, amd, nv]),
            Err(IdentityResolveError::Ambiguous(vec![0, 2]))
        );
    }

    #[test]
    fn backend_id_as_str_is_stable() {
        assert_eq!(BackendId::Cpu.as_str(), "cpu");
        assert_eq!(BackendId::Cuda.as_str(), "cuda");
        assert_eq!(BackendId::Vulkan.as_str(), "vulkan");
        assert_eq!(BackendId::Metal.as_str(), "metal");
    }

    #[test]
    fn equivalence_key_collapses_identical_cuda_devices() {
        let make = |idx: u32| DeviceDescriptor {
            backend: BackendId::Cuda,
            device_index: idx,
            hardware_sku: "NVIDIA GeForce RTX 4090".to_string(),
            vendor_id: 0x10DE,
            device_id: 0x2684,
            compute_capability: Some((8, 9)),
            subgroup_width: Some(32), // NVIDIA warp
            driver_version: "550.54.14".to_string(),
            total_memory_bytes: 25_769_803_776,
            location: DeviceLocation::Cuda {
                gpu_id: idx as usize,
            },
        };
        let a = make(0);
        let b = make(1);
        // Same silicon, different ordinals → same equivalence class.
        assert_eq!(a.equivalence_key(), b.equivalence_key());
        // Different device_index → descriptors still distinct overall.
        assert_ne!(a, b);
    }

    #[test]
    fn equivalence_key_splits_same_silicon_across_backends() {
        let cuda = DeviceDescriptor {
            backend: BackendId::Cuda,
            device_index: 0,
            hardware_sku: "NVIDIA GeForce RTX 4090".to_string(),
            vendor_id: 0x10DE,
            device_id: 0x2684,
            compute_capability: Some((8, 9)),
            subgroup_width: Some(32),
            driver_version: "550.54.14".to_string(),
            total_memory_bytes: 25_769_803_776,
            location: DeviceLocation::Cuda { gpu_id: 0 },
        };
        let vulkan = DeviceDescriptor {
            backend: BackendId::Vulkan,
            device_index: 0,
            hardware_sku: "NVIDIA GeForce RTX 4090".to_string(),
            vendor_id: 0x10DE,
            device_id: 0x2684,
            compute_capability: None, // Vulkan doesn't expose CUDA CC.
            subgroup_width: Some(32),
            driver_version: "550.54.14".to_string(),
            total_memory_bytes: 25_769_803_776,
            location: DeviceLocation::Cpu, // placeholder; vulkane uses a different variant
        };
        // Same chip, different submission paths → different classes.
        assert_ne!(cuda.equivalence_key(), vulkan.equivalence_key());
    }

    #[test]
    fn equivalence_key_splits_on_driver_version() {
        let base = DeviceDescriptor {
            backend: BackendId::Cuda,
            device_index: 0,
            hardware_sku: "X".to_string(),
            vendor_id: 1,
            device_id: 2,
            compute_capability: Some((8, 9)),
            subgroup_width: Some(32),
            driver_version: "550".to_string(),
            total_memory_bytes: 0,
            location: DeviceLocation::Cuda { gpu_id: 0 },
        };
        let newer = DeviceDescriptor {
            driver_version: "560".to_string(),
            ..base.clone()
        };
        assert_ne!(base.equivalence_key(), newer.equivalence_key());
    }

    /// `subgroup_width` is a function of `(vendor_id, device_id)`, which the
    /// key already carries, so it must NOT participate in the equivalence
    /// key — otherwise a backend that learns to report it (or stops) would
    /// churn every cached Judge profile for no discriminating gain.
    /// Pins the documented invariant on `DeviceDescriptor::subgroup_width`.
    #[test]
    fn equivalence_key_ignores_subgroup_width() {
        let base = DeviceDescriptor {
            backend: BackendId::Vulkan,
            device_index: 0,
            hardware_sku: "X".to_string(),
            vendor_id: 1,
            device_id: 2,
            compute_capability: None,
            subgroup_width: None,
            driver_version: "550".to_string(),
            total_memory_bytes: 0,
            location: DeviceLocation::Vulkan { gpu_id: 0 },
        };
        let with_width = DeviceDescriptor {
            subgroup_width: Some(64),
            ..base.clone()
        };
        // Descriptors differ...
        assert_ne!(base, with_width);
        // ...but they profile identically, so the key must collapse them.
        assert_eq!(base.equivalence_key(), with_width.equivalence_key());
    }

    /// Every GPU ordinal maps to the same backend, and the CPU to the CPU.
    #[test]
    fn location_to_backend_ignores_the_ordinal() {
        assert_eq!(DeviceLocation::Cpu.backend(), BackendId::Cpu);
        for gpu_id in [0usize, 1, 7] {
            assert_eq!(DeviceLocation::Cuda { gpu_id }.backend(), BackendId::Cuda);
            assert_eq!(
                DeviceLocation::Vulkan { gpu_id }.backend(),
                BackendId::Vulkan
            );
            assert_eq!(DeviceLocation::Metal { gpu_id }.backend(), BackendId::Metal);
        }
    }

    /// The default location is ordinal 0 (or the CPU), and mapping it back gives the same backend,
    /// for every backend this crate knows.
    #[test]
    fn default_location_round_trips_for_every_backend() {
        for b in [
            BackendId::Cpu,
            BackendId::Cuda,
            BackendId::Vulkan,
            BackendId::Metal,
        ] {
            assert_eq!(b.default_location().backend(), b, "{b:?}");
        }
        assert_eq!(
            BackendId::Cuda.default_location(),
            DeviceLocation::Cuda { gpu_id: 0 }
        );
        assert_eq!(BackendId::Cpu.default_location(), DeviceLocation::Cpu);
    }

    fn ident(uuid: Option<u8>, pci: Option<(u32, u32, u32, Option<u32>)>) -> DeviceIdentity {
        DeviceIdentity {
            uuid: uuid.map(|b| [b; 16]),
            pci: pci.map(|(domain, bus, device, function)| PciAddress {
                domain,
                bus,
                device,
                function,
            }),
        }
    }

    /// UUIDs decide when both sides have one; the PCI address cannot overrule them.
    #[test]
    fn identity_uuid_decides_when_both_sides_have_one() {
        let a = ident(Some(1), Some((0, 1, 0, None)));
        assert!(a.same_physical_device(&ident(Some(1), Some((0, 9, 9, None)))));
        assert!(!a.same_physical_device(&ident(Some(2), Some((0, 1, 0, None)))));
    }

    /// An all-zero UUID is absent, not a value two devices share (the vulkane V1_0 trap).
    #[test]
    fn identity_zero_uuid_is_absent_so_pci_decides() {
        let zero = DeviceIdentity {
            uuid: Some([0; 16]),
            pci: Some(PciAddress {
                domain: 0,
                bus: 3,
                device: 0,
                function: Some(0),
            }),
        };
        let other_zero_other_bus = DeviceIdentity {
            uuid: Some([0; 16]),
            pci: Some(PciAddress {
                domain: 0,
                bus: 4,
                device: 0,
                function: Some(0),
            }),
        };
        assert!(!zero.same_physical_device(&other_zero_other_bus));
        assert!(zero.same_physical_device(&zero));
    }

    /// PCI fallback: a missing function on one side matches (CUDA does not report it); a present,
    /// different function does not. Neither source having anything is "unknown", not "same".
    #[test]
    fn identity_pci_fallback_and_unknown() {
        let cuda = ident(None, Some((0, 1, 0, None)));
        assert!(cuda.same_physical_device(&ident(None, Some((0, 1, 0, Some(0))))));
        assert!(!cuda.same_physical_device(&ident(None, Some((0, 2, 0, Some(0))))));
        assert!(
            !ident(Some(1), Some((0, 1, 0, Some(1))))
                .same_physical_device(&ident(None, Some((0, 1, 0, Some(0)))))
        );
        assert!(!ident(None, None).same_physical_device(&ident(None, None)));
    }

    const NV: &str = "05c56c06-f951-3d45-5c8a-c76bbcb1deaa";

    #[test]
    fn selector_parses_uuid_in_every_written_form() {
        let want: DeviceIdentity = NV.parse().unwrap();
        assert_eq!(want.uuid.unwrap()[0], 0x05);
        assert_eq!(want.uuid.unwrap()[15], 0xaa);
        assert_eq!(want.pci, None);
        assert_eq!(format!("GPU-{NV}").parse::<DeviceIdentity>(), Ok(want));
        assert_eq!(NV.replace('-', "").parse::<DeviceIdentity>(), Ok(want));
        assert_eq!(
            format!("  {}\n", NV.to_uppercase()).parse::<DeviceIdentity>(),
            Ok(want)
        );
    }

    #[test]
    fn selector_parses_pci_addresses() {
        let p = |s: &str| s.parse::<DeviceIdentity>().unwrap().pci.unwrap();
        let a = p("01:00.0");
        assert_eq!((a.domain, a.bus, a.device, a.function), (0, 1, 0, Some(0)));
        let b = p("0000:0a:1f.7");
        assert_eq!(
            (b.domain, b.bus, b.device, b.function),
            (0, 0x0a, 0x1f, Some(7))
        );
        let c = p("0001:81:00");
        assert_eq!((c.domain, c.bus, c.device, c.function), (1, 0x81, 0, None));
        assert_eq!("01:00.0".parse::<DeviceIdentity>().unwrap().uuid, None);
    }

    #[test]
    fn selector_refuses_what_is_not_a_device() {
        use IdentityParseError::*;
        let e = |s: &str| s.parse::<DeviceIdentity>().unwrap_err();
        assert_eq!(e(""), Empty);
        assert_eq!(e("   "), Empty);
        assert_eq!(e("0"), BadUuid);
        assert_eq!(e("zzzzzzzz-f951-3d45-5c8a-c76bbcb1deaa"), BadUuid);
        assert_eq!(e(&NV[..35]), BadUuid);
        assert_eq!(e("00000000-0000-0000-0000-000000000000"), ZeroUuid);
        assert_eq!(e("01:00.8"), BadPci); // function is 3 bits
        assert_eq!(e("01:20.0"), BadPci); // device is 5 bits
        assert_eq!(e("01:"), BadPci);
        assert_eq!(e("1:2:3:4"), BadPci);
        assert_eq!(e("012:00.0"), BadPci); // bus is 2 digits
    }

    /// A parsed selector resolves through the same matcher the constructors use: the UUID picks
    /// the right device whatever the enumeration order, and a PCI-only selector falls back to the
    /// bus address (the iGPU case: no usable UUID).
    #[test]
    fn selector_resolves_against_an_enumeration_in_either_order() {
        let nv: DeviceIdentity = NV.parse().unwrap();
        let other: DeviceIdentity = "11111111-2222-3333-4444-555555555555".parse().unwrap();
        assert_eq!(resolve_identity_ordinal(&nv, &[nv, other]), Ok(0));
        assert_eq!(resolve_identity_ordinal(&nv, &[other, nv]), Ok(1));
        let gpu = |bus| DeviceIdentity {
            uuid: None,
            pci: Some(PciAddress {
                domain: 0,
                bus,
                device: 0,
                function: None,
            }),
        };
        let want: DeviceIdentity = "0000:03:00.0".parse().unwrap();
        assert_eq!(resolve_identity_ordinal(&want, &[gpu(1), gpu(3)]), Ok(1));
        assert_eq!(
            resolve_identity_ordinal(&want, &[gpu(1), gpu(2)]),
            Err(IdentityResolveError::NoMatch)
        );
    }

    #[test]
    fn env_selector_is_unset_empty_good_or_a_named_error() {
        let var = "FUEL_CUDA_DEVICE";
        assert_eq!(identity_from_value(var, None), Ok(None));
        assert_eq!(identity_from_value(var, Some("  ")), Ok(None));
        assert_eq!(
            identity_from_value(var, Some(NV)),
            Ok(Some(NV.parse().unwrap()))
        );
        let err = identity_from_value(var, Some("not-a-gpu")).unwrap_err();
        assert!(
            err.contains(var) && err.contains("not a device UUID"),
            "{err}"
        );
    }
}
