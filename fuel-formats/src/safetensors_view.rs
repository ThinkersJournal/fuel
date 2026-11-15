// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safetensors mmap/buffer views: [`MmapedSafetensors`] and [`BufferedSafetensors`] hand out
//! `safetensors::TensorView`s and nothing more (the caller decodes the bytes).
//!
//! Moved here from `fuel-loaders` (ROADMAP row 156, fuel#680, the ARC-004 cure): model constructors named
//! the type in 168 public signatures, so it lives with the format crate, next to `MmapedFile`. It is the
//! same type as before: `fuel_formats::safetensors` re-exports it, so no caller changes.
use fuel_ir::{Error, Result};
use mlmf_core::{DType as MlmfDType, Encoding, TensorContainer, UnrecognizedKind};
use mlmf_safetensors::{SafetensorsTensors, parse_header, parse_tensors};
use safetensors::tensor as st;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// MLMF move, slice B (docs/design-notes/2026-10-24-mlmf-move-survey.md): the header and the
// tensor directory are read by `mlmf-safetensors`; this module keeps the mmap, the public API
// and the `safetensors::TensorView` type the 226 callers already use. Equivalence with the
// upstream-`safetensors` reader this replaced: `tests/safetensors_equivalence.rs`.

/// One file's parsed directory, borrowing the bytes it was parsed from.
#[derive(yoke::Yokeable)]
struct Parsed<'a> {
    tensors: SafetensorsTensors<'a>,
    /// Tensors whose dtype `mlmf-safetensors` cannot resolve: omitted from the directory, so
    /// `get` names them instead of reporting them missing.
    declined: HashSet<String>,
}

fn mlmf_err(what: &str, origin: &str, e: impl std::fmt::Display) -> Error {
    Error::Msg(format!("safetensors {what} of {origin}: {e}")).bt()
}

/// Parse `bytes` (one whole safetensors file) into a directory, refusing a file whose tensors
/// point outside it (the upstream reader refused those at open, and so do we).
fn parse_bytes<'a>(bytes: &'a [u8], origin: &str) -> Result<Parsed<'a>> {
    let header = parse_header(bytes).map_err(|e| mlmf_err("header", origin, e))?;
    let (tensors, report) =
        parse_tensors(bytes, &header, origin).map_err(|e| mlmf_err("directory", origin, e))?;
    let declined = report
        .entries()
        .iter()
        .filter_map(|u| match &u.kind {
            UnrecognizedKind::TensorEncoding { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    let len = bytes.len() as u64;
    for d in tensors.tensors() {
        if d.bytes.end > len {
            return Err(mlmf_err(
                "directory",
                origin,
                format!(
                    "tensor {:?} ends at byte {} of a {len}-byte file (truncated?)",
                    d.name, d.bytes.end
                ),
            ));
        }
    }
    Ok(Parsed { tensors, declined })
}

fn st_dtype(d: MlmfDType) -> Option<st::Dtype> {
    Some(match d {
        MlmfDType::F64 => st::Dtype::F64,
        MlmfDType::F32 => st::Dtype::F32,
        MlmfDType::F16 => st::Dtype::F16,
        MlmfDType::BF16 => st::Dtype::BF16,
        MlmfDType::F8E4M3 => st::Dtype::F8_E4M3,
        MlmfDType::F8E5M2 => st::Dtype::F8_E5M2,
        MlmfDType::I64 => st::Dtype::I64,
        MlmfDType::I32 => st::Dtype::I32,
        MlmfDType::I16 => st::Dtype::I16,
        MlmfDType::I8 => st::Dtype::I8,
        MlmfDType::U64 => st::Dtype::U64,
        MlmfDType::U32 => st::Dtype::U32,
        MlmfDType::U16 => st::Dtype::U16,
        MlmfDType::U8 => st::Dtype::U8,
        MlmfDType::Bool => st::Dtype::BOOL,
        MlmfDType::C64 => st::Dtype::C64,
        _ => return None,
    })
}

impl<'a> Parsed<'a> {
    fn view(&self, name: &str) -> Result<st::TensorView<'_>> {
        let Some(d) = self.tensors.tensor(name) else {
            let why = if self.declined.contains(name) {
                "its dtype is not one `mlmf-safetensors` resolves"
            } else {
                "no such tensor"
            };
            return Err(Error::Msg(format!("safetensors tensor {name:?}: {why}")).bt());
        };
        let Encoding::Dense(dtype) = d.encoding else {
            return Err(
                Error::Msg(format!("safetensors tensor {name:?}: not a dense tensor")).bt(),
            );
        };
        let dtype = st_dtype(dtype).ok_or_else(|| {
            Error::Msg(format!(
                "safetensors tensor {name:?}: dtype {dtype:?} has no safetensors spelling"
            ))
            .bt()
        })?;
        let bytes = match self.tensors.tensor_bytes(d) {
            Ok(std::borrow::Cow::Borrowed(b)) => b,
            Ok(std::borrow::Cow::Owned(_)) => {
                return Err(Error::Msg(format!(
                    "safetensors tensor {name:?}: bytes are not borrowed"
                ))
                .bt());
            }
            Err(e) => return Err(mlmf_err("tensor bytes", name, e)),
        };
        st::TensorView::new(dtype, d.shape.dims().to_vec(), bytes)
            .map_err(|e| mlmf_err("tensor view", name, format!("{e:?}")))
    }

    fn views(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.tensors
            .tensors()
            .iter()
            .filter_map(|d| self.view(&d.name).ok().map(|v| (d.name.clone(), v)))
            .collect()
    }
}

/// Memory-mapped access to one or more safetensors files.
///
/// This is the recommended way to load large models because tensors are read from disk on
/// demand via `mmap` rather than being fully loaded into memory upfront. When multiple files
/// are provided (via [`MmapedSafetensors::multi`]), a routing table maps tensor names to the
/// correct file.
///
/// # Safety
///
/// Construction is `unsafe` because it relies on memory-mapped I/O
/// ([`memmap2::MmapOptions`]). The caller must ensure the underlying files are not modified
/// or truncated while the `MmapedSafetensors` is alive.
///
/// # Example
///
/// ```no_run
/// use fuel_formats::safetensors::MmapedSafetensors;
/// // SAFETY: the file must not be modified while the mapping is alive.
/// let st = unsafe { MmapedSafetensors::new("weights.safetensors")? };
/// let view = st.get("weight")?;   // raw TensorView; caller decodes the bytes
/// # Ok::<(), fuel_ir::Error>(())
/// ```
pub struct MmapedSafetensors {
    safetensors: Vec<yoke::Yoke<Parsed<'static>, memmap2::Mmap>>,
    routing: Option<HashMap<String, usize>>,
}

fn map_file(p: &Path) -> Result<yoke::Yoke<Parsed<'static>, memmap2::Mmap>> {
    let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
    // SAFETY: inherited from the caller of `new`/`multi`, who promises the file is not modified
    // or truncated while the mapping is alive.
    let map = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| Error::from(e).with_path(p))?
    };
    let origin = p.display().to_string();
    yoke::Yoke::<Parsed<'static>, memmap2::Mmap>::try_attach_to_cart(map, |data: &[u8]| {
        parse_bytes(data, &origin)
    })
}

impl MmapedSafetensors {
    /// Creates a wrapper around a memory mapped file and deserialize the safetensors header.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        Ok(Self {
            safetensors: vec![map_file(p.as_ref())?],
            routing: None,
        })
    }

    /// Creates a wrapper around multiple memory mapped file and deserialize the safetensors headers.
    ///
    /// If a tensor name appears in multiple files, the last entry is returned.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut routing = HashMap::new();
        let mut safetensors = vec![];
        for (index, p) in paths.iter().enumerate() {
            let data = map_file(p.as_ref())?;
            for d in data.get().tensors.tensors() {
                routing.insert(d.name.clone(), index);
            }
            safetensors.push(data)
        }
        Ok(Self {
            safetensors,
            routing: Some(routing),
        })
    }

    /// Return metadata (name, dtype, shape) for every tensor across all mapped files.
    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        let mut tensors = vec![];
        for safetensors in self.safetensors.iter() {
            tensors.push(safetensors.get().views())
        }
        tensors.into_iter().flatten().collect()
    }

    /// Retrieve the raw `TensorView` for a tensor by name without loading it onto a device.
    ///
    /// This is useful for inspecting tensor metadata (dtype, shape) before deciding
    /// whether to materialize it.
    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        let index = match &self.routing {
            None => 0,
            Some(routing) => {
                let index = routing.get(name).ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: name.to_string(),
                    }
                    .bt()
                })?;
                *index
            }
        };
        self.safetensors[index].get().view(name)
    }
}

/// Owning wrapper around a `Vec<u8>` containing safetensors data.
///
/// Use this when the data source (e.g. a file read or a network download) hands
/// you an owned `Vec<u8>` rather than a path. For the memory-mapped equivalent,
/// see [`MmapedSafetensors`].
///
/// # Example
///
/// ```no_run
/// use fuel_formats::safetensors::BufferedSafetensors;
/// let bytes: Vec<u8> = std::fs::read("weights.safetensors")?;
/// let st = BufferedSafetensors::new(bytes)?;
/// let view = st.get("weight")?;   // raw TensorView; caller decodes the bytes
/// # Ok::<(), fuel_ir::Error>(())
/// ```
pub struct BufferedSafetensors {
    safetensors: yoke::Yoke<Parsed<'static>, Vec<u8>>,
}

impl BufferedSafetensors {
    /// Creates a wrapper around a binary buffer and deserialize the safetensors header.
    pub fn new(buffer: Vec<u8>) -> Result<Self> {
        let safetensors =
            yoke::Yoke::<Parsed<'static>, Vec<u8>>::try_attach_to_cart(buffer, |data: &[u8]| {
                parse_bytes(data, "an in-memory buffer")
            })?;
        Ok(Self { safetensors })
    }

    /// Return metadata for every tensor in the buffer.
    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors.get().views()
    }

    /// Retrieve the raw `TensorView` for a tensor by name without loading it onto a device.
    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        self.safetensors.get().view(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The view surface must hand back raw bytes without ever materializing a
    /// tensor — this is the exact path the lazy stack takes (`get` → `data()`).
    #[test]
    fn buffered_get_returns_raw_view() -> Result<()> {
        // A minimal hand-built safetensors buffer: one U8 tensor `x` = [1, 3].
        let bytes: &[u8] =
            b"8\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   \x01\x03";
        let st = BufferedSafetensors::new(bytes.to_vec())?;

        let view = st.get("x")?;
        assert_eq!(view.shape(), &[2]);
        assert_eq!(view.dtype(), st::Dtype::U8);
        assert_eq!(view.data(), &[1, 3]);

        let names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["x".to_string()]);
        Ok(())
    }

    /// A missing name is an error, not a panic.
    #[test]
    fn buffered_get_missing_name_is_an_error() {
        let bytes: &[u8] =
            b"8\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   \x01\x03";
        let st = BufferedSafetensors::new(bytes.to_vec()).unwrap();
        assert!(st.get("nope").is_err());
    }
}
