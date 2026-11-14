// SPDX-License-Identifier: MIT OR Apache-2.0
//! Zero-copy GGUF loading via memory-mapped files.
//!
//! `MmapedContent` opens a GGUF file once, mmaps it, parses the header
//! using a `Cursor` over the mmap bytes (no syscalls), and lets the
//! caller pull tensors by slicing directly into the mmap — no seek,
//! no `read_exact`, no per-tensor allocation. For models with hundreds
//! of tensors this is 2-10x faster than the seek-based `Content::tensor`
//! path.
//!
//! The `Arc<Mmap>` is retained for the lifetime of `MmapedContent`, so
//! callers may slice into the mapping for as long as they hold the
//! struct (or a clone of the inner `Arc`, via [`MmapedContent::mmap`]).
//!
//! B6 removed the eager `tensor(&self, name, device) -> QTensor` accessor.
//! Callers now take [`MmapedContent::mmap`] plus the `TensorInfo` from
//! [`MmapedContent::content`] and decode the bytes themselves — which is
//! what the lazy loader already did.
//!
//! **Sharded models.** A `gguf-split` model (`<prefix>-00001-of-0000N.gguf`) opens as ONE
//! [`MmapedContent`]: pass any [`MmapedContent::from_path`] the first shard's path (the other
//! shards are found next to it by name) or give the whole set to [`MmapedContent::from_paths`].
//! Every shard is mapped; none is merged or copied. Tensor positions are then in the shards laid
//! end to end (see [`Content::from_shards`]), so read bytes with [`MmapedContent::bytes_at`],
//! which resolves a position to its shard, rather than slicing [`MmapedContent::mmap`].

use super::arch::{Architecture, detect_from_gguf};
use super::gguf_file::{Content, Value};
use crate::model_progress::{ProgressEvent, ProgressReporter};
use fuel_ir::error::Result;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A GGUF file parsed from an mmapped backing buffer. Tensor reads are
/// zero-syscall slice copies from the mmap.
pub struct MmapedContent {
    /// One mapping per shard, in `split.no` order; a single file is one entry.
    shards: Vec<Arc<Mmap>>,
    /// `bases[i]` is where shard `i` starts in the end-to-end position space.
    bases: Vec<u64>,
    content: Content,
    path: PathBuf,
}

impl MmapedContent {
    /// Open `path`, mmap it, and parse the GGUF header.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_path_with_progress(path, &ProgressReporter::silent())
    }

    /// Like [`from_path`](Self::from_path), but announces `OpeningFile`
    /// and `MetadataParsed` events through `progress` so UIs can show
    /// activity during the header parse (the parse itself is fast, but
    /// the mmap + page fault on a cold model file can take a second
    /// or two on spinning disks).
    pub fn from_path_with_progress<P: AsRef<Path>>(
        path: P,
        progress: &ProgressReporter,
    ) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        if let Some(siblings) = split_siblings(&path_buf)? {
            return Self::from_paths_with_progress(&siblings, progress);
        }
        progress.emit(&ProgressEvent::OpeningFile {
            path: path_buf.clone(),
            format: "gguf",
        });
        let file = File::open(&path_buf)?;
        // SAFETY: We hold the file-backed mmap in an Arc for the life
        // of this struct, and we never mutate the mapped region.
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);
        let content = Content::from_bytes(&mmap[..], &path_buf)?;
        progress.emit(&ProgressEvent::MetadataParsed {
            format: "gguf",
            tensor_count: content.tensor_infos.len(),
        });
        if progress.is_enabled() {
            let arch = detect_from_gguf(&content);
            progress.emit(&ProgressEvent::ArchitectureDetected {
                architecture: arch.as_str().to_string(),
            });
        }
        Ok(Self {
            shards: vec![mmap],
            bases: vec![0],
            content,
            path: path_buf,
        })
    }

    /// Open the shards of one `gguf-split` model, given in any order, as one model. They must be
    /// exactly one complete set (`mlmf-gguf` checks count, numbering and `split.*` keys); a set
    /// with a missing or foreign shard is a typed error naming what is wrong.
    pub fn from_paths<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let paths: Vec<PathBuf> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();
        Self::from_paths_with_progress(&paths, &ProgressReporter::silent())
    }

    /// Like [`from_paths`](Self::from_paths), announcing progress for each shard opened.
    pub fn from_paths_with_progress(
        paths: &[PathBuf],
        progress: &ProgressReporter,
    ) -> Result<Self> {
        let first = paths.first().ok_or_else(|| {
            fuel_ir::error::Error::Msg("gguf: an empty list of shard paths".to_string()).bt()
        })?;
        let mut maps = Vec::with_capacity(paths.len());
        let mut names = Vec::with_capacity(paths.len());
        for p in paths {
            progress.emit(&ProgressEvent::OpeningFile {
                path: p.clone(),
                format: "gguf",
            });
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| {
                    fuel_ir::error::Error::Msg("gguf: a shard path has no UTF-8 file name".into())
                        .with_path(p)
                })?
                .to_string();
            let file = File::open(p).map_err(|e| {
                fuel_ir::error::Error::Msg(format!("gguf: opening shard: {e}")).with_path(p)
            })?;
            // SAFETY: each mapping is held in an Arc for the life of this struct and never
            // mutated.
            let mmap = unsafe { Mmap::map(&file)? };
            maps.push(Arc::new(mmap));
            names.push(name);
        }
        let inputs: Vec<(&str, &[u8])> = names
            .iter()
            .zip(&maps)
            .map(|(n, m)| (n.as_str(), &m[..]))
            .collect();
        let (content, order) = Content::from_shards(&inputs, first)?;
        // Put the mappings in `split.no` order: the end-to-end positions use that order.
        let mut shards = Vec::with_capacity(order.len());
        for n in &order {
            let i = names.iter().position(|x| x == n).ok_or_else(|| {
                fuel_ir::error::Error::Msg(format!("gguf: shard {n:?} missing after ordering"))
                    .with_path(first)
            })?;
            shards.push(Arc::clone(&maps[i]));
        }
        let mut bases = Vec::with_capacity(shards.len());
        let mut at = 0u64;
        for m in &shards {
            bases.push(at);
            at += m.len() as u64;
        }
        progress.emit(&ProgressEvent::MetadataParsed {
            format: "gguf",
            tensor_count: content.tensor_infos.len(),
        });
        if progress.is_enabled() {
            let arch = detect_from_gguf(&content);
            progress.emit(&ProgressEvent::ArchitectureDetected {
                architecture: arch.as_str().to_string(),
            });
        }
        Ok(Self {
            shards,
            bases,
            content,
            path: first.clone(),
        })
    }

    /// How many files back this model (1 for an ordinary GGUF).
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// The `len` bytes at tensor position `start` (`content().tensor_data_offset +
    /// info.offset as u64` for a tensor `info`): a slice of the one mapping, or of the shard the
    /// position falls in. A range outside every shard, or one that crosses a shard boundary, is a
    /// typed error, never a panic or a read into the next shard.
    pub fn bytes_at(&self, start: usize, len: usize) -> Result<&[u8]> {
        let start = start as u64;
        let shard = self
            .bases
            .partition_point(|b| *b <= start)
            .saturating_sub(1);
        let (map, base) = (&self.shards[shard], self.bases[shard]);
        let local = (start - base) as usize;
        local
            .checked_add(len)
            .and_then(|end| map.get(local..end))
            .ok_or_else(|| {
                fuel_ir::error::Error::Msg(format!(
                    "gguf: tensor bytes {start}..+{len} are outside shard {shard} ({} bytes, starts at {base}) of {} shard(s)",
                    map.len(),
                    self.shards.len()
                ))
                .with_path(&self.path)
            })
    }

    /// Raw metadata map, identical to `Content::metadata`.
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.content.metadata
    }

    /// Borrow the underlying `Content` for tokenizer construction or
    /// any caller that wants the raw tensor-info table / metadata.
    pub fn content(&self) -> &Content {
        &self.content
    }

    /// Return the underlying `Arc<Mmap>` so callers can keep the
    /// mapping alive independently (e.g. when handing tensors into
    /// a model that outlives this struct, keep a clone of the Arc).
    ///
    /// **Single file only.** For a sharded model this is the FIRST shard's mapping, and a tensor
    /// position is not an offset into it: use [`MmapedContent::bytes_at`] or
    /// [`MmapedContent::shard_mmaps`].
    pub fn mmap(&self) -> Arc<Mmap> {
        Arc::clone(&self.shards[0])
    }

    /// Every shard's mapping in `split.no` order (one entry for an ordinary GGUF).
    pub fn shard_mmaps(&self) -> &[Arc<Mmap>] {
        &self.shards
    }

    /// List tensor names known to the file.
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.content.tensor_infos.keys()
    }

    /// Detect the model architecture using GGUF `general.architecture`
    /// metadata, falling back to tensor-name pattern matching.
    pub fn architecture(&self) -> Architecture {
        detect_from_gguf(&self.content)
    }

    /// Path this file was opened from (useful for progress/log sinks).
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Consume `self` and return the parsed `Content` alongside the
    /// Arc'd mmap. Use this when you need to hand ownership of
    /// `Content` to an existing loader API (e.g.
    /// `ModelWeights::from_gguf(ct, reader, device)`) while keeping
    /// the mmap alive for zero-syscall tensor reads via
    /// `Cursor::new(&mmap[..])`.
    ///
    /// **Single file only** (the first shard's mapping for a sharded model; see
    /// [`MmapedContent::mmap`]).
    pub fn into_parts(self) -> (Arc<Mmap>, Content) {
        let first = Arc::clone(&self.shards[0]);
        (first, self.content)
    }
}

/// `Some(all shard paths)` when `path` is named like a `gguf-split` shard
/// (`<prefix>-NNNNN-of-MMMMM.gguf`): the whole set next to it, found by name. `None` for any other
/// file name, which opens as a single GGUF exactly as before.
fn split_siblings(path: &Path) -> Result<Option<Vec<PathBuf>>> {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return Ok(None);
    };
    let Some(parsed) = mlmf_gguf::split::parse_split_file_name(name) else {
        return Ok(None);
    };
    if parsed.count <= 1 {
        return Ok(None);
    }
    let names = mlmf_gguf::split::shard_file_names(&parsed.prefix, parsed.count).map_err(|e| {
        fuel_ir::error::Error::Msg(format!("gguf: shard names: {e}")).with_path(path)
    })?;
    let dir = path.parent().unwrap_or_else(|| Path::new(""));
    Ok(Some(names.iter().map(|n| dir.join(n)).collect()))
}
