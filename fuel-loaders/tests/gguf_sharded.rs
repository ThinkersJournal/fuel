// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sharded (`gguf-split`) GGUF loading (fuel#660, Lightbulb B5): a model split into
//! `<prefix>-00001-of-0000N.gguf` files opens as ONE `MmapedContent`, and each tensor's bytes are
//! read from the shard that holds them. The fixtures are synthetic GGUF files built byte for byte
//! (the same builders as `gguf_open_mlmf_adapter.rs`); each shard's tensor holds DISTINCT values,
//! so a read that landed in the wrong shard would show different floats, not just a different
//! length.

use byteorder::{LittleEndian, WriteBytesExt};
use fuel_loaders::quantized::gguf_mmap::MmapedContent;
use std::path::PathBuf;

const GGUF_MAGIC: u32 = 0x4655_4747; // b"GGUF" little-endian
const ALIGN: u64 = 32;

fn string_entry(key: &str, value: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_u64::<LittleEndian>(key.len() as u64).unwrap();
    b.extend_from_slice(key.as_bytes());
    b.write_u32::<LittleEndian>(8).unwrap(); // String
    b.write_u64::<LittleEndian>(value.len() as u64).unwrap();
    b.extend_from_slice(value.as_bytes());
    b
}

fn u32_entry(key: &str, value: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_u64::<LittleEndian>(key.len() as u64).unwrap();
    b.extend_from_slice(key.as_bytes());
    b.write_u32::<LittleEndian>(4).unwrap(); // U32
    b.write_u32::<LittleEndian>(value).unwrap();
    b
}

fn i32_entry(key: &str, value: i32) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_u64::<LittleEndian>(key.len() as u64).unwrap();
    b.extend_from_slice(key.as_bytes());
    b.write_u32::<LittleEndian>(5).unwrap(); // I32
    b.write_i32::<LittleEndian>(value).unwrap();
    b
}

fn tensor_info(name: &str, dims: &[u64], ggml_code: u32, offset: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_u64::<LittleEndian>(name.len() as u64).unwrap();
    b.extend_from_slice(name.as_bytes());
    b.write_u32::<LittleEndian>(dims.len() as u32).unwrap();
    for d in dims {
        b.write_u64::<LittleEndian>(*d).unwrap();
    }
    b.write_u32::<LittleEndian>(ggml_code).unwrap();
    b.write_u64::<LittleEndian>(offset).unwrap();
    b
}

fn build_gguf(metadata: &[Vec<u8>], infos: &[Vec<u8>], data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(infos.len() as u64).unwrap();
    buf.write_u64::<LittleEndian>(metadata.len() as u64)
        .unwrap();
    for m in metadata {
        buf.extend_from_slice(m);
    }
    for t in infos {
        buf.extend_from_slice(t);
    }
    let pos = buf.len() as u64;
    buf.resize((pos.div_ceil(ALIGN) * ALIGN) as usize, 0);
    buf.extend_from_slice(data);
    buf
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A `count`-shard model: shard `i` (0-based) holds one F32 tensor `t{i}` of 4 elements whose
/// values are `10*i + 1 ..= 10*i + 4`. Only shard 0 carries `general.name`; the rest carry only
/// the `split.*` keys, as `gguf-split` writes them.
fn write_shards(dir: &std::path::Path, prefix: &str, count: u32) -> Vec<PathBuf> {
    (0..count)
        .map(|i| {
            let mut meta = vec![
                u32_entry("split.no", i),
                u32_entry("split.count", count),
                i32_entry("split.tensors.count", count as i32),
            ];
            if i == 0 {
                meta.insert(0, string_entry("general.name", "sharded-test"));
            }
            let base = 10.0 * i as f32;
            let data = f32_bytes(&[base + 1.0, base + 2.0, base + 3.0, base + 4.0]);
            let gguf = build_gguf(&meta, &[tensor_info(&format!("t{i}"), &[4], 0, 0)], &data);
            let path = dir.join(format!("{prefix}-{:05}-of-{count:05}.gguf", i + 1));
            std::fs::write(&path, gguf).unwrap();
            path
        })
        .collect()
}

fn read_f32(mc: &MmapedContent, name: &str) -> Vec<f32> {
    let c = mc.content();
    let info = &c.tensor_infos[name];
    let start = c.tensor_data_offset as usize + info.offset as usize;
    let bytes = mc.bytes_at(start, 16).unwrap();
    (0..4)
        .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn first_shard_path_opens_the_whole_set_and_each_tensor_reads_from_its_own_shard() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_shards(dir.path(), "model", 3);
    let mc = MmapedContent::from_path(&paths[0]).unwrap();
    assert_eq!(mc.shard_count(), 3);
    assert_eq!(mc.content().tensor_infos.len(), 3);
    assert_eq!(read_f32(&mc, "t0"), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_f32(&mc, "t1"), [11.0, 12.0, 13.0, 14.0]);
    assert_eq!(read_f32(&mc, "t2"), [21.0, 22.0, 23.0, 24.0]);
    // Metadata is shard 1's.
    assert!(mc.metadata().contains_key("general.name"));
}

#[test]
fn a_shard_list_in_any_order_gives_the_same_model() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = write_shards(dir.path(), "m", 3);
    paths.reverse();
    let mc = MmapedContent::from_paths(&paths).unwrap();
    assert_eq!(mc.shard_count(), 3);
    assert_eq!(read_f32(&mc, "t0"), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_f32(&mc, "t2"), [21.0, 22.0, 23.0, 24.0]);
}

#[test]
fn a_missing_shard_is_a_typed_error_not_a_partial_model() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_shards(dir.path(), "m", 3);
    std::fs::remove_file(&paths[1]).unwrap();
    // Opening by the first shard looks for the siblings by name and cannot open shard 2.
    assert!(MmapedContent::from_path(&paths[0]).is_err());
    // A list that is not the whole set is refused by the set check.
    let partial = [paths[0].clone(), paths[2].clone()];
    let err = MmapedContent::from_paths(&partial)
        .err()
        .expect("incomplete set must fail");
    assert!(err.to_string().contains("shard"), "{err}");
}

#[test]
fn a_read_that_crosses_a_shard_boundary_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_shards(dir.path(), "m", 2);
    let mc = MmapedContent::from_path(&paths[0]).unwrap();
    let first_len = std::fs::metadata(&paths[0]).unwrap().len() as usize;
    // The last 8 bytes of shard 1 plus the first 8 of shard 2.
    assert!(mc.bytes_at(first_len - 8, 16).is_err());
    // Past the end of the last shard.
    let total = first_len + std::fs::metadata(&paths[1]).unwrap().len() as usize;
    assert!(mc.bytes_at(total - 4, 16).is_err());
}

#[test]
fn an_ordinary_file_still_opens_as_one_shard() {
    let dir = tempfile::tempdir().unwrap();
    let data = f32_bytes(&[5.0, 6.0, 7.0, 8.0]);
    let gguf = build_gguf(&[], &[tensor_info("w", &[4], 0, 0)], &data);
    let path = dir.path().join("plain.gguf");
    std::fs::write(&path, gguf).unwrap();
    let mc = MmapedContent::from_path(&path).unwrap();
    assert_eq!(mc.shard_count(), 1);
    assert_eq!(read_f32(&mc, "w"), [5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn a_file_named_like_a_shard_but_alone_in_its_directory_is_an_error() {
    // `-00001-of-00002` says a sibling exists; there is none.
    let dir = tempfile::tempdir().unwrap();
    let paths = write_shards(dir.path(), "m", 2);
    std::fs::remove_file(&paths[1]).unwrap();
    assert!(MmapedContent::from_path(&paths[0]).is_err());
}
