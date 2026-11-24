// SPDX-License-Identifier: MIT OR Apache-2.0
//! Finding a HuggingFace `*-AWQ` checkpoint's quantization parameters and its packed linear
//! layers. MLMF move, slice A (`docs/design-notes/2026-10-24-mlmf-move-survey.md`): the parsing
//! is `mlmf-awq` (`AwqConfig`, `locate_layers`) over `mlmf-safetensors`' header and directory;
//! this module only adds fuel's policy on top, namely which AWQ variants the kernels can read.
//!
//! Like the rest of this crate it knows nothing about `Tensor`, `Storage` or `Device`.
//!
//! # Scope
//!
//! One safetensors file. A sharded checkpoint would have to merge per-shard reports (a layer's
//! three tensors can sit in different shards, which a per-shard scan reports as incomplete), and
//! nothing here does that yet.

use fuel_ir::error::{Error, Result};
pub use mlmf_awq::AwqLayerBytes;
use mlmf_awq::{
    AwqConfig, AwqTensorBytes, LocateReport, PackedLinearLayer, locate_layers, read_layer,
};
use mlmf_safetensors::{SafetensorsTensors, parse_header, parse_tensors};

/// What fuel's AWQ path can read: 4-bit, zero-point, GEMM-version packing.
const BITS: u64 = 4;

/// The checkpoint's quantization group size and its located AWQ-packed linear layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwqLayout {
    /// `quantization_config.group_size`.
    pub group_size: usize,
    /// Every fully located, geometrically consistent packed layer, sorted by prefix.
    pub layers: Vec<PackedLinearLayer>,
}

fn refuse(text: String) -> Error {
    Error::Msg(format!("AWQ checkpoint: {text}")).bt()
}

/// `quantization_config.group_size` from a model's `config.json`, after refusing every
/// quantization the loaders here do not read: a different method, bit width, packing or
/// zero-point convention would load as plausible garbage.
pub fn awq_group_size(config_json: &[u8]) -> Result<usize> {
    let cfg = parse_config(config_json)?;
    check_variant(&cfg)?;
    group_size_of(&cfg)
}

fn parse_config(config_json: &[u8]) -> Result<AwqConfig> {
    let cfg = AwqConfig::parse_from_model_config(config_json)
        .map_err(|e| refuse(format!("config.json: {e}")))?
        .ok_or_else(|| refuse("config.json has no AWQ `quantization_config`".to_string()))?;
    if !cfg.malformed.is_empty() {
        return Err(refuse(format!(
            "unreadable quantization_config fields: {:?}",
            cfg.malformed
        )));
    }
    Ok(cfg)
}

/// The method, bit width, packing and zero-point convention must be exactly what the kernels read.
fn check_variant(cfg: &AwqConfig) -> Result<()> {
    if cfg.quant_method.as_deref() != Some("awq") {
        return Err(refuse(format!(
            "quant_method is {:?}, not \"awq\" (a missing method is not accepted)",
            cfg.quant_method
        )));
    }
    check_packing(cfg)?;
    if cfg.zero_point != Some(true) {
        return Err(refuse(format!(
            "only zero-point AWQ is read, got zero_point {:?}",
            cfg.zero_point
        )));
    }
    Ok(())
}

fn check_packing(cfg: &AwqConfig) -> Result<()> {
    if cfg.bits != Some(BITS) {
        return Err(refuse(format!(
            "only {BITS}-bit AWQ is read, got bits {:?}",
            cfg.bits
        )));
    }
    let gemm = cfg
        .version
        .as_deref()
        .is_some_and(|v| v.eq_ignore_ascii_case("gemm"));
    if !gemm {
        return Err(refuse(format!(
            "only the GEMM packing is read, got version {:?}",
            cfg.version
        )));
    }
    Ok(())
}

fn group_size_of(cfg: &AwqConfig) -> Result<usize> {
    let group_size = cfg
        .group_size
        .filter(|g| *g > 0)
        .ok_or_else(|| refuse("quantization_config.group_size is missing or zero".to_string()))?;
    usize::try_from(group_size).map_err(|_| refuse(format!("group_size {group_size} overflows")))
}

/// One safetensors file's tensor directory, parsed once, to read located layers' tensors from.
///
/// MLMF move, the AWQ tensor read (survey finding F6): the lookup of a layer's `qweight` / `qzeros` /
/// `scales`, with the stored dtype, shape and bytes, is `mlmf-awq`'s `read_layer`. The bytes are
/// exactly as the file holds them (HF layout, little-endian, unaligned); validating them against the
/// geometry the model config declares, and repacking them for the kernels, stay with the caller.
pub struct AwqFile<'a> {
    tensors: SafetensorsTensors<'a>,
}

impl<'a> AwqFile<'a> {
    /// Parse `safetensors`' header and tensor directory.
    pub fn open(safetensors: &'a [u8]) -> Result<Self> {
        let header =
            parse_header(safetensors).map_err(|e| refuse(format!("safetensors header: {e}")))?;
        let (tensors, _report) = parse_tensors(safetensors, &header, "awq checkpoint")
            .map_err(|e| refuse(format!("safetensors directory: {e}")))?;
        Ok(Self { tensors })
    }

    /// Every fully located, geometrically consistent packed layer of this file for `group_size`, sorted by
    /// prefix. A layer MLMF reports as incomplete or inconsistent is an error naming every offender.
    pub fn locate(&self, group_size: usize) -> Result<Vec<PackedLinearLayer>> {
        let report = locate_layers(&self.tensors, BITS, group_size as u64)
            .map_err(|e| refuse(format!("locating layers: {e}")))?;
        reject_unusable(&report)?;
        Ok(report.layers)
    }

    /// The three tensors (and the bias, if any) of a located layer, as stored. A tensor the layer
    /// names that is missing or block-encoded is an error naming it.
    pub fn read_layer(&self, layer: &PackedLinearLayer) -> Result<AwqLayerBytes<'_>> {
        read_layer(&self.tensors, layer)
            .map_err(|e| refuse(format!("reading layer `{}`: {e}", layer.prefix)))
    }
}

/// Locate every AWQ-packed linear layer in one safetensors file's bytes. A layer MLMF reports
/// as incomplete or geometrically inconsistent is an error here (a half-loaded model is worse
/// than a refused one), with every offender named.
pub fn locate_awq_layers(config_json: &[u8], safetensors: &[u8]) -> Result<AwqLayout> {
    let group_size = awq_group_size(config_json)?;
    let header =
        parse_header(safetensors).map_err(|e| refuse(format!("safetensors header: {e}")))?;
    let (tensors, _report) = parse_tensors(safetensors, &header, "awq checkpoint")
        .map_err(|e| refuse(format!("safetensors directory: {e}")))?;
    let layers = AwqFile { tensors }.locate(group_size)?;
    Ok(AwqLayout { group_size, layers })
}

fn reject_unusable(report: &LocateReport) -> Result<()> {
    if report.incomplete.is_empty() && report.malformed.is_empty() {
        return Ok(());
    }
    Err(refuse(format!(
        "{} layer(s) missing qzeros or scales: {:?}; {} layer(s) with inconsistent geometry: {:?}",
        report.incomplete.len(),
        report.incomplete,
        report.malformed.len(),
        report.malformed
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> Vec<u8> {
        format!(
            r#"{{"quantization_config": {{"quant_method": "awq", "bits": 4, "version": "gemm",
                "zero_point": true, "group_size": 128 {extra}}}}}"#
        )
        .into_bytes()
    }

    #[test]
    fn reads_the_group_size_of_a_gemm_zero_point_4_bit_config() {
        assert_eq!(awq_group_size(&config("")).unwrap(), 128);
        // The two conventions differ in casing; GEMM and gemm are the same packing.
        let upper = String::from_utf8(config(""))
            .unwrap()
            .replace("\"gemm\"", "\"GEMM\"");
        assert_eq!(awq_group_size(upper.as_bytes()).unwrap(), 128);
    }

    /// Each refusal is its own arm: a config that differs from the accepted one in exactly one
    /// way must be refused, so no arm is decoration.
    #[test]
    fn refuses_every_variant_the_kernels_do_not_read() {
        let ok = String::from_utf8(config("")).unwrap();
        for (label, bad) in [
            ("8-bit", ok.replace("\"bits\": 4", "\"bits\": 8")),
            ("gemv packing", ok.replace("\"gemm\"", "\"gemv\"")),
            (
                "no zero point",
                ok.replace("\"zero_point\": true", "\"zero_point\": false"),
            ),
            ("gptq", ok.replace("\"awq\"", "\"gptq\"")),
            ("no method", ok.replace("\"quant_method\": \"awq\",", "")),
            (
                "zero group",
                ok.replace("\"group_size\": 128", "\"group_size\": 0"),
            ),
            ("no group", ok.replace(", \"group_size\": 128", "")),
            (
                "group size as a string",
                ok.replace("\"group_size\": 128", "\"group_size\": \"128\""),
            ),
        ] {
            assert!(
                awq_group_size(bad.as_bytes()).is_err(),
                "{label} must be refused"
            );
        }
        assert!(awq_group_size(b"{}").is_err(), "no quantization_config");
        assert!(awq_group_size(b"not json").is_err());
    }
}

/// A layer's three tensors in the HF layout, validated against the geometry the model config declares
/// and decoded from the stored little-endian bytes. Scales are raw IEEE half bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfAwqBuffers {
    /// `[ic, oc / 8]` packed weights (an `I32` or `U32` tensor; the nibbles are read unsigned).
    pub qweight: Vec<u32>,
    /// `[ic / group_size, oc / 8]` packed zero points.
    pub qzeros: Vec<u32>,
    /// `[ic / group_size, oc]` scales as `f16` bit patterns.
    pub scale_bits: Vec<u16>,
}

fn words(t: &AwqTensorBytes<'_>, rows: usize, cols: usize) -> Result<Vec<u32>> {
    let key = &t.name;
    let dtype = format!("{:?}", t.dtype);
    if dtype != "I32" && dtype != "U32" {
        return Err(refuse(format!(
            "load_awq_layer: {key:?} dtype={dtype}, expected I32 or U32"
        )));
    }
    if t.shape != [rows, cols] {
        return Err(refuse(format!(
            "load_awq_layer: {key:?} shape={:?}, expected [{rows}, {cols}]",
            t.shape
        )));
    }
    Ok(t.bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

fn half_bits(t: &AwqTensorBytes<'_>, groups: usize, oc: usize) -> Result<Vec<u16>> {
    let key = &t.name;
    let dtype = format!("{:?}", t.dtype);
    if dtype != "F16" {
        return Err(refuse(format!(
            "load_awq_layer: {key:?} dtype={dtype}, expected F16"
        )));
    }
    if t.shape != [groups, oc] {
        return Err(refuse(format!(
            "load_awq_layer: {key:?} shape={:?}, expected [{groups}, {oc}]",
            t.shape
        )));
    }
    Ok(t.bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect())
}

/// Validate one layer's stored tensors against `ic` / `oc` / `group_size` (from the model config) and
/// decode them. A mismatched geometry, dtype or shape errors here rather than reading a
/// plausible-looking wrong layer.
pub fn hf_buffers(
    layer: &AwqLayerBytes<'_>,
    ic: usize,
    oc: usize,
    group_size: usize,
) -> Result<HfAwqBuffers> {
    if group_size == 0 || !ic.is_multiple_of(group_size) || !oc.is_multiple_of(8) {
        return Err(refuse(format!(
            "load_awq_layer: ic={ic} must be a multiple of group_size={group_size} and oc={oc} a multiple of 8"
        )));
    }
    let groups = ic / group_size;
    Ok(HfAwqBuffers {
        qweight: words(&layer.qweight, ic, oc / 8)?,
        qzeros: words(&layer.qzeros, groups, oc / 8)?,
        scale_bits: half_bits(&layer.scales, groups, oc)?,
    })
}
