// SPDX-License-Identifier: MIT OR Apache-2.0
//! MLMF move, slice C (`docs/design-notes/2026-10-24-mlmf-move-survey.md`, finding F7): the HF-AWQ unpack and the
//! kernel repack are CONVERSIONS, which live in `ml-convertant-awq` (crates.io), not in fuel and not in MLMF.
//! Before fuel's `fuel_quantized::awq_repack::repack_hf_awq_for_kernel` is deleted, the two must agree.
//!
//! Equivalence, on the REAL `Orion-zhen/Qwen3-0.6B-AWQ` checkpoint (`FUEL_QWEN3_AWQ_DIR`; a `Hardware::Hub`
//! skip when absent): for EVERY located layer (28 x 7), `ml_convertant_awq::repack_for_kernel(unpack_hf(hf))`
//! equals fuel's repack of the same HF buffers, field for field (`weight`, `zeros`, `scales` as bits,
//! `zeros_w`). Positive control: 196 layers and more than a million kernel words. Sabotage sibling: flipping
//! one stored nibble changes the kernel output, so the equality is not one that holds for any input.

use std::path::PathBuf;

use fuel_loaders::awq::{AwqFile, hf_buffers, locate_awq_layers};
use fuel_quantized::awq_repack::repack_hf_awq_for_kernel;
use fuel_test_support::hardware::{Hardware, Missing, skip};
use half::f16;
use ml_convertant_awq::{HfAwq, repack_for_kernel, unpack_hf};

fn checkpoint_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("FUEL_QWEN3_AWQ_DIR").map(PathBuf::from) else {
        skip(
            Hardware::Hub,
            Missing::device("FUEL_QWEN3_AWQ_DIR is not set (Orion-zhen/Qwen3-0.6B-AWQ, ~540 MB)"),
        );
        return None;
    };
    for f in ["config.json", "model.safetensors"] {
        if !dir.join(f).is_file() {
            skip(
                Hardware::Hub,
                Missing::device(format!("{} is missing {f}", dir.display())),
            );
            return None;
        }
    }
    Some(dir)
}

fn bits(v: &[f16]) -> Vec<u16> {
    v.iter().map(|h| h.to_bits()).collect()
}

#[test]
fn ml_convertant_repack_equals_fuels_repack_for_every_layer_of_the_real_checkpoint() {
    let Some(dir) = checkpoint_dir() else { return };
    let config = std::fs::read(dir.join("config.json")).expect("config.json");
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("model.safetensors");
    let layout = locate_awq_layers(&config, &bytes).expect("locate");
    let file = AwqFile::open(&bytes).expect("open");
    let mut kernel_words = 0usize;
    for l in &layout.layers {
        let (ic, oc, g) = (
            l.in_features as usize,
            l.out_features as usize,
            layout.group_size,
        );
        let b = hf_buffers(&file.read_layer(l).expect("read"), ic, oc, g).expect("hf_buffers");
        let scales: Vec<f16> = b.scale_bits.iter().map(|x| f16::from_bits(*x)).collect();
        let old = repack_hf_awq_for_kernel(&b.qweight, &b.qzeros, &scales, ic, oc, g).expect("old");
        let hf = HfAwq {
            ic,
            oc,
            group_size: g,
            qweight: b.qweight.clone(),
            qzeros: b.qzeros.clone(),
            scales: scales.clone(),
        };
        let new = repack_for_kernel(&unpack_hf(&hf).expect("unpack")).expect("repack");
        assert_eq!(new.weight, old.weight, "{}: weight", l.prefix);
        assert_eq!(new.zeros, old.zeros, "{}: zeros", l.prefix);
        assert_eq!(bits(&new.scales), bits(&old.scales), "{}: scales", l.prefix);
        assert_eq!(new.zeros_w, old.zeros_w, "{}: zeros_w", l.prefix);
        kernel_words += new.weight.len() + new.zeros.len() + new.scales.len();
    }
    println!(
        "{} layers, {kernel_words} kernel words equal",
        layout.layers.len()
    );
    assert_eq!(layout.layers.len(), 196, "positive control");
    assert!(kernel_words > 1_000_000);
}

#[test]
fn one_flipped_nibble_changes_the_kernel_output() {
    let Some(dir) = checkpoint_dir() else { return };
    let config = std::fs::read(dir.join("config.json")).expect("config.json");
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("model.safetensors");
    let layout = locate_awq_layers(&config, &bytes).expect("locate");
    let file = AwqFile::open(&bytes).expect("open");
    let l = &layout.layers[0];
    let (ic, oc, g) = (
        l.in_features as usize,
        l.out_features as usize,
        layout.group_size,
    );
    let b = hf_buffers(&file.read_layer(l).expect("read"), ic, oc, g).expect("hf_buffers");
    let scales: Vec<f16> = b.scale_bits.iter().map(|x| f16::from_bits(*x)).collect();
    let hf = |qw: Vec<u32>| HfAwq {
        ic,
        oc,
        group_size: g,
        qweight: qw,
        qzeros: b.qzeros.clone(),
        scales: scales.clone(),
    };
    let pristine = repack_for_kernel(&unpack_hf(&hf(b.qweight.clone())).unwrap()).unwrap();
    let mut qw = b.qweight.clone();
    qw[0] ^= 0xF;
    let flipped = repack_for_kernel(&unpack_hf(&hf(qw)).unwrap()).unwrap();
    assert_ne!(
        pristine.weight, flipped.weight,
        "the sabotage must be visible"
    );
}
