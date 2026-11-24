// SPDX-License-Identifier: MIT OR Apache-2.0
//! MLMF move, the AWQ tensor read (`docs/design-notes/2026-10-24-mlmf-move-survey.md`, finding F6: "locating
//! the three tensors ... is FORMAT and moves"). `fuel_loaders::awq::AwqFile::read_layer` reads a located
//! layer's `qweight` / `qzeros` / `scales` through `mlmf-awq`'s `read_layer`; fuel's own reader
//! (`fuel_tensor::awq`'s `read_hf_layer`, over `MmapedSafetensors`) stays until this equivalence holds.
//!
//! Equivalence, on the REAL `Orion-zhen/Qwen3-0.6B-AWQ` checkpoint (`FUEL_QWEN3_AWQ_DIR`; a `Hardware::Hub`
//! skip when absent): for EVERY located layer (28 layers x 7 projections), each of the three tensors read
//! through `mlmf-awq` has the same dtype, the same shape and the same BYTES as the old reader's view.
//!
//! Positive control: the comparison covers more than 100 layers and 3 tensors each. Sabotage sibling: a
//! single flipped byte inside one layer's `qweight` in a copy of the file makes the new read DISAGREE with
//! the pristine old reader, so a passing comparison is not one that cannot fail.

use std::path::PathBuf;

use fuel_formats::safetensors::MmapedSafetensors;
use fuel_loaders::awq::{AwqFile, locate_awq_layers};
use fuel_test_support::hardware::{Hardware, Missing, skip};

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

#[test]
fn the_mlmf_read_equals_the_old_reader_for_every_layer_of_the_real_checkpoint() {
    let Some(dir) = checkpoint_dir() else { return };
    let config = std::fs::read(dir.join("config.json")).expect("config.json");
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("model.safetensors");
    let layout = locate_awq_layers(&config, &bytes).expect("locate");
    let file = AwqFile::open(&bytes).expect("open");
    // SAFETY: a read-only checkpoint this test does not modify, mapped for the test's duration.
    let old = unsafe { MmapedSafetensors::new(dir.join("model.safetensors")) }.expect("mmap");

    let mut tensors_compared = 0usize;
    for l in &layout.layers {
        let new = file.read_layer(l).expect("read_layer");
        for (name, got) in [
            (&l.qweight, &new.qweight),
            (&l.qzeros, &new.qzeros),
            (&l.scales, &new.scales),
        ] {
            let view = old.get(name).expect(name);
            assert_eq!(&got.name, name);
            assert_eq!(got.shape, view.shape(), "{name}: shape");
            assert_eq!(
                format!("{:?}", got.dtype),
                format!("{:?}", view.dtype()),
                "{name}: dtype"
            );
            assert_eq!(&*got.bytes, view.data(), "{name}: bytes");
            tensors_compared += 1;
        }
    }
    println!(
        "{} layers, {tensors_compared} tensors byte-identical",
        layout.layers.len()
    );
    assert!(
        layout.layers.len() > 100,
        "positive control: a real layer set"
    );
    assert_eq!(tensors_compared, layout.layers.len() * 3);
}

#[test]
fn a_flipped_byte_in_one_qweight_makes_the_new_read_disagree_with_the_old_reader() {
    let Some(dir) = checkpoint_dir() else { return };
    let config = std::fs::read(dir.join("config.json")).expect("config.json");
    let pristine = std::fs::read(dir.join("model.safetensors")).expect("model.safetensors");
    let layout = locate_awq_layers(&config, &pristine).expect("locate");
    let target = &layout.layers[0];

    // Where the target's qweight sits in the file: the new read borrows into the buffer it was given.
    let offset = {
        let file = AwqFile::open(&pristine).expect("open");
        let q = file.read_layer(target).expect("read_layer").qweight;
        q.bytes.as_ptr() as usize - pristine.as_ptr() as usize
    };
    let mut corrupted = pristine.clone();
    corrupted[offset] ^= 0xFF;

    let file = AwqFile::open(&corrupted).expect("open corrupted");
    let new = file.read_layer(target).expect("read_layer");
    // SAFETY: as above; the OLD reader maps the pristine file.
    let old = unsafe { MmapedSafetensors::new(dir.join("model.safetensors")) }.expect("mmap");
    let view = old.get(&target.qweight).expect("old qweight");
    assert_ne!(
        &*new.qweight.bytes,
        view.data(),
        "the sabotage must be visible to the comparison"
    );
    // ... and the untouched tensors still agree, so the disagreement is the flipped byte and nothing else.
    assert_eq!(
        &*new.qzeros.bytes,
        old.get(&target.qzeros).expect("qzeros").data()
    );
    assert_eq!(
        &*new.scales.bytes,
        old.get(&target.scales).expect("scales").data()
    );
}
