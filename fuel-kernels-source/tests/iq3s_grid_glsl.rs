// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 512-entry `iq3s_grid` in `fuel-kernels-source/kernels/qmatvec_iq3_s.glsl` must equal
//! `fuel_quantized::iq_grids::IQ3S_GRID` entry by entry (ROADMAP row 119). The GLSL table is generated
//! by `scripts/gen_iq3s_glsl.py` from the Rust one, which `scripts/gen_iq_grids.py check` compares to
//! llama.cpp's `ggml-common.h` at `9d57ce456c`. This test needs no GPU. It lives in `fuel-kernels-source` (next to the GLSL, so it reads nothing outside its own
//! package) with `fuel-quantized` as a dev-dependency (dev-dependencies add no layering edge).

use fuel_quantized::iq_grids::IQ3S_GRID;

const GLSL: &str = include_str!("../kernels/qmatvec_iq3_s.glsl");

/// The `0x........u` entries between the `BEGIN IQ3S_GRID` and `END IQ3S_GRID` markers.
fn glsl_grid(src: &str) -> Vec<u32> {
    let begin = src.find("// BEGIN IQ3S_GRID").expect("BEGIN marker");
    let end = src.find("// END IQ3S_GRID").expect("END marker");
    src[begin..end]
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|t| t.strip_prefix("0x").and_then(|h| h.strip_suffix('u')))
        .map(|h| u32::from_str_radix(h, 16).expect("hex entry"))
        .collect()
}

#[test]
fn iq3s_glsl_grid_matches_rust_grid() {
    let g = glsl_grid(GLSL);
    assert_eq!(g.len(), IQ3S_GRID.len(), "entry count");
    for (i, (a, b)) in g.iter().zip(IQ3S_GRID.iter()).enumerate() {
        assert_eq!(a, b, "entry {i}: glsl {a:#010x} vs rust {b:#010x}");
    }
}

/// The sabotage sibling: a one-entry change in the GLSL text must be seen by the comparison above.
#[test]
fn a_one_entry_change_in_the_glsl_grid_is_caught() {
    let first = format!("{:#010x}u", IQ3S_GRID[0]);
    assert_eq!(
        GLSL.matches(&first).count().min(1),
        1,
        "control: the first entry is present"
    );
    let mutated = GLSL.replacen(&first, "0x01010107u", 1);
    assert_ne!(mutated, GLSL, "the mutation applied");
    assert_ne!(
        glsl_grid(&mutated),
        IQ3S_GRID.to_vec(),
        "the one-entry change was not caught"
    );
}
