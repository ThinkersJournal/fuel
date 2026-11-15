// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuel × KISS `structure_key` **derivation** vectors: Fuel's sk5 deriver
//! against the 36 cell-carrying vectors in
//! `fixtures/kiss-corpus/structure_key_derivation_vectors.json`.
//!
//! **Provenance (a recorded stamp, not an enforced check):** KISS `60c375c`
//! (conformance 0.2.30, KISS#566), `conformance/corpus/structure_key_derivation_vectors.json`,
//! blob `697aa543` (36 vectors; `git rev-parse origin/main:<path>` at KISS `0ffd6af`),
//! vendored 2026-10-15 with `git show origin/main:<path>` (a commit object, never a
//! working tree). It REPLACES the `ae16d2a7` copy (blob `299811b1`, 33 vectors):
//! the only change is three vectors for the innermost active axis
//! (`innermost_axis_extent_one_is_axis_rank_minus_1`,
//! `innermost_axis_extent_eight_after_unit_axis`,
//! `innermost_axis_extent_zero_is_axis_rank_minus_1`; KISS-CLASSIFY-6.3-0011 and
//! 6.5-0012), all of which Fuel's deriver passes under both feeding disciplines
//! (including `[8,0]`). The schema version and the vector count are asserted
//! below; the bytes are not hashed.
//!
//! Unlike `kiss_structure_key_byte_match.rs` (whose cells are hand-built Rust
//! literals keyed by vector name), every vector here CARRIES its cell
//! (operands in call order, index slot, reduced axes, accumulator, math
//! fidelity), so the cell is built mechanically from the JSON and the deriver
//! is asked what it produces. Nothing in this file names a vector except the
//! pinned outcome tables, so a vector added upstream with no pin fails loudly.
//!
//! # What is measured
//!
//! Each vector gets an outcome under TWO feeding disciplines, because KISS says
//! canonicalization is "the deriver's job" while Fuel's
//! `derive_structure_key_token` documents that it TAKES canonical order and
//! declines anything else:
//!
//! - **as-given** — operands passed in the vector's CALL order and the index
//!   slot unmapped. This is the literal reading of the vectors.
//! - **canonicalized** — the harness does the §6.6-0014 stable partition
//!   (inputs and in-place operands in call order, then outputs) and maps the
//!   index slot, which is what a Fuel caller must do today.
//!
//! Outcomes: `PASS` (bytes equal), `FAIL` (different bytes), `NOT-EXPRESSIBLE`
//! (Fuel has no way to state the cell, with the reason), `DECLINE-DIFFERS`
//! (a decline was expected and a token came out, or the reverse). A decline's
//! KISS reason code (`ReservedDtype`, ...) IS compared, through
//! `derive_structure_key_token_typed` (GAP-354); `REASON-DIFFERS` means both
//! declined with different reasons.
//!
//! # Honest limits
//!
//! - Only the 10 vectors carrying `spec_golden` reproduce an Appendix A.1 token
//!   whose bytes were hand-derived in the spec text; the other 17 token vectors
//!   are KISS's own deriver's output, so a PASS there is agreement with another
//!   deriver, not with the spec.
//! - `decline_contraction_out_of_scope` is not derived: the batch role is
//!   undefined upstream (KISS#543), and a `GemCell` needs M/N/K the vector does
//!   not carry.
//! - A token-level match cannot see derivation: a deriver reading a different
//!   fact that agrees on every sampled cell passes.

#![cfg(feature = "telemetry")]

use fuel_dispatch::telemetry::structure_key::FdxOperandDesc;
use fuel_dispatch::telemetry::structure_key_derive::{
    AccMp, CellOperand, DeclaredFacts, DequantForm, GemCell, GemMathPrecision, ReduceAxes,
    derive_structure_key_token_typed,
};
use fuel_ir::{DType, Layout, Shape};
use std::collections::BTreeMap;

const VECTORS: &str = include_str!("../fixtures/kiss-corpus/structure_key_derivation_vectors.json");

/// Fuel's `DType` for a KISS dtype spelling, or `None` if Fuel has no such
/// dtype (`i4`, `u4`, `b1`, `u16`, `u64`, the complex tokens, the `fnuz` pair).
fn dtype_of(token: &str) -> Option<DType> {
    Some(match token {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        "f64" => DType::F64,
        "i8" => DType::I8,
        "i16" => DType::I16,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "u8" => DType::U8,
        "u32" => DType::U32,
        "bool" => DType::Bool,
        "f8e4m3fn" => DType::F8E4M3,
        "f8e5m2" => DType::F8E5M2,
        "f8e8m0" => DType::F8E8M0,
        "f8e6m2" => DType::F8E6M2,
        "f4e2m1" => DType::F4,
        "f6e2m3" => DType::F6E2M3,
        "f6e3m2" => DType::F6E3M2,
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Pass,
    /// Different bytes: `(got, want)`.
    Fail(String, String),
    /// A decline was expected and a token came out.
    DeclineDiffers(String),
    /// A token was expected and the deriver declined.
    UnexpectedDecline(String),
    /// Both declined, with different reasons: `(got, want)`.
    ReasonDiffers(String, String),
    NotExpressible(String),
}

impl Outcome {
    fn tag(&self) -> &'static str {
        match self {
            Outcome::Pass => "PASS",
            Outcome::Fail(..) => "FAIL",
            Outcome::DeclineDiffers(_) => "DECLINE-DIFFERS",
            Outcome::UnexpectedDecline(_) => "UNEXPECTED-DECLINE",
            Outcome::ReasonDiffers(..) => "REASON-DIFFERS",
            Outcome::NotExpressible(_) => "NOT-EXPRESSIBLE",
        }
    }
}

struct Built {
    facts: DeclaredFacts,
    operands: Vec<CellOperand>,
    target: String,
    acc_mp: Option<AccMp>,
}

/// The `shape`/`strides` of one operand, from the vector's JSON arrays.
fn layout_of(o: &serde_json::Value) -> Result<Layout, String> {
    let extents: Vec<usize> = o["extents"]
        .as_array()
        .ok_or("extents")?
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let strides: Vec<isize> = o["strides"]
        .as_array()
        .ok_or("strides")?
        .iter()
        .map(|x| x.as_i64().unwrap() as isize)
        .collect();
    Ok(Layout::new(
        Shape::from_dims(&extents),
        strides.into_iter().collect(),
        0,
    ))
}

/// An operand's `use` as a `CellOperand` role.
fn with_role(use_: &str, desc: FdxOperandDesc) -> Result<CellOperand, String> {
    match use_ {
        "in" => Ok(CellOperand::input(desc)),
        "out" => Ok(CellOperand::output(desc)),
        "inout" => Ok(CellOperand::in_place(desc)),
        other => Err(format!("unknown operand use `{other}`")),
    }
}

/// An operand's optional `dequant_form`.
fn with_dequant(cell: CellOperand, form: Option<&str>) -> Result<CellOperand, String> {
    match form {
        None => Ok(cell),
        Some("dl" | "linear") => Ok(cell.with_dequant_form(DequantForm::Linear)),
        Some("dc" | "codebook") => Ok(cell.with_dequant_form(DequantForm::Codebook)),
        Some(other) => Err(format!("unknown dequant_form `{other}`")),
    }
}

/// The descriptor of one operand: layout, dtype, and the stated alignment.
fn desc_of(o: &serde_json::Value) -> Result<FdxOperandDesc, String> {
    let dtok = o["dtype"].as_str().ok_or("operand dtype")?;
    let dtype = dtype_of(dtok)
        .ok_or_else(|| format!("no Fuel DType for `{dtok}` (GAP-097 absent surface)"))?;
    let mut desc = FdxOperandDesc::from_layout(&layout_of(o)?, dtype);
    // The vector states the base alignment outright; `from_layout` would model
    // 256 for offset 0, which is the only thing it cannot be told.
    desc.align_bytes = o["alignment"].as_u64().ok_or("alignment")? as u32;
    Ok(desc)
}

/// One operand of a vector, as Fuel's `CellOperand`.
fn operand_of(o: &serde_json::Value) -> Result<CellOperand, String> {
    let cell = with_role(o["use"].as_str().ok_or("use")?, desc_of(o)?)?;
    with_dequant(cell, o["dequant_form"].as_str())
}

/// The declared reduced axes as Fuel's `Keepdim` bitmask (bit `d` = frame axis `d`).
fn reduce_axes_of(v: &serde_json::Value) -> Result<Option<ReduceAxes>, String> {
    let axes = v["reduced_axes"].as_array().ok_or("reduced_axes")?;
    if axes.is_empty() {
        return Ok(None);
    }
    let mut mask: u32 = 0;
    for a in axes {
        let bit = a.as_u64().ok_or("axis")? as u32;
        // An axis past the mask width cannot be stated at all.
        if bit >= 8 {
            return Err(format!("reduced axis {bit} does not fit Fuel's u8 mask"));
        }
        mask |= 1 << bit;
    }
    Ok(Some(ReduceAxes::Keepdim(mask as u8)))
}

/// §6.6-0014 stable partition (everything that is not an output, then outputs),
/// returning the new order and the index slot mapped through it.
fn canonicalized(
    cells: Vec<(usize, CellOperand)>,
    index: Option<usize>,
) -> (Vec<(usize, CellOperand)>, Option<usize>) {
    use fuel_dispatch::telemetry::structure_key_derive::OperandRole;
    let (mut front, back): (Vec<_>, Vec<_>) = cells
        .into_iter()
        .partition(|(_, c)| c.role != OperandRole::Output);
    front.extend(back);
    let index = index.and_then(|slot| front.iter().position(|(call, _)| *call == slot));
    (front, index)
}

/// The `(acc + mp)` coordinate: accumulator null = the compute dtype.
fn acc_mp_of(v: &serde_json::Value, compute: DType) -> Result<AccMp, String> {
    let math_precision = match v["math_fidelity"].as_str().ok_or("math_fidelity")? {
        "st" => GemMathPrecision::BitStable,
        "rm" => GemMathPrecision::ReducedMantissa,
        other => return Err(format!("unknown math_fidelity `{other}`")),
    };
    let acc_dtype = match v["accumulator"].as_str() {
        None => compute,
        Some(t) => dtype_of(t).ok_or_else(|| format!("no Fuel DType for accumulator `{t}`"))?,
    };
    Ok(AccMp {
        acc_dtype,
        math_precision,
    })
}

/// Why a vector's cell could not be handed to the deriver.
enum BuildErr {
    /// Fuel has no way to state the cell (an absent surface).
    NotExpressible(String),
    /// The role hints themselves are invalid: a decline (KISS reason code) made
    /// while turning operand-axis roles into the extents a `GemCell` carries.
    /// KISS's reference decides these from the roles, before any key exists.
    Decline(&'static str),
}

impl From<String> for BuildErr {
    fn from(s: String) -> Self {
        Self::NotExpressible(s)
    }
}

impl From<&str> for BuildErr {
    fn from(s: &str) -> Self {
        Self::NotExpressible(s.to_string())
    }
}

/// An `{operand, axis}` role as the extent of that operand axis, declining on
/// an out-of-range slot or axis. Also returns the iteration-frame axis it
/// occupies (right-aligned, §6.6-0013).
fn role_extent(
    ops: &[serde_json::Value],
    frame_rank: usize,
    role: &serde_json::Value,
) -> Result<(i64, usize, (u64, u64)), BuildErr> {
    let slot = role["operand"].as_u64().ok_or("role operand")?;
    let axis = role["axis"].as_u64().ok_or("role axis")?;
    let op = ops
        .get(slot as usize)
        .ok_or(BuildErr::Decline("ContractionAxisInvalid"))?;
    let ext = op["extents"].as_array().ok_or("extents")?;
    let e = ext
        .get(axis as usize)
        .ok_or(BuildErr::Decline("ContractionAxisInvalid"))?;
    let frame_axis = frame_rank - ext.len() + axis as usize;
    Ok((e.as_i64().ok_or("extent")?, frame_axis, (slot, axis)))
}

/// The slot a `weight`/`scale` role names, declining when it is out of range.
fn role_slot(
    v: &serde_json::Value,
    key: &str,
    n_operands: usize,
) -> Result<Option<usize>, BuildErr> {
    match v["contraction"][key].as_u64() {
        None => Ok(None),
        Some(s) if (s as usize) < n_operands => Ok(Some(s as usize)),
        Some(_) => Err(BuildErr::Decline("RoleSlotOutOfRange")),
    }
}

/// The batch role: frame axes (outermost = 0) validated against the frame, the
/// M/N/K roles and the output, returning the batch extent (their product), or
/// `None` when the cell is not batched.
fn batch_extent(
    ops: &[serde_json::Value],
    frame: &[i64],
    mnk_frame_axes: &[usize],
    axes: &[serde_json::Value],
) -> Result<Option<i64>, BuildErr> {
    if axes.is_empty() {
        return Ok(None);
    }
    let out_rank = ops
        .iter()
        .find(|o| o["use"] == "out")
        .map(|o| o["extents"].as_array().map_or(0, |e| e.len()))
        .unwrap_or(0);
    let mut seen: Vec<usize> = Vec::new();
    let mut product = 1i64;
    for a in axes {
        let ax = a.as_u64().ok_or("batch axis")? as usize;
        if ax >= frame.len() {
            return Err(BuildErr::Decline("BatchAxisOutOfRange"));
        }
        if seen.contains(&ax) {
            return Err(BuildErr::Decline("BatchAxisDuplicate"));
        }
        if mnk_frame_axes.contains(&ax) {
            return Err(BuildErr::Decline("BatchAxisOverlapsContractionRole"));
        }
        if ax < frame.len() - out_rank {
            return Err(BuildErr::Decline("BatchAxisNotInOutput"));
        }
        if frame[ax] < 1 {
            return Err(BuildErr::Decline("BatchAxisNonPositiveExtent"));
        }
        seen.push(ax);
        product *= frame[ax];
    }
    Ok(Some(product))
}

/// The element-count product of the weight's `quant` `block_shape`: the block
/// length (§6.6-0022a), `None` when the weight carries no `block_shape`.
fn block_length_of(weight: &serde_json::Value) -> Option<u32> {
    let shape = weight["block_shape"].as_array()?;
    if shape.is_empty() {
        return None;
    }
    Some(
        shape
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as u32)
            .product(),
    )
}

/// The `ctr` contraction facts of a vector: the roles (operand axes) turned
/// into the extents and dtype coordinates a [`GemCell`] carries.
fn contraction_of(v: &serde_json::Value) -> Result<GemCell, BuildErr> {
    let ops = v["operands"].as_array().ok_or("no operands")?;
    let c = &v["contraction"];
    let rank = ops
        .iter()
        .map(|o| o["extents"].as_array().map_or(0, |e| e.len()))
        .max()
        .unwrap_or(0);
    // Per-axis maximum extent (§6.5-0010); an axis no operand has stays 1, and a
    // zero every operand agrees on stays 0 (a max seeded at 1 would hide it).
    let mut frame: Vec<Option<i64>> = vec![None; rank];
    for o in ops {
        let ext = o["extents"].as_array().ok_or("extents")?;
        let off = rank - ext.len();
        for (i, e) in ext.iter().enumerate() {
            let e = e.as_i64().ok_or("extent")?;
            frame[off + i] = Some(frame[off + i].map_or(e, |f| f.max(e)));
        }
    }
    let frame: Vec<i64> = frame.into_iter().map(|f| f.unwrap_or(1)).collect();
    let weight = role_slot(v, "weight", ops.len())?;
    let scale = role_slot(v, "scale", ops.len())?;
    let (m, mf, mr) = role_extent(ops, rank, &c["m"])?;
    let (n, nf, nr) = role_extent(ops, rank, &c["n"])?;
    let (k, kf, kr) = role_extent(ops, rank, &c["k"])?;
    if mr == nr || mr == kr || nr == kr {
        return Err(BuildErr::Decline("ContractionAxisInvalid"));
    }
    let out = ops
        .iter()
        .find(|o| o["use"] == "out")
        .ok_or(BuildErr::Decline("ContractionNoOutput"))?;
    let batch = batch_extent(
        ops,
        &frame,
        &[mf, nf, kf],
        c["batch_axes"].as_array().ok_or("batch_axes")?,
    )?;
    let dt = |o: &serde_json::Value| {
        let tok = o["dtype"].as_str().ok_or("operand dtype")?;
        dtype_of(tok).ok_or_else(|| format!("no Fuel DType for `{tok}` (GAP-097 absent surface)"))
    };
    // The weight defaults to operand 1, the usual rhs, when none is designated.
    let weight_op = &ops[weight.unwrap_or(1.min(ops.len() - 1))];
    let compute = dt(&ops[0])?;
    let acc_dtype = match v["accumulator"].as_str() {
        None => compute,
        Some(t) => dtype_of(t).ok_or_else(|| format!("no Fuel DType for accumulator `{t}`"))?,
    };
    let math_precision = match v["math_fidelity"].as_str().ok_or("math_fidelity")? {
        "st" => GemMathPrecision::BitStable,
        "rm" => GemMathPrecision::ReducedMantissa,
        other => return Err(format!("unknown math_fidelity `{other}`").into()),
    };
    Ok(GemCell {
        m,
        n,
        k,
        batch,
        weight_dtype: dt(weight_op)?,
        acc_dtype,
        out_dtype: dt(out)?,
        math_precision,
        scale_dtype: scale.map(|s| dt(&ops[s])).transpose()?,
        // A designated scale needs the weight's block length; with none the
        // deriver is handed `None` and declines `BlockLengthMissing`.
        block_length: scale.and_then(|_| block_length_of(weight_op)),
    })
}

/// The operands in CALL order, each with its call-order slot.
fn cells_of(v: &serde_json::Value) -> Result<Vec<(usize, CellOperand)>, String> {
    let ops = v["operands"].as_array().ok_or("no operands")?;
    ops.iter()
        .enumerate()
        .map(|(i, o)| operand_of(o).map(|c| (i, c)))
        .collect()
}

/// The declared facts: index slot and reduced axes.
fn facts_of(v: &serde_json::Value, index: Option<usize>) -> Result<DeclaredFacts, String> {
    let mut facts = index.map_or_else(DeclaredFacts::none, DeclaredFacts::indexed);
    facts.reduced_axes = reduce_axes_of(v)?;
    Ok(facts)
}

/// Build Fuel's cell from one vector. `canonicalize` applies the §6.6-0014
/// stable partition and maps the call-order index slot through it.
fn build(v: &serde_json::Value, canonicalize: bool) -> Result<Built, BuildErr> {
    let mut cells = cells_of(v)?;
    let mut index = v["index_operand"].as_u64().map(|x| x as usize);
    if canonicalize {
        (cells, index) = canonicalized(cells, index);
    }
    let mut facts = facts_of(v, index)?;
    // §6.8-0018: the target namespace's declared vector-access cap, when the vector states one.
    if let Some(cap) = v["target_vec_access_cap_bytes"].as_u64() {
        facts = facts.with_vec_access_cap_bytes(cap as u32);
    }
    let contraction = v["contraction_declared"].as_bool() == Some(true);
    if contraction {
        facts.contraction = Some(contraction_of(v)?);
    }
    let operands: Vec<CellOperand> = cells.into_iter().map(|(_, c)| c).collect();
    let compute = operands
        .first()
        .map(|o| o.desc.dtype)
        .ok_or("no operands")?;
    Ok(Built {
        facts,
        operands,
        target: v["target"].as_str().ok_or("target")?.to_string(),
        // §6.7-0013: a contraction carries its precision in the `ctr` field, never
        // beside it, so the `(acc + mp)` coordinate is supplied only otherwise.
        acc_mp: if contraction {
            None
        } else {
            Some(acc_mp_of(v, compute)?)
        },
    })
}

fn evaluate(v: &serde_json::Value, canonicalize: bool) -> Outcome {
    let built = match build(v, canonicalize) {
        Ok(b) => b,
        Err(BuildErr::NotExpressible(why)) => return Outcome::NotExpressible(why),
        // A role-level decline is the harness's own, so it compares by name.
        Err(BuildErr::Decline(wire)) => {
            return match (v["expect"].as_str(), v["value"].as_str()) {
                (Some("decline"), Some(want)) if want == wire => Outcome::Pass,
                (Some("decline"), Some(want)) => {
                    Outcome::ReasonDiffers(wire.to_string(), want.to_string())
                }
                _ => Outcome::UnexpectedDecline(format!("{wire}; the vector expects a token")),
            };
        }
    };
    // KISS (cell_note, `call_order_is_not_canonical_order`): a deriver whose input
    // contract is canonical order reports NOT-EXPRESSIBLE on call-order input,
    // "which is not a defect". Fuel's deriver takes canonical order (caller's
    // contract), so as-given call order is not a cell it can be asked.
    if !canonicalize {
        let mut seen_output = false;
        for o in &built.operands {
            use fuel_dispatch::telemetry::structure_key_derive::OperandRole;
            if o.role == OperandRole::Output {
                seen_output = true;
            } else if seen_output {
                return Outcome::NotExpressible(
                    "input contract is canonical order (KISS: not a defect)".to_string(),
                );
            }
        }
    }
    let got = derive_structure_key_token_typed(
        &built.facts,
        &built.operands,
        &built.target,
        built.acc_mp,
    );
    let want = v["value"].as_str().unwrap().to_string();
    match (v["expect"].as_str(), got) {
        (Some("token"), Ok(t)) if t == want => Outcome::Pass,
        (Some("token"), Ok(t)) => Outcome::Fail(t, want),
        (Some("token"), Err(why)) => {
            Outcome::UnexpectedDecline(format!("{}; want {want}", why.wire()))
        }
        (Some("decline"), Err(why)) if why.wire() == want => Outcome::Pass,
        (Some("decline"), Err(why)) => Outcome::ReasonDiffers(why.wire().to_string(), want),
        (Some("decline"), Ok(t)) => Outcome::DeclineDiffers(t),
        (other, _) => panic!("unknown `expect` {other:?}"),
    }
}

fn load() -> Vec<serde_json::Value> {
    let doc: serde_json::Value = serde_json::from_str(VECTORS).expect("vectors parse");
    assert_eq!(
        doc["structure_key_schema_version"].as_u64(),
        Some(8),
        "schema version moved: re-measure"
    );
    doc["vectors"].as_array().expect("vectors").clone()
}

/// Vectors where a DECLINE is required and Fuel's deriver EMITS A TOKEN instead.
/// Pinned as a SET, both directions (a repair removes a name and must edit this
/// list in the same change; a regression adds one and fails here).
///
/// These are deriver-vs-spec divergences, the dangerous direction: the deriver's
/// own contract is "a typed decline, never a wrong token".
///
/// (An earlier revision also listed `decline_reserved_dtype`: it put a reserved
/// dtype at operand 1, which KISS then ruled is NOT a key position, so Fuel's
/// keying it was right and the vector was corrected, see the provenance note.)
const EMITS_WHERE_KISS_DECLINES: &[&str] = &[];

/// Vectors Fuel cannot state at all (an absent surface, not a mismatch). Every
/// one needs a `DType` Fuel does not have (GAP-097): `i4` (no sub-byte integer)
/// or `f8e4m3fnuz` (no `fnuz` layouts), so the cell cannot be built and a
/// decline, where one is required, is by construction. The block-length and
/// scale-role logic these `i4` weight cells exercise is measured on `f32`
/// weights instead, modulo the `i4` coordinate, in
/// `kiss_structure_key_byte_match.rs`.
const NOT_EXPRESSIBLE: &[&str] = &[
    // `i4` weight or primary.
    "ctr_block_length_is_the_product_of_block_shape",
    "ctr_quantized_no_scale_designated",
    "ctr_quantized_scale_e8m0",
    "ctr_quantized_scale_f16",
    "ctr_weight_first_undeclared_accumulator_defaults_to_operand0_dtype",
    "ctr_weight_role_not_operand_one",
    "ctr_weight_second_undeclared_accumulator_defaults_to_operand0_dtype",
    "decline_ctr_block_length_outside_closed_set",
    "decline_ctr_scale_without_block_shape",
    "quantized_operand_dequant_form",
    "sub_byte_dtype_i4",
    // `f8e4m3fnuz`, the reserved dtype.
    "decline_ctr_reserved_weight_dtype",
    "decline_reserved_dtype",
    "decline_reserved_dtype_in_nonprimary_operand",
];

/// Fed in CALL order, Fuel's deriver declines because it documents that it takes
/// canonical order (`is_canonical_order`); KISS says canonicalization is the
/// deriver's job. Passes once the caller canonicalizes.
const NEEDS_CANONICAL_INPUT: &[&str] = &["call_order_is_not_canonical_order"];

fn rows() -> BTreeMap<String, (Outcome, Outcome, bool)> {
    let mut rows = BTreeMap::new();
    for v in &load() {
        let name = v["name"].as_str().unwrap().to_string();
        let golden = v.get("spec_golden").is_some_and(|g| !g.is_null());
        rows.insert(name, (evaluate(v, false), evaluate(v, true), golden));
    }
    rows
}

/// The measured report, printed so a human can read it.
#[test]
fn report_every_vector_under_both_feeding_disciplines() {
    let rows = rows();
    assert_eq!(rows.len(), 75, "vector count changed: re-pin the tables");
    for (name, (given, canon, golden)) in &rows {
        println!(
            "[kiss-deriv] {:<44} golden={:<5} as-given={:<18} canonicalized={}",
            name,
            golden,
            given.tag(),
            canon.tag()
        );
        for (label, o) in [("as-given", given), ("canonicalized", canon)] {
            match o {
                Outcome::Fail(got, want) => {
                    println!(
                        "[kiss-deriv]    {label}: got  {got}
[kiss-deriv]    {label}: want {want}"
                    )
                }
                Outcome::DeclineDiffers(t) => {
                    println!("[kiss-deriv]    {label}: expected a decline, got token {t}")
                }
                Outcome::UnexpectedDecline(want) => {
                    println!("[kiss-deriv]    {label}: declined, want {want}")
                }
                Outcome::ReasonDiffers(got, want) => {
                    println!("[kiss-deriv]    {label}: declined {got}, want {want}")
                }
                Outcome::NotExpressible(why) => println!("[kiss-deriv]    {label}: {why}"),
                Outcome::Pass => {}
            }
        }
    }
    let count = |f: &dyn Fn(&(Outcome, Outcome, bool)) -> &Outcome| {
        let mut m: BTreeMap<&'static str, usize> = BTreeMap::new();
        for r in rows.values() {
            *m.entry(f(r).tag()).or_default() += 1;
        }
        m
    };
    println!("[kiss-deriv] as-given:      {:?}", count(&|r| &r.0));
    println!("[kiss-deriv] canonicalized: {:?}", count(&|r| &r.1));
}

/// Every vector whose bytes the spec text itself pins (`spec_golden`) reproduces,
/// when fed canonically. A FAIL here is a deriver-vs-spec defect and not a
/// disagreement between two derivers.
#[test]
fn all_ten_spec_golden_vectors_byte_match_when_fed_canonically() {
    let rows = rows();
    let goldens: Vec<_> = rows.iter().filter(|(_, r)| r.2).collect();
    assert_eq!(goldens.len(), 10, "spec_golden count changed: re-measure");
    let bad: Vec<_> = goldens
        .iter()
        .filter(|(_, r)| r.1 != Outcome::Pass)
        .map(|(n, r)| format!("{n}: {:?}", r.1))
        .collect();
    assert!(bad.is_empty(), "golden vectors failed: {bad:?}");
}

/// The three outcome classes are pinned as SETS: a correct count can hide a wrong
/// distribution, so names (not numbers) are compared, in both directions.
#[test]
fn divergences_and_absent_surfaces_are_exactly_the_pinned_sets() {
    let rows = rows();
    let names = |f: &dyn Fn(&(Outcome, Outcome, bool)) -> bool| -> Vec<String> {
        rows.iter()
            .filter(|(_, r)| f(r))
            .map(|(n, _)| n.clone())
            .collect()
    };
    let sorted = |xs: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = xs.iter().map(|s| s.to_string()).collect();
        v.sort();
        v
    };

    // CONTROL: the harness can see a mismatch at all. The corpus and Fuel agree
    // on every decline now, so an empty set is a finding and not evidence of an
    // inert harness only if a deliberately WRONG expectation is reported. Take a
    // passing token vector and corrupt its expected token; and take a passing
    // decline vector and corrupt its expected reason.
    let vs = load();
    let tok = vs.iter().find(|v| v["expect"] == "token").unwrap();
    let mut wrong = tok.clone();
    wrong["value"] = serde_json::Value::String("sk8|not-the-token".to_string());
    assert!(matches!(evaluate(&wrong, true), Outcome::Fail(..)));
    let dec = vs
        .iter()
        .find(|v| v["name"] == "decline_collapsed_reduction_output")
        .unwrap();
    let mut wrong = dec.clone();
    wrong["value"] = serde_json::Value::String("ReservedDtype".to_string());
    assert!(matches!(evaluate(&wrong, true), Outcome::ReasonDiffers(..)));

    assert_eq!(
        names(&|r| matches!(r.1, Outcome::DeclineDiffers(_))),
        sorted(EMITS_WHERE_KISS_DECLINES),
        "the set of vectors where Fuel EMITS a token and KISS requires a decline changed"
    );
    assert_eq!(
        names(&|r| matches!(r.1, Outcome::NotExpressible(_))),
        sorted(NOT_EXPRESSIBLE),
        "the set of vectors Fuel cannot state changed"
    );
    assert_eq!(
        names(&|r| r.0 != r.1 && r.1 == Outcome::Pass),
        sorted(NEEDS_CANONICAL_INPUT),
        "the set of vectors that pass only when the caller canonicalizes changed"
    );
    // Everything else passes in BOTH disciplines.
    let accounted =
        EMITS_WHERE_KISS_DECLINES.len() + NOT_EXPRESSIBLE.len() + NEEDS_CANONICAL_INPUT.len();
    let both_pass = rows
        .values()
        .filter(|r| r.0 == Outcome::Pass && r.1 == Outcome::Pass)
        .count();
    assert_eq!(both_pass + accounted, 75, "an unpinned outcome exists");
}
