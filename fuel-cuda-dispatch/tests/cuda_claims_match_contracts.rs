// SPDX-License-Identifier: MIT OR Apache-2.0
//! The CUDA accuracy claims against the CUDA contracts and the registered entries. Moved from
//! `fuel-dispatch/src/fkc/verify/seed_cuda_accuracy.rs` (its `tests` module) with the contracts, ROADMAP row 137
//! step 6.2: the claim constants stay in `fuel-dispatch` (the measurement code needs them) and arrive here
//! through `fuel_dispatch::dev_tools::cuda_accuracy_claim_rows`. Hermetic: reads text and populates a table.

#![cfg(feature = "cuda")]

use fuel_cuda_dispatch::register_baracuda_cuda_kernels;
use fuel_dispatch::dev_tools::{cuda_accuracy_claim_rows, cuda_qmatmul_f32_claim};
use fuel_dispatch::kernel::KernelBindingTable;
use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;

fn contract(stem: &str) -> String {
    let path = format!(
        "{}/kernel-contracts/cuda/{stem}.fkc.md",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
        .replace("\r\n", "\n")
}

fn find_entry<'a>(
    table: &'a KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
) -> &'a fuel_dispatch::kernel::BindingEntry {
    table
        .iter_entries()
        .find(|(o, d, b, _)| *o == op && *d == dtypes && *b == BackendId::Cuda)
        .map(|(_, _, _, e)| e)
        .unwrap_or_else(|| panic!("no Cuda {op:?} {dtypes:?} entry"))
}

/// The claim constants and their contract sections say the same numbers: a claim edited in one place only
/// would be earned against a bound the contract does not declare.
#[test]
fn claim_constants_match_the_contracts() {
    let rows = cuda_accuracy_claim_rows();
    assert!(rows.len() > 20, "only {} claim rows", rows.len());
    for (stem, kernel, max_relative, max_absolute, _, _) in rows {
        let text = contract(stem);
        let marker = format!("kernel: {kernel}\n");
        let at = text
            .find(&marker)
            .unwrap_or_else(|| panic!("section `{kernel}` present in {stem}.fkc.md"));
        let rest = &text[at..];
        let prec = &rest[rest.find("precision:").expect("precision block")..];
        let prec = &prec[..prec.find("audited:").expect("precision block ends")];
        assert!(
            prec.contains(&format!("max_relative: {max_relative}")),
            "{kernel}: contract max_relative differs from the claim constant"
        );
        assert!(
            prec.contains(&format!("max_absolute: {max_absolute}")),
            "{kernel}: contract max_absolute differs from the claim constant"
        );
    }
}

#[test]
fn the_qmatmul_claim_matches_the_contract_and_reaches_the_entry() {
    let (max_relative, max_absolute) = cuda_qmatmul_f32_claim();
    let text = contract("qmatmul");
    let at = text.find("kernel: qmatmul_f32\n").expect("section present");
    let rest = &text[at..];
    let prec = &rest[rest.find("precision:").expect("precision block")..];
    let prec = &prec[..prec.find("audited:").expect("precision block ends")];
    assert!(prec.contains(&format!("max_relative: {max_relative}")));
    assert!(prec.contains(&format!("max_absolute: {max_absolute}")));
    let mut table = KernelBindingTable::new();
    register_baracuda_cuda_kernels(&mut table);
    let e = find_entry(
        &table,
        OpKind::QMatMul,
        &[DType::F32, DType::U32, DType::F32],
    );
    assert_eq!(e.precision.max_relative, Some(max_relative));
    assert_eq!(e.precision.max_absolute, Some(max_absolute));
    assert!(fuel_dispatch::ranker::filters::meets_accuracy_bound(
        &e.precision,
        &fuel_dispatch::ranker::filters::DEFAULT_ACCURACY_BOUND
    ));
}

/// What the import gate leaves on the registered entries once the ledger backs the claims: each F32 entry
/// carries its numeric claim AND keeps `bit_stable`, and the F16 entry of the same op (a different section,
/// no numeric claim) is not collapsed to UNAUDITED by it.
#[test]
fn the_ledger_backs_the_f32_claims_and_leaves_the_other_dtypes_audited() {
    let mut table = KernelBindingTable::new();
    register_baracuda_cuda_kernels(&mut table);
    for (_, kernel, max_relative, max_absolute, op, arity) in cuda_accuracy_claim_rows() {
        let f32e = find_entry(&table, op, &vec![DType::F32; arity]);
        assert!(f32e.precision.bit_stable_on_same_hardware, "{kernel}");
        assert_eq!(f32e.precision.max_relative, Some(max_relative), "{kernel}");
        assert_eq!(f32e.precision.max_absolute, Some(max_absolute), "{kernel}");
        if kernel == "sign" {
            continue; // F32-only op: there is no other-dtype section to keep audited
        }
        let f16e = find_entry(&table, op, &vec![DType::F16; arity]);
        assert!(f16e.precision.bit_stable_on_same_hardware, "{kernel} f16");
        assert_eq!(f16e.precision.max_relative, None, "{kernel} f16");
    }
}
