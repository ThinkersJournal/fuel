// SPDX-License-Identifier: MIT OR Apache-2.0
//! Each Vulkan accuracy claim constant and its contract section say the same numbers: a claim edited in one
//! place only would be earned against a bound the contract does not declare. Moved here from the unit tests of
//! `fuel-dispatch`'s `seed_vulkan_matmul_accuracy` (ROADMAP row 137 step 6.1) together with the contracts it
//! reads; the claims arrive through the hidden `fuel_dispatch::dev_tools::vulkan_f32_accuracy_claims`.

use fuel_dispatch::dev_tools;

fn section<'a>(text: &'a str, kernel: &str) -> &'a str {
    let marker = format!("kernel: {kernel}");
    let mut start = 0;
    loop {
        let at = text[start..]
            .find(&marker)
            .unwrap_or_else(|| panic!("section `{kernel}` present"));
        let after = start + at + marker.len();
        // `kernel: rms_norm_last_dim` is a prefix of `kernel: rms_norm_last_dim_f32`:
        // require the name to END at the line.
        if matches!(text.as_bytes().get(after), Some(b'\r' | b'\n')) {
            let begin = start + at;
            let rest = &text[begin..];
            return &rest[..rest.find("```").expect("section ends")];
        }
        start = after;
    }
}

#[test]
fn claim_constants_match_the_contracts() {
    let contract = |kernel: &str| match kernel {
        "matmul_f32" => include_str!("../kernel-contracts/vulkan/matmul.fkc.md"),
        "rms_norm_last_dim_f32" | "softmax_last_dim_f32" => {
            include_str!("../kernel-contracts/vulkan/norm.fkc.md")
        }
        "rope_f32" => include_str!("../kernel-contracts/vulkan/rope.fkc.md"),
        "qmatmul_vk" => include_str!("../kernel-contracts/vulkan/qmatmul.fkc.md"),
        other => panic!("no contract file known for claim `{other}`"),
    };
    let claims = dev_tools::vulkan_f32_accuracy_claims();
    // Positive control: all five claims arrive, so a loop over an empty list cannot pass.
    assert_eq!(claims.len(), 5);
    for (kernel, max_relative, max_absolute) in claims {
        let s = section(contract(kernel), kernel);
        assert!(
            s.contains(&format!("max_relative: {max_relative}")),
            "{kernel}: contract max_relative differs from the claim constant"
        );
        assert!(
            s.contains(&format!("max_absolute: {max_absolute}")),
            "{kernel}: contract max_absolute differs from the claim constant"
        );
    }
}

/// GAP-393: the default placement rule drops a kernel that is neither bit-stable nor claims an accuracy bound
/// whenever a bit-stable CPU kernel is offered, so an unclaimed Vulkan `QMatMul` never won placement and a loaded
/// quantized model ran its `QMatMul`s on the CPU. The registered binding (precision as the ledger gate leaves it)
/// must meet the default bound; the control is the CPU binding, which meets it by being bit-stable.
#[test]
fn the_vulkan_qmatmul_binding_can_win_default_placement() {
    use fuel_dispatch::kernel::KernelBindingTable;
    use fuel_dispatch::ranker::filters::{DEFAULT_ACCURACY_BOUND, meets_accuracy_bound};
    use fuel_ir::DType::{F32, U32};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::probe::BackendId;

    let mut table = KernelBindingTable::new();
    fuel_vulkan_dispatch::register_vulkan_kernels(&mut table);
    let alts = table.lookup_alternatives(OpKind::QMatMul, &[F32, U32, F32], BackendId::Vulkan);
    assert_eq!(alts.len(), 1, "one Vulkan QMatMul binding is registered");
    let p = &alts[0].precision;
    assert!(
        !p.bit_stable_on_same_hardware,
        "premise: the Vulkan reduction order is not pinned, so only a claim can earn placement"
    );
    assert!(
        meets_accuracy_bound(p, &DEFAULT_ACCURACY_BOUND),
        "the Vulkan QMatMul binding does not meet the default accuracy bound, so the placement rule drops it \
         in favour of the CPU kernel: {p:?}"
    );
}
