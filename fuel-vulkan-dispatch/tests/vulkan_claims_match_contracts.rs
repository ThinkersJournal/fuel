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
        other => panic!("no contract file known for claim `{other}`"),
    };
    let claims = dev_tools::vulkan_f32_accuracy_claims();
    // Positive control: all four claims arrive, so a loop over an empty list cannot pass.
    assert_eq!(claims.len(), 4);
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
