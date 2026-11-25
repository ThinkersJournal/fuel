// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Metal `argmax`/`argmin` comparator is spelled the way the CPU rule is (a NaN WINS, the
//! FIRST NaN's index is returned; no NaN: ties take the first index; `-0.0 == +0.0`).
//!
//! **UNMEASURED.** This is a TEXT-level test: there is no Metal device on this box, so the
//! kernel has never run against the spec vectors. It pins that the comparator source says the
//! rule, so a later edit cannot silently drop it; it does NOT claim Metal parity. The measurement
//! is the row "measure Metal arg_reduce on a Mac" (the Vulkan counterpart,
//! `fuel-vulkan-dispatch/tests/vulkan_arg_nan_vectors_live.rs`, IS measured).
//!
//! The NaN test is a BIT-PATTERN test (`arg_is_nan`), because the shaders compile with fast math
//! ON by default (`FUEL_METAL_ENABLE_FAST_MATH`), under which `x != x` may be folded to false; the
//! generic overload returns false for the integer types that share the comparator.

use std::path::PathBuf;

fn reduce_metal() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("fuel-metal-kernels")
        .join("src")
        .join("metal_src")
        .join("reduce.metal");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// The body of `operator<` / `operator>` for `indexed<T>`.
fn comparator_body(src: &str, op: &str) -> String {
    let head = format!("METAL_FUNC bool operator{op}(indexed<T> lhs, indexed<T> rhs)");
    let at = src
        .find(&head)
        .unwrap_or_else(|| panic!("`{head}` not found in reduce.metal"));
    let open = at + src[at..].find('{').expect("comparator has a body");
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return src[open..=open + i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces in the `{op}` comparator");
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn both_indexed_comparators_let_a_nan_win_and_take_the_first_nan() {
    let src = reduce_metal();
    for op in ["<", ">"] {
        let body = norm(&comparator_body(&src, op));
        for needle in [
            "bool ln = arg_is_nan(lhs.val);",
            "bool rn = arg_is_nan(rhs.val);",
            "if (ln || rn) return ln && (!rn || lhs.i < rhs.i);",
            "(lhs.val == rhs.val && lhs.i < rhs.i)",
        ] {
            assert!(
                body.contains(needle),
                "operator{op} for indexed<T> must contain `{needle}` (NaN wins, first NaN index); body: {body}"
            );
        }
    }
}

/// The sabotage sibling: the OLD comparator (NaN compares false, so a NaN is ignored) must fail
/// the same check, so the check above can go red.
#[test]
fn the_old_comparator_text_fails_the_check() {
    let old = "constexpr METAL_FUNC bool operator>(indexed<T> lhs, indexed<T> rhs) { return lhs.val > rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i); }";
    assert!(!norm(old).contains("if (ln || rn) return ln && (!rn || lhs.i < rhs.i);"));
}

/// The helper must cover every float element type the arg-reduce kernels instantiate, by bit
/// pattern: a missing overload falls to the generic `false` and would silently ignore NaN.
#[test]
fn the_nan_helper_covers_float_half_and_bfloat_by_bit_pattern() {
    let src = norm(&reduce_metal());
    for needle in [
        "bool arg_is_nan(float x) { return (as_type<uint>(x) & 0x7fffffffu) > 0x7f800000u; }",
        "bool arg_is_nan(half x) { return (as_type<ushort>(x) & 0x7fffu) > 0x7c00u; }",
        "bool arg_is_nan(bfloat x) { return (as_type<ushort>(x) & 0x7fffu) > 0x7f80u; }",
        "template<typename T> METAL_FUNC bool arg_is_nan(T) { return false; }",
    ] {
        assert!(src.contains(needle), "reduce.metal must contain `{needle}`");
    }
}
