// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`scripts/gpu-run.ps1` takes the machine-wide GPU mutex from the canonical helper, byte for byte.**
//!
//! `Global\gpu-run` is one mutex for the whole machine, so every repo's script must create and open it
//! the same way. Fuel's copy created it inline with `MutexAcl.Create(..)`, which asks for
//! `MUTEX_ALL_ACCESS` even when the mutex exists, so a second process (any account) got "access
//! denied" instead of waiting while another account held it (Fuel waited from ~01:43Z on 2026-11-12
//! behind Baracuda's CI). The canonical fix (Baracuda#355; copies in `C:/Projects/tools/gpu-run`)
//! opens the existing mutex first and creates it only when absent.
//!
//! This test pins three things:
//!
//! 1. `scripts/gpu-run-mutex.ps1` and `scripts/test-gpu-run-interleave.ps1` hash to the lines in
//!    `scripts/gpu-run-canonical.SHA256SUMS` (copied from the canonical `SHA256SUMS`). `gpu-run.ps1`
//!    itself is NOT in that list: it legitimately differs from the canonical copy (Fuel's
//!    `FUEL_REQUIRE_*` documentation), so it is checked structurally instead.
//! 2. `gpu-run.ps1` dot-sources the helper, carries the stale-lockfile re-read, is version 4, and has
//!    no inline mutex creation (`inline_mutex_creation`).
//! 3. Both checks can fail: a flipped byte changes the hash and a synthetic old-style script is
//!    flagged (the sabotage siblings below).
//!
//! **What this cannot see.** It reads files; it does not run PowerShell. That two concurrent
//! `gpu-run` calls alternate is shown by `scripts/test-gpu-run-interleave.ps1` (run it under `pwsh`
//! and `powershell`); `.gitattributes` pins the byte-compared files to `eol=lf` so a Windows
//! checkout does not change their hash.

use std::path::PathBuf;

fn scripts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("scripts")
}

/// SHA-256 (FIPS 180-4), std only, so the test adds no dependency. Known-answer tests below.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// True when the script creates the GPU mutex itself instead of through the canonical helper.
fn inline_mutex_creation(script: &str) -> bool {
    script.contains("MutexAcl]::Create") || script.contains("System.Threading.Mutex($false")
}

/// The `(hex, file name)` pairs of `scripts/gpu-run-canonical.SHA256SUMS` (`<hex> *<name>` lines).
fn recorded_sums() -> Vec<(String, String)> {
    let text = std::fs::read_to_string(scripts_dir().join("gpu-run-canonical.SHA256SUMS"))
        .expect("scripts/gpu-run-canonical.SHA256SUMS exists");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (h, name) = l.split_once(" *").expect("`<hex> *<name>` line");
            (h.to_string(), name.trim().to_string())
        })
        .collect()
}

#[test]
fn sha256_known_answers() {
    // CONTROL for the hash below: the empty string and "abc" (FIPS 180-4 examples).
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // A two-block message (> 55 bytes), which exercises the padding path.
    assert_eq!(
        hex(&sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn canonical_helper_files_match_their_recorded_sha256() {
    let sums = recorded_sums();
    assert_eq!(
        sums.len(),
        2,
        "CONTROL: the sums file lists the helper and its interleave test, got {sums:?}"
    );
    for (want, name) in &sums {
        let bytes = std::fs::read(scripts_dir().join(name))
            .unwrap_or_else(|e| panic!("scripts/{name} is readable: {e}"));
        assert_eq!(
            &hex(&sha256(&bytes)),
            want,
            "scripts/{name} is not byte for byte the canonical copy (C:/Projects/tools/gpu-run, \
             Baracuda#355). Take the canonical file again; do not edit it here. If a Windows checkout \
             changed it, check the `eol=lf` pin in .gitattributes."
        );
    }
}

#[test]
fn a_changed_byte_is_detected() {
    // Sabotage sibling: the same comparison must FAIL on a one-byte change, so the test above
    // cannot pass vacuously.
    let (want, name) = recorded_sums().remove(0);
    let mut bytes = std::fs::read(scripts_dir().join(&name)).expect("helper readable");
    let i = bytes.len() / 2;
    bytes[i] ^= 0x01;
    assert_ne!(
        hex(&sha256(&bytes)),
        want,
        "a flipped byte must change the hash"
    );
}

#[test]
fn gpu_run_takes_the_canonical_mutex_helper() {
    let script = std::fs::read_to_string(scripts_dir().join("gpu-run.ps1")).expect("gpu-run.ps1");
    assert!(
        script.contains("gpu-run-mutex.ps1") && script.contains("New-GpuRunMutex"),
        "gpu-run.ps1 must dot-source gpu-run-mutex.ps1 and call New-GpuRunMutex"
    );
    assert!(
        !inline_mutex_creation(&script),
        "gpu-run.ps1 creates the mutex inline: any second process, in any account, gets 'access denied' \
         while another holds it. Use New-GpuRunMutex from the canonical helper."
    );
    assert!(
        script.contains("$GpuRunVersion = 4"),
        "gpu-run.ps1 must be version 4 (the open-first mutex)"
    );
    assert!(
        script.contains("Re-read the lockfile NOW"),
        "gpu-run.ps1 must re-read the lockfile after acquiring (the stale-lock check)"
    );
}

#[test]
fn an_old_style_inline_creation_is_flagged() {
    // Sabotage sibling for `inline_mutex_creation`: the old Fuel code and the .NET Framework spelling.
    assert!(inline_mutex_creation(
        "$mutex = [System.Threading.MutexAcl]::Create($false, $MutexName, [ref]$createdNew, $mutexSec)"
    ));
    assert!(inline_mutex_creation(
        "$mutex = New-Object System.Threading.Mutex($false, $MutexName, [ref]$createdNew, $mutexSec)"
    ));
    assert!(!inline_mutex_creation(
        "$mutex        = New-GpuRunMutex $MutexName"
    ));
}
