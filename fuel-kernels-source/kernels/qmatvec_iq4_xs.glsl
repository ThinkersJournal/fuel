#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused IQ4_XS x F32 gemv (ROADMAP row 119): out[col] = sum_k a[k] * dequant(W[col])[k], one workgroup
// per output column, one activation row per dispatch (the host loops rows, as for qmatvec_q5_k).
//
// PROVENANCE: the block layout, the 16-entry non-linear codebook `kvalues_iq4nl` and the dequantisation
// arithmetic are llama.cpp's / ggml's (MIT, https://github.com/ggml-org/llama.cpp) at commit 9d57ce456c:
// `ggml/src/ggml-common.h` (`block_iq4_xs`, `kvalues_iq4nl`) and `ggml/src/ggml-quants.c`
// (`dequantize_row_iq4_xs`), as recorded in `fuel-quantized/src/iq4.rs`, which this shader is checked
// against on the dequantised weight (`BlockIq4Xs::to_float`). Licence text: `THIRD-PARTY-NOTICES.md`
// (section "llama.cpp / ggml"). The code below is written from that arithmetic, not copied from any shader.
//
// Block (136 bytes, 256 elements):
//   bytes 0..1     f16 d
//   bytes 2..3     scales_h (u16, little endian)
//   bytes 4..7     scales_l[4]
//   bytes 8..135   qs[128]
// For sub-block ib in 0..8: ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 15) | (((scales_h >> 2 ib) & 3) << 4),
//   dl = d * (ls - 32); element 32 ib + j = dl * kv[qs[16 ib + j] & 15], element 32 ib + j + 16 = dl * kv[qs[16 ib + j] >> 4].

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer ABuf { float A[]; };
layout(set = 0, binding = 1, std430) readonly buffer WBuf { uint W[]; };
layout(set = 0, binding = 2, std430) buffer OBuf { float O[]; };

layout(set = 0, binding = 3, std140) uniform Params {
    uint n;
    uint k;
    uint blocks_per_row;
    uint _pad;
} p;

shared float partials[128];

const int KV[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

uint byte_at(uint off) {
    return (W[off >> 2] >> ((off & 3u) * 8u)) & 0xFFu;
}

void main() {
    uint col = gl_WorkGroupID.x;
    if (col >= p.n) return;
    uint tid = gl_LocalInvocationID.x;

    float acc = 0.0;
    uint units = p.blocks_per_row * 8u; // (block, sub-block of 32) pairs
    for (uint u = tid; u < units; u += 128u) {
        uint blk = u >> 3;
        uint ib = u & 7u;
        uint base = (col * p.blocks_per_row + blk) * 136u; // byte offset of the block
        float d = unpackHalf2x16(W[base >> 2]).x;
        uint scales_h = W[base >> 2] >> 16u;
        uint sh = 2u * ib;
        uint ls = ((byte_at(base + 4u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u) | (((scales_h >> sh) & 3u) << 4u);
        float dl = d * float(int(ls) - 32);
        uint a0 = blk * 256u + ib * 32u;
        uint qs = base + 8u + ib * 16u;
        for (uint j = 0u; j < 16u; j++) {
            uint q = byte_at(qs + j);
            acc += A[a0 + j] * (dl * float(KV[q & 15u]))
                 + A[a0 + j + 16u] * (dl * float(KV[q >> 4u]));
        }
    }
    partials[tid] = acc;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        barrier();
    }
    if (tid == 0u) O[col] = partials[0];
}
