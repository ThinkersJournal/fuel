#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused Q5_K x F32 gemv (ROADMAP row 119): out[col] = sum_k a[k] * dequant(W[col])[k], one workgroup
// per output column, one activation row per dispatch (the host loops rows, as for qmatvec_q6_k).
//
// PROVENANCE: the block layout and the dequantisation arithmetic are ggml / llama.cpp's Q5_K
// (`block_q5_K` and `dequantize_row_q5_K`, MIT, https://github.com/ggml-org/llama.cpp), reached through
// candle's port (`candle-core/src/quantized/k_quants.rs`, MIT OR Apache-2.0) as `BlockQ5K::to_float` in
// `fuel-quantized/src/k_quants.rs`, which this shader is checked against bit for bit on the dequantised
// weight. Licence texts: `THIRD-PARTY-NOTICES.md` (sections "llama.cpp / ggml" and "candle"). The code
// below is written from that arithmetic, not copied from any shader.
//
// Block (176 bytes = 44 u32 words, 256 elements):
//   word 0        f16 d (low half), f16 dmin (high half)
//   words 1..3    scales[12]  (6-bit scale/min pairs, the get_scale_min_k4 packing)
//   words 4..11   qh[32]      (the 5th bit of each element)
//   words 12..43  qs[128]     (low/high nibbles)
// For group g in 0..4 (64 elements): ql = qs[32g .. 32g+32], u1 = 1 << 2g, u2 = 2 << 2g,
//   y[64g + i]      = d * sc[2g]   * ((ql[i] & 15) + (qh[i] & u1 ? 16 : 0)) - dmin * m[2g]
//   y[64g + 32 + i] = d * sc[2g+1] * ((ql[i] >> 4) + (qh[i] & u2 ? 16 : 0)) - dmin * m[2g+1]

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

uint byte_at(uint off) {
    return (W[off >> 2] >> ((off & 3u) * 8u)) & 0xFFu;
}

// ggml get_scale_min_k4: (scale, min) for sub-block j of a block whose scales start at byte `s`.
uvec2 scale_min(uint j, uint s) {
    if (j < 4u) {
        return uvec2(byte_at(s + j) & 63u, byte_at(s + j + 4u) & 63u);
    }
    uint lo = byte_at(s + j + 4u);
    return uvec2((lo & 0xFu) | ((byte_at(s + j - 4u) >> 6u) << 4u),
                 (lo >> 4u) | ((byte_at(s + j) >> 6u) << 4u));
}

void main() {
    uint col = gl_WorkGroupID.x;
    if (col >= p.n) return;
    uint tid = gl_LocalInvocationID.x;

    float acc = 0.0;
    uint units = p.blocks_per_row * 4u; // (block, group of 64) pairs
    for (uint u = tid; u < units; u += 128u) {
        uint blk = u >> 2;
        uint g = u & 3u;
        uint base = (col * p.blocks_per_row + blk) * 176u; // byte offset of the block
        vec2 dd = unpackHalf2x16(W[base >> 2]);
        uvec2 s1 = scale_min(2u * g, base + 4u);
        uvec2 s2 = scale_min(2u * g + 1u, base + 4u);
        float d1 = dd.x * float(s1.x), m1 = dd.y * float(s1.y);
        float d2 = dd.x * float(s2.x), m2 = dd.y * float(s2.y);
        uint u1 = 1u << (2u * g);
        uint u2 = 2u << (2u * g);
        uint a0 = blk * 256u + g * 64u;
        uint qs = base + 48u + g * 32u;
        uint qh = base + 16u;
        for (uint i = 0u; i < 32u; i++) {
            uint ql = byte_at(qs + i);
            uint h = byte_at(qh + i);
            float lo = d1 * float((ql & 15u) + (((h & u1) != 0u) ? 16u : 0u)) - m1;
            float hi = d2 * float((ql >> 4u) + (((h & u2) != 0u) ? 16u : 0u)) - m2;
            acc += A[a0 + i] * lo + A[a0 + 32u + i] * hi;
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
