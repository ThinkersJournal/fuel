#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused Q3_K x F32 gemv (ROADMAP row 119): out[col] = sum_k a[k] * dequant(W[col])[k], one workgroup
// per output column, one activation row per dispatch (the host loops rows, as for qmatvec_q5_k).
//
// PROVENANCE: the block layout and the dequantisation arithmetic are ggml / llama.cpp's Q3_K
// (`block_q3_K` and `dequantize_row_q3_K`, MIT, https://github.com/ggml-org/llama.cpp), reached through
// candle's port (`candle-core/src/quantized/k_quants.rs`, MIT OR Apache-2.0) as `BlockQ3K::to_float` in
// `fuel-quantized/src/k_quants.rs`, which this shader is checked against on the dequantised weight.
// Licence texts: `THIRD-PARTY-NOTICES.md` (sections "llama.cpp / ggml" and "candle"). The code below is
// written from that arithmetic, not copied from any shader.
//
// Block (110 bytes, 256 elements; NOT a multiple of 4, so every read goes through byte_at):
//   bytes 0..31    hmask[32]  (the high bit of each element, inverted: a CLEAR bit means "minus 4")
//   bytes 32..95   qs[64]     (2 low bits of each element, 4 elements per byte)
//   bytes 96..107  scales[12] (sixteen 6-bit scales, the ggml kmask packing)
//   bytes 108..109 f16 d
// For half h in 0..2 and t in 0..4 (32 elements, output index 128h + 32t + 16s + i, s in 0..2, i in 0..16):
//   sc = scale[8h + 2t + s] - 32
//   y  = d * sc * (((qs[32h + 16s + i] >> 2t) & 3) - ((hmask[16s + i] & (1 << (4h + t))) == 0 ? 4 : 0))

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

// The j-th (0..16) unsigned 6-bit scale of the block whose scales start at byte `s` (ggml aux[] unpack).
uint scale6(uint j, uint s) {
    uint q = j & 3u;
    uint hi = byte_at(s + 8u + q);
    uint lo_byte = byte_at(s + ((j >> 2) & 1u) * 4u + q);
    uint lo = (j < 8u) ? (lo_byte & 0xFu) : (lo_byte >> 4u);
    uint hi2 = (hi >> (2u * (j >> 2))) & 3u;
    return lo | (hi2 << 4u);
}

void main() {
    uint col = gl_WorkGroupID.x;
    if (col >= p.n) return;
    uint tid = gl_LocalInvocationID.x;

    float acc = 0.0;
    uint units = p.blocks_per_row * 8u; // (block, half, 32-element group) triples
    for (uint u = tid; u < units; u += 128u) {
        uint blk = u >> 3;
        uint h = (u >> 2) & 1u;
        uint t = u & 3u;
        uint base = (col * p.blocks_per_row + blk) * 110u; // byte offset of the block
        uint dh = byte_at(base + 108u) | (byte_at(base + 109u) << 8u);
        float d = unpackHalf2x16(dh).x;
        uint mbit = 1u << (4u * h + t);
        for (uint s = 0u; s < 2u; s++) {
            float dl = d * (float(scale6(8u * h + 2u * t + s, base + 96u)) - 32.0);
            uint a0 = blk * 256u + 128u * h + 32u * t + 16u * s;
            for (uint i = 0u; i < 16u; i++) {
                uint q = (byte_at(base + 32u + 32u * h + 16u * s + i) >> (2u * t)) & 3u;
                float v = float(int(q) - (((byte_at(base + 16u * s + i) & mbit) == 0u) ? 4 : 0));
                acc += A[a0 + i] * (dl * v);
            }
        }
    }
    partials[tid] = acc;
    barrier();
    for (uint sh = 64u; sh > 0u; sh >>= 1) {
        if (tid < sh) partials[tid] += partials[tid + sh];
        barrier();
    }
    if (tid == 0u) O[col] = partials[0];
}
