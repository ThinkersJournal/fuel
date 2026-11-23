#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
// Fused IQ4_NL x F32 gemv (ROADMAP row 119): out[col] = sum_k a[k] * dequant(W[col])[k], one workgroup
// per output column, one activation row per dispatch (the host loops rows, as for qmatvec_q5_k).
//
// PROVENANCE: the block layout, the 16-entry non-linear codebook `kvalues_iq4nl` and the dequantisation
// arithmetic are llama.cpp's / ggml's (MIT, https://github.com/ggml-org/llama.cpp) at commit 9d57ce456c:
// `ggml/src/ggml-common.h` (`QK4_NL = 32`, `block_iq4_nl`, `kvalues_iq4nl`) and `ggml/src/ggml-quants.c`
// (`dequantize_row_iq4_nl`), as recorded in `fuel-quantized/src/iq4.rs`, which this shader is checked
// against on the dequantised weight (`BlockIq4Nl::to_float`). Licence text: `THIRD-PARTY-NOTICES.md`
// (section "llama.cpp / ggml"). The code below is written from that arithmetic, not copied from any shader.
//
// Block (18 bytes, 32 elements; NOT a multiple of 4, so every read goes through byte_at):
//   bytes 0..1   f16 d
//   bytes 2..17  qs[16]: element j = d * kv[qs[j] & 15], element j + 16 = d * kv[qs[j] >> 4]

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
    uint units = p.blocks_per_row * 4u; // (block, group of 4 bytes) pairs
    for (uint u = tid; u < units; u += 128u) {
        uint blk = u >> 2;
        uint g = u & 3u;
        uint base = (col * p.blocks_per_row + blk) * 18u; // byte offset of the block
        float d = unpackHalf2x16(byte_at(base) | (byte_at(base + 1u) << 8u)).x;
        uint a0 = blk * 32u;
        for (uint jj = 0u; jj < 4u; jj++) {
            uint j = 4u * g + jj;
            uint q = byte_at(base + 2u + j);
            acc += A[a0 + j] * (d * float(KV[q & 15u]))
                 + A[a0 + j + 16u] * (d * float(KV[q >> 4u]));
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
