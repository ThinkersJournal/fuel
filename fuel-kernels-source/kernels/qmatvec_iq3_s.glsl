#version 450
// SPDX-License-Identifier: MIT OR Apache-2.0
// GENERATED FILE: edit `templates/qmatvec_iq3_s.glsl.in`, then run `python scripts/gen_iq3s_glsl.py write`.
// Fused IQ3_S x F32 gemv (ROADMAP row 119): out[col] = sum_k a[k] * dequant(W[col])[k], one workgroup
// per output column, one activation row per dispatch (the host loops rows, as for qmatvec_q5_k).
//
// PROVENANCE: the block layout, the dequantisation arithmetic and the 512-entry `iq3s_grid` codebook are
// llama.cpp's / ggml's (MIT, https://github.com/ggml-org/llama.cpp) at commit 9d57ce456c:
// `ggml/src/ggml-common.h` (`block_iq3_s`, `iq3s_grid`, `kmask_iq2xs`) and `ggml/src/ggml-quants.c`
// (`dequantize_row_iq3_s`). The grid below is NOT typed or pasted: it is generated from
// `IQ3S_GRID` in `fuel-quantized/src/iq_grids.rs`, itself generated from `ggml-common.h` at that commit by
// `scripts/gen_iq_grids.py` (whose `check` mode compares it to upstream); `scripts/gen_iq3s_glsl.py check`
// and the test `iq3s_glsl_grid_matches_rust_grid` compare this table to the Rust one entry by entry. The
// shader is checked against `BlockIq3S::decode`. Licence text: `THIRD-PARTY-NOTICES.md` (section
// "llama.cpp / ggml"). The remaining code is written from the ggml arithmetic, not copied from any shader.
//
// Block (110 bytes, 256 elements; NOT a multiple of 4, so every read goes through byte_at):
//   bytes 0..1     f16 d
//   bytes 2..65    qs[64]       bytes 66..73  qh[8]      bytes 74..105 signs[32]     bytes 106..109 scales[4]
// For sub-block ib32 in 0..8 (pair = ib32 / 2): db = d * (1 + 2 * (scales[pair] nibble ib32 % 2)); for l in 0..4:
//   i1 = qs[8 ib32 + 2l]     | ((qh[ib32] << (8 - 2l)) & 256),  i2 = qs[8 ib32 + 2l + 1] | ((qh[ib32] << (7 - 2l)) & 256)
//   element 32 ib32 + 8l + j     = db * byte j of grid[i1] * (signs[4 ib32 + l] & (1 << j)       ? -1 : 1)   (j in 0..4)
//   element 32 ib32 + 8l + j + 4 = db * byte j of grid[i2] * (signs[4 ib32 + l] & (1 << (j + 4)) ? -1 : 1)

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

// BEGIN IQ3S_GRID
const uint IQ3S_GRID[512] = uint[512](
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u
);
// END IQ3S_GRID

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
        uint ib32 = u & 7u;
        uint base = (col * p.blocks_per_row + blk) * 110u; // byte offset of the block
        float d = unpackHalf2x16(byte_at(base) | (byte_at(base + 1u) << 8u)).x;
        uint sc = byte_at(base + 106u + (ib32 >> 1));
        float db = d * float(1u + 2u * ((ib32 & 1u) == 0u ? (sc & 0xFu) : (sc >> 4u)));
        uint qh = byte_at(base + 66u + ib32);
        uint a0 = blk * 256u + ib32 * 32u;
        for (uint l = 0u; l < 4u; l++) {
            uint i1 = byte_at(base + 2u + 8u * ib32 + 2u * l) | ((qh << (8u - 2u * l)) & 256u);
            uint i2 = byte_at(base + 2u + 8u * ib32 + 2u * l + 1u) | ((qh << (7u - 2u * l)) & 256u);
            uint g1 = IQ3S_GRID[i1];
            uint g2 = IQ3S_GRID[i2];
            uint sg = byte_at(base + 74u + 4u * ib32 + l);
            for (uint j = 0u; j < 4u; j++) {
                float s1 = ((sg & (1u << j)) != 0u) ? -1.0 : 1.0;
                float s2 = ((sg & (1u << (j + 4u))) != 0u) ? -1.0 : 1.0;
                acc += A[a0 + 8u * l + j] * (db * float((g1 >> (8u * j)) & 0xFFu) * s1)
                     + A[a0 + 8u * l + j + 4u] * (db * float((g2 >> (8u * j)) & 0xFFu) * s2);
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
