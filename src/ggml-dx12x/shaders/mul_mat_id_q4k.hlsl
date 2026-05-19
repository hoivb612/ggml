// mul_mat_id_q4k.hlsl -- K-direction reduction MUL_MAT_ID for Q4_K.
//
// Algorithm: 1 workgroup == 1 output element. 256 threads cooperate over
// the K dimension via strided per-thread accumulation, then a wave reduce
// plus groupshared reduction collapses partials into a single result.
//
// Compared to the prior "1 thread = 1 output, K serial per thread" design
// (in mul_mat_id_quant.hlsli), this:
//   1. drops per-thread K iters from K down to K/256, and
//   2. dispatches 256x more groups, taking Mixtral MoE
//      (K=4096 for gate/up, K=14336 for down, N up to 14336) from ~32-112
//      groups -- which leaves most of XSX's 52 CUs idle -- to thousands of
//      groups for full CU occupancy.
//
// Dispatch (CPU side: ggml-dx12.cpp MUL_MAT_ID switch case):
//   total_groups = ggml_nelements(node)   // NOT divided by 256
//   2D fallback when total > 65535:
//     groups_x = 65535, groups_y = ceil(total/65535)
//
// Q4_K dequant is inlined here (mirrors the MMID_Q4_K branch of
// mul_mat_id_quant.hlsli) so we don't drag in that header's [numthreads]
// main. The naive include path is still used by the other quant types
// (q8_0, q5_0/q5_1, q4_0/q4_1, q5k, q6k, iq4_nl) until they get the same
// treatment.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE 256
#endif

// Q4_K constants -- must match block_q4_K in ggml-quants.h:
//   per 256-element super-block, 144 bytes:
//   d (f16) | dmin (f16) | scales[12B] | qs[128B]
#define Q4K_QK     256
#define Q4K_BSIZE  144

// WAVE_SIZE is defined at compile time via -D WAVE_SIZE=N, so
// GROUP_SIZE / WARP_SIZE is a compile-time constant.
groupshared float shared_acc[GROUP_SIZE / WARP_SIZE];

uint q4k_read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

// Dequantize one Q4_K element at (row_off, k). Identical math to the
// MMID_Q4_K branch of mul_mat_id_quant.hlsli.
float q4k_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / Q4K_QK) * Q4K_BSIZE;
    uint elem      = k % Q4K_QK;
    uint dm_raw    = buf.Load(block_off);
    float dall     = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin_val = f16_to_f32(dm_raw >> 16);

    uint il            = elem / 64;
    uint elem_in_chunk = elem % 64;
    bool is_high       = (elem_in_chunk >= 32);
    uint elem_in_half  = elem_in_chunk % 32;
    uint is            = 2 * il;
    uint is_eff        = is_high ? (is + 1) : is;
    uint scales_off    = block_off + 4;

    uint scidx0   = (is < 4) ? is_eff : (is_eff + 4);
    uint scidx1   = (is < 4) ? is_eff : (is_eff - 4);
    uint scmask1  = (is < 4) ? 0x30u : 0xC0u;
    uint scshift1 = (is < 4) ? 0u : 2u;
    uint mbidx0   = is_eff + 4;
    uint mbidx1   = (is < 4) ? is_eff + 4 : is_eff;
    uint mbmask0  = (is < 4) ? 0x0Fu : 0xF0u;
    uint mbshift0 = (is < 4) ? 0u : 4u;
    uint mbmask1  = (is < 4) ? 0x30u : 0xC0u;
    uint mbshift1 = (is < 4) ? 0u : 2u;

    uint sc = (q4k_read_byte(buf, scales_off + scidx0) & 0x0Fu) |
              ((q4k_read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
    uint mb = ((q4k_read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
              ((q4k_read_byte(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    uint qs = q4k_read_byte(buf, block_off + 16 + il * 32 + elem_in_half);
    uint q  = is_high ? (qs >> 4) : (qs & 0x0Fu);
    return dall * (float)sc * (float)q - dmin_val * (float)mb;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    // Each workgroup owns one output element. 2D-dispatch reconstruction
    // (group_id.y * 65535 + group_id.x) matches the CPU-side fallback for
    // total_groups > 65535. No *256 multiplier here because group != tile.
    uint out_idx = group_id.y * 65535u + group_id.x;
    uint total   = ne0 * ne1 * ne2 * ne3;
    if (out_idx >= total) return;

    uint i0 = out_idx % ne0; uint rem = out_idx / ne0;
    uint i1 = rem % ne1;     rem      = rem / ne1;
    uint i2 = rem % ne2;     uint i3  = rem / ne2;

    // Expert id for this output column. All 256 threads load the same
    // value -- the driver collapses this into a single broadcast.
    uint ids_off  = op0 + i1 * op1 + i2 * op2;
    int expert_id = asint(src2.Load(ids_off));

    uint K        = ne00;
    uint i3_src0  = i3 * ne03 / ne3;
    uint src0_row = src0_offset + i0 * nb01 + (uint)expert_id * nb02 + i3_src0 * nb03;
    uint i1_src1  = i1 * ne11 / ne1;
    uint src1_row = src1_offset + i1_src1 * nb11 + i2 * nb12 + i3 * nb13;

    // Strided K accumulation. 256 threads x ceil(K/256) iters per thread.
    // K=4096 -> 16 iters, K=14336 -> 56 iters.
    float acc = 0.0f;
    for (uint k = local_id; k < K; k += GROUP_SIZE) {
        float w = q4k_dequant(src0, src0_row, k);
        float x = load_auto(src1, src1_row + k * nb10, src1_esize);
        acc += w * x;
    }

    // Stage 1: wave reduce -> num_waves partials in groupshared.
    float wave_sum = WaveActiveSum(acc);
    uint  wave_id  = local_id / WARP_SIZE;
    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Stage 2: first wave folds the partials and emits. num_waves
    // is at most 16 (wave16) and at least 4 (wave64), all <= WARP_SIZE,
    // so a single WaveActiveSum in wave 0 finishes the reduction.
    const uint num_waves = GROUP_SIZE / WARP_SIZE;
    if (wave_id == 0) {
        float v = (local_id < num_waves) ? shared_acc[local_id] : 0.0f;
        v = WaveActiveSum(v);
        if (local_id == 0) {
            uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d, v, dst_esize);
        }
    }
}
