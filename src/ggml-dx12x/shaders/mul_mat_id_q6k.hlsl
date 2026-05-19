// mul_mat_id_q6k.hlsl -- K-direction reduction MUL_MAT_ID for Q6_K.
//
// Same algorithm as mul_mat_id_q4k.hlsl (1 workgroup == 1 output element,
// 256 threads stride across K, wave reduce + groupshared fold). Only the
// dequant body and BSIZE differ.
//
// This is the down-projection path for Mixtral 8x7B Q4_K_M (every other
// down_proj layer ships as Q6_K). The old naive shader spent ~88% of the
// decode graph on grp=32 dispatches across 52 CUs (XSX), one block per
// row, K=14336 serial loads per thread. This swaps that out for grp=8192
// with 56 iters/thread.
//
// Dispatch (CPU side: ggml-dx12.cpp MUL_MAT_ID switch case):
//   total_groups = ggml_nelements(node)   // NOT divided by 256
//   2D fallback when total > 65535:
//     groups_x = 65535, groups_y = ceil(total/65535)

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE 256
#endif

// Q6_K constants -- must match block_q6_K in ggml-quants.h:
//   per 256-element super-block, 210 bytes:
//   ql[128B] | qh[64B] | scales[16B int8] | d (f16, 2B)
#define Q6K_QK     256
#define Q6K_BSIZE  210

// WAVE_SIZE is defined at compile time via -D WAVE_SIZE=N, so
// GROUP_SIZE / WARP_SIZE is a compile-time constant.
groupshared float shared_acc[GROUP_SIZE / WARP_SIZE];

uint q6k_read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int q6k_read_sbyte(ByteAddressBuffer buf, uint byte_off) {
    uint b = q6k_read_byte(buf, byte_off);
    return (b < 128u) ? (int)b : (int)b - 256;
}

// Dequantize one Q6_K element at (row_off, k). Identical math to the
// MMID_Q6_K branch of mul_mat_id_quant.hlsli.
float q6k_dequant(ByteAddressBuffer buf, uint row_off, uint k) {
    uint block_off = row_off + (k / Q6K_QK) * Q6K_BSIZE;
    uint elem      = k % Q6K_QK;
    uint d_off     = block_off + 208;
    uint d_word    = buf.Load(d_off & ~3u);
    float d        = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

    uint ip   = elem / 128;
    uint il   = elem % 128;
    int scale = q6k_read_sbyte(buf, block_off + 192 + 8 * ip + il / 16);
    uint ql   = q6k_read_byte (buf, block_off       + 64 * ip + (il % 64));
    uint qh   = q6k_read_byte (buf, block_off + 128 + 32 * ip + (il % 32));

    int q;
    if (il < 32) {
        q = (int)((ql & 0x0Fu) | (((qh >> 0) & 3u) << 4)) - 32;
    } else if (il < 64) {
        q = (int)((ql & 0x0Fu) | (((qh >> 2) & 3u) << 4)) - 32;
    } else if (il < 96) {
        q = (int)((ql >> 4)    | (((qh >> 4) & 3u) << 4)) - 32;
    } else {
        q = (int)((ql >> 4)    | (((qh >> 6) & 3u) << 4)) - 32;
    }
    return d * (float)scale * (float)q;
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
    // K=14336 (Mixtral down_proj) -> 56 iters.
    float acc = 0.0f;
    for (uint k = local_id; k < K; k += GROUP_SIZE) {
        float w = q6k_dequant(src0, src0_row, k);
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

    // Stage 2: first wave folds the partials and emits.
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
