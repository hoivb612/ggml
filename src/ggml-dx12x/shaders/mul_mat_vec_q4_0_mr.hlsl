// mul_mat_vec_q4_0_mr.hlsl - Multi-row Q4_0 matvec (M=1, 2 rows/group)
//
// Q4_0 block: d(f16) + qs[16] = 18 bytes per 32 elements
// 32 threads (1 wave), shares activation loads across 2 rows.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK4_0 32
#define Q4_0_BSIZE 18

groupshared float shared_acc[64];

uint read_u32_fast(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

// Per-element decode.  elem in [0,31]. Range [-8, 7].
int dequant_q4_0_qs(ByteAddressBuffer buf, uint block_off, uint elem) {
    uint qs_idx = (elem < 16) ? elem : (elem - 16);
    uint qs_word = read_u32_fast(buf, block_off + 2 + (qs_idx & ~3u));
    uint qs_byte = (qs_word >> ((qs_idx & 3u) * 8u)) & 0xFFu;

    if (elem < 16) {
        return (int)(qs_byte & 0x0Fu) - 8;
    } else {
        return (int)(qs_byte >> 4) - 8;
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK4_0;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    uint elem = local_id;

    for (uint block = 0; block < num_blocks; block++) {
        // Shared: load activation value once
        uint k = block * QK4_0 + elem;
        float x = asfloat(src1.Load(src1_base + k * 4));

        uint blk_off0 = src0_row0 + block * Q4_0_BSIZE;
        uint blk_off1 = src0_row1 + block * Q4_0_BSIZE;

        // Per-block scalars (d) are uniform across the wave - broadcast from
        // lane 0 instead of each of 32 lanes redundantly hitting memory.
        float d0 = WaveReadLaneFirst(read_f16_v(src0, blk_off0));
        float d1 = WaveReadLaneFirst(read_f16_v(src0, blk_off1));

        int val0 = dequant_q4_0_qs(src0, blk_off0, elem);
        acc0 += d0 * float(val0) * x;

        int val1 = dequant_q4_0_qs(src0, blk_off1, elem);
        acc1 += d1 * float(val1) * x;
    }

    // Wave reduction
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = local_id / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction across waves (correct for any wave size)
    for (uint s = num_waves / 2; s > 0; s /= 2) {
        if (local_id < s) {
            shared_acc[local_id] += shared_acc[local_id + s];
            shared_acc[32 + local_id] += shared_acc[32 + local_id + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[32];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
