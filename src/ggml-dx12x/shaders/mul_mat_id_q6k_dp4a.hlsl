// mul_mat_id_q6k_dp4a.hlsl -- dp4a-accelerated MUL_MAT_ID Q6_K.
//
// Adapted from mul_mat_vec_q6k_dp4a.hlsl with MUL_MAT_ID indexing:
//   - group_id.z decodes into (i1, i2, i3) where i1 is the used-expert axis
//   - src2 holds the ids buffer; expert_id picks the per-expert weight slab
//   - Q8_1 src1 is laid out flat over [k_block, i1, i2, i3], so the
//     per-(i1,i2,i3) Q8_1 base picks up an extra ne11 dim
//   - no bias fusion path
//
// Q6_K reconstructs values as q = (ql_nibble | (qh_2bit << 4)) - 32.
// We keep q in [0,63] (high bit clear) so it interprets as positive
// signed int8, then subtract the bias 32*sum(q8) at the end via psum.
//
// 2 output rows per workgroup; 16 threads per super-block, one thread per
// 16-element subblock (one scale value), 4 dp4a ops per thread.
//
// Q6_K block (210 bytes, NOT 4-byte aligned):
//   ql[128] + qh[64] + scales[16](int8) + d(f16)
//
// Dispatch (CPU side: GGML_OP_MUL_MAT_ID dp4a path):
//   groups_x = (N + 1) / 2
//   groups_y = 1
//   groups_z = ne1 * ne2 * ne3

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q6K_BSIZE   210
#define Q8_1_BSIZE  36
#define NUM_ROWS    2

groupshared float shared_acc[64];

// Unaligned 4-byte load: Q6_K block stride is 210 bytes so block_off
// may not be 4-byte aligned.
uint load_u32_u(ByteAddressBuffer buf, uint byte_off) {
    uint align_off = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint w0 = buf.Load(align_off);
    if (shift == 0) return w0;
    uint w1 = buf.Load(align_off + 4);
    return (w0 >> shift) | (w1 << (32u - shift));
}

uint read_byte_q6(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

// Decode Q6_K data for one row's superblock for thread `t` (subblock index 0..15).
void decode_q6k_row(uint block_off, uint t,
                    out float d_super, out int scale_int8,
                    out uint uq0, out uint uq1, out uint uq2, out uint uq3) {
    uint d_off = block_off + 208;
    uint d_word = src0.Load(d_off & ~3u);
    d_super = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

    scale_int8 = (int)read_byte_q6(src0, block_off + 192 + t);
    if (scale_int8 >= 128) scale_int8 -= 256;

    uint ip  = t >> 3;
    uint sub = t & 7u;
    uint ql_base_in_block = 64u * ip + 16u * (sub & 3u);
    uint qh_base_in_block = 128u + 32u * ip + 16u * (sub & 1u);
    uint qh_shift = (sub & ~1u);

    bool high_nib = (sub >= 4u);

    uint ql_w0 = load_u32_u(src0, block_off + ql_base_in_block + 0);
    uint ql_w1 = load_u32_u(src0, block_off + ql_base_in_block + 4);
    uint ql_w2 = load_u32_u(src0, block_off + ql_base_in_block + 8);
    uint ql_w3 = load_u32_u(src0, block_off + ql_base_in_block + 12);

    uint qh_w0 = load_u32_u(src0, block_off + qh_base_in_block + 0);
    uint qh_w1 = load_u32_u(src0, block_off + qh_base_in_block + 4);
    uint qh_w2 = load_u32_u(src0, block_off + qh_base_in_block + 8);
    uint qh_w3 = load_u32_u(src0, block_off + qh_base_in_block + 12);

    if (high_nib) {
        ql_w0 = (ql_w0 >> 4) & 0x0F0F0F0Fu;
        ql_w1 = (ql_w1 >> 4) & 0x0F0F0F0Fu;
        ql_w2 = (ql_w2 >> 4) & 0x0F0F0F0Fu;
        ql_w3 = (ql_w3 >> 4) & 0x0F0F0F0Fu;
    } else {
        ql_w0 = ql_w0 & 0x0F0F0F0Fu;
        ql_w1 = ql_w1 & 0x0F0F0F0Fu;
        ql_w2 = ql_w2 & 0x0F0F0F0Fu;
        ql_w3 = ql_w3 & 0x0F0F0F0Fu;
    }

    qh_w0 = (qh_w0 >> qh_shift) & 0x03030303u;
    qh_w1 = (qh_w1 >> qh_shift) & 0x03030303u;
    qh_w2 = (qh_w2 >> qh_shift) & 0x03030303u;
    qh_w3 = (qh_w3 >> qh_shift) & 0x03030303u;

    uq0 = ql_w0 | (qh_w0 << 4);
    uq1 = ql_w1 | (qh_w1 << 4);
    uq2 = ql_w2 | (qh_w2 << 4);
    uq3 = ql_w3 | (qh_w3 << 4);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * NUM_ROWS;
    if (row0 >= ne0) return;

    // MUL_MAT_ID: group_id.z is flat over (i1, i2, i3) where i1 is used-expert
    uint flat_batch = group_id.z;
    uint i1 = flat_batch % ne1;
    uint tmp_b = flat_batch / ne1;
    uint i2 = tmp_b % ne2;
    uint i3 = tmp_b / ne2;

    uint ids_off  = op0 + i1 * op1 + i2 * op2;
    int expert_id = asint(src2.Load(ids_off));

    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;
    uint num_q8_per_vec = K / 32;

    uint src0_base = src0_offset + (uint)expert_id * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    uint i1_q8 = i1 * ne11 / ne1;
    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset +
        (((i3_q8 * ne12 + i2_q8) * ne11) + i1_q8) * num_q8_per_vec * Q8_1_BSIZE;

    // 16 threads cooperate per Q6_K superblock (256 elems / 16 = 16 elems/thread)
    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    // Q8_1 block: each thread covers half of one Q8_1 block (16 of 32 elems)
    uint q8_blk = itid / 2;
    uint q8_byte_off = 16u * (itid & 1u);  // 0 or 16

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // --- Load Q8_1 data (shared across both rows) ---
        uint q8_super_base = q8_vec_base + block_idx * 8u * Q8_1_BSIZE;
        uint q8_off = q8_super_base + q8_blk * Q8_1_BSIZE;

        uint ds = src1.Load(q8_off);
        float q8d = f16_to_f32(ds & 0xFFFFu);

        uint q8_qs0 = src1.Load(q8_off + 4 + q8_byte_off + 0);
        uint q8_qs1 = src1.Load(q8_off + 4 + q8_byte_off + 4);
        uint q8_qs2 = src1.Load(q8_off + 4 + q8_byte_off + 8);
        uint q8_qs3 = src1.Load(q8_off + 4 + q8_byte_off + 12);

        int q8_psum = 0;
        q8_psum = dot4add_i8packed(0x01010101u, q8_qs0, q8_psum);
        q8_psum = dot4add_i8packed(0x01010101u, q8_qs1, q8_psum);
        q8_psum = dot4add_i8packed(0x01010101u, q8_qs2, q8_psum);
        q8_psum = dot4add_i8packed(0x01010101u, q8_qs3, q8_psum);

        // --- Row 0 ---
        {
            float d_super; int scale_int8;
            uint uq0, uq1, uq2, uq3;
            decode_q6k_row(src0_row0 + block_idx * Q6K_BSIZE, itid,
                           d_super, scale_int8, uq0, uq1, uq2, uq3);

            int isx = 0;
            isx = dot4add_i8packed(uq0, q8_qs0, isx);
            isx = dot4add_i8packed(uq1, q8_qs1, isx);
            isx = dot4add_i8packed(uq2, q8_qs2, isx);
            isx = dot4add_i8packed(uq3, q8_qs3, isx);

            float scale_f = d_super * float(scale_int8) * q8d;
            acc0 = mad(scale_f, float(isx - 32 * q8_psum), acc0);
        }

        // --- Row 1 ---
        {
            float d_super; int scale_int8;
            uint uq0, uq1, uq2, uq3;
            decode_q6k_row(src0_row1 + block_idx * Q6K_BSIZE, itid,
                           d_super, scale_int8, uq0, uq1, uq2, uq3);

            int isx = 0;
            isx = dot4add_i8packed(uq0, q8_qs0, isx);
            isx = dot4add_i8packed(uq1, q8_qs1, isx);
            isx = dot4add_i8packed(uq2, q8_qs2, isx);
            isx = dot4add_i8packed(uq3, q8_qs3, isx);

            float scale_f = d_super * float(scale_int8) * q8d;
            acc1 = mad(scale_f, float(isx - 32 * q8_psum), acc1);
        }
    }

    // Two-level tree reduction (matches mul_mat_vec dp4a)
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[32 + tid] += shared_acc[32 + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float result0 = shared_acc[0];
        uint off_d0 = offset_4d(row0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[32];
            uint off_d1 = offset_4d(row0 + 1, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
