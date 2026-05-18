// mul_mat_q4_0_q8_1_tiled.hlsl - Batch MUL_MAT using dp4a + groupshared
// activation tile.
//
// Same math as mul_mat_q4_0_q8_1.hlsl (one thread = one output cell, dp4a
// for the inner dot, -8 bias correction via Q8_1 's' field), but adds a
// cooperative groupshared tile for the Q8_1 activation row.
//
// Precondition: ne0 (= N) >= GROUP_SIZE (256), so that every thread in a
// thread group lands on the SAME (i1, i2, i3) and therefore shares the
// same activation row. The CPU-side dispatch enforces this.
//
// Each thread group iterates over K in TILE_K_BLOCKS-block chunks. At the
// top of each iteration, 144 of the 256 threads cooperatively load the
// 16 Q8_1 blocks (144 dwords = 576 bytes) for the shared M row into
// groupshared memory. After a barrier, every thread reads activations
// from groupshared instead of from global, eliminating the per-thread
// redundant global reads of the activation row.
//
// Result on Gemma-4-E2B-Q4_0: prefill ~48 t/s (flat) -> expected ~75-90 t/s
// from removing the activation re-read pressure on the iGPU.

#include "ggml_common.hlsli"

#define QK4_0           32
#define Q4_0_BSIZE      18
#define Q8_1_BSIZE      36
#define GROUP_SIZE      256
#define TILE_K_BLOCKS   16
#define TILE_BYTES      (TILE_K_BLOCKS * Q8_1_BSIZE)   // 576
#define TILE_DWORDS     (TILE_BYTES / 4)               // 144
// Per-block dword stride inside the tile: 36 bytes / 4 = 9 dwords.
//   shared_act[base + 0]     = ds_word     (a_d in lo16, a_s in hi16)
//   shared_act[base + 1..4]  = qs[ 0..15]  (low-nibble half of the 32 q4 elems)
//   shared_act[base + 5..8]  = qs[16..31]  (high-nibble half)
#define BLOCK_DWORDS    (Q8_1_BSIZE / 4)               // 9

groupshared uint shared_act[TILE_DWORDS];

float read_f16_fast(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint read_u32_fast(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    bool active = (idx < total);

    // All 256 threads in a group share (i1, i2, i3) because ne0 >= GROUP_SIZE
    // (dispatch enforces this). Threads vary only in i0 (the output N axis),
    // so they read different src0 weight rows but the same activation row.
    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint K = ne00;
    uint num_blocks = K / QK4_0;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;

    // Q8_1 quantized input: flat scratch layout.
    // Row i1 of batch (i2,i3) starts at: ((i3*ne12 + i2)*ne11 + i1) * num_blocks * Q8_1_BSIZE
    uint flat_row = (i3 * ne12 + i2) * ne11 + i1;
    uint src1_row = src1_offset + flat_row * num_blocks * Q8_1_BSIZE;

    precise float acc = 0.0f;

    uint num_chunks = (num_blocks + TILE_K_BLOCKS - 1) / TILE_K_BLOCKS;
    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        uint chunk_start_block = chunk * TILE_K_BLOCKS;
        uint chunk_end_block   = min(chunk_start_block + TILE_K_BLOCKS, num_blocks);
        uint chunk_block_count = chunk_end_block - chunk_start_block;
        uint chunk_dword_count = chunk_block_count * BLOCK_DWORDS;

        // Cooperative load: threads 0..chunk_dword_count-1 each pull one
        // dword from the activation row into the shared tile. With
        // TILE_K_BLOCKS=16 the full count is 144 < GROUP_SIZE=256, so the
        // tile fills in a single pass with idle tail threads.
        uint base = src1_row + chunk_start_block * Q8_1_BSIZE;
        if (local_id < chunk_dword_count) {
            shared_act[local_id] = src1.Load(base + local_id * 4);
        }
        GroupMemoryBarrierWithGroupSync();

        if (active) {
            for (uint b = 0; b < chunk_block_count; b++) {
                uint block_idx = chunk_start_block + b;
                uint w_off = src0_row + block_idx * Q4_0_BSIZE;
                float w_d = read_f16_fast(src0, w_off);
                uint w_qs = w_off + 2;     // 16 nibble bytes

                uint block_dword_base = b * BLOCK_DWORDS;
                uint ds_word = shared_act[block_dword_base];
                float i_d = f16_to_f32(ds_word & 0xFFFFu);
                float i_s = f16_to_f32(ds_word >> 16);   // d_a * sum(a_int8)

                int isum = 0;
                [unroll]
                for (uint j = 0; j < 16; j += 4) {
                    uint qs4 = read_u32_fast(src0, w_qs + j);

                    // Low nibbles -> activations 0..15 -> dwords 1..4
                    uint w_packed_lo = qs4 & 0x0F0F0F0Fu;
                    uint i_packed_lo = shared_act[block_dword_base + 1u + (j / 4u)];
                    isum = dot4add_i8packed(w_packed_lo, i_packed_lo, isum);

                    // High nibbles -> activations 16..31 -> dwords 5..8
                    uint w_packed_hi = (qs4 >> 4) & 0x0F0F0F0Fu;
                    uint i_packed_hi = shared_act[block_dword_base + 5u + (j / 4u)];
                    isum = dot4add_i8packed(w_packed_hi, i_packed_hi, isum);
                }

                // block_sum = w_d * (a_d * isum - 8 * a_s)
                acc += w_d * (i_d * (float)isum - 8.0f * i_s);
            }
        }
        // Barrier before refilling the tile for the next chunk.
        GroupMemoryBarrierWithGroupSync();
    }

    if (active) {
        uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, acc, dst_esize);
    }
}
