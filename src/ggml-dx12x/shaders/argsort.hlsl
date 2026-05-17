// argsort.hlsl -- per-row argsort.
//
// dst[i] = the index permutation such that gathering src0 by dst yields the
// row sorted by op_params[0] (0 = ASC, 1 = DESC).
//
// src0: GGML_TYPE_F32, shape [ne00, ne01, ne02, ne03], dim-0 contiguous
//       (CPU reference asserts nb00 == sizeof(float)).
// dst:  GGML_TYPE_I32, same shape as src0.
//
// One workgroup per row. ncols (== ne00) must be <= BLOCK_SIZE; this is
// gated by dx12_supports_op() in ggml-dx12.cpp. The row is padded
// internally to the next power of two, with index-based OOB sentinels
// driving the bitonic comparator.

#include "ggml_common.hlsli"

#define BLOCK_SIZE 256u
#define ASC        0u

// .x = original column index, .y = bit-cast f32 value. uint2 keeps the
// per-thread shared cell at 8 bytes, so the whole table is 256 * 8 = 2 KiB.
groupshared uint2 dst_row[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint col = gtid.x;
    const uint row = gid.x;

    const uint total_rows = ne01 * ne02 * ne03;
    if (row >= total_rows) return;

    const uint ncols = ne00;
    const uint order = op0;

    // Decompose flat row into (i1, i2, i3) for stride-correct src0 addressing.
    const uint i3   = row / (ne01 * ne02);
    const uint rem3 = row % (ne01 * ne02);
    const uint i2   = rem3 / ne01;
    const uint i1   = rem3 % ne01;

    // Round ncols up to a power of two in [1, BLOCK_SIZE].
    uint ncols_padded = 1u;
    while (ncols_padded < ncols) ncols_padded <<= 1u;

    // Initial load. Real cells get (col, value); padded cells get a sentinel
    // index (col, which is >= ncols) and a don't-care value -- the OOB-aware
    // comparator never reads padded values.
    if (col < ncols) {
        const uint off = offset_4d(col, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        dst_row[col] = uint2(col, src0.Load(off));
    } else if (col < ncols_padded) {
        dst_row[col] = uint2(col, 0u);
    }
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort, ascending. Each stage: pair (col, ixj=col^j); only the
    // lower-index thread of the pair performs the compare-exchange. Within
    // a pair, (col & k) == 0 selects the ascending half (smaller at lower
    // index), otherwise the descending half (larger at lower index).
    [loop] for (uint k = 2u; k <= ncols_padded; k <<= 1u) {
        [loop] for (uint j = k >> 1; j > 0u; j >>= 1u) {
            if (col < ncols_padded) {
                const uint ixj = col ^ j;
                if (ixj > col) {
                    const uint2 a = dst_row[col];
                    const uint2 b = dst_row[ixj];
                    const bool a_oob    = a.x >= ncols;
                    const bool b_oob    = b.x >= ncols;
                    const bool asc_half = (col & k) == 0u;
                    bool swap;
                    if (a_oob && b_oob) {
                        // Both padded -- nothing meaningful to order.
                        swap = false;
                    } else if (a_oob) {
                        // Padded sorts after real. Ascending half wants
                        // smaller at col, so swap padded out of col.
                        swap = asc_half;
                    } else if (b_oob) {
                        // Real sorts before padded. Ascending half wants
                        // smaller at col, so leave real at col.
                        swap = !asc_half;
                    } else {
                        const float fa = asfloat(a.y);
                        const float fb = asfloat(b.y);
                        swap = asc_half ? (fa > fb) : (fa < fb);
                    }
                    if (swap) {
                        dst_row[col] = b;
                        dst_row[ixj] = a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Writeback. dst_row[0..ncols) holds the ASC permutation; reverse for DESC.
    if (col < ncols) {
        const uint dst_col = (order == ASC) ? col : (ncols - 1u - col);
        const uint dst_off = offset_4d(dst_col, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        dst.Store(dst_off, dst_row[col].x);
    }
}
