// Mat-mat (prompt batches) for Imagination PowerVR GPUs (Q4_K, Q6_K).
//
// The tiled shaders stage both operands in workgroup memory, which is slow
// there, and read through read_write bindings, which PowerVR reads several
// times slower than read-only ones. Here each thread computes a 2x2 block of
// dst straight from read-only bindings: the weights (SRC0_RO, when they are
// in another buffer than dst) and a copy of src1 (mul_mat_pvr_stage.wgsl).
// The 8x8 threads of a workgroup cover 16 rows and 16 columns, rows and
// columns interleaved by 8. Larger blocks per thread run slower.
//
// LAYOUT_A: the weights were repacked at load (see mul_mat_vec_pvr.wgsl);
// rb0 and rb1 are then the rows, and stride_01 the rows of the weights.

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

#ifdef SRC0_RO
#define SRC0_ACCESS read
#else
#define SRC0_ACCESS read_write
#endif

#ifdef Q4_K
@group(0) @binding(0) var<storage, SRC0_ACCESS> src0: array<vec4<u32>>;
#endif
#ifdef Q6_K
@group(0) @binding(0) var<storage, SRC0_ACCESS> src0: array<u32>;
#endif

#ifdef LAYOUT_A
// block b of row r
fn blk_index(r: u32, b: u32) -> u32 {
    return b * params.stride_01 + r;
}
#endif
@group(0) @binding(1) var<storage, read> src1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: MulMatParams;

const LANES = 8u;          // threads along rows and along columns
const TILE = 2u * LANES;   // rows (and columns) per workgroup

#ifdef Q4_K
// 144-byte blocks = 9 vec4<u32>: [0] = d, dmin, scales[12]; [1..8] = qs.

// d * scale * 255 and dmin * min of sub-blocks 2jj (xy) and 2jj + 1 (zw)
fn q4k_scales(h: vec4<u32>, jj: u32) -> vec4<f32> {
    let dm = unpack2x16float(h.x);
    let A = h.y;
    let B = h.z;
    let C = h.w;
    let s = 2u * jj;
    let sha = 8u * (s & 3u);
    let shb = sha + 8u;
    var sc1: u32;
    var m1: u32;
    var sc2: u32;
    var m2: u32;
    if (s >= 4u) {
        sc1 = ((C >> sha) & 0xFu) | (((A >> sha) >> 2u) & 0x30u);
        m1  = (((C >> sha) >> 4u) & 0xFu) | (((B >> sha) >> 2u) & 0x30u);
        sc2 = ((C >> shb) & 0xFu) | (((A >> shb) >> 2u) & 0x30u);
        m2  = (((C >> shb) >> 4u) & 0xFu) | (((B >> shb) >> 2u) & 0x30u);
    } else {
        sc1 = (A >> sha) & 63u;
        m1  = (B >> sha) & 63u;
        sc2 = (A >> shb) & 63u;
        m2  = (B >> shb) & 63u;
    }
    // unpack4x8unorm divides by 255
    return vec4<f32>(dm.x * f32(sc1) * 255.0, dm.y * f32(m1), dm.x * f32(sc2) * 255.0, dm.y * f32(m2));
}

// one qs word of rows 0 and 1 (4 low-nibble and 4 high-nibble values each)
// against 4 + 4 values of columns 0 and 1: deltas for (r0c0, r0c1, r1c0, r1c1)
fn q4k_step(qa: u32, qb: u32, s0: vec4<f32>, s1: vec4<f32>, xl0: vec4<f32>, xh0: vec4<f32>, xl1: vec4<f32>,
            xh1: vec4<f32>) -> vec4<f32> {
    let wl0 = unpack4x8unorm(qa & 0x0F0F0F0Fu) * s0.x - s0.y;
    let wh0 = unpack4x8unorm((qa >> 4u) & 0x0F0F0F0Fu) * s0.z - s0.w;
    let wl1 = unpack4x8unorm(qb & 0x0F0F0F0Fu) * s1.x - s1.y;
    let wh1 = unpack4x8unorm((qb >> 4u) & 0x0F0F0F0Fu) * s1.z - s1.w;
    return vec4<f32>(dot(wl0, xl0) + dot(wh0, xh0), dot(wl0, xl1) + dot(wh0, xh1), dot(wl1, xl0) + dot(wh1, xh0),
                     dot(wl1, xl1) + dot(wh1, xh1));
}

fn row_block_dot(rb0: u32, rb1: u32, b: u32, xb0: u32, xb1: u32) -> vec4<f32> {
#ifdef LAYOUT_A
    let hq = params.stride_01 * (params.k / 256u);
    let h0 = src0[blk_index(rb0, b)];
    let h1 = src0[blk_index(rb1, b)];
    let qb0 = hq + blk_index(rb0, b) * 8u;
    let qb1 = hq + blk_index(rb1, b) * 8u;
#else
    let bb0 = (rb0 + b) * 9u;
    let bb1 = (rb1 + b) * 9u;
    let h0 = src0[bb0];
    let h1 = src0[bb1];
#endif
    var acc = vec4<f32>(0.0);
    for (var p = 0u; p < 8u; p++) {
        let jj = p >> 1u;
#ifdef LAYOUT_A
        let q0 = src0[qb0 + p];
        let q1 = src0[qb1 + p];
#else
        let q0 = src0[bb0 + 1u + p];
        let q1 = src0[bb1 + 1u + p];
#endif
        let s0 = q4k_scales(h0, jj);
        let s1 = q4k_scales(h1, jj);
        // qs bytes 16p..16p+15: values 64jj + l (low nibbles) and 64jj + 32 + l (high)
        let xo0 = xb0 + b * 64u + jj * 16u + (p & 1u) * 4u;
        let xo1 = xb1 + b * 64u + jj * 16u + (p & 1u) * 4u;
        acc += q4k_step(q0.x, q1.x, s0, s1, src1[xo0], src1[xo0 + 8u], src1[xo1], src1[xo1 + 8u]);
        acc += q4k_step(q0.y, q1.y, s0, s1, src1[xo0 + 1u], src1[xo0 + 9u], src1[xo1 + 1u], src1[xo1 + 9u]);
        acc += q4k_step(q0.z, q1.z, s0, s1, src1[xo0 + 2u], src1[xo0 + 10u], src1[xo1 + 2u], src1[xo1 + 10u]);
        acc += q4k_step(q0.w, q1.w, s0, s1, src1[xo0 + 3u], src1[xo0 + 11u], src1[xo1 + 3u], src1[xo1 + 11u]);
    }
    return acc;
}
#endif

#ifdef Q6_K
// 210-byte blocks: ql[128], qh[64], scales[16] (i8), d (f16), only 2-byte aligned.

// the 32-bit word at a 2-byte aligned byte offset
fn ld(byte: u32) -> u32 {
    let i = byte >> 2u;
    let lo = src0[i];
    let hi = src0[i + 1u];
    return select((lo >> 16u) | (hi << 16u), lo, (byte & 2u) == 0u);
}

fn sbyte(w: u32, i: u32) -> f32 {
    return f32(bitcast<i32>(w << (24u - 8u * i)) >> 24u);
}

// 4 values (6 bits, biased by 32) of rows 0 and 1 against 4 values of
// columns 0 and 1, with scales s0 and s1 (times d): (r0c0, r0c1, r1c0, r1c1)
fn q6k_step(qa: u32, qb: u32, s0: f32, s1: f32, x0: vec4<f32>, x1: vec4<f32>) -> vec4<f32> {
    let w0 = (unpack4x8unorm(qa) * 255.0 - 32.0) * s0;
    let w1 = (unpack4x8unorm(qb) * 255.0 - 32.0) * s1;
    return vec4<f32>(dot(w0, x0), dot(w0, x1), dot(w1, x0), dot(w1, x1));
}

#ifdef LAYOUT_A
// planes of 32-bit words (see mul_mat_pvr_dequant.wgsl): the ql word at byte
// 64 n + 32 h + 4 w of a block, its qh word at 128 + 32 n + 4 w, scale word i, d
fn q6a_ql(blk: u32, n: u32, h: u32, w: u32) -> u32 {
    return src0[blk * 32u + (4u * n + (w >> 1u)) * 4u + 2u * h + (w & 1u)];
}
fn q6a_qh(blk: u32, n: u32, w: u32) -> u32 {
    return src0[params.stride_01 * (params.k / 256u) * 32u + blk * 16u + (4u * n + (w >> 1u)) * 2u + (w & 1u)];
}
fn q6a_sc(blk: u32, i: u32) -> u32 {
    return src0[params.stride_01 * (params.k / 256u) * 48u + blk * 4u + i];
}
fn q6a_d(blk: u32) -> f32 {
    let dw = src0[params.stride_01 * (params.k / 256u) * 52u + blk / 2u];
    return unpack2x16float(select(dw & 0xFFFFu, dw >> 16u, (blk & 1u) == 1u)).x;
}
#endif

fn row_block_dot(rb0: u32, rb1: u32, b: u32, xb0: u32, xb1: u32) -> vec4<f32> {
#ifdef LAYOUT_A
    let k0 = blk_index(rb0, b);
    let k1 = blk_index(rb1, b);
    let d0 = q6a_d(k0);
    let d1 = q6a_d(k1);
#else
    let base0 = (rb0 + b) * 210u;
    let base1 = (rb1 + b) * 210u;
    let d0 = unpack2x16float(ld(base0 + 208u) & 0xFFFFu).x;
    let d1 = unpack2x16float(ld(base1 + 208u) & 0xFFFFu).x;
#endif
    var acc = vec4<f32>(0.0);
    for (var n = 0u; n < 2u; n++) {
        // scales 8n .. 8n + 7
#ifdef LAYOUT_A
        let sc00 = q6a_sc(k0, 2u * n);
        let sc01 = q6a_sc(k0, 2u * n + 1u);
        let sc10 = q6a_sc(k1, 2u * n);
        let sc11 = q6a_sc(k1, 2u * n + 1u);
#else
        let sc00 = ld(base0 + 192u + 8u * n);
        let sc01 = ld(base0 + 196u + 8u * n);
        let sc10 = ld(base1 + 192u + 8u * n);
        let sc11 = ld(base1 + 196u + 8u * n);
#endif
        for (var w = 0u; w < 8u; w++) {
            let l0 = 4u * w;
            let od = w >> 2u;  // l0 / 16
#ifdef LAYOUT_A
            let a0 = q6a_ql(k0, n, 0u, w);
            let b0 = q6a_ql(k0, n, 1u, w);
            let h0 = q6a_qh(k0, n, w);
            let a1 = q6a_ql(k1, n, 0u, w);
            let b1 = q6a_ql(k1, n, 1u, w);
            let h1 = q6a_qh(k1, n, w);
#else
            let a0 = ld(base0 + 64u * n + l0);
            let b0 = ld(base0 + 64u * n + l0 + 32u);
            let h0 = ld(base0 + 128u + 32u * n + l0);
            let a1 = ld(base1 + 64u * n + l0);
            let b1 = ld(base1 + 64u * n + l0 + 32u);
            let h1 = ld(base1 + 128u + 32u * n + l0);
#endif
            let xo0 = xb0 + b * 64u + 32u * n + w;
            let xo1 = xb1 + b * 64u + 32u * n + w;
            // y[l], y[l + 32], y[l + 64], y[l + 96] of this half
            acc += q6k_step((a0 & 0x0F0F0F0Fu) | ((h0 << 4u) & 0x30303030u),
                            (a1 & 0x0F0F0F0Fu) | ((h1 << 4u) & 0x30303030u), d0 * sbyte(sc00, od),
                            d1 * sbyte(sc10, od), src1[xo0], src1[xo1]);
            acc += q6k_step((b0 & 0x0F0F0F0Fu) | ((h0 << 2u) & 0x30303030u),
                            (b1 & 0x0F0F0F0Fu) | ((h1 << 2u) & 0x30303030u), d0 * sbyte(sc00, 2u + od),
                            d1 * sbyte(sc10, 2u + od), src1[xo0 + 8u], src1[xo1 + 8u]);
            acc += q6k_step(((a0 >> 4u) & 0x0F0F0F0Fu) | (h0 & 0x30303030u),
                            ((a1 >> 4u) & 0x0F0F0F0Fu) | (h1 & 0x30303030u), d0 * sbyte(sc01, od),
                            d1 * sbyte(sc11, od), src1[xo0 + 16u], src1[xo1 + 16u]);
            acc += q6k_step(((b0 >> 4u) & 0x0F0F0F0Fu) | ((h0 >> 2u) & 0x30303030u),
                            ((b1 >> 4u) & 0x0F0F0F0Fu) | ((h1 >> 2u) & 0x30303030u), d0 * sbyte(sc01, 2u + od),
                            d1 * sbyte(sc11, 2u + od), src1[xo0 + 24u], src1[xo1 + 24u]);
        }
    }
    return acc;
}
#endif

@compute @workgroup_size(LANES * LANES)
fn main(@builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let row_tiles = (params.m + TILE - 1u) / TILE;
    let row0 = (wg_linear % row_tiles) * TILE + li % LANES;
    let col0 = (wg_linear / row_tiles) * TILE + li / LANES;
    let row1 = row0 + LANES;
    let col1 = col0 + LANES;
    // past the end: compute on the last row/column, store nothing
#ifdef LAYOUT_A
    let rb0 = min(row0, params.m - 1u);
    let rb1 = min(row1, params.m - 1u);
#else
    let rb0 = params.offset_src0 + min(row0, params.m - 1u) * params.stride_01;
    let rb1 = params.offset_src0 + min(row1, params.m - 1u) * params.stride_01;
#endif
    let xb0 = (params.offset_src1 + min(col0, params.n - 1u) * params.stride_11) / 4u;
    let xb1 = (params.offset_src1 + min(col1, params.n - 1u) * params.stride_11) / 4u;

    var acc = vec4<f32>(0.0);
    let nb = params.k / 256u;
    for (var b = 0u; b < nb; b++) {
        acc += row_block_dot(rb0, rb1, b, xb0, xb1);
    }

    if (row0 < params.m && col0 < params.n) {
        dst[params.offset_dst + col0 * params.m + row0] = acc.x;
    }
    if (row0 < params.m && col1 < params.n) {
        dst[params.offset_dst + col1 * params.m + row0] = acc.y;
    }
    if (row1 < params.m && col0 < params.n) {
        dst[params.offset_dst + col0 * params.m + row1] = acc.z;
    }
    if (row1 < params.m && col1 < params.n) {
        dst[params.offset_dst + col1 * params.m + row1] = acc.w;
    }
}
