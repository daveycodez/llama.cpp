// Mat-vec for Imagination PowerVR GPUs (Q4_K, Q6_K): one column, or NCOL
// columns at once (the tokens of a speculative batch, or the few at the end
// of a prompt), the weights, which bound the time, being read once for all.
//
// On these GPUs, reading through a read_write storage binding is several
// times slower than through a read-only one, so the weights are bound
// read-only (SRC0_RO, when they sit in a different buffer than dst) and the
// vector is read from a read-only copy (mul_mat_pvr_stage.wgsl). Each
// group of TPR threads handles one row, and neighbouring lanes load
// neighbouring 16-byte words of a block.
//
// The body of main() is written out per column by the shader library
// (ggml_webgpu_mul_mat_vec_pvr_body), in place of the marker below: these
// GPUs run indexed arrays from slow memory, and a column's arithmetic must
// not depend on NCOL. Each column of a several-column product is bit for
// bit the one-column product of that column, so that a speculative batch
// scores its tokens exactly as decoding them one at a time does. With
// subgroups, the TPR lanes of a row add up their sums with shuffles, in the
// order of the workgroup-memory tree used otherwise.
//
// LAYOUT_A: the weights were repacked at load (see ggml_webgpu_pvr_repacked_marker
// in ggml-webgpu.cpp): block b of row r is block b * rows + r, so that
// neighbouring rows' blocks are neighbours, and each part of a block sits in a
// plane of its own, every load aligned. The values loaded, and so the
// arithmetic, are those of the original layout; only which thread computes
// what may differ (a Q6_K thread may take two lanes' pieces, TPR 4).

#ifdef USE_SUBGROUPS
enable subgroups;
#endif

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
#if defined(LAYOUT_A) && !defined(Q6_A_LD)
@group(0) @binding(0) var<storage, SRC0_ACCESS> src0: array<vec2<u32>>;
#else
@group(0) @binding(0) var<storage, SRC0_ACCESS> src0: array<u32>;
#endif
#endif
@group(0) @binding(1) var<storage, read> src1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(PARAMS_BINDING) var<uniform> params: MulMatParams;

// Fused in (see ggml_webgpu_mul_mat_vec_pvr_fused), bound after dst in this order, then params:
// ADD: dst = the product + addv (at params.stride_12; ADD_IN_DST: dst itself)
#ifdef ADD
@group(0) @binding(ADD_BINDING) var<storage, read_write> addv: array<f32>;
#endif
// GLU: dst = SwiGLU of the products with src0 (the gate) and src0b (at params.stride_02)
#ifdef GLU
#ifdef Q4_K
@group(0) @binding(SRC0B_BINDING) var<storage, SRC0_ACCESS> src0b: array<vec4<u32>>;
#endif
#ifdef Q6_K
#if defined(LAYOUT_A) && !defined(Q6_A_LD)
@group(0) @binding(SRC0B_BINDING) var<storage, SRC0_ACCESS> src0b: array<vec2<u32>>;
#else
@group(0) @binding(SRC0B_BINDING) var<storage, SRC0_ACCESS> src0b: array<u32>;
#endif
#endif
#endif
// STAGE_OUT: dst also written packed into staged_out, the next product's src1
#ifdef STAGE_OUT
@group(0) @binding(STAGE_BINDING) var<storage, read_write> staged_out: array<f32>;
#endif

// RB rows per thread (consecutive): each value of the vector it loads serves RB rows
const TPR = WG_SIZE * RB / ROWS_PER_WG;
const_assert TPR * ROWS_PER_WG == WG_SIZE * RB;

#ifndef USE_SUBGROUPS
var<workgroup> partial_sums: array<f32, WG_SIZE * NCOL>;
#endif

#ifdef Q4_K
// 144-byte blocks = 9 vec4<u32>: [0] = d, dmin, scales[12]; [1..8] = qs.
// LPB lanes share a block; a lane handles qs words pg and pg + 4.
const LPB = 4u;

// the qs word p of a block, unpacked: low nibbles (values 64 jj + 16 (p & 1) + 0..15) in l, high in h
struct Q4Piece {
    l0: vec4<f32>, l1: vec4<f32>, l2: vec4<f32>, l3: vec4<f32>,
    h0: vec4<f32>, h1: vec4<f32>, h2: vec4<f32>, h3: vec4<f32>,
    // d * scale * 255 and dmin * min of the low (xy) and high (zw) sub-blocks
    sc: vec4<f32>,
};

#ifdef LAYOUT_A
// HELPERS_BEGIN (the shader library copies these for src0b)
// word p of block blk (headers, 16 bytes per block, then quants, 8 x 16
// bytes from hq): q4k_unpack, reading those addresses
fn q4k_unpack_a(blk: u32, hq: u32, p: u32) -> Q4Piece {
    let hdr = src0[blk];
    let dm = unpack2x16float(hdr.x);
    let A = hdr.y;
    let B = hdr.z;
    let C = hdr.w;
    let q = src0[hq + blk * 8u + p];
    let jj = p >> 1u;
    // scales and mins of sub-blocks 2jj (low nibbles) and 2jj + 1 (high nibbles)
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
    return Q4Piece(unpack4x8unorm(q.x & 0x0F0F0F0Fu), unpack4x8unorm(q.y & 0x0F0F0F0Fu),
                   unpack4x8unorm(q.z & 0x0F0F0F0Fu), unpack4x8unorm(q.w & 0x0F0F0F0Fu),
                   unpack4x8unorm((q.x >> 4u) & 0x0F0F0F0Fu), unpack4x8unorm((q.y >> 4u) & 0x0F0F0F0Fu),
                   unpack4x8unorm((q.z >> 4u) & 0x0F0F0F0Fu), unpack4x8unorm((q.w >> 4u) & 0x0F0F0F0Fu),
                   vec4<f32>(dm.x * 255.0 * f32(sc1), dm.y * f32(m1), dm.x * 255.0 * f32(sc2), dm.y * f32(m2)));
}

// HELPERS_END
#else
// HELPERS_BEGIN (the shader library copies these for src0b)
fn q4k_unpack(bb: u32, p: u32) -> Q4Piece {
    let hdr = src0[bb];
    let dm = unpack2x16float(hdr.x);
    let A = hdr.y;
    let B = hdr.z;
    let C = hdr.w;
    let q = src0[bb + 1u + p];
    let jj = p >> 1u;
    // scales and mins of sub-blocks 2jj (low nibbles) and 2jj + 1 (high nibbles)
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
    return Q4Piece(unpack4x8unorm(q.x & 0x0F0F0F0Fu), unpack4x8unorm(q.y & 0x0F0F0F0Fu),
                   unpack4x8unorm(q.z & 0x0F0F0F0Fu), unpack4x8unorm(q.w & 0x0F0F0F0Fu),
                   unpack4x8unorm((q.x >> 4u) & 0x0F0F0F0Fu), unpack4x8unorm((q.y >> 4u) & 0x0F0F0F0Fu),
                   unpack4x8unorm((q.z >> 4u) & 0x0F0F0F0Fu), unpack4x8unorm((q.w >> 4u) & 0x0F0F0F0Fu),
                   vec4<f32>(dm.x * 255.0 * f32(sc1), dm.y * f32(m1), dm.x * 255.0 * f32(sc2), dm.y * f32(m2)));
}

// HELPERS_END
#endif
// the piece against one column; xo is where the word's values start in the column
fn q4k_col(w: Q4Piece, xo: u32) -> f32 {
    let xl0 = src1[xo];
    let xl1 = src1[xo + 1u];
    let xl2 = src1[xo + 2u];
    let xl3 = src1[xo + 3u];
    let xh0 = src1[xo + 8u];
    let xh1 = src1[xo + 9u];
    let xh2 = src1[xo + 10u];
    let xh3 = src1[xo + 11u];
    let sl = w.l0 * xl0 + w.l1 * xl1 + w.l2 * xl2 + w.l3 * xl3;
    let sh = w.h0 * xh0 + w.h1 * xh1 + w.h2 * xh2 + w.h3 * xh3;
    let one = vec4<f32>(1.0);
    return w.sc.x * dot(sl, one) + w.sc.z * dot(sh, one) - w.sc.y * dot(xl0 + xl1 + xl2 + xl3, one) -
           w.sc.w * dot(xh0 + xh1 + xh2 + xh3, one);
}
#endif

#ifdef Q6_K
// 210-byte blocks: ql[128], qh[64], scales[16] (i8), d (f16). Blocks are only
// 2-byte aligned. 8 lanes share a block, each handling 32 values.
const LPB = 8u;

fn sbyte(w: u32, i: u32) -> f32 {
    return f32(bitcast<i32>(w << (24u - 8u * i)) >> 24u);
}

// 4 values (6-bit, biased by 32) times 4 x values
fn grp(q: u32, x: vec4<f32>) -> f32 {
    return dot(unpack4x8unorm(q), x) * 255.0 - 32.0 * (x.x + x.y + x.z + x.w);
}

// a lane's 32 values of a block (8 words of 4) and their 4 scales
struct Q6Piece {
    qa0: u32, qa1: u32, qb0: u32, qb1: u32, qc0: u32, qc1: u32, qd0: u32, qd1: u32,
    s: vec4<f32>,
};

#ifdef LAYOUT_A
#ifdef Q6_A_LD
// HELPERS_BEGIN
// Q6_A_LD (several columns, or SwiGLU): the same words read through ld(), as
// in the original layout's code, at layout A's addresses. With a runtime zero
// (z) in the address the compiler keeps ld()'s alignment select, and then
// compiles the arithmetic as for the original layout, which keeps each column
// bit for bit the one-column product (with plain loads it fuses differently).
fn ld(byte: u32) -> u32 {
    let i = byte >> 2u;
    let lo = src0[i];
    let hi = src0[i + 1u];
    return select((lo >> 16u) | (hi << 16u), lo, (byte & 2u) == 0u);
}

fn q6k_unpack_a(blk: u32, nblk: u32, p: u32) -> Q6Piece {
    let z = params.offset_src0 & 2u;  // 0: repacked weights are bound at their start
    let n = p >> 2u;
    let l0 = 8u * (p & 3u);
    let a0 = ld(z + blk * 128u + 16u * p);
    let a1 = ld(z + blk * 128u + 16u * p + 4u);
    let b0 = ld(z + blk * 128u + 16u * p + 8u);
    let b1 = ld(z + blk * 128u + 16u * p + 12u);
    let h0 = ld(z + nblk * 128u + blk * 64u + 8u * p);
    let h1 = ld(z + nblk * 128u + blk * 64u + 8u * p + 4u);
    let scw0 = ld(z + nblk * 192u + blk * 16u + 8u * n);
    let scw1 = ld(z + nblk * 192u + blk * 16u + 8u * n + 4u);
    let od = l0 >> 4u;
    let d = unpack2x16float(ld(z + nblk * 208u + blk * 2u) & 0xFFFFu).x;
    return Q6Piece((a0 & 0x0F0F0F0Fu) | ((h0 << 4u) & 0x30303030u), (a1 & 0x0F0F0F0Fu) | ((h1 << 4u) & 0x30303030u),
                   (b0 & 0x0F0F0F0Fu) | ((h0 << 2u) & 0x30303030u), (b1 & 0x0F0F0F0Fu) | ((h1 << 2u) & 0x30303030u),
                   ((a0 >> 4u) & 0x0F0F0F0Fu) | (h0 & 0x30303030u), ((a1 >> 4u) & 0x0F0F0F0Fu) | (h1 & 0x30303030u),
                   ((b0 >> 4u) & 0x0F0F0F0Fu) | ((h0 >> 2u) & 0x30303030u), ((b1 >> 4u) & 0x0F0F0F0Fu) | ((h1 >> 2u) & 0x30303030u),
                   vec4<f32>(d * sbyte(scw0, od), d * sbyte(scw0, 2u + od), d * sbyte(scw1, od), d * sbyte(scw1, 2u + od)));
}
// HELPERS_END
#else
// HELPERS_BEGIN
// piece p of block blk (of nblk): ql (piece p's words a0 a1 b0 b1, 16 bytes), qh
// (piece p's h0 h1, 8 bytes), scales (16 bytes) and d (2 bytes), in planes
fn q6k_unpack_a(blk: u32, nblk: u32, p: u32) -> Q6Piece {
    let qa = src0[blk * 16u + 2u * p];
    let qb = src0[blk * 16u + 2u * p + 1u];
    let qh = src0[nblk * 16u + blk * 8u + p];
    let sc = src0[nblk * 24u + blk * 2u + (p >> 2u)];
    let dw = src0[nblk * 26u + (blk >> 2u)];
    let dd = select(dw.x, dw.y, (blk & 2u) != 0u);
    let d = unpack2x16float(select(dd & 0xFFFFu, dd >> 16u, (blk & 1u) != 0u)).x;
    let a0 = qa.x;
    let a1 = qa.y;
    let b0 = qb.x;
    let b1 = qb.y;
    let h0 = qh.x;
    let h1 = qh.y;
    let scw0 = sc.x;
    let scw1 = sc.y;
    let od = (p & 3u) >> 1u;
    return Q6Piece((a0 & 0x0F0F0F0Fu) | ((h0 << 4u) & 0x30303030u), (a1 & 0x0F0F0F0Fu) | ((h1 << 4u) & 0x30303030u),
                   (b0 & 0x0F0F0F0Fu) | ((h0 << 2u) & 0x30303030u), (b1 & 0x0F0F0F0Fu) | ((h1 << 2u) & 0x30303030u),
                   ((a0 >> 4u) & 0x0F0F0F0Fu) | (h0 & 0x30303030u), ((a1 >> 4u) & 0x0F0F0F0Fu) | (h1 & 0x30303030u),
                   ((b0 >> 4u) & 0x0F0F0F0Fu) | ((h0 >> 2u) & 0x30303030u), ((b1 >> 4u) & 0x0F0F0F0Fu) | ((h1 >> 2u) & 0x30303030u),
                   vec4<f32>(d * sbyte(scw0, od), d * sbyte(scw0, 2u + od), d * sbyte(scw1, od), d * sbyte(scw1, 2u + od)));
}
// HELPERS_END
#endif
#else
// HELPERS_BEGIN
// the 32-bit word at a 2-byte aligned byte offset
fn ld(byte: u32) -> u32 {
    let i = byte >> 2u;
    let lo = src0[i];
    let hi = src0[i + 1u];
    return select((lo >> 16u) | (hi << 16u), lo, (byte & 2u) == 0u);
}

fn q6k_unpack(base: u32, p: u32) -> Q6Piece {
    let n = p >> 2u;
    let l0 = 8u * (p & 3u);
    let a0 = ld(base + 64u * n + l0);
    let a1 = ld(base + 64u * n + l0 + 4u);
    let b0 = ld(base + 64u * n + l0 + 32u);
    let b1 = ld(base + 64u * n + l0 + 36u);
    let h0 = ld(base + 128u + 32u * n + l0);
    let h1 = ld(base + 132u + 32u * n + l0);
    let scw0 = ld(base + 192u + 8u * n);
    let scw1 = ld(base + 196u + 8u * n);
    let od = l0 >> 4u;
    let d = unpack2x16float(ld(base + 208u) & 0xFFFFu).x;
    return Q6Piece((a0 & 0x0F0F0F0Fu) | ((h0 << 4u) & 0x30303030u), (a1 & 0x0F0F0F0Fu) | ((h1 << 4u) & 0x30303030u),
                   (b0 & 0x0F0F0F0Fu) | ((h0 << 2u) & 0x30303030u), (b1 & 0x0F0F0F0Fu) | ((h1 << 2u) & 0x30303030u),
                   ((a0 >> 4u) & 0x0F0F0F0Fu) | (h0 & 0x30303030u), ((a1 >> 4u) & 0x0F0F0F0Fu) | (h1 & 0x30303030u),
                   ((b0 >> 4u) & 0x0F0F0F0Fu) | ((h0 >> 2u) & 0x30303030u), ((b1 >> 4u) & 0x0F0F0F0Fu) | ((h1 >> 2u) & 0x30303030u),
                   vec4<f32>(d * sbyte(scw0, od), d * sbyte(scw0, 2u + od), d * sbyte(scw1, od), d * sbyte(scw1, 2u + od)));
}
// HELPERS_END
#endif

// the piece against one column; x0 is where the lane's values start in the column
fn q6k_col(q: Q6Piece, x0: u32) -> f32 {
    return q.s.x * (grp(q.qa0, src1[x0]) + grp(q.qa1, src1[x0 + 1u])) + q.s.y * (grp(q.qb0, src1[x0 + 8u]) + grp(q.qb1, src1[x0 + 9u])) +
           q.s.z * (grp(q.qc0, src1[x0 + 16u]) + grp(q.qc1, src1[x0 + 17u])) + q.s.w * (grp(q.qd0, src1[x0 + 24u]) + grp(q.qd1, src1[x0 + 25u]));
}
#endif

#ifdef LAYOUT_A
// a thread takes LPB / TPR lanes' pieces of every block of its rows (the loop
// over blocks is written as in the original layout's, which compiles faster)
const BLOCKS_PER_ITER = 1u;
const_assert TPR > 0u;
const_assert LPB % TPR == 0u;
#else
const BLOCKS_PER_ITER = TPR / LPB;
const_assert BLOCKS_PER_ITER > 0u;
const_assert BLOCKS_PER_ITER * LPB == TPR;
#endif

@compute @workgroup_size(WG_SIZE)
#ifdef USE_SUBGROUPS
// the shuffles pair subgroup lanes, so a thread's place in the workgroup comes
// from its subgroup and lane (WGSL doesn't say local_invocation_index runs
// through the lanes in order); TPR divides the subgroup size, so a row's TPR
// threads are TPR neighbouring lanes of one subgroup
fn main(@builtin(subgroup_invocation_id) sg_lane: u32,
        @builtin(subgroup_id) sg_id: u32,
        @builtin(subgroup_size) sg_size: u32,
        @builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    let li = sg_id * sg_size + sg_lane;
#else
fn main(@builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
#endif
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let t = li % TPR;
    // the thread's first row (rows past the end compute on the last row and store nothing)
    let row_first = wg_linear * ROWS_PER_WG + (li / TPR) * RB;
    let nb = params.k / 256u;
    let x0 = params.offset_src1 / 4u;
    let xs = params.k / 4u;  // between columns of the staged src1
    let lane = t % LPB;
#ifdef LAYOUT_A
    let la_m = params.stride_01;  // rows: block b of row r is block b * la_m + r
    let la_nblk = la_m * nb;
#endif

MAIN_BODY
}
