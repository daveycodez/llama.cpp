// Mat-mat for prompt batches on Imagination PowerVR GPUs, on f16 weights
// (mul_mat_pvr_dequant.wgsl, in a texture) and f16 activations
// (mul_mat_pvr_stage.wgsl, OUT_F16, in a buffer bound read-only), both read 8
// values at a time; the sums are f32. Workgroup memory tiles run slower
// there: each thread computes 4 rows by 8 columns of dst straight from the
// bindings, the 8 x 8 threads of a workgroup interleaving their rows and
// columns by 8 (32 rows, 64 columns). The code is unrolled by hand, as loops
// over arrays run slower there. About 800 GFLOPS on a Pixel 10 Pro.

struct Params {
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    offset_glu: u32  // GLU: offset of the other operand
};

@group(0) @binding(0) var src0: texture_2d<u32>;                  // [m][k] f16, TEX_W texels per texture row
@group(0) @binding(1) var<storage, read> src1: array<vec4<u32>>;  // [n][k] f16
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
#ifdef GLU
// SwiGLU fused in: dst = silu(gate) * up, with this product the gate
// (MM_IS_GATE) or the up operand, and the other operand read from glu
@group(0) @binding(3) var<storage, read_write> glu: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// the value stored at dst index i (relative to offset_dst) for the sum a
fn out(a: f32, i: u32) -> f32 {
    let o = glu[params.offset_glu + i];
#ifdef MM_IS_GATE
    return silu(a) * o;
#else
    return silu(o) * a;
#endif
}
#else
@group(0) @binding(3) var<uniform> params: Params;
#endif

const LANES = 8u;
const ROWS = 4u * LANES;
const COLS = 8u * LANES;

// the i-th 8 weights
fn ld(i: u32) -> vec4<u32> {
    return textureLoad(src0, vec2<u32>(i % TEX_W, i / TEX_W), 0);
}

fn lo(u: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y));
}

fn hi(u: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(u.z), unpack2x16float(u.w));
}

@compute @workgroup_size(LANES * LANES)
fn main(@builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let row_tiles = (params.m + ROWS - 1u) / ROWS;
    let r = (wg_linear % row_tiles) * ROWS + li % LANES;
    let c = (wg_linear / row_tiles) * COLS + li / LANES;
    let k8 = params.k / 8u;
    // past the end: compute on the last row/column, store nothing
    let w0 = min(r, params.m - 1u) * k8;
    let w1 = min(r + LANES, params.m - 1u) * k8;
    let w2 = min(r + 2u * LANES, params.m - 1u) * k8;
    let w3 = min(r + 3u * LANES, params.m - 1u) * k8;
    let x0 = min(c, params.n - 1u) * k8;
    let x1 = min(c + LANES, params.n - 1u) * k8;
    let x2 = min(c + 2u * LANES, params.n - 1u) * k8;
    let x3 = min(c + 3u * LANES, params.n - 1u) * k8;
    let x4 = min(c + 4u * LANES, params.n - 1u) * k8;
    let x5 = min(c + 5u * LANES, params.n - 1u) * k8;
    let x6 = min(c + 6u * LANES, params.n - 1u) * k8;
    let x7 = min(c + 7u * LANES, params.n - 1u) * k8;

    var s00 = 0.0; var s01 = 0.0; var s02 = 0.0; var s03 = 0.0; var s04 = 0.0; var s05 = 0.0; var s06 = 0.0; var s07 = 0.0;
    var s10 = 0.0; var s11 = 0.0; var s12 = 0.0; var s13 = 0.0; var s14 = 0.0; var s15 = 0.0; var s16 = 0.0; var s17 = 0.0;
    var s20 = 0.0; var s21 = 0.0; var s22 = 0.0; var s23 = 0.0; var s24 = 0.0; var s25 = 0.0; var s26 = 0.0; var s27 = 0.0;
    var s30 = 0.0; var s31 = 0.0; var s32 = 0.0; var s33 = 0.0; var s34 = 0.0; var s35 = 0.0; var s36 = 0.0; var s37 = 0.0;
    for (var kk = 0u; kk < k8; kk++) {
        let u0 = ld(w0 + kk);
        let l0 = lo(u0);
        let h0 = hi(u0);
        let u1 = ld(w1 + kk);
        let l1 = lo(u1);
        let h1 = hi(u1);
        let u2 = ld(w2 + kk);
        let l2 = lo(u2);
        let h2 = hi(u2);
        let u3 = ld(w3 + kk);
        let l3 = lo(u3);
        let h3 = hi(u3);
        let v0 = src1[x0 + kk];
        let xl0 = lo(v0);
        let xh0 = hi(v0);
        s00 += dot(l0, xl0) + dot(h0, xh0);
        s10 += dot(l1, xl0) + dot(h1, xh0);
        s20 += dot(l2, xl0) + dot(h2, xh0);
        s30 += dot(l3, xl0) + dot(h3, xh0);
        let v1 = src1[x1 + kk];
        let xl1 = lo(v1);
        let xh1 = hi(v1);
        s01 += dot(l0, xl1) + dot(h0, xh1);
        s11 += dot(l1, xl1) + dot(h1, xh1);
        s21 += dot(l2, xl1) + dot(h2, xh1);
        s31 += dot(l3, xl1) + dot(h3, xh1);
        let v2 = src1[x2 + kk];
        let xl2 = lo(v2);
        let xh2 = hi(v2);
        s02 += dot(l0, xl2) + dot(h0, xh2);
        s12 += dot(l1, xl2) + dot(h1, xh2);
        s22 += dot(l2, xl2) + dot(h2, xh2);
        s32 += dot(l3, xl2) + dot(h3, xh2);
        let v3 = src1[x3 + kk];
        let xl3 = lo(v3);
        let xh3 = hi(v3);
        s03 += dot(l0, xl3) + dot(h0, xh3);
        s13 += dot(l1, xl3) + dot(h1, xh3);
        s23 += dot(l2, xl3) + dot(h2, xh3);
        s33 += dot(l3, xl3) + dot(h3, xh3);
        let v4 = src1[x4 + kk];
        let xl4 = lo(v4);
        let xh4 = hi(v4);
        s04 += dot(l0, xl4) + dot(h0, xh4);
        s14 += dot(l1, xl4) + dot(h1, xh4);
        s24 += dot(l2, xl4) + dot(h2, xh4);
        s34 += dot(l3, xl4) + dot(h3, xh4);
        let v5 = src1[x5 + kk];
        let xl5 = lo(v5);
        let xh5 = hi(v5);
        s05 += dot(l0, xl5) + dot(h0, xh5);
        s15 += dot(l1, xl5) + dot(h1, xh5);
        s25 += dot(l2, xl5) + dot(h2, xh5);
        s35 += dot(l3, xl5) + dot(h3, xh5);
        let v6 = src1[x6 + kk];
        let xl6 = lo(v6);
        let xh6 = hi(v6);
        s06 += dot(l0, xl6) + dot(h0, xh6);
        s16 += dot(l1, xl6) + dot(h1, xh6);
        s26 += dot(l2, xl6) + dot(h2, xh6);
        s36 += dot(l3, xl6) + dot(h3, xh6);
        let v7 = src1[x7 + kk];
        let xl7 = lo(v7);
        let xh7 = hi(v7);
        s07 += dot(l0, xl7) + dot(h0, xh7);
        s17 += dot(l1, xl7) + dot(h1, xh7);
        s27 += dot(l2, xl7) + dot(h2, xh7);
        s37 += dot(l3, xl7) + dot(h3, xh7);
    }

    // stores written out flat: nested branches or a store helper run slower
    if (r < params.m && c < params.n) {
        let i = (c) * params.m + r;
#ifdef GLU
        dst[params.offset_dst + i] = out(s00, i);
#else
        dst[params.offset_dst + i] = s00;
#endif
    }
    if (r + LANES < params.m && c < params.n) {
        let i = (c) * params.m + r + LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s10, i);
#else
        dst[params.offset_dst + i] = s10;
#endif
    }
    if (r + 2u * LANES < params.m && c < params.n) {
        let i = (c) * params.m + r + 2u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s20, i);
#else
        dst[params.offset_dst + i] = s20;
#endif
    }
    if (r + 3u * LANES < params.m && c < params.n) {
        let i = (c) * params.m + r + 3u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s30, i);
#else
        dst[params.offset_dst + i] = s30;
#endif
    }
    if (r < params.m && c + LANES < params.n) {
        let i = (c + LANES) * params.m + r;
#ifdef GLU
        dst[params.offset_dst + i] = out(s01, i);
#else
        dst[params.offset_dst + i] = s01;
#endif
    }
    if (r + LANES < params.m && c + LANES < params.n) {
        let i = (c + LANES) * params.m + r + LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s11, i);
#else
        dst[params.offset_dst + i] = s11;
#endif
    }
    if (r + 2u * LANES < params.m && c + LANES < params.n) {
        let i = (c + LANES) * params.m + r + 2u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s21, i);
#else
        dst[params.offset_dst + i] = s21;
#endif
    }
    if (r + 3u * LANES < params.m && c + LANES < params.n) {
        let i = (c + LANES) * params.m + r + 3u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s31, i);
#else
        dst[params.offset_dst + i] = s31;
#endif
    }
    if (r < params.m && c + 2u * LANES < params.n) {
        let i = (c + 2u * LANES) * params.m + r;
#ifdef GLU
        dst[params.offset_dst + i] = out(s02, i);
#else
        dst[params.offset_dst + i] = s02;
#endif
    }
    if (r + LANES < params.m && c + 2u * LANES < params.n) {
        let i = (c + 2u * LANES) * params.m + r + LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s12, i);
#else
        dst[params.offset_dst + i] = s12;
#endif
    }
    if (r + 2u * LANES < params.m && c + 2u * LANES < params.n) {
        let i = (c + 2u * LANES) * params.m + r + 2u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s22, i);
#else
        dst[params.offset_dst + i] = s22;
#endif
    }
    if (r + 3u * LANES < params.m && c + 2u * LANES < params.n) {
        let i = (c + 2u * LANES) * params.m + r + 3u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s32, i);
#else
        dst[params.offset_dst + i] = s32;
#endif
    }
    if (r < params.m && c + 3u * LANES < params.n) {
        let i = (c + 3u * LANES) * params.m + r;
#ifdef GLU
        dst[params.offset_dst + i] = out(s03, i);
#else
        dst[params.offset_dst + i] = s03;
#endif
    }
    if (r + LANES < params.m && c + 3u * LANES < params.n) {
        let i = (c + 3u * LANES) * params.m + r + LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s13, i);
#else
        dst[params.offset_dst + i] = s13;
#endif
    }
    if (r + 2u * LANES < params.m && c + 3u * LANES < params.n) {
        let i = (c + 3u * LANES) * params.m + r + 2u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s23, i);
#else
        dst[params.offset_dst + i] = s23;
#endif
    }
    if (r + 3u * LANES < params.m && c + 3u * LANES < params.n) {
        let i = (c + 3u * LANES) * params.m + r + 3u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s33, i);
#else
        dst[params.offset_dst + i] = s33;
#endif
    }
    if (r < params.m && c + 4u * LANES < params.n) {
        let i = (c + 4u * LANES) * params.m + r;
#ifdef GLU
        dst[params.offset_dst + i] = out(s04, i);
#else
        dst[params.offset_dst + i] = s04;
#endif
    }
    if (r + LANES < params.m && c + 4u * LANES < params.n) {
        let i = (c + 4u * LANES) * params.m + r + LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s14, i);
#else
        dst[params.offset_dst + i] = s14;
#endif
    }
    if (r + 2u * LANES < params.m && c + 4u * LANES < params.n) {
        let i = (c + 4u * LANES) * params.m + r + 2u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s24, i);
#else
        dst[params.offset_dst + i] = s24;
#endif
    }
    if (r + 3u * LANES < params.m && c + 4u * LANES < params.n) {
        let i = (c + 4u * LANES) * params.m + r + 3u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s34, i);
#else
        dst[params.offset_dst + i] = s34;
#endif
    }
    if (r < params.m && c + 5u * LANES < params.n) {
        let i = (c + 5u * LANES) * params.m + r;
#ifdef GLU
        dst[params.offset_dst + i] = out(s05, i);
#else
        dst[params.offset_dst + i] = s05;
#endif
    }
    if (r + LANES < params.m && c + 5u * LANES < params.n) {
        let i = (c + 5u * LANES) * params.m + r + LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s15, i);
#else
        dst[params.offset_dst + i] = s15;
#endif
    }
    if (r + 2u * LANES < params.m && c + 5u * LANES < params.n) {
        let i = (c + 5u * LANES) * params.m + r + 2u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s25, i);
#else
        dst[params.offset_dst + i] = s25;
#endif
    }
    if (r + 3u * LANES < params.m && c + 5u * LANES < params.n) {
        let i = (c + 5u * LANES) * params.m + r + 3u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s35, i);
#else
        dst[params.offset_dst + i] = s35;
#endif
    }
    if (r < params.m && c + 6u * LANES < params.n) {
        let i = (c + 6u * LANES) * params.m + r;
#ifdef GLU
        dst[params.offset_dst + i] = out(s06, i);
#else
        dst[params.offset_dst + i] = s06;
#endif
    }
    if (r + LANES < params.m && c + 6u * LANES < params.n) {
        let i = (c + 6u * LANES) * params.m + r + LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s16, i);
#else
        dst[params.offset_dst + i] = s16;
#endif
    }
    if (r + 2u * LANES < params.m && c + 6u * LANES < params.n) {
        let i = (c + 6u * LANES) * params.m + r + 2u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s26, i);
#else
        dst[params.offset_dst + i] = s26;
#endif
    }
    if (r + 3u * LANES < params.m && c + 6u * LANES < params.n) {
        let i = (c + 6u * LANES) * params.m + r + 3u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s36, i);
#else
        dst[params.offset_dst + i] = s36;
#endif
    }
    if (r < params.m && c + 7u * LANES < params.n) {
        let i = (c + 7u * LANES) * params.m + r;
#ifdef GLU
        dst[params.offset_dst + i] = out(s07, i);
#else
        dst[params.offset_dst + i] = s07;
#endif
    }
    if (r + LANES < params.m && c + 7u * LANES < params.n) {
        let i = (c + 7u * LANES) * params.m + r + LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s17, i);
#else
        dst[params.offset_dst + i] = s17;
#endif
    }
    if (r + 2u * LANES < params.m && c + 7u * LANES < params.n) {
        let i = (c + 7u * LANES) * params.m + r + 2u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s27, i);
#else
        dst[params.offset_dst + i] = s27;
#endif
    }
    if (r + 3u * LANES < params.m && c + 7u * LANES < params.n) {
        let i = (c + 7u * LANES) * params.m + r + 3u * LANES;
#ifdef GLU
        dst[params.offset_dst + i] = out(s37, i);
#else
        dst[params.offset_dst + i] = s37;
#endif
    }
}
