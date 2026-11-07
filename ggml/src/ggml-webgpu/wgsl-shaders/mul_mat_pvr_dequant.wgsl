// Dequantizes Q4_K or Q6_K weights to f16, rows packed, for
// mul_mat_pvr_gemm.wgsl (prompt batches on Imagination PowerVR GPUs). Each
// thread writes 8 consecutive values; 32 threads cover a 256-value block.
// They go to an rgba32uint texture, 8 values per texel and TEX_W texels per
// texture row: PowerVR writes textures about 3 times faster than buffers,
// and the product reads them faster too.
//
// LAYOUT_A: the weights were repacked at load (see mul_mat_vec_pvr.wgsl):
// block b of row r is block b * rows + r, in planes of headers and quants.

struct Params {
    offset_src0: u32,  // blocks
    stride_01: u32,    // blocks between rows (LAYOUT_A: rows)
    m: u32,            // rows
    k: u32             // values per row
};

#ifdef Q4_K
@group(0) @binding(0) var<storage, read> src0: array<vec4<u32>>;
#endif
#ifdef Q6_K
@group(0) @binding(0) var<storage, read> src0: array<u32>;
#endif
@group(0) @binding(1) var dst: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(2) var<uniform> params: Params;

fn pack8(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
    return vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw));
}

#ifdef Q4_K
// 144-byte blocks = 9 vec4<u32>: [0] = d, dmin, scales[12]; [1..8] = qs.
// sub = 8 jj + t: values 64 jj + 32 (t / 4) + 8 (t % 4) .. + 7
fn dequant8(blk: u32, sub: u32) -> vec4<u32> {
#ifdef LAYOUT_A
    let h = src0[blk];
    let qs = params.stride_01 * (params.k / 256u) + blk * 8u;
#else
    let bb = blk * 9u;
    let h = src0[bb];
#endif
    let dm = unpack2x16float(h.x);
    let jj = sub / 8u;
    let t = sub % 8u;
    let s = 2u * jj + t / 4u;  // sub-block
    let sh = 8u * (s & 3u);
    var sc: u32;
    var mn: u32;
    if (s >= 4u) {
        sc = ((h.w >> sh) & 0xFu) | (((h.y >> sh) >> 2u) & 0x30u);
        mn = (((h.w >> sh) >> 4u) & 0xFu) | (((h.z >> sh) >> 2u) & 0x30u);
    } else {
        sc = (h.y >> sh) & 63u;
        mn = (h.z >> sh) & 63u;
    }
    // qs bytes 32 jj + 8 (t % 4) .. + 7
#ifdef LAYOUT_A
    let qv = src0[qs + 2u * jj + (t % 4u) / 2u];
#else
    let qv = src0[bb + 1u + 2u * jj + (t % 4u) / 2u];
#endif
    var q0 = select(qv.x, qv.z, (t & 1u) == 1u);
    var q1 = select(qv.y, qv.w, (t & 1u) == 1u);
    if (t >= 4u) {
        q0 = q0 >> 4u;
        q1 = q1 >> 4u;
    }
    // unpack4x8unorm divides by 255
    let a = dm.x * f32(sc) * 255.0;
    let b = dm.y * f32(mn);
    return pack8(unpack4x8unorm(q0 & 0x0F0F0F0Fu) * a - b, unpack4x8unorm(q1 & 0x0F0F0F0Fu) * a - b);
}
#endif

#ifdef Q6_K
// 210-byte blocks: ql[128], qh[64], scales[16] (i8), d (f16), only 2-byte aligned.

#ifdef LAYOUT_A
// planes of 32-bit words: ql, piece p's words a0 a1 b0 b1 (p = 4 n + l0 / 8);
// qh, piece p's h0 h1; scales; d (two per word)
fn dequant8(blk: u32, sub: u32) -> vec4<u32> {
    let nblk = params.stride_01 * (params.k / 256u);
    let n = sub / 16u;
    let g = (sub / 4u) % 4u;
    let l0 = 8u * (sub % 4u);
    let p = 4u * n + l0 / 8u;
    let qlo = blk * 32u + p * 4u + 2u * (g & 1u);
    var ql0 = src0[qlo];
    var ql1 = src0[qlo + 1u];
    if (g >= 2u) {
        ql0 = ql0 >> 4u;
        ql1 = ql1 >> 4u;
    }
    let qho = nblk * 32u + blk * 16u + p * 2u;
    let qsh = 2u * g;
    let q0 = (ql0 & 0x0F0F0F0Fu) | (((src0[qho] >> qsh) & 0x03030303u) << 4u);
    let q1 = (ql1 & 0x0F0F0F0Fu) | (((src0[qho + 1u] >> qsh) & 0x03030303u) << 4u);
    let si = 8u * n + l0 / 16u + 2u * g;
    let scw = src0[nblk * 48u + blk * 4u + si / 4u];
    let sc = f32(bitcast<i32>(scw << (24u - 8u * (si & 3u))) >> 24u);
    let dw = src0[nblk * 52u + blk / 2u];
    let d = unpack2x16float(select(dw & 0xFFFFu, dw >> 16u, (blk & 1u) == 1u)).x * sc;
    return pack8((unpack4x8unorm(q0) * 255.0 - 32.0) * d, (unpack4x8unorm(q1) * 255.0 - 32.0) * d);
}
#else
// the 32-bit word at a 2-byte aligned byte offset
fn ld(byte: u32) -> u32 {
    let i = byte >> 2u;
    let lo = src0[i];
    let hi = src0[i + 1u];
    return select((lo >> 16u) | (hi << 16u), lo, (byte & 2u) == 0u);
}

// sub = 16 n + 4 g + l0 / 8: values 128 n + 32 g + l0 .. + 7
fn dequant8(blk: u32, sub: u32) -> vec4<u32> {
    let base = blk * 210u;
    let n = sub / 16u;
    let g = (sub / 4u) % 4u;
    let l0 = 8u * (sub % 4u);
    let qlo = base + 64u * n + 32u * (g & 1u) + l0;
    var ql0 = ld(qlo);
    var ql1 = ld(qlo + 4u);
    if (g >= 2u) {
        ql0 = ql0 >> 4u;
        ql1 = ql1 >> 4u;
    }
    let qho = base + 128u + 32u * n + l0;
    let qsh = 2u * g;
    let q0 = (ql0 & 0x0F0F0F0Fu) | (((ld(qho) >> qsh) & 0x03030303u) << 4u);
    let q1 = (ql1 & 0x0F0F0F0Fu) | (((ld(qho + 4u) >> qsh) & 0x03030303u) << 4u);
    let si = 8u * n + l0 / 16u + 2u * g;
    let scw = ld(base + 192u + (si & ~1u));
    let sc = f32(bitcast<i32>(scw << (24u - 8u * (si & 1u))) >> 24u);
    let d = unpack2x16float(ld(base + 208u) & 0xFFFFu).x * sc;
    return pack8((unpack4x8unorm(q0) * 255.0 - 32.0) * d, (unpack4x8unorm(q1) * 255.0 - 32.0) * d);
}
#endif
#endif

@compute @workgroup_size(WG_SIZE)
#ifdef LAYOUT_A
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let per_row = params.k / 8u;
    // a workgroup takes block column b of WG_SIZE / 32 neighbouring rows, which
    // are neighbours in layout A too; the workgroups of a block column follow
    // each other
    const RPW = WG_SIZE / 32u;
    const_assert RPW * 32u == WG_SIZE;
    let w = gid.y * num_wg.x + gid.x / WG_SIZE;
    let groups = (params.m + RPW - 1u) / RPW;
    let b = w / groups;
    let row = (w % groups) * RPW + li / 32u;
    if (b >= params.k / 256u || row >= params.m) {
        return;
    }
    let c = b * 32u + li % 32u;
    let i = row * per_row + c;  // index of the 8 values written
    let blk = b * params.stride_01 + row;
#else
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
    let i = gid.y * num_wg.x * WG_SIZE + gid.x;  // index of the 8 values written
    let per_row = params.k / 8u;
    if (i >= params.m * per_row) {
        return;
    }
    let row = i / per_row;
    let c = i % per_row;
    textureStore(dst, vec2<u32>(i % TEX_W, i / TEX_W), dequant8(params.offset_src0 + row * params.stride_01 + c / 32u, c % 32u));
#endif
#ifdef LAYOUT_A
    textureStore(dst, vec2<u32>(i % TEX_W, i / TEX_W), dequant8(blk, c % 32u));
#endif
}
