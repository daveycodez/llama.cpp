// Copies the src1 of a mat-vec or mat-mat into a buffer of its own, columns
// packed, so that mul_mat_vec_pvr.wgsl, mul_mat_pvr.wgsl and
// mul_mat_pvr_gemm.wgsl can bind it read-only (see there). OUT_F16 packs it
// as f16 pairs, for mul_mat_pvr_gemm.wgsl.

struct Params {
    offset_src: u32,
    k: u32,
    stride_src: u32,  // elements between columns
    n: u32
};

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
#ifdef OUT_F16
@group(0) @binding(1) var<storage, read_write> staged: array<u32>;
#else
@group(0) @binding(1) var<storage, read_write> staged: array<f32>;
#endif
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
#ifdef OUT_F16
    // one f16 pair per thread
    if (2u * gid.x < params.k && gid.y < params.n) {
        let s = params.offset_src + gid.y * params.stride_src + 2u * gid.x;
        staged[gid.y * (params.k / 2u) + gid.x] = pack2x16float(vec2<f32>(src[s], src[s + 1u]));
    }
#else
    if (gid.x < params.k && gid.y < params.n) {
        staged[gid.y * params.k + gid.x] = src[params.offset_src + gid.y * params.stride_src + gid.x];
    }
#endif
}
