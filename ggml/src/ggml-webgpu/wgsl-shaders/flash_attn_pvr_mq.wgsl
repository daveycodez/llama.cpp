// Flash attention for prompt batches (several query tokens) on Imagination
// PowerVR GPUs (f32 Q, f16 K/V, f16 mask). The generic tiled shaders read
// every operand through read_write bindings, which PowerVR reads several
// times slower, and stage tiles in workgroup memory.
//
// A workgroup takes WG_SIZE / SPLIT query rows of one query head. SPLIT
// neighbouring threads share a row, each keeping every SPLIT-th 8 values of
// its Q and of its output in registers, and add up their parts of each score
// with subgroup shuffles. The threads walk the KV positions in chunks of 4,
// all of them reading the same K and V rows (8 values per load) from
// read-only bindings when K and V live in another buffer than dst (the KV
// cache). Chunks the mask hides from a whole subgroup are skipped.

enable f16;
#if SPLIT > 1
enable subgroups;
// the KV loop skips chunks only when subgroupAll() says the whole subgroup
// may, so the shuffles run in subgroup-uniform control flow
diagnostic(off, subgroup_uniformity);
#endif

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_mask: u32,
    offset_dst: u32,
    n_kv: u32,
    n_q: u32,
    n_heads: u32,
    stride_q1: u32,     // Q elements between query rows
    stride_q2: u32,     // Q elements between heads
    stride_k1: u32,     // K elements between KV positions
    stride_k2: u32,     // K elements between KV heads
    stride_v1: u32,
    stride_v2: u32,
    stride_mask1: u32,  // mask elements between query rows
    head_ratio: u32,    // query heads per KV head
    scale: f32
};

#ifdef KV_RO
#define KV_ACCESS read
#else
#define KV_ACCESS read_write
#endif

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, KV_ACCESS> k: array<vec4<u32>>;  // 8 f16 each
@group(0) @binding(2) var<storage, KV_ACCESS> v: array<vec4<u32>>;
#ifdef HAS_MASK
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
#define DST_BINDING 4
#else
#define DST_BINDING 3
#endif
@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<f32>;
@group(0) @binding(DST_BINDING + 1) var<uniform> params: Params;

const ROWS = WG_SIZE / SPLIT;
const NC = HEAD_DIM / 8u / SPLIT;  // 8-value chunks per thread
const NEG_BIG = -3.0e38;

fn lo(u: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y));
}

fn hi(u: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(u.z), unpack2x16float(u.w));
}

@compute @workgroup_size(WG_SIZE)
#if SPLIT > 1
// the shuffles pair subgroup lanes, so a thread's place in the workgroup comes
// from its subgroup and lane (WGSL doesn't say local_invocation_index runs
// through the lanes in order); SPLIT divides the subgroup size, so a row's
// threads are SPLIT neighbouring lanes of one subgroup
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(subgroup_invocation_id) sg_lane: u32,
        @builtin(subgroup_id) sg_id: u32,
        @builtin(subgroup_size) sg_size: u32) {
    let lid = sg_id * sg_size + sg_lane;
#else
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
#endif
    let h = wg_id.y;
    let kvh = h / params.head_ratio;
    let part = lid % SPLIT;
    let row = wg_id.x * ROWS + lid / SPLIT;
    // rows past the end compute on the last row and store nothing
    let qi = min(row, params.n_q - 1u);

    // this thread's chunks: c = SPLIT * i + part
    var ql: array<vec4<f32>, NC>;
    var qh: array<vec4<f32>, NC>;
    var ol: array<vec4<f32>, NC>;
    var oh: array<vec4<f32>, NC>;
    let q_base = params.offset_q + qi * params.stride_q1 + h * params.stride_q2 + 8u * part;
    for (var i = 0u; i < NC; i++) {
        let b = q_base + 8u * SPLIT * i;
        ql[i] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]) * params.scale;
        qh[i] = vec4<f32>(q[b + 4u], q[b + 5u], q[b + 6u], q[b + 7u]) * params.scale;
        ol[i] = vec4<f32>(0.0);
        oh[i] = vec4<f32>(0.0);
    }

    var m = NEG_BIG;
    var l = 0.0;
    let k_head = (params.offset_k + kvh * params.stride_k2) / 8u + part;
    let v_head = (params.offset_v + kvh * params.stride_v2) / 8u + part;
    let k_step = params.stride_k1 / 8u;
    let v_step = params.stride_v1 / 8u;
    let mask_row = params.offset_mask + qi * params.stride_mask1;
    let last = params.n_kv - 1u;

    for (var j = 0u; j < params.n_kv; j += 4u) {
        let jv = min(vec4<u32>(j, j + 1u, j + 2u, j + 3u), vec4<u32>(last));
        // mask (or -inf past the end) of positions j .. j + 3
        var mk = vec4<f32>(0.0);
#ifdef HAS_MASK
        mk = vec4<f32>(f32(mask[mask_row + jv.x]), f32(mask[mask_row + jv.y]), f32(mask[mask_row + jv.z]),
                       f32(mask[mask_row + jv.w]));
#endif
        mk = select(mk, vec4<f32>(NEG_BIG), vec4<u32>(j, j + 1u, j + 2u, j + 3u) > vec4<u32>(last));
        let hidden = all(mk <= vec4<f32>(NEG_BIG));
#if SPLIT > 1
        if (subgroupAll(hidden)) {
            continue;
        }
#else
        if (hidden) {
            continue;
        }
#endif
        let kb = k_head + jv * k_step;
        var s = vec4<f32>(0.0);
        for (var i = 0u; i < NC; i++) {
            let c = SPLIT * i;
            let k0 = k[kb.x + c];
            let k1 = k[kb.y + c];
            let k2 = k[kb.z + c];
            let k3 = k[kb.w + c];
            s += vec4<f32>(dot(ql[i], lo(k0)) + dot(qh[i], hi(k0)), dot(ql[i], lo(k1)) + dot(qh[i], hi(k1)),
                           dot(ql[i], lo(k2)) + dot(qh[i], hi(k2)), dot(ql[i], lo(k3)) + dot(qh[i], hi(k3)));
        }
#if SPLIT > 1
        s += subgroupShuffleXor(s, 1u);
#endif
#if SPLIT > 2
        s += subgroupShuffleXor(s, 2u);
#endif
        s += mk;
        let m_new = max(m, max(max(s.x, s.y), max(s.z, s.w)));
        let corr = exp(m - m_new);
        let p = select(exp(s - vec4<f32>(m_new)), vec4<f32>(0.0), mk <= vec4<f32>(NEG_BIG));
        l = l * corr + p.x + p.y + p.z + p.w;
        m = m_new;
        let vb = v_head + jv * v_step;
        for (var i = 0u; i < NC; i++) {
            let c = SPLIT * i;
            let v0 = v[vb.x + c];
            let v1 = v[vb.y + c];
            let v2 = v[vb.z + c];
            let v3 = v[vb.w + c];
            ol[i] = ol[i] * corr + p.x * lo(v0) + p.y * lo(v1) + p.z * lo(v2) + p.w * lo(v3);
            oh[i] = oh[i] * corr + p.x * hi(v0) + p.y * hi(v1) + p.z * hi(v2) + p.w * hi(v3);
        }
    }

    if (row < params.n_q) {
        // a row the mask hides entirely gets zeros
        let inv_l = select(1.0 / l, 0.0, l == 0.0);
        let d = params.offset_dst + (row * params.n_heads + h) * HEAD_DIM + 8u * part;
        for (var i = 0u; i < NC; i++) {
            let b = d + 8u * SPLIT * i;
            let a = ol[i] * inv_l;
            let c = oh[i] * inv_l;
            dst[b] = a.x;
            dst[b + 1u] = a.y;
            dst[b + 2u] = a.z;
            dst[b + 3u] = a.w;
            dst[b + 4u] = c.x;
            dst[b + 5u] = c.y;
            dst[b + 6u] = c.z;
            dst[b + 7u] = c.w;
        }
    }
}
