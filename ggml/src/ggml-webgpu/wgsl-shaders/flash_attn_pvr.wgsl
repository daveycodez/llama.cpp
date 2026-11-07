// Flash attention for one query token, or a few, on Imagination PowerVR GPUs
// (f32 Q, f16 K/V). The generic vector path takes milliseconds per call
// there. One workgroup per query token and head: each thread walks every
// WG_SIZE-th KV position with its own running max, sum and output, then the
// threads merge them.
// K and V are bound read-only when they live in another buffer than dst (the
// KV cache), as PowerVR reads read_write bindings much slower.

enable f16;

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_mask: u32,
    offset_dst: u32,
    n_kv: u32,
    stride_q2: u32,   // Q elements between heads
    stride_k1: u32,   // K elements between KV positions
    stride_k2: u32,   // K elements between KV heads
    stride_v1: u32,
    stride_v2: u32,
    head_ratio: u32,  // query heads per KV head
    scale: f32,
    stride_q1: u32,     // Q elements between query tokens
    stride_mask1: u32   // mask elements between query tokens
};

#ifdef KV_RO
#define KV_ACCESS read
#else
#define KV_ACCESS read_write
#endif

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, KV_ACCESS> k: array<vec4<f16>>;
@group(0) @binding(2) var<storage, KV_ACCESS> v: array<vec4<f16>>;
#ifdef HAS_MASK
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
#define DST_BINDING 4
#else
#define DST_BINDING 3
#endif
@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<f32>;
@group(0) @binding(DST_BINDING + 1) var<uniform> params: Params;

const D4 = HEAD_DIM / 4u;
const NEG_BIG = -3.0e38;

var<workgroup> sh_m: array<f32, WG_SIZE>;
var<workgroup> sh_l: array<f32, WG_SIZE>;
var<workgroup> sh_o: array<vec4<f32>, WG_SIZE * D4>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    // the query tokens of a head are neighbouring workgroups (the rows of a
    // speculative batch), which run at once and share K and V through the cache
    let row = wg_id.x;  // query token
    let h = wg_id.y;
    let kvh = h / params.head_ratio;

    var qv: array<vec4<f32>, D4>;
    let q_base = params.offset_q + row * params.stride_q1 + h * params.stride_q2;
    for (var i = 0u; i < D4; i++) {
        qv[i] = vec4<f32>(q[q_base + 4u * i], q[q_base + 4u * i + 1u], q[q_base + 4u * i + 2u], q[q_base + 4u * i + 3u]) *
                params.scale;
    }

    var m = NEG_BIG;
    var l = 0.0;
    var o: array<vec4<f32>, D4>;
    for (var i = 0u; i < D4; i++) {
        o[i] = vec4<f32>(0.0);
    }

    let k_head = (params.offset_k + kvh * params.stride_k2) / 4u;
    let v_head = (params.offset_v + kvh * params.stride_v2) / 4u;
    for (var j = lid; j < params.n_kv; j += WG_SIZE) {
#ifdef HAS_MASK
        // a position the mask hides (-inf: f16 holds nothing else this low) contributes nothing: skip
        // its K row (the sum below would be -inf, or NaN, and fail the test after it alike)
        let mv = f32(mask[params.offset_mask + row * params.stride_mask1 + j]);
        if (mv <= NEG_BIG) {
            continue;
        }
#endif
        let kb = k_head + j * (params.stride_k1 / 4u);
        var s = 0.0;
        for (var i = 0u; i < D4; i++) {
            s += dot(qv[i], vec4<f32>(k[kb + i]));
        }
#ifdef HAS_MASK
        s += mv;
#endif
        if (s > NEG_BIG) {
            let m_new = max(m, s);
            let corr = exp(m - m_new);
            let p = exp(s - m_new);
            l = l * corr + p;
            let vb = v_head + j * (params.stride_v1 / 4u);
            for (var i = 0u; i < D4; i++) {
                o[i] = o[i] * corr + p * vec4<f32>(v[vb + i]);
            }
            m = m_new;
        }
    }

    // merge the threads
    sh_m[lid] = m;
    workgroupBarrier();
    var m_all = NEG_BIG;
    for (var t = 0u; t < WG_SIZE; t++) {
        m_all = max(m_all, sh_m[t]);
    }
    let f = select(exp(m - m_all), 0.0, m <= NEG_BIG);
    sh_l[lid] = l * f;
    for (var i = 0u; i < D4; i++) {
        sh_o[lid * D4 + i] = o[i] * f;
    }
    workgroupBarrier();
    // head dims above 4 * WG_SIZE take several passes
    for (var c = lid; c < D4; c += WG_SIZE) {
        var l_all = 0.0;
        var acc = vec4<f32>(0.0);
        for (var t = 0u; t < WG_SIZE; t++) {
            l_all += sh_l[t];
            acc += sh_o[t * D4 + c];
        }
        // a row the mask hides entirely gets zeros
        let r = select(acc / l_all, vec4<f32>(0.0), l_all == 0.0);
        let d = params.offset_dst + (row * num_wg.y + h) * HEAD_DIM + 4u * c;
        dst[d] = r.x;
        dst[d + 1u] = r.y;
        dst[d + 2u] = r.z;
        dst[d + 3u] = r.w;
    }
}
