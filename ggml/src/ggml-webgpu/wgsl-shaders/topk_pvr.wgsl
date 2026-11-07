// The logits' K largest values per row, with their indices, for the sampler,
// on Imagination PowerVR GPUs (llama.cpp's TOP_K, GET_ROWS, CPY and CONCAT
// of the logits in one; see ggml_webgpu_topk_pvr). Each output row is the K
// values, largest first (ties: lower index first), then their K indices as
// floats (exact below 2^24); a row that couldn't be done (too many
// candidates, see STAGE2) gets index -1 everywhere, and the sampler then
// reads the logits instead.
//
// Three dispatches over a scratch area (a buffer of its own): per row, the
// maxima of its groups of GROUP values (256, or fewer for small vocabularies:
// there must be K groups at least), a candidate count and up to CAP
// candidates (key, index).
// STAGE1: the group maxima. STAGE2: T0, the K-th largest maximum of the
// whole groups (a last group of a few values, whose maximum may be low, is
// left out: there are K whole groups at least), is at most the K-th largest
// value (K groups have a value at least T0), so the values at least T0 are
// the candidates; they're few for logits.
// STAGE3: each candidate's rank among them (by key, then index) is its
// place in the output, if below K. A key is the float's bits mapped so that
// unsigned order is float order.

struct Params {
    offset_src: u32,  // elements of the logits
    stride_src: u32,  // elements between rows of the logits
    n: u32,           // values per row
    offset_dst: u32,  // elements of the output (STAGE3)
    stride_dst: u32,  // elements between output rows (STAGE3)
};

const WG = 256u;
// GROUP: values per group (a define)
const GPW = 16u;              // groups per STAGE1/STAGE2 workgroup
const CHUNK = GROUP * GPW;    // values per STAGE1/STAGE2 workgroup
// scratch per row (u32 words): G_MAX group maxima, the count, CAP (key, index) pairs
const ROW_WORDS = G_MAX + 1u + 2u * CAP;
const_assert K <= 256u;
const_assert G_MAX % 8u == 0u;
const_assert CAP % WG == 0u;
const_assert GROUP % 16u == 0u && GROUP <= WG;

#ifdef STAGE1
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
#endif
#ifdef STAGE2
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<atomic<u32>>;
#endif
#ifdef STAGE3
@group(0) @binding(0) var<storage, read> scratch: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
#endif
@group(0) @binding(2) var<uniform> params: Params;

fn key_of(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn value_of(k: u32) -> f32 {
    return bitcast<f32>(select(~k, k & 0x7fffffffu, (k & 0x80000000u) != 0u));
}

#ifdef STAGE1
var<workgroup> part: array<u32, WG>;

// thread t takes values (t % 16) + 16 i of group t / 16 of the workgroup's chunk
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wg_id.y;
    let g = wg_id.x * GPW + lid / 16u;  // group of the row
    let first = g * GROUP + lid % 16u;
    let base = params.offset_src + row * params.stride_src;
    var m = 0u;  // below every float's key: an empty group never counts
    for (var i = 0u; i < GROUP / 16u; i++) {
        let e = first + 16u * i;
        if (e < params.n) {
            m = max(m, key_of(src[base + e]));
        }
    }
    part[lid] = m;
    workgroupBarrier();
    let srow = row * ROW_WORDS;
    // (the last workgroup's groups past the row's last are no groups: not written over the count)
    if (lid % 16u == 0u && g * GROUP < params.n) {
        var mm = part[lid];
        for (var j = 1u; j < 16u; j++) {
            mm = max(mm, part[lid + j]);
        }
        scratch[srow + g] = mm;
    }
    if (wg_id.x == 0u && lid == 0u) {
        scratch[srow + G_MAX] = 0u;  // the candidate count
    }
}
#endif

#ifdef STAGE2
var<workgroup> gmax: array<vec4<u32>, G_MAX / 4u>;
var<workgroup> t0: u32;
var<workgroup> n_wg: atomic<u32>;
var<workgroup> wg_base: u32;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wg_id.y;
    let srow = row * ROW_WORDS;
    let n_groups = params.n / GROUP;  // the whole ones
    for (var i = lid; i < G_MAX / 4u; i += WG) {
        var v = vec4<u32>(0u);
        for (var c = 0u; c < 4u; c++) {
            if (4u * i + c < n_groups) {
                v[c] = atomicLoad(&scratch[srow + 4u * i + c]);
            }
        }
        gmax[i] = v;
    }
    if (lid == 0u) {
        t0 = 0u;
        atomicStore(&n_wg, 0u);
    }
    workgroupBarrier();
    // T0: the whole groups' maximum ranked K - 1 (by key, then group; the others rank last)
    for (var i = lid; i < n_groups; i += WG) {
        let mi = gmax[i / 4u][i % 4u];
        let mi4 = vec4<u32>(mi);
        let ii4 = vec4<u32>(i);
        var rank4 = vec4<u32>(0u);
        for (var j = 0u; j < G_MAX / 4u; j += 2u) {
            let a = gmax[j];
            let b = gmax[j + 1u];
            let ja = vec4<u32>(4u * j) + vec4<u32>(0u, 1u, 2u, 3u);
            let jb = ja + vec4<u32>(4u);
            rank4 += select(vec4<u32>(0u), vec4<u32>(1u), (a > mi4) | ((a == mi4) & (ja < ii4)));
            rank4 += select(vec4<u32>(0u), vec4<u32>(1u), (b > mi4) | ((b == mi4) & (jb < ii4)));
        }
        if (rank4.x + rank4.y + rank4.z + rank4.w == K - 1u) {
            t0 = mi;
        }
    }
    let thr = workgroupUniformLoad(&t0);
    // the candidates: every value of the chunk at least T0, placed with one
    // global atomic per workgroup (few values pass)
    let base = params.offset_src + row * params.stride_src;
    var n_mine = 0u;
    for (var i = 0u; i < CHUNK / WG; i++) {
        let e = wg_id.x * CHUNK + lid + WG * i;
        if (e < params.n && key_of(src[base + e]) >= thr) {
            n_mine++;
        }
    }
    let at = atomicAdd(&n_wg, n_mine);
    workgroupBarrier();
    if (lid == 0u) {
        wg_base = atomicAdd(&scratch[srow + G_MAX], atomicLoad(&n_wg));
    }
    var p = workgroupUniformLoad(&wg_base) + at;
    for (var i = 0u; i < CHUNK / WG; i++) {
        let e = wg_id.x * CHUNK + lid + WG * i;
        if (e < params.n) {
            let k = key_of(src[base + e]);
            if (k >= thr) {
                if (p < CAP) {
                    atomicStore(&scratch[srow + G_MAX + 1u + 2u * p], k);
                    atomicStore(&scratch[srow + G_MAX + 2u + 2u * p], e);
                }
                p++;
            }
        }
    }
}
#endif

#ifdef STAGE3
var<workgroup> ckey: array<vec4<u32>, CAP / 4u>;
var<workgroup> cidx: array<vec4<u32>, CAP / 4u>;
var<workgroup> wg_count: u32;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wg_id.x;
    let srow = row * ROW_WORDS;
    if (lid == 0u) {
        wg_count = scratch[srow + G_MAX];
    }
    let count = workgroupUniformLoad(&wg_count);
    let o = params.offset_dst + row * params.stride_dst;
    if (count > CAP || count < K) {
        // too many candidates (or, impossibly, too few): the sampler reads the logits
        for (var i = lid; i < K; i += WG) {
            dst[o + i] = 0.0;
            dst[o + K + i] = -1.0;
        }
        return;
    }
    // padded to a multiple of 8 with keys that rank last
    let count8 = (count + 7u) & ~7u;
    for (var i = lid; i < count8 / 4u; i += WG) {
        var kv = vec4<u32>(0u);
        var iv = vec4<u32>(0xffffffffu);
        for (var c = 0u; c < 4u; c++) {
            let e = 4u * i + c;
            if (e < count) {
                kv[c] = scratch[srow + G_MAX + 1u + 2u * e];
                iv[c] = scratch[srow + G_MAX + 2u + 2u * e];
            }
        }
        ckey[i] = kv;
        cidx[i] = iv;
    }
    workgroupBarrier();
    for (var i = lid; i < count; i += WG) {
        let ki = ckey[i / 4u][i % 4u];
        let ii = cidx[i / 4u][i % 4u];
        let ki4 = vec4<u32>(ki);
        let ii4 = vec4<u32>(ii);
        var rank4 = vec4<u32>(0u);
        for (var j = 0u; j < count8 / 4u; j += 2u) {
            let ka = ckey[j];
            let kb = ckey[j + 1u];
            let ia = cidx[j];
            let ib = cidx[j + 1u];
            rank4 += select(vec4<u32>(0u), vec4<u32>(1u), (ka > ki4) | ((ka == ki4) & (ia < ii4)));
            rank4 += select(vec4<u32>(0u), vec4<u32>(1u), (kb > ki4) | ((kb == ki4) & (ib < ii4)));
        }
        let rank = rank4.x + rank4.y + rank4.z + rank4.w;
        if (rank < K) {
            dst[o + rank] = value_of(ki);
            dst[o + K + rank] = f32(ii);
        }
    }
}
#endif
