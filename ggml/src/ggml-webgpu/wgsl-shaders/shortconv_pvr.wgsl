// LFM2/LFM3's short convolution on N tokens (1 to 8) of one sequence, on
// Imagination PowerVR GPUs, in one dispatch instead of seven or more (see
// ggml_webgpu_pvr_shortconv_nodes): with the in-projection's output bcx
// (B, C and X per token), the old conv state (D = width - 1 values per
// channel: LFM2's conv has 3 taps, LFM3's 7) gathered from the recurrent
// cache, and the conv weights, each thread takes a channel and computes
//   v      = [old state, B * X of each token]   (the CONCAT)
//   conv_t = v[t] * w0 + v[t + 1] * w1 + ... + v[t + D] * wD   (SSM_CONV: summed in that order from 0,
//            as fused multiply-adds, which is how ssm_conv.wgsl's loop compiles on PowerVR)
//   y_t    = C_t * conv_t                        (the MUL)
// writes y (and copies it into the staging buffer, as the next product's
// src1), and writes the rollback snapshots of the state into the cache
// (slot s holds the state s tokens back: v[N - s] .. v[N - s + D - 1]). Each
// value is computed as the separate ops compute it, so the results are
// theirs bit for bit. The old state is read before any snapshot is written
// over it, in the same thread.

// W_RO: the weights bound read-only (in another buffer than what's written)
#ifdef W_RO
#define W_ACCESS read
#else
#define W_ACCESS read_write
#endif

struct Params {
    offset_b: u32,      // elements of bcx: B of token 0 (C at offset_c, X at offset_x)
    offset_c: u32,
    offset_x: u32,
    stride_bcx: u32,    // elements between tokens of bcx
    offset_state: u32,  // elements of the cache binding: row 0 of the gathered view
    stride_state: u32,  // elements between rows of that view
    offset_idx: u32,    // elements of the row index (i32)
    offset_w: u32,      // elements of the conv weights ([width, channels])
    offset_y: u32,      // elements of y
    stride_y: u32,      // elements between tokens of y
    channels: u32,
    n_snap: u32,        // snapshots to write (at most N)
    snap0: u32,         // elements of the cache binding: snapshot rows 0 .. 7
    snap1: u32,
    snap2: u32,
    snap3: u32,
    snap4: u32,
    snap5: u32,
    snap6: u32,
    snap7: u32,
};

@group(0) @binding(0) var<storage, read_write> bcx: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<i32>;
@group(0) @binding(3) var<storage, W_ACCESS> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> staged: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

const_assert N >= 1u && N <= 8u;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= params.channels) {
        return;
    }
    let row = u32(idx[params.offset_idx]);

// the body, written out token by token by the shader library (see
// ggml_webgpu_shortconv_pvr_body): plain variables w0, w1, ..., v0, v1, ...
// rather than indexed arrays, which PowerVR keeps in slow memory
MAIN_BODY
}
