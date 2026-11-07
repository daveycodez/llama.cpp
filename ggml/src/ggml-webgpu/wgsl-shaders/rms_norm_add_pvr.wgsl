// PowerVR: a block's post-norm, the residual add and the next norm as one
// dispatch (models with norms after the attention/conv and the FFN, as LFM3):
//   a   = x + rms_norm(o) * w1     (a is kept: the next residual add reads it)
//   out = rms_norm(a) * w2         (also packed into staged, the next mat-vec's src1)
// Each step's arithmetic is that of rms_norm_mul.wgsl and of the add shader
// (the same sums in the same order, the same reduction tree and the same
// expressions), so the results are those of the three dispatches it replaces.
// A workgroup takes a row; every column belongs to one thread throughout, so
// a, out, x and o may share memory.

struct Params {
    offset_o: u32,  // elements, in act
    offset_x: u32,
    offset_a: u32,
    offset_out: u32,
    offset_w1: u32,  // elements, in w1 and w2
    offset_w2: u32,
    ne0: u32,
    nrows: u32,
    eps1: f32,
    eps2: f32
};

// o, x, a and out: one binding over all four (they share a buffer)
@group(0) @binding(0) var<storage, read_write> act: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;
@group(0) @binding(2) var<storage, read> w2: array<f32>;
@group(0) @binding(3) var<storage, read_write> staged: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scratch: array<f32, WG_SIZE>;

// the sum over the workgroup, as rms_norm_mul.wgsl's tree adds it
fn wg_sum(lid: u32, v: f32) -> f32 {
    scratch[lid] = v;
    workgroupBarrier();
    var offset: u32 = WG_SIZE / 2;
    while (offset > 0) {
        if (lid < offset) {
            scratch[lid] += scratch[lid + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    let s = scratch[0];
    // every thread has read the sum before scratch is reused
    workgroupBarrier();
    return s;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * num_wg.x;
    if (row >= params.nrows) {
        return;
    }
    let base = row * params.ne0;
    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;

    var sum = 0.0f;
    var col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        sum += pow(act[params.offset_o + base + col], 2.0);
        col += WG_SIZE;
    }
    let scale1 = 1.0/sqrt(wg_sum(lid.x, sum)/f32(params.ne0) + params.eps1);

    sum = 0.0f;
    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        let m = scale1 * act[params.offset_o + base + col] * w1[params.offset_w1 + col];
        act[params.offset_a + base + col] = act[params.offset_x + base + col] + m;
        sum += pow(act[params.offset_a + base + col], 2.0);
        col += WG_SIZE;
    }
    let scale2 = 1.0/sqrt(wg_sum(lid.x, sum)/f32(params.ne0) + params.eps2);

    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        let v = scale2 * act[params.offset_a + base + col] * w2[params.offset_w2 + col];
        act[params.offset_out + base + col] = v;
        staged[base + col] = v;
        col += WG_SIZE;
    }
}
