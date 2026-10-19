// Partial sums of a field: each workgroup sums CHUNK consecutive values.
const CHUNK: u32 = 4096u;

struct Params {
    total: u32,
    out_offset: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wg.y * nwg.x + wg.x;
    let base = g * CHUNK;
    var s = 0.0;
    for (var k = lid; k < CHUNK; k += 256u) {
        let i = base + k;
        if (i < params.total) {
            s += a[i];
        }
    }
    partial[lid] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        out[params.out_offset + g] = partial[0];
    }
}
