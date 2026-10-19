// Pack two real channels into one complex field: z = a + i*b.
struct Params {
    total: u32,
    has_b: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= params.total) {
        return;
    }
    var bv = 0.0;
    if (params.has_b != 0u) {
        bv = b[i];
    }
    z[i] = vec2<f32>(a[i], bv);
}
