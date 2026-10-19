// Multiply a complex spectrum by a real spectrum (stored in .x).
struct P {
    total: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> z: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= p.total) {
        return;
    }
    z[i] = z[i] * k[i].x;
}
