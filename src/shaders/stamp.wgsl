// Copy a pattern (one channel) into the world at a given origin, with wrap.
struct Params {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    dims: vec3<u32>,
    _p3: u32,
    origin: vec3<i32>,
    _p4: u32,
};

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> pattern: array<f32>;
@group(0) @binding(2) var<storage, read> wall: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = params.dims;
    if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) {
        return;
    }
    let n = i32(params.n);
    let p = params.origin + vec3<i32>(gid);
    let c = vec3<u32>(((p % n) + n) % n);
    let i = (c.z * params.n + c.y) * params.n + c.x;
    let v = pattern[(gid.z * d.y + gid.y) * d.x + gid.x];
    a[i] = v * (1.0 - wall[i]);
}
