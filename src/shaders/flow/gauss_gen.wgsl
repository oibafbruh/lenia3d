// Truncated, normalized Gaussian sensing kernel centered at the origin (periodic).
struct P {
    n: u32,
    total: u32,
    _p0: u32,
    _p1: u32,
    sigma: f32,
    rmax: f32,
    inv_sum: f32,
    _p2: f32,
};

@group(0) @binding(0) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= p.total) {
        return;
    }
    let n = p.n;
    let x = i % n;
    let y = (i / n) % n;
    let z = i / (n * n);
    let o = vec3<f32>(f32(min(x, n - x)), f32(min(y, n - y)), f32(min(z, n - z)));
    let d2 = dot(o, o);
    var v = 0.0;
    if (d2 < p.rmax * p.rmax) {
        v = exp(-d2 / (2.0 * p.sigma * p.sigma)) * p.inv_sum;
    }
    out[i] = vec2<f32>(v, 0.0);
}
