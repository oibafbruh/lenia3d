// Apply one kernel's growth: acc (+)= h * G(U). The last kernel feeding a
// channel also performs the Euler update A = clip(A + dt * acc, 0, 1).
struct Params {
    total: u32,
    part: u32,
    gn: u32,
    flags: u32, // bit0: first kernel for this channel, bit1: last kernel
    m: f32,
    s: f32,
    h: f32,
    scale: f32, // 1 / N^3 (inverse FFT normalization)
    dt: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

@group(0) @binding(0) var<storage, read> w: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> acc: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read> wall: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn growth(u: f32) -> f32 {
    let d = u - params.m;
    let s = params.s;
    if (params.gn == 2u) {
        return exp(-d * d / (2.0 * s * s)) * 2.0 - 1.0;
    } else if (params.gn == 3u) {
        return select(-1.0, 1.0, abs(d) <= s);
    }
    let q = max(0.0, 1.0 - d * d / (9.0 * s * s));
    return q * q * q * q * 2.0 - 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= params.total) {
        return;
    }
    let wv = w[i];
    let u = select(wv.x, wv.y, params.part != 0u) * params.scale;
    var v = params.h * growth(u);
    if ((params.flags & 1u) == 0u) {
        v += acc[i];
    }
    if ((params.flags & 2u) != 0u) {
        let na = clamp(a[i] + params.dt * v, 0.0, 1.0) * (1.0 - wall[i]);
        a[i] = na;
    } else {
        acc[i] = v;
    }
}
