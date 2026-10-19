// Affinity of one kernel with per-cell weight h_k from the genome:
// acc (+)= h_k(x) * G(U_k(x)).
struct P {
    total: u32,
    part: u32,
    gn: u32,
    first: u32,
    m: f32,
    s: f32,
    h: f32,
    scale: f32,
    k: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> w: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> acc: array<f32>;
@group(0) @binding(2) var<storage, read> genome: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: P;

fn growth(u: f32) -> f32 {
    let d = u - p.m;
    if (p.gn == 2u) {
        return exp(-d * d / (2.0 * p.s * p.s)) * 2.0 - 1.0;
    } else if (p.gn == 3u) {
        return select(-1.0, 1.0, abs(d) <= p.s);
    }
    let q = max(0.0, 1.0 - d * d / (9.0 * p.s * p.s));
    return q * q * q * q * 2.0 - 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= p.total) {
        return;
    }
    let wv = w[i];
    let u = select(wv.x, wv.y, p.part != 0u) * p.scale;
    var h = p.h;
    if (p.k < 6u) {
        let gv = genome[i * 4u + p.k / 4u];
        h = gv[p.k % 4u];
    }
    var v = h * growth(u);
    if (p.first == 0u) {
        v += acc[i];
    }
    acc[i] = v;
}
