// Write two normalized Lenia kernels (centered at the origin, periodic) into
// one complex field: re = kernel 1, im = kernel 2. After a forward FFT the
// real and imaginary parts are the (real, even) spectra of the two kernels.
struct Params {
    n: u32,
    has2: u32,
    nb1: u32,
    nb2: u32,
    kn1: u32,
    kn2: u32,
    total: u32,
    _p0: u32,
    rad1: f32,
    rad2: f32,
    inv_sum1: f32,
    inv_sum2: f32,
    b1: vec4<f32>,
    b2: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

fn core(f: f32, kn: u32) -> f32 {
    if (kn == 2u) {
        if (f > 0.0 && f < 1.0) {
            return exp(4.0 - 1.0 / (f * (1.0 - f)));
        }
        return 0.0;
    } else if (kn == 3u) {
        return select(0.0, 1.0, f >= 0.25 && f <= 0.75);
    } else if (kn == 4u) {
        if (f >= 0.25 && f <= 0.75) {
            return 1.0;
        }
        return select(0.0, 0.5, f < 0.25);
    }
    let q = 4.0 * f * (1.0 - f);
    return q * q * q * q;
}

fn shell(d: f32, nb: u32, b: vec4<f32>, kn: u32) -> f32 {
    if (d >= 1.0) {
        return 0.0;
    }
    let br = f32(nb) * d;
    let fi = floor(br);
    let idx = min(u32(fi), nb - 1u);
    return b[idx] * core(br - fi, kn);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= params.total) {
        return;
    }
    let n = params.n;
    let x = i % n;
    let y = (i / n) % n;
    let z = i / (n * n);
    let o = vec3<f32>(f32(min(x, n - x)), f32(min(y, n - y)), f32(min(z, n - z)));
    let dist = sqrt(dot(o, o));
    let k1 = shell(dist / params.rad1, params.nb1, params.b1, params.kn1) * params.inv_sum1;
    var k2 = 0.0;
    if (params.has2 != 0u) {
        k2 = shell(dist / params.rad2, params.nb2, params.b2, params.kn2) * params.inv_sum2;
    }
    out[i] = vec2<f32>(k1, k2);
}
