// Spectral multiply for a pair of kernels.
// za / zb hold FFT(a + i b) of packed channel pairs; part selects which of the
// two real channels a kernel reads. ks = (K1_hat, K2_hat), both real.
// Output w = K1_hat * S1 + i * K2_hat * S2, whose inverse FFT is
// (K1 * s1) + i (K2 * s2).
struct Params {
    n: u32,
    total: u32,
    part1: u32,
    part2: u32,
};

@group(0) @binding(0) var<storage, read> za: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> zb: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> ks: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> w: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

// Spectrum of one real channel from the packed spectrum z at k and -k.
fn extract(z: vec2<f32>, zn: vec2<f32>, part: u32) -> vec2<f32> {
    let c = vec2<f32>(zn.x, -zn.y);
    if (part == 0u) {
        return (z + c) * 0.5;
    }
    let d = z - c;
    return vec2<f32>(d.y, -d.x) * 0.5;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= params.total) {
        return;
    }
    let n = params.n;
    let m = n - 1u;
    let x = i % n;
    let y = (i / n) % n;
    let z = i / (n * n);
    let j = ((n - z) & m) * n * n + ((n - y) & m) * n + ((n - x) & m);
    let s1 = extract(za[i], za[j], params.part1);
    let s2 = extract(zb[i], zb[j], params.part2);
    let k = ks[i];
    w[i] = vec2<f32>(k.x * s1.x - k.y * s2.y, k.x * s1.y + k.y * s2.x);
}
