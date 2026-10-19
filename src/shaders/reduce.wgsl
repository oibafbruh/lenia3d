// Per-workgroup partial sums of a field: mass and periodic first moments
// (sum a*cos(2*pi*x/n), sum a*sin(2*pi*x/n) per axis) for the circular mean.
// Each workgroup handles CHUNK consecutive cells and writes 8 floats.
const CHUNK: u32 = 4096u;
const TAU: f32 = 6.283185307;

struct Params {
    total: u32,
    out_offset: u32,
    n: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial: array<array<f32, 7>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wg.y * nwg.x + wg.x;
    let base = g * CHUNK;
    let n = params.n;
    let k = TAU / f32(n);
    var s = array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var j = lid; j < CHUNK; j += 256u) {
        let i = base + j;
        if (i < params.total) {
            let v = a[i];
            if (v > 0.0) {
                let p = vec3<f32>(f32(i % n), f32((i / n) % n), f32(i / (n * n))) * k;
                s[0] += v;
                s[1] += v * cos(p.x);
                s[2] += v * sin(p.x);
                s[3] += v * cos(p.y);
                s[4] += v * sin(p.y);
                s[5] += v * cos(p.z);
                s[6] += v * sin(p.z);
            }
        }
    }
    partial[lid] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            for (var c = 0u; c < 7u; c++) {
                partial[lid][c] += partial[lid + stride][c];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let o = (params.out_offset + g) * 8u;
        for (var c = 0u; c < 7u; c++) {
            out[o + c] = partial[0][c];
        }
        out[o + 7u] = 0.0;
    }
}
