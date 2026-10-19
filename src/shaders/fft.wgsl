// In-place 3D complex FFT, one axis per dispatch.
// Each workgroup loads LINES full lines of length N into workgroup memory,
// runs a radix-2 Cooley-Tukey FFT on them and writes them back.
// Templated constants: N, LOG2N, LINES (lines per workgroup), WG (threads).
// Layout: index = z*N*N + y*N + x.

const N: u32 = {{N}}u;
const LOG2N: u32 = {{LOG2N}}u;
const LINES: u32 = {{LINES}}u;
const WG: u32 = {{WG}}u;
const ELEMS: u32 = N * LINES;
const HALF: u32 = N / 2u;

struct Params {
    axis: u32,
    inverse: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
// twiddle[k] = (cos(2*pi*k/N), sin(2*pi*k/N)) for k < N/2, computed in f64 on the CPU
@group(0) @binding(1) var<storage, read> twiddle: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sh: array<vec2<f32>, ELEMS>;

fn bitrev(x: u32) -> u32 {
    return reverseBits(x) >> (32u - LOG2N);
}

// Global index of element j of line l in this workgroup's tile.
fn global_index(wg: vec3<u32>, l: u32, j: u32) -> u32 {
    let a = wg.x * LINES + l;
    let b = wg.y;
    if (params.axis == 0u) {
        // lines along x, tile over y, wg.y = z
        return b * N * N + a * N + j;
    } else if (params.axis == 1u) {
        // lines along y, tile over x, wg.y = z
        return b * N * N + j * N + a;
    }
    // lines along z, tile over x, wg.y = y
    return j * N * N + b * N + a;
}

// Thread -> (line, element) mapping that keeps global memory access coalesced.
fn split(i: u32) -> vec2<u32> {
    if (params.axis == 0u) {
        return vec2<u32>(i / N, i % N);
    }
    return vec2<u32>(i % LINES, i / LINES);
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    for (var i = lid; i < ELEMS; i += WG) {
        let lj = split(i);
        sh[lj.x * N + bitrev(lj.y)] = data[global_index(wg, lj.x, lj.y)];
    }
    workgroupBarrier();

    let sign = select(-1.0, 1.0, params.inverse != 0u);
    for (var s = 0u; s < LOG2N; s++) {
        let half = 1u << s;
        let tw_step = N >> (s + 1u);
        for (var t = lid; t < ELEMS / 2u; t += WG) {
            let l = t / HALF;
            let bf = t % HALF;
            let pos = bf & (half - 1u);
            let i0 = l * N + ((bf >> s) << (s + 1u)) + pos;
            let i1 = i0 + half;
            let tw = twiddle[pos * tw_step];
            let w = vec2<f32>(tw.x, sign * tw.y);
            let x0 = sh[i0];
            let x1 = sh[i1];
            let p = vec2<f32>(x1.x * w.x - x1.y * w.y, x1.x * w.y + x1.y * w.x);
            sh[i0] = x0 + p;
            sh[i1] = x0 - p;
        }
        workgroupBarrier();
    }

    for (var i = lid; i < ELEMS; i += WG) {
        let lj = split(i);
        data[global_index(wg, lj.x, lj.y)] = sh[lj.x * N + lj.y];
    }
}
