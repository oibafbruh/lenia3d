// Budding: move half of the matter inside a sphere by `offset` and give the
// moved half the parent's genome plus one shared mutation `delta`.
// Source sphere and target sphere never overlap (|offset| > 2 * radius).
struct P {
    n: u32,
    nch: u32,
    extent: u32,
    _p0: u32,
    origin: vec3<i32>,
    radius: f32,
    center: vec3<f32>,
    _p1: f32,
    offset: vec3<i32>,
    _p2: u32,
    delta: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<storage, read_write> a0: array<f32>;
@group(0) @binding(1) var<storage, read_write> a1: array<f32>;
@group(0) @binding(2) var<storage, read_write> a2: array<f32>;
@group(0) @binding(3) var<storage, read_write> genome: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> wall: array<f32>;
@group(0) @binding(5) var<uniform> p: P;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = p.extent;
    if (gid.x >= e || gid.y >= e || gid.z >= e) {
        return;
    }
    let q = p.origin + vec3<i32>(gid);
    if (length(vec3<f32>(q) - p.center) > p.radius) {
        return;
    }
    let src = cell_index(q, p.n);
    let dst = cell_index(q + p.offset, p.n);
    if (wall[dst] > 0.5) {
        return;
    }
    var moved = 0.0;
    let h0 = a0[src] * 0.5;
    a0[src] -= h0;
    a0[dst] += h0;
    moved += h0;
    if (p.nch > 1u) {
        let h1 = a1[src] * 0.5;
        a1[src] -= h1;
        a1[dst] += h1;
        moved += h1;
    }
    if (p.nch > 2u) {
        let h2 = a2[src] * 0.5;
        a2[src] -= h2;
        a2[dst] += h2;
        moved += h2;
    }
    if (moved > 0.0) {
        var g = array<vec4<f32>, 4>(
            genome[src * 4u] + p.delta[0],
            genome[src * 4u + 1u] + p.delta[1],
            genome[src * 4u + 2u] + p.delta[2],
            genome[src * 4u + 3u] + p.delta[3]);
        g = clamp_genome(g);
        for (var k = 0u; k < 4u; k++) {
            genome[dst * 4u + k] = g[k];
        }
    }
}
