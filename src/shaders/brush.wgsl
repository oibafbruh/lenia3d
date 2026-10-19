// Spherical brush operations on one channel and the wall field.
// Dispatched over the brush bounding cube (with periodic wrap).
struct Params {
    n: u32,
    mode: u32, // 0 noise blob, 1 add density, 2 erase, 3 add wall, 4 remove wall
    seed: u32,
    extent: u32,
    origin: vec3<i32>,
    radius: f32,
    center: vec3<f32>,
    strength: f32,
};

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> wall: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn pcg3d(v0: vec3<u32>) -> vec3<u32> {
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn rand01(c: vec3<u32>, seed: u32) -> f32 {
    let h = pcg3d(c ^ vec3<u32>(seed, seed * 747796405u, seed * 2891336453u));
    return f32(h.x >> 8u) / 16777216.0;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = params.extent;
    if (gid.x >= e || gid.y >= e || gid.z >= e) {
        return;
    }
    let p = params.origin + vec3<i32>(gid);
    let off = vec3<f32>(p) - params.center;
    let d = length(off) / params.radius;
    if (d > 1.0) {
        return;
    }
    let n = i32(params.n);
    let c = vec3<u32>(((p % n) + n) % n);
    let i = (c.z * params.n + c.y) * params.n + c.x;
    let falloff = 1.0 - smoothstep(0.55, 1.0, d);
    switch params.mode {
        case 0u: {
            let r = rand01(c, params.seed) * params.strength;
            a[i] = clamp(mix(a[i], r, falloff), 0.0, 1.0) * (1.0 - wall[i]);
        }
        case 1u: {
            a[i] = clamp(a[i] + params.strength * falloff * 0.25, 0.0, 1.0) * (1.0 - wall[i]);
        }
        case 2u: {
            a[i] = a[i] * (1.0 - falloff * params.strength);
        }
        case 3u: {
            wall[i] = 1.0;
            a[i] = 0.0;
        }
        case 4u: {
            wall[i] = 0.0;
        }
        default: {}
    }
}
