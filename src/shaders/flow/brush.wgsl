// Flow-mode brushes over a sphere (with periodic wrap):
// 0 seed a blob of matter carrying `genes`, 1 add fertility (food source),
// 2 toxic zone, 3 neutral ground, 4 set `genes` where matter exists,
// 5 set `genes` everywhere in the sphere.
struct P {
    n: u32,
    mode: u32,
    seed: u32,
    extent: u32,
    origin: vec3<i32>,
    radius: f32,
    center: vec3<f32>,
    strength: f32,
    density: f32,
    nch: u32,
    default_fert: f32,
    _p0: f32,
    genes: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<storage, read_write> a0: array<f32>;
@group(0) @binding(1) var<storage, read_write> a1: array<f32>;
@group(0) @binding(2) var<storage, read_write> a2: array<f32>;
@group(0) @binding(3) var<storage, read_write> genome: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> food: array<f32>;
@group(0) @binding(5) var<storage, read_write> fert: array<f32>;
@group(0) @binding(6) var<storage, read> wall: array<f32>;
@group(0) @binding(7) var<uniform> p: P;

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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = p.extent;
    if (gid.x >= e || gid.y >= e || gid.z >= e) {
        return;
    }
    let q = p.origin + vec3<i32>(gid);
    let d = length(vec3<f32>(q) - p.center) / p.radius;
    if (d > 1.0) {
        return;
    }
    let n = i32(p.n);
    let c = vec3<u32>(((q % n) + n) % n);
    let i = (c.z * p.n + c.y) * p.n + c.x;
    if (wall[i] > 0.5) {
        return;
    }
    switch p.mode {
        case 0u: {
            let t = 1.0 - d;
            let w = t * t * (3.0 - 2.0 * t);
            var placed = false;
            for (var ch = 0u; ch < p.nch; ch++) {
                let r = vec2<f32>(pcg3d(c ^ vec3<u32>(p.seed + ch * 7919u, p.seed * 747796405u, p.seed * 2891336453u + ch)).xy >> vec2<u32>(8u)) / 16777216.0;
                let v = select(0.0, r.y * w, r.x < p.density);
                if (v > 0.0) {
                    placed = true;
                    if (ch == 0u) {
                        a0[i] = max(a0[i], v);
                    } else if (ch == 1u) {
                        a1[i] = max(a1[i], v);
                    } else {
                        a2[i] = max(a2[i], v);
                    }
                }
            }
            if (placed) {
                for (var k = 0u; k < 4u; k++) {
                    genome[i * 4u + k] = p.genes[k];
                }
            }
        }
        case 1u: {
            let falloff = 1.0 - smoothstep(0.5, 1.0, d);
            fert[i] = clamp(fert[i] + p.strength * falloff * 0.25, -1.0, 1.0);
            food[i] = max(food[i], fert[i]);
        }
        case 2u: {
            let falloff = 1.0 - smoothstep(0.5, 1.0, d);
            fert[i] = clamp(fert[i] - p.strength * falloff * 0.25, -1.0, 1.0);
        }
        case 3u: {
            fert[i] = p.default_fert;
        }
        case 4u: {
            var msum = a0[i];
            if (p.nch > 1u) {
                msum += a1[i];
            }
            if (p.nch > 2u) {
                msum += a2[i];
            }
            if (msum > 0.0) {
                for (var k = 0u; k < 4u; k++) {
                    genome[i * 4u + k] = p.genes[k];
                }
            }
        }
        case 5u: {
            for (var k = 0u; k < 4u; k++) {
                genome[i * 4u + k] = p.genes[k];
            }
        }
        default: {}
    }
}
