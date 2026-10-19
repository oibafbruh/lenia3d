// Coarse statistics for creature tracking, 20 floats per block of f^3 cells:
// [0] mass, [1..16] mass-weighted genes, [17] mass*cos(hue), [18] mass*sin(hue),
// [19] food.
struct P {
    n: u32,
    m: u32,
    f: u32,
    nch: u32,
    has_genome: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    scale: f32,
    _p3: f32,
    _p4: f32,
    _p5: f32,
};

@group(0) @binding(0) var<storage, read> a0: array<f32>;
@group(0) @binding(1) var<storage, read> a1: array<f32>;
@group(0) @binding(2) var<storage, read> a2: array<f32>;
@group(0) @binding(3) var<storage, read> genome: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> food: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = p.m;
    if (gid.x >= m || gid.y >= m || gid.z >= m) {
        return;
    }
    var mass = 0.0;
    var gs = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var hc = 0.0;
    var hs = 0.0;
    var fsum = 0.0;
    let base = gid * p.f;
    for (var z = 0u; z < p.f; z++) {
        for (var y = 0u; y < p.f; y++) {
            for (var x = 0u; x < p.f; x++) {
                let c = base + vec3<u32>(x, y, z);
                let i = (c.z * p.n + c.y) * p.n + c.x;
                var a = a0[i];
                if (p.nch > 1u) {
                    a += a1[i];
                }
                if (p.nch > 2u) {
                    a += a2[i];
                }
                mass += a;
                if (p.has_genome != 0u) {
                    if (a > 0.0) {
                        for (var k = 0u; k < 4u; k++) {
                            gs[k] += a * genome[i * 4u + k];
                        }
                        let hue = genome[i * 4u + 3u].z;
                        hc += a * cos(6.2831853 * hue);
                        hs += a * sin(6.2831853 * hue);
                    }
                    fsum += food[i];
                }
            }
        }
    }
    let o = ((gid.z * m + gid.y) * m + gid.x) * 20u;
    out[o] = mass;
    for (var k = 0u; k < 4u; k++) {
        out[o + 1u + k * 4u] = gs[k].x;
        out[o + 2u + k * 4u] = gs[k].y;
        out[o + 3u + k * 4u] = gs[k].z;
        out[o + 4u + k * 4u] = gs[k].w;
    }
    out[o + 17u] = hc;
    out[o + 18u] = hs;
    out[o + 19u] = fsum;
}
