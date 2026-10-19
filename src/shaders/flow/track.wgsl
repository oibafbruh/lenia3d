// Coarse statistics for creature tracking: per block of f^3 cells, 12 floats:
// [mass, eye, ear, voice, mouth, claw, pitch, hue_cos, hue_sin, timbre, food, sound]
// (traits are mass-weighted sums).
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
@group(0) @binding(5) var<storage, read> sense: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> p: P;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = p.m;
    if (gid.x >= m || gid.y >= m || gid.z >= m) {
        return;
    }
    var s = array<f32, 12>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
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
                s[0] += a;
                if (p.has_genome != 0u) {
                    if (a > 0.0) {
                        let g2 = genome[i * 4u + 2u];
                        let g3 = genome[i * 4u + 3u];
                        s[1] += a * g2.x;
                        s[2] += a * g2.y;
                        s[3] += a * g2.z;
                        s[4] += a * g2.w;
                        s[5] += a * g3.x;
                        s[6] += a * g3.y;
                        s[7] += a * cos(6.2831853 * g3.z);
                        s[8] += a * sin(6.2831853 * g3.z);
                        s[9] += a * g3.w;
                    }
                    s[10] += food[i];
                    s[11] += max(sense[i].y * p.scale, 0.0);
                }
            }
        }
    }
    let o = ((gid.z * m + gid.y) * m + gid.x) * 12u;
    for (var k = 0u; k < 12u; k++) {
        out[o + k] = s[k];
    }
}
