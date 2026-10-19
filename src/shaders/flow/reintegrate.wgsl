// Reintegration tracking (gather form) with stochastic, claw-biased genome inheritance.
@group(0) @binding(0) var<storage, read> d0: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> d1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> d2: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> genome: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> genome_next: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> n0: array<f32>;
@group(0) @binding(6) var<storage, read_write> n1: array<f32>;
@group(0) @binding(7) var<storage, read_write> n2: array<f32>;
@group(0) @binding(8) var<storage, read> wall: array<f32>;
@group(0) @binding(9) var<uniform> fp: FlowParams;
@group(0) @binding(10) var<storage, read> state: array<u32>;

fn overlap(p: vec3<f32>, s: f32) -> f32 {
    let ov = max(vec3<f32>(0.0), min(p + s, vec3<f32>(0.5)) - max(p - s, vec3<f32>(-0.5)));
    return ov.x * ov.y * ov.z;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= fp.total) {
        return;
    }
    let n = fp.n;
    if (wall[i] > 0.5) {
        n0[i] = 0.0;
        if (fp.nch > 1u) {
            n1[i] = 0.0;
        }
        if (fp.nch > 2u) {
            n2[i] = 0.0;
        }
        for (var k = 0u; k < 4u; k++) {
            genome_next[i * 4u + k] = genome[i * 4u + k];
        }
        return;
    }
    let p = cell_coord(i, n);
    let s = fp.s;
    let inv = 1.0 / (8.0 * s * s * s);
    var acc = vec3<f32>(0.0);
    var seed = pcg(i ^ pcg(state[0] * 2654435761u));
    var wsum = 0.0;
    var chosen = i;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let o = vec3<i32>(dx, dy, dz);
                let j = cell_index(p + o, n);
                let ofs = vec3<f32>(o);
                var m = 0.0;
                let v0 = d0[j];
                let c0 = v0.w * overlap(ofs + v0.xyz, s) * inv;
                acc.x += c0;
                m += c0;
                if (fp.nch > 1u) {
                    let v1 = d1[j];
                    let c1 = v1.w * overlap(ofs + v1.xyz, s) * inv;
                    acc.y += c1;
                    m += c1;
                }
                if (fp.nch > 2u) {
                    let v2 = d2[j];
                    let c2 = v2.w * overlap(ofs + v2.xyz, s) * inv;
                    acc.z += c2;
                    m += c2;
                }
                if (m > 1e-9) {
                    let claw = genome[j * 4u + 3u].x;
                    let wt = m * (1.0 + fp.claw_bias * claw);
                    wsum += wt;
                    // weighted reservoir sampling
                    if (rnd(&seed) * wsum < wt) {
                        chosen = j;
                    }
                }
            }
        }
    }
    n0[i] = acc.x;
    if (fp.nch > 1u) {
        n1[i] = acc.y;
    }
    if (fp.nch > 2u) {
        n2[i] = acc.z;
    }
    for (var k = 0u; k < 4u; k++) {
        genome_next[i * 4u + k] = genome[chosen * 4u + k];
    }
}
