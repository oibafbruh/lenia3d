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
@group(0) @binding(11) var<storage, read_write> food: array<f32>;

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
    // resident matter and genome (for the compatibility gate)
    var resident = d0[i].w;
    if (fp.nch > 1u) {
        resident += d1[i].w;
    }
    if (fp.nch > 2u) {
        resident += d2[i].w;
    }
    let r0 = genome[i * 4u];
    let r3 = genome[i * 4u + 3u];
    let occupied = resident > 0.02;
    let kill = clamp(fp.combat * r3.x * fp.dt, 0.0, 1.0);
    var destroyed = 0.0;
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
                // overlap of each channel's moved box with this cell
                var c = vec3<f32>(0.0);
                let v0 = d0[j];
                c.x = v0.w * overlap(ofs + v0.xyz, s) * inv;
                if (fp.nch > 1u) {
                    let v1 = d1[j];
                    c.y = v1.w * overlap(ofs + v1.xyz, s) * inv;
                }
                if (fp.nch > 2u) {
                    let v2 = d2[j];
                    c.z = v2.w * overlap(ofs + v2.xyz, s) * inv;
                }
                var m = c.x + c.y + c.z;
                if (m > 1e-9) {
                    let s0 = genome[j * 4u];
                    let s3 = genome[j * 4u + 3u];
                    var foreign = 0.0;
                    if (occupied && genetic_distance(r0, r3, s0, s3) > fp.compat) {
                        foreign = 1.0;
                        // the resident's claws destroy incoming foreign matter
                        destroyed += m * kill;
                        c *= 1.0 - kill;
                        m *= 1.0 - kill;
                    }
                    acc += c;
                    // foreign claws win contested cells (assimilation)
                    let wt = m * (1.0 + fp.claw_bias * s3.x * foreign);
                    wsum += wt;
                    // weighted reservoir sampling
                    if (wt > 0.0 && rnd(&seed) * wsum < wt) {
                        chosen = j;
                    }
                } else {
                    acc += c;
                }
            }
        }
    }
    if (destroyed > 0.0) {
        food[i] += destroyed;
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
