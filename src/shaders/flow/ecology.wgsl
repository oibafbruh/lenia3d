// Eating, metabolism, food regrowth and per-cell mutation. Reads the
// reintegrated state (n*, genome_next) and writes the new state (a*, genome).
@group(0) @binding(0) var<storage, read> n0: array<f32>;
@group(0) @binding(1) var<storage, read> n1: array<f32>;
@group(0) @binding(2) var<storage, read> n2: array<f32>;
@group(0) @binding(3) var<storage, read_write> a0: array<f32>;
@group(0) @binding(4) var<storage, read_write> a1: array<f32>;
@group(0) @binding(5) var<storage, read_write> a2: array<f32>;
@group(0) @binding(6) var<storage, read> genome_next: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> genome: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> food: array<f32>;
@group(0) @binding(9) var<storage, read> fert: array<f32>;
@group(0) @binding(10) var<storage, read> sense_a: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read> wall: array<f32>;
@group(0) @binding(12) var<uniform> fp: FlowParams;
@group(0) @binding(13) var<storage, read> state: array<u32>;
@group(0) @binding(14) var<storage, read> sense_b: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= fp.total) {
        return;
    }
    var m = vec3<f32>(n0[i], 0.0, 0.0);
    if (fp.nch > 1u) {
        m.y = n1[i];
    }
    if (fp.nch > 2u) {
        m.z = n2[i];
    }
    var g = array<vec4<f32>, 4>(genome_next[i * 4u], genome_next[i * 4u + 1u], genome_next[i * 4u + 2u], genome_next[i * 4u + 3u]);
    var f = food[i];
    if (wall[i] > 0.5) {
        m = vec3<f32>(0.0);
        f = 0.0;
    } else if (fp.eco != 0u) {
        let dt = fp.dt;
        let fe = fert[i];
        let eye = g[2].x;
        let ear = g[2].y;
        let voice = g[2].z;
        let mouth = g[2].w;
        let claw = g[3].x;
        let msum = m.x + m.y + m.z;
        // eating
        let ss = max(hearing(sense_a[i].y, sense_b[i].x, g[1].z) * fp.scale, 0.0);
        let sat = clamp((msum - fp.eat_min) / max(1.0 - fp.eat_min, 1e-3), 0.0, 1.0);
        let eat = min(f, fp.eat * mouth * f * sat * (1.0 + fp.social * ss / (ss + 0.1)) * dt);
        f -= eat;
        if (msum > 1e-9) {
            m += eat * m / msum;
        } else {
            m.x += eat;
        }
        // metabolism
        let upkeep = (fp.meta_base + fp.meta_trait * (eye + abs(ear) + voice + mouth + claw)) * (1.0 + fp.toxicity * max(-fe, 0.0));
        let dec = min(m, upkeep * m * dt);
        m -= dec;
        f += fp.recycle * (dec.x + dec.y + dec.z);
        // regrowth
        if (fe > f) {
            f += fp.regrow * (fe - f) * dt;
        }
        // mutation
        if (fp.mutation_rate > 0.0 && msum > 0.01) {
            var seed = pcg(i * 9781u ^ pcg(state[0] + 0x9E3779B9u));
            if (rnd(&seed) < fp.mutation_rate) {
                let j = min(u32(rnd(&seed) * 16.0), 15u);
                let u1 = max(rnd(&seed), 1e-7);
                let u2 = rnd(&seed);
                let gauss = sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
                g[j / 4u][j % 4u] += fp.mutation_size * gauss;
                g = clamp_genome(g);
            }
        }
    }
    a0[i] = m.x;
    if (fp.nch > 1u) {
        a1[i] = m.y;
    }
    if (fp.nch > 2u) {
        a2[i] = m.z;
    }
    for (var k = 0u; k < 4u; k++) {
        genome[i * 4u + k] = g[k];
    }
    food[i] = f;
}
