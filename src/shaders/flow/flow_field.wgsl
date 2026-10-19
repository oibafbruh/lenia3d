// Flow field and displacement for one channel; also packs the channel's mass
// into .w so reintegration reads a single buffer per channel.
struct C {
    c: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read> aff: array<f32>;
@group(0) @binding(1) var<storage, read> a0: array<f32>;
@group(0) @binding(2) var<storage, read> a1: array<f32>;
@group(0) @binding(3) var<storage, read> a2: array<f32>;
@group(0) @binding(4) var<storage, read> sense_a: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> genome: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> wall: array<f32>;
@group(0) @binding(7) var<storage, read_write> disp: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> fp: FlowParams;
@group(0) @binding(9) var<uniform> cp: C;
@group(0) @binding(10) var<storage, read> sense_b: array<vec2<f32>>;

fn asum(i: u32) -> f32 {
    var s = a0[i];
    if (fp.nch > 1u) {
        s += a1[i];
    }
    if (fp.nch > 2u) {
        s += a2[i];
    }
    return s;
}

fn clip1(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    return select(v, v / l, l > 1.0);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= fp.total) {
        return;
    }
    let n = fp.n;
    let p = cell_coord(i, n);
    let ix = vec2<u32>(cell_index(p + vec3<i32>(1, 0, 0), n), cell_index(p - vec3<i32>(1, 0, 0), n));
    let iy = vec2<u32>(cell_index(p + vec3<i32>(0, 1, 0), n), cell_index(p - vec3<i32>(0, 1, 0), n));
    let iz = vec2<u32>(cell_index(p + vec3<i32>(0, 0, 1), n), cell_index(p - vec3<i32>(0, 0, 1), n));

    let g_aff = vec3<f32>(aff[ix.x] - aff[ix.y], aff[iy.x] - aff[iy.y], aff[iz.x] - aff[iz.y]) * 0.5;
    let g_a = vec3<f32>(asum(ix.x) - asum(ix.y), asum(iy.x) - asum(iy.y), asum(iz.x) - asum(iz.y)) * 0.5;
    let alpha = clamp(pow(max(asum(i), 0.0) / fp.theta, fp.nexp), 0.0, 1.0);
    var f = (1.0 - alpha) * g_aff - alpha * g_a;

    let g2 = genome[i * 4u + 2u];
    let eye = g2.x;
    let ear = g2.y;
    if (eye != 0.0 || ear != 0.0) {
        let sc = fp.scale;
        let tune = genome[i * 4u + 1u].z;
        let h = (1.0 - tune) * sense_a[i].y + tune * sense_b[i].x;
        let ax = (sense_a[ix.x] - sense_a[ix.y]) * (0.5 * sc);
        let ay = (sense_a[iy.x] - sense_a[iy.y]) * (0.5 * sc);
        let az = (sense_a[iz.x] - sense_a[iz.y]) * (0.5 * sc);
        let bx = (sense_b[ix.x].x - sense_b[ix.y].x) * (0.5 * sc);
        let by = (sense_b[iy.x].x - sense_b[iy.y].x) * (0.5 * sc);
        let bz = (sense_b[iz.x].x - sense_b[iz.y].x) * (0.5 * sc);
        let g_food = vec3<f32>(ax.x, ay.x, az.x);
        let g_hear = (1.0 - tune) * vec3<f32>(ax.y, ay.y, az.y) + tune * vec3<f32>(bx, by, bz);
        f += fp.w_eye * eye * clip1(g_food / (max(sense_a[i].x * sc, 0.0) + 0.05));
        f += fp.w_ear * ear * clip1(g_hear / (max(h * sc, 0.0) + 0.05));
    }

    let lim = 1.0 - fp.s;
    var d = clamp(fp.dt * f, vec3<f32>(-lim), vec3<f32>(lim));
    // matter cannot move into walls
    if (d.x != 0.0 && wall[select(ix.y, ix.x, d.x > 0.0)] > 0.5) {
        d.x = 0.0;
    }
    if (d.y != 0.0 && wall[select(iy.y, iy.x, d.y > 0.0)] > 0.5) {
        d.y = 0.0;
    }
    if (d.z != 0.0 && wall[select(iz.y, iz.x, d.z > 0.0)] > 0.5) {
        d.z = 0.0;
    }
    var m = a0[i];
    if (cp.c == 1u) {
        m = a1[i];
    } else if (cp.c == 2u) {
        m = a2[i];
    }
    disp[i] = vec4<f32>(d, m);
}
