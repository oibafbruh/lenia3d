// Copy channel buffers and the wall field into an rgba16f 3D texture,
// optionally smoothing with a 3x3x3 binomial filter (periodic wrap) so the
// raymarched surfaces don't show the voxel grid.
struct Params {
    n: u32,
    nch: u32,
    smooth_on: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read> a0: array<f32>;
@group(0) @binding(1) var<storage, read> a1: array<f32>;
@group(0) @binding(2) var<storage, read> a2: array<f32>;
@group(0) @binding(3) var<storage, read> wall: array<f32>;
@group(0) @binding(4) var out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params: Params;

fn fetch(i: u32) -> vec4<f32> {
    var v = vec4<f32>(a0[i], 0.0, 0.0, wall[i]);
    if (params.nch > 1u) {
        v.y = a1[i];
    }
    if (params.nch > 2u) {
        v.z = a2[i];
    }
    return v;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) {
        return;
    }
    if (params.smooth_on == 0u) {
        textureStore(out, gid, fetch((gid.z * n + gid.y) * n + gid.x));
        return;
    }
    var acc = vec4<f32>(0.0);
    let w = array<f32, 3>(0.25, 0.5, 0.25);
    for (var dz = 0u; dz < 3u; dz++) {
        let z = (gid.z + n + dz - 1u) % n;
        for (var dy = 0u; dy < 3u; dy++) {
            let y = (gid.y + n + dy - 1u) % n;
            for (var dx = 0u; dx < 3u; dx++) {
                let x = (gid.x + n + dx - 1u) % n;
                acc += fetch((z * n + y) * n + x) * (w[dz] * w[dy] * w[dx]);
            }
        }
    }
    // keep walls crisp
    acc.w = wall[(gid.z * n + gid.y) * n + gid.x];
    textureStore(out, gid, acc);
}
