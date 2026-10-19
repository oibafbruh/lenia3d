// Per 8^3 brick: max channel-sum density and max wall value, including a
// one-voxel apron so trilinear filtering never reads outside an empty brick.
const B: i32 = 8;

struct Params {
    n: u32,
    nb: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var vol: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nb = params.nb;
    if (gid.x >= nb || gid.y >= nb || gid.z >= nb) {
        return;
    }
    let n = i32(params.n);
    let lo = max(vec3<i32>(gid) * B - 1, vec3<i32>(0));
    let hi = min(vec3<i32>(gid) * B + B + 1, vec3<i32>(n));
    var md = 0.0;
    var mw = 0.0;
    for (var z = lo.z; z < hi.z; z++) {
        for (var y = lo.y; y < hi.y; y++) {
            for (var x = lo.x; x < hi.x; x++) {
                let v = textureLoad(vol, vec3<i32>(x, y, z), 0);
                md = max(md, v.x + v.y + v.z);
                mw = max(mw, v.w);
            }
        }
    }
    out[(gid.z * nb + gid.y) * nb + gid.x] = vec2<f32>(md, mw);
}
