// Periodic translation of a field: dst[p] = src[p - shift].
struct Params {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    shift: vec3<i32>,
    _p3: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) {
        return;
    }
    let ni = i32(n);
    let s = vec3<u32>(((vec3<i32>(gid) - params.shift) % ni + ni) % ni);
    dst[(gid.z * n + gid.y) * n + gid.x] = src[(s.z * n + s.y) * n + s.x];
}
