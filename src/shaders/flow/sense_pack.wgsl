// Pack food and sound emission (voice * mass) into one complex field.
@group(0) @binding(0) var<storage, read> a0: array<f32>;
@group(0) @binding(1) var<storage, read> a1: array<f32>;
@group(0) @binding(2) var<storage, read> a2: array<f32>;
@group(0) @binding(3) var<storage, read> genome: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> food: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> fp: FlowParams;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + lid;
    if (i >= fp.total) {
        return;
    }
    var asum = a0[i];
    if (fp.nch > 1u) {
        asum += a1[i];
    }
    if (fp.nch > 2u) {
        asum += a2[i];
    }
    let voice = genome[i * 4u + 2u].z;
    out[i] = vec2<f32>(food[i], voice * asum);
}
