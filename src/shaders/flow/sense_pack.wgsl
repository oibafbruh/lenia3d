// Pack the sensed sources into two complex fields:
//   sense_a = (food, low-band sound), sense_b = (high-band sound, claw scent).
// Voice gets louder (alarm call) when the cell smelled claws last step.
@group(0) @binding(0) var<storage, read> a0: array<f32>;
@group(0) @binding(1) var<storage, read> a1: array<f32>;
@group(0) @binding(2) var<storage, read> a2: array<f32>;
@group(0) @binding(3) var<storage, read> genome: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> food: array<f32>;
@group(0) @binding(5) var<storage, read_write> sense_a: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> sense_b: array<vec2<f32>>;
@group(0) @binding(7) var<uniform> fp: FlowParams;

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
    let g1 = genome[i * 4u + 1u];
    let g2 = genome[i * 4u + 2u];
    let g3 = genome[i * 4u + 3u];
    let claw = g3.x;
    let pitch = g3.y;
    let alarm = g1.w;
    // claw scent sensed last step (others' claws; a clawed cell ignores its own kind)
    let threat = max(sense_b[i].y * fp.scale, 0.0);
    let call = min(g2.z + alarm * min(threat * 8.0, 1.0) * (1.0 - claw), 1.5);
    let e = call * asum;
    sense_a[i] = vec2<f32>(food[i], e * (1.0 - pitch));
    sense_b[i] = vec2<f32>(e * pitch, claw * asum);
}
