// Shared declarations for Flow mode shaders (prepended to each flow shader).

struct FlowParams {
    n: u32,
    total: u32,
    nch: u32,
    eco: u32,
    dt: f32,
    theta: f32,
    nexp: f32,
    s: f32,
    w_eye: f32,
    w_ear: f32,
    claw_bias: f32,
    scale: f32, // 1 / N^3 (inverse FFT normalization)
    eat: f32,
    social: f32,
    meta_base: f32,
    meta_trait: f32,
    toxicity: f32,
    recycle: f32,
    regrow: f32,
    mutation_rate: f32,
    mutation_size: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

// Genome: 16 floats per cell stored as 4 x vec4.
// [0..7] kernel weights h_k, [8] eye, [9] ear, [10] voice, [11] mouth,
// [12] claw, [13] pitch, [14] hue, [15] timbre.

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rnd(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed >> 8u) / 16777216.0;
}

fn wrapi(v: i32, n: i32) -> u32 {
    return u32(((v % n) + n) % n);
}

fn cell_index(p: vec3<i32>, n: u32) -> u32 {
    let ni = i32(n);
    return (wrapi(p.z, ni) * n + wrapi(p.y, ni)) * n + wrapi(p.x, ni);
}

fn cell_coord(i: u32, n: u32) -> vec3<i32> {
    return vec3<i32>(i32(i % n), i32((i / n) % n), i32(i / (n * n)));
}

fn clamp_genome(g: array<vec4<f32>, 4>) -> array<vec4<f32>, 4> {
    var o = g;
    o[0] = clamp(o[0], vec4<f32>(0.0), vec4<f32>(1.0));
    o[1] = clamp(o[1], vec4<f32>(0.0), vec4<f32>(1.0));
    o[2] = clamp(o[2], vec4<f32>(0.0, -1.0, 0.0, 0.0), vec4<f32>(1.0));
    let hue = fract(o[3].z);
    o[3] = clamp(o[3], vec4<f32>(0.0), vec4<f32>(1.0));
    o[3].z = hue;
    return o;
}
