// HDR + bloom -> tonemapped display color.
struct Params {
    exposure: f32,
    bloom: f32,
    encode_srgb: f32,
    vignette: f32,
};

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> params: Params;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let xy = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return o;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var c = textureSampleLevel(hdr, samp, in.uv, 0.0).rgb;
    let b = textureSampleLevel(bloom_tex, samp, in.uv, 0.0).rgb;
    c = c + b * params.bloom;
    c = aces(c * params.exposure);
    let q = in.uv - 0.5;
    c *= 1.0 - params.vignette * dot(q, q) * 1.6;
    if (params.encode_srgb > 0.5) {
        c = srgb(c);
    }
    // tiny dither against banding in dark gradients
    let n = fract(sin(dot(in.pos.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) - 0.5;
    c += vec3<f32>(n / 255.0);
    return vec4<f32>(c, 1.0);
}
