// Bloom: 13-tap downsample and 9-tap tent upsample (Jimenez, SIGGRAPH 2014).
struct Params {
    texel: vec2<f32>, // 1 / source size
    first: f32,       // 1 on the first downsample (applies a soft clamp)
    _p: f32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

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

fn tap(uv: vec2<f32>, o: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, samp, uv + o * params.texel, 0.0).rgb;
}

@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let a = tap(uv, vec2<f32>(-2.0, 2.0));
    let b = tap(uv, vec2<f32>(0.0, 2.0));
    let c = tap(uv, vec2<f32>(2.0, 2.0));
    let d = tap(uv, vec2<f32>(-2.0, 0.0));
    let e = tap(uv, vec2<f32>(0.0, 0.0));
    let f = tap(uv, vec2<f32>(2.0, 0.0));
    let g = tap(uv, vec2<f32>(-2.0, -2.0));
    let h = tap(uv, vec2<f32>(0.0, -2.0));
    let i = tap(uv, vec2<f32>(2.0, -2.0));
    let j = tap(uv, vec2<f32>(-1.0, 1.0));
    let k = tap(uv, vec2<f32>(1.0, 1.0));
    let l = tap(uv, vec2<f32>(-1.0, -1.0));
    let m = tap(uv, vec2<f32>(1.0, -1.0));
    var col = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    if (params.first > 0.5) {
        // soft clamp against fireflies
        let lum = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
        col = col / (1.0 + max(lum - 4.0, 0.0) * 0.25);
    }
    return vec4<f32>(col, 1.0);
}

@fragment
fn fs_up(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    var col = tap(uv, vec2<f32>(0.0, 0.0)) * 4.0;
    col += (tap(uv, vec2<f32>(-1.0, 0.0)) + tap(uv, vec2<f32>(1.0, 0.0)) + tap(uv, vec2<f32>(0.0, -1.0)) + tap(uv, vec2<f32>(0.0, 1.0))) * 2.0;
    col += tap(uv, vec2<f32>(-1.0, -1.0)) + tap(uv, vec2<f32>(1.0, -1.0)) + tap(uv, vec2<f32>(-1.0, 1.0)) + tap(uv, vec2<f32>(1.0, 1.0));
    return vec4<f32>(col / 16.0, 1.0);
}
