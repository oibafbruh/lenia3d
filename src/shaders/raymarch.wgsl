// Volume raymarcher for the Lenia world.
// World box is [-1, 1]^3; the volume texture holds rgb = channels, a = wall.

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,     // xyz, w = time
    light_dir: vec4<f32>,   // xyz (normalized, towards the light), w = intensity
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
    core_col: vec4<f32>,
    bg_top: vec4<f32>,
    bg_bot: vec4<f32>,
    brush: vec4<f32>,       // xyz center, w radius (<= 0: hidden)
    p0: vec4<f32>,          // iso, core_iso, membrane opacity, density gain
    p1: vec4<f32>,          // emission, step scale, plane y, clip position
    p2: vec4<f32>,          // sss, shadow strength, ao strength, floor y
    ip: vec4<u32>,          // mode, n, channels, flags
    screen: vec4<f32>,      // width, height, frame, pixel angle
    p3: vec4<f32>,          // floor grid offset x, z (follow mode scrolling)
    p4: vec4<f32>,          // food haze opacity, organ scale, sound rings, -
    ip2: vec4<u32>,         // creature count, flags2, -, -
};

const F2_GENOME: u32 = 1u;
const F2_FOOD: u32 = 2u;
const F2_ORGANS: u32 = 4u;
const F2_SOUND: u32 = 8u;

struct Creature {
    center: vec3<f32>,
    radius: f32,
    forward: vec3<f32>,
    eye: f32,
    mouth: f32,
    claw: f32,
    ear: f32,
    voice: f32,
    hue: f32,
    level: f32,
    eat: f32,
    selected: f32,
};

const FLAG_FLOOR: u32 = 1u;
const FLAG_BOX: u32 = 2u;
const FLAG_COLORMAP: u32 = 4u;
const FLAG_CLIP_FLIP: u32 = 32u;
const FLAG_WALLS: u32 = 64u;
const FLAG_PLANE: u32 = 128u;
const BRICK: u32 = 8u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> bricks: array<vec2<f32>>;
@group(0) @binding(4) var aux: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> creatures: array<Creature>;

fn has2(flag: u32) -> bool {
    return (u.ip2.y & flag) != 0u;
}

// HSV (display space) to linear RGB.
fn hsv(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let q = abs(fract(vec3<f32>(h) + k) * 6.0 - 3.0);
    let c = v * mix(vec3<f32>(1.0), clamp(q - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
    return pow(c, vec3<f32>(2.2));
}

fn genome_hue(p: vec3<f32>) -> f32 {
    let a = textureSampleLevel(aux, samp, p * 0.5 + 0.5, 0.0);
    return atan2(a.z, a.y) / 6.2831853;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let xy = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
    var o: VsOut;
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.ndc = xy;
    return o;
}

fn has(flag: u32) -> bool {
    return (u.ip.w & flag) != 0u;
}

fn voxel() -> f32 {
    return 2.0 / f32(u.ip.y);
}

fn sample_vol(p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(vol, samp, p * 0.5 + 0.5, 0.0);
}

fn dens(s: vec4<f32>) -> f32 {
    return s.x + s.y + s.z;
}

fn density_at(p: vec3<f32>) -> f32 {
    return dens(sample_vol(p));
}

fn box_hit(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / rd;
    let ta = (bmin - ro) * inv;
    let tb = (bmax - ro) * inv;
    let t0 = min(ta, tb);
    let t1 = max(ta, tb);
    return vec2<f32>(max(max(t0.x, t0.y), t0.z), min(min(t1.x, t1.y), t1.z));
}

fn clip_axis() -> u32 {
    return (u.ip.w >> 3u) & 3u;
}

fn clip_min() -> vec3<f32> {
    var m = vec3<f32>(-1.0);
    let a = clip_axis();
    if (a != 0u && has(FLAG_CLIP_FLIP)) {
        m[a - 1u] = u.p1.w;
    }
    return m;
}

fn clip_max() -> vec3<f32> {
    var m = vec3<f32>(1.0);
    let a = clip_axis();
    if (a != 0u && !has(FLAG_CLIP_FLIP)) {
        m[a - 1u] = u.p1.w;
    }
    return m;
}

fn brick_coord(p: vec3<f32>) -> vec3<u32> {
    let nb = u.ip.y / BRICK;
    let c = clamp(vec3<i32>(floor((p * 0.5 + 0.5) * f32(nb))), vec3<i32>(0), vec3<i32>(i32(nb) - 1));
    return vec3<u32>(c);
}

fn brick_at(c: vec3<u32>) -> vec2<f32> {
    let nb = u.ip.y / BRICK;
    return bricks[(c.z * nb + c.y) * nb + c.x];
}

// Ray parameter where the ray leaves brick c.
fn brick_exit(ro: vec3<f32>, rd: vec3<f32>, c: vec3<u32>) -> f32 {
    let nb = f32(u.ip.y / BRICK);
    let lo = vec3<f32>(c) / nb * 2.0 - 1.0;
    let hi = (vec3<f32>(c) + 1.0) / nb * 2.0 - 1.0;
    let inv = 1.0 / rd;
    let ta = (lo - ro) * inv;
    let tb = (hi - ro) * inv;
    let t1 = max(ta, tb);
    return min(min(t1.x, t1.y), t1.z);
}

fn brick_empty(b: vec2<f32>, thr: f32) -> bool {
    return b.x < thr && (b.y < 0.5 || !has(FLAG_WALLS));
}

// Interleaved gradient noise for ray jitter.
fn ign(p: vec2<f32>) -> f32 {
    let f = u.screen.z * 5.588238;
    return fract(52.9829189 * fract(dot(p + f, vec2<f32>(0.06711056, 0.00583715))));
}

fn palette(x: f32) -> vec3<f32> {
    // stops are sRGB; convert to linear for shading
    return pow(palette_srgb(x), vec3<f32>(2.2));
}

fn palette_srgb(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);
    let c0 = vec3<f32>(0.03, 0.05, 0.18);
    let c1 = vec3<f32>(0.05, 0.42, 0.62);
    let c2 = vec3<f32>(0.30, 0.85, 0.65);
    let c3 = vec3<f32>(1.00, 0.80, 0.30);
    let c4 = vec3<f32>(1.00, 0.35, 0.45);
    if (t < 0.25) {
        return mix(c0, c1, t * 4.0);
    } else if (t < 0.5) {
        return mix(c1, c2, (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        return mix(c2, c3, (t - 0.5) * 4.0);
    }
    return mix(c3, c4, (t - 0.75) * 4.0);
}

fn channel_color(s: vec4<f32>) -> vec3<f32> {
    // cubed weights: the dominant channel's hue wins instead of mixing to gray
    let w = s.xyz * s.xyz * s.xyz;
    let sum = max(w.x + w.y + w.z, 1e-9);
    return (u.col0.rgb * w.x + u.col1.rgb * w.y + u.col2.rgb * w.z) / sum;
}

fn gradient(p: vec3<f32>, wall: bool) -> vec3<f32> {
    let h = voxel() * 1.25;
    var g: vec3<f32>;
    if (wall) {
        g = vec3<f32>(
            sample_vol(p + vec3<f32>(h, 0.0, 0.0)).w - sample_vol(p - vec3<f32>(h, 0.0, 0.0)).w,
            sample_vol(p + vec3<f32>(0.0, h, 0.0)).w - sample_vol(p - vec3<f32>(0.0, h, 0.0)).w,
            sample_vol(p + vec3<f32>(0.0, 0.0, h)).w - sample_vol(p - vec3<f32>(0.0, 0.0, h)).w);
    } else {
        g = vec3<f32>(
            density_at(p + vec3<f32>(h, 0.0, 0.0)) - density_at(p - vec3<f32>(h, 0.0, 0.0)),
            density_at(p + vec3<f32>(0.0, h, 0.0)) - density_at(p - vec3<f32>(0.0, h, 0.0)),
            density_at(p + vec3<f32>(0.0, 0.0, h)) - density_at(p - vec3<f32>(0.0, 0.0, h)));
    }
    return g;
}

fn surface_normal(p: vec3<f32>, rd: vec3<f32>, wall: bool) -> vec3<f32> {
    let g = gradient(p, wall);
    let l = length(g);
    if (l < 1e-6) {
        return -rd;
    }
    return -g / l;
}

// Soft shadow: transmittance from p towards the light through the volume.
fn shadow(p: vec3<f32>) -> f32 {
    let L = u.light_dir.xyz;
    let th = box_hit(p, L, clip_min(), clip_max());
    if (th.y <= 0.0 || th.x > th.y) {
        return 1.0;
    }
    let vx = voxel();
    let stp = vx * 1.5;
    var t = max(th.x, 0.0) + vx;
    var acc = 0.0;
    for (var i = 0; i < 160; i++) {
        if (t > th.y || acc > 8.0) {
            break;
        }
        let q = p + L * t;
        let c = brick_coord(q);
        if (brick_empty(brick_at(c), 0.01)) {
            t = brick_exit(p, L, c) + vx * 0.25;
            continue;
        }
        let s = sample_vol(q);
        var w = 0.0;
        if (has(FLAG_WALLS)) {
            w = s.w;
        }
        acc += (dens(s) + w * 6.0) * 1.5;
        t += stp;
    }
    return mix(1.0, exp(-acc * 0.45), u.p2.y);
}

fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let vx = voxel();
    var occ = 0.0;
    var d = vx * 1.5;
    for (var i = 1; i <= 4; i++) {
        let s = sample_vol(p + n * d);
        var w = 0.0;
        if (has(FLAG_WALLS)) {
            w = s.w;
        }
        occ += clamp(dens(s) + w, 0.0, 1.0) / f32(i);
        d *= 2.0;
    }
    return clamp(1.0 - occ * 0.5 * u.p2.z, 0.15, 1.0);
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
    return mix(vec3<f32>(0.22, 0.19, 0.17), vec3<f32>(0.38, 0.48, 0.65), dir.y * 0.5 + 0.5);
}

fn shade(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, base: vec3<f32>, sss: f32, gloss: f32) -> vec3<f32> {
    let L = u.light_dir.xyz;
    let li = vec3<f32>(1.0, 0.95, 0.88) * u.light_dir.w;
    let vx = voxel();
    let ndl = dot(n, L);
    let diff = max((ndl + 0.25) / 1.25, 0.0);
    let sh = shadow(p + n * vx * 1.5);
    let ao = ambient_occlusion(p, n);
    let h = normalize(L + v);
    let ndv = max(dot(n, v), 0.0);
    let fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    let spec = pow(max(dot(n, h), 0.0), 64.0) * gloss * (0.2 + fres) * 1.6;
    let rim = pow(1.0 - ndv, 3.0);
    // light scattered through thin parts of the body
    let thick = density_at(p - n * vx * 2.0) + density_at(p - n * vx * 4.0);
    let trans = exp(-thick * 1.2) * (0.35 + 0.65 * max(-ndl, 0.0) + 0.5 * pow(max(dot(-v, L), 0.0), 4.0));
    var c = base * (sky(n) * 0.45 * ao + diff * sh * li * 0.85);
    c += spec * sh * li;
    c += rim * ao * (base * 0.3 + sky(reflect(-v, n)) * 0.15);
    c += base * trans * sss * li * (0.3 + 0.4 * sh);
    return c;
}

fn refine(ro: vec3<f32>, rd: vec3<f32>, ta: f32, tb: f32, level: f32, wall: bool) -> f32 {
    var a = ta;
    var b = tb;
    for (var i = 0; i < 6; i++) {
        let m = 0.5 * (a + b);
        let s = sample_vol(ro + rd * m);
        var v = dens(s);
        if (wall) {
            v = s.w;
        }
        if (v >= level) {
            b = m;
        } else {
            a = m;
        }
    }
    return b;
}

struct March {
    col: vec3<f32>,
    trans: f32,
    t_hit: f32,
};

fn shade_wall(ro: vec3<f32>, rd: vec3<f32>, ta: f32, tb: f32) -> vec3<f32> {
    let th = refine(ro, rd, ta, tb, 0.5, true);
    let p = ro + rd * th;
    let n = surface_normal(p, rd, true);
    // subtle stone texture from position hash
    let q = floor((p * 0.5 + 0.5) * f32(u.ip.y));
    let hsh = fract(sin(dot(q, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
    let base = vec3<f32>(0.46, 0.44, 0.42) * (0.85 + 0.15 * hsh);
    return shade(p, n, -rd, base, 0.0, 0.25);
}

fn cut_face_normal() -> vec3<f32> {
    var n = vec3<f32>(0.0);
    let a = clip_axis();
    if (a != 0u) {
        n[a - 1u] = select(1.0, -1.0, has(FLAG_CLIP_FLIP));
    }
    return n;
}

fn march_organic(ro: vec3<f32>, rd: vec3<f32>, t0: f32, t1: f32, jitter: f32, entered_clip: bool) -> March {
    var m: March;
    m.col = vec3<f32>(0.0);
    m.trans = 1.0;
    m.t_hit = 1e9;
    let vx = voxel();
    let stp = vx * u.p1.y;
    let iso = u.p0.x;
    let core = u.p0.y;
    let mem_a = u.p0.z;
    let opaque_shell = core <= iso || mem_a >= 0.999;
    let skip_thr = min(iso, core);

    var t = t0;
    var prev_t = t;
    var s = sample_vol(ro + rd * t);
    var inside = dens(s) >= iso;
    if (inside && entered_clip) {
        // cross-section through the cut plane
        let d = dens(s);
        let n = cut_face_normal();
        var base = palette(d * 0.9 + 0.1);
        if (!has(FLAG_COLORMAP) && u.ip.z > 1u) {
            base = channel_color(s) * (0.4 + 0.8 * d);
        }
        m.col = shade(ro + rd * t, n, -rd, base, 0.0, 0.1);
        m.trans = 0.0;
        m.t_hit = t;
        return m;
    }
    t += stp * jitter;
    for (var i = 0; i < 4096; i++) {
        if (t > t1) {
            break;
        }
        let p = ro + rd * t;
        if (!inside) {
            let c = brick_coord(p);
            if (brick_empty(brick_at(c), skip_thr)) {
                t = brick_exit(ro, rd, c) + stp * jitter;
                prev_t = t;
                continue;
            }
        }
        s = sample_vol(p);
        let d = dens(s);
        if (has(FLAG_WALLS) && s.w >= 0.5) {
            m.col += m.trans * shade_wall(ro, rd, prev_t, t);
            m.t_hit = min(m.t_hit, t);
            m.trans = 0.0;
            break;
        }
        if (!inside && d >= iso) {
            let th = refine(ro, rd, prev_t, t, iso, false);
            let ph = ro + rd * th;
            m.t_hit = min(m.t_hit, th);
            let n = surface_normal(ph, rd, false);
            let sc = sample_vol(ph - n * vx);
            var base = channel_color(sc);
            if (has2(F2_GENOME)) {
                base = hsv(genome_hue(ph - n * vx), 0.6, 0.9);
            }
            if (has(FLAG_COLORMAP)) {
                base = palette(0.15 + 0.85 * density_at(ph - n * vx * 3.0));
            }
            let c = shade(ph, n, -rd, base, u.p2.x, 1.0);
            if (opaque_shell) {
                m.col += m.trans * c;
                m.trans = 0.0;
                break;
            }
            // thin-film skin: clear face-on, opaque at grazing angles
            let edge = pow(1.0 - abs(dot(n, rd)), 2.0);
            let a = clamp(mem_a * (0.3 + 1.2 * edge), 0.0, 1.0);
            m.col += m.trans * a * c;
            m.trans *= 1.0 - a;
            inside = true;
        } else if (inside && d < iso) {
            inside = false;
        }
        if (inside) {
            if (d >= core) {
                let th = refine(ro, rd, prev_t, t, core, false);
                let ph = ro + rd * th;
                let n = surface_normal(ph, rd, false);
                var base = u.core_col.rgb;
                if (u.ip.z > 1u) {
                    base = channel_color(sample_vol(ph)) * 1.15;
                }
                if (has2(F2_GENOME)) {
                    base = hsv(genome_hue(ph) + 0.06, 0.8, 1.0);
                }
                if (has(FLAG_COLORMAP)) {
                    base = palette(0.55 + 0.45 * density_at(ph - n * vx * 2.0));
                }
                m.col += m.trans * shade(ph, n, -rd, base, u.p2.x * 0.5, 0.8);
                m.trans = 0.0;
                break;
            }
            // jelly: absorption and faint inner glow
            let a = 1.0 - exp(-d * u.p0.w * u.p1.y * 0.05);
            // cytoplasm: skin color at low density shading towards the core color
            var core_tint = u.core_col.rgb;
            if (u.ip.z > 1u) {
                core_tint = channel_color(s) * 1.15;
            }
            var skin = channel_color(s);
            if (has2(F2_GENOME)) {
                let hue = genome_hue(p);
                skin = hsv(hue, 0.6, 0.9);
                core_tint = hsv(hue + 0.06, 0.8, 1.0);
            }
            var glow = mix(skin, core_tint, smoothstep(iso, core, d));
            if (has(FLAG_COLORMAP)) {
                glow = palette(d);
            }
            m.col += m.trans * a * glow * (0.1 + 0.6 * d);
            m.trans *= 1.0 - a;
            if (m.trans < 0.02) {
                break;
            }
        }
        prev_t = t;
        t += stp;
    }
    return m;
}

fn march_glow(ro: vec3<f32>, rd: vec3<f32>, t0: f32, t1: f32, jitter: f32, mip: bool) -> March {
    var m: March;
    m.col = vec3<f32>(0.0);
    m.trans = 1.0;
    m.t_hit = 1e9;
    let vx = voxel();
    let stp = vx * u.p1.y * 2.0;
    let step_vox = u.p1.y * 2.0;
    var t = t0 + stp * jitter;
    var prev_t = t;
    var maxd = 0.0;
    for (var i = 0; i < 4096; i++) {
        if (t > t1) {
            break;
        }
        let p = ro + rd * t;
        let c = brick_coord(p);
        if (brick_empty(brick_at(c), 0.004)) {
            t = brick_exit(ro, rd, c) + stp * jitter;
            prev_t = t;
            continue;
        }
        let s = sample_vol(p);
        if (has(FLAG_WALLS) && s.w >= 0.5) {
            if (!mip) {
                m.col += m.trans * shade_wall(ro, rd, prev_t, t) * 0.5;
                m.trans = 0.0;
            }
            m.t_hit = min(m.t_hit, t);
            break;
        }
        let d = dens(s);
        if (d > 0.002) {
            m.t_hit = min(m.t_hit, t);
            if (mip) {
                maxd = max(maxd, d);
            } else {
                var e = u.col0.rgb * s.x * s.x + u.col1.rgb * s.y * s.y + u.col2.rgb * s.z * s.z;
                if (has2(F2_GENOME)) {
                    e = hsv(genome_hue(p), 0.7, 1.0) * d * d;
                }
                if (has(FLAG_COLORMAP)) {
                    e = palette(d) * d;
                }
                let a = 1.0 - exp(-d * u.p0.w * step_vox * 0.06);
                m.col += m.trans * e * u.p1.x * step_vox * 0.12;
                m.trans *= 1.0 - a;
                if (m.trans < 0.01) {
                    break;
                }
            }
        }
        prev_t = t;
        t += stp;
    }
    if (mip) {
        m.col = palette(maxd) * (0.3 + maxd) * u.p1.x;
        m.trans = 1.0 - clamp(maxd * 3.0, 0.0, 1.0);
    }
    return m;
}

fn background(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var c = mix(u.bg_bot.rgb, u.bg_top.rgb, smoothstep(-0.4, 0.6, rd.y));
    if (has(FLAG_FLOOR) && rd.y < 0.0) {
        let fy = u.p2.w;
        let tf = (fy - ro.y) / rd.y;
        if (tf > 0.0) {
            let p = ro + rd * tf;
            let px = tf * u.screen.w;
            // grid lines every 1/4 world unit with pixel-width antialiasing
            let g = abs(fract((p.xz - u.p3.xy) * 4.0 + 0.5) - 0.5) / max(px * 4.0, 1e-4);
            let line = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
            let inside = step(abs(p.x), 1.0) * step(abs(p.z), 1.0);
            let sh = shadow(p);
            let fade = exp(-dot(p.xz, p.xz) * 0.08);
            var fc = vec3<f32>(0.03, 0.033, 0.04) * (0.3 + 0.7 * sh) * u.light_dir.w;
            fc += vec3<f32>(0.10, 0.14, 0.18) * line * (0.15 + 0.35 * inside) * fade;
            c = mix(c, fc, fade * 0.9);
        }
    }
    return c;
}

// Distance between the ray and segment ab; returns (distance, ray t).
fn seg_dist(ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec2<f32> {
    let d2 = b - a;
    let oa = ro - a;
    let a12 = dot(rd, d2);
    let a22 = dot(d2, d2);
    let b1 = dot(rd, oa);
    let b2 = dot(d2, oa);
    let den = a22 - a12 * a12;
    var s = 0.0;
    if (abs(den) > 1e-8) {
        s = clamp((b2 - a12 * b1) / den, 0.0, 1.0);
    }
    // re-solve s for clamped t if needed
    let t = max(s * a12 - b1, 0.0);
    s = clamp((dot(ro + rd * t - a, d2)) / a22, 0.0, 1.0);
    let dist = length(ro + rd * t - (a + d2 * s));
    return vec2<f32>(dist, t);
}

fn box_edges(col: vec3<f32>, ro: vec3<f32>, rd: vec3<f32>, t_hit: f32) -> vec3<f32> {
    var c = col;
    var a = 0.0;
    for (var e = 0u; e < 12u; e++) {
        let axis = e / 4u;
        let k = e % 4u;
        let s1 = select(-1.0, 1.0, (k & 1u) != 0u);
        let s2 = select(-1.0, 1.0, (k & 2u) != 0u);
        var pa = vec3<f32>(0.0);
        var pb = vec3<f32>(0.0);
        if (axis == 0u) {
            pa = vec3<f32>(-1.0, s1, s2);
            pb = vec3<f32>(1.0, s1, s2);
        } else if (axis == 1u) {
            pa = vec3<f32>(s1, -1.0, s2);
            pb = vec3<f32>(s1, 1.0, s2);
        } else {
            pa = vec3<f32>(s1, s2, -1.0);
            pb = vec3<f32>(s1, s2, 1.0);
        }
        let r = seg_dist(ro, rd, pa, pb);
        if (r.y < t_hit) {
            let w = r.y * u.screen.w;
            a = max(a, 1.0 - smoothstep(w * 0.4, w * 1.4, r.x));
        }
    }
    return mix(c, vec3<f32>(0.45, 0.55, 0.65), a * 0.18);
}

fn overlays(col: vec3<f32>, ro: vec3<f32>, rd: vec3<f32>, t_hit: f32) -> vec3<f32> {
    var c = col;
    let px = u.screen.w;
    if (has(FLAG_PLANE) && abs(rd.y) > 1e-5) {
        let tp = (u.p1.z - ro.y) / rd.y;
        if (tp > 0.0 && tp < t_hit) {
            let p = ro + rd * tp;
            if (abs(p.x) <= 1.0 && abs(p.z) <= 1.0) {
                let cells = f32(u.ip.y) / 16.0;
                let g = abs(fract(p.xz * 0.5 * cells + 0.5) - 0.5) / max(tp * px * 0.5 * cells, 1e-4);
                let line = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
                c = mix(c, vec3<f32>(0.35, 0.75, 1.0), 0.06 + line * 0.22);
                if (u.brush.w > 0.0) {
                    let dr = abs(length(p.xz - u.brush.xz) - u.brush.w) / max(tp * px, 1e-5);
                    c = mix(c, vec3<f32>(1.0, 0.9, 0.4), (1.0 - clamp(dr - 0.5, 0.0, 1.0)) * 0.9);
                }
            }
        }
    }
    if (u.brush.w > 0.0) {
        let oc = ro - u.brush.xyz;
        let b = dot(oc, rd);
        let cc = dot(oc, oc) - u.brush.w * u.brush.w;
        let h = b * b - cc;
        if (h > 0.0) {
            let ts = -b - sqrt(h);
            if (ts > 0.0 && ts < t_hit) {
                let n = normalize(ro + rd * ts - u.brush.xyz);
                let rim = pow(1.0 - abs(dot(n, rd)), 2.5);
                c = mix(c, vec3<f32>(1.0, 0.88, 0.45), 0.07 + 0.45 * rim);
            }
        }
    }
    return c;
}


// ---------------------------------------------------------------- organs

fn dot2(v: vec3<f32>) -> f32 {
    return dot(v, v);
}

fn sd_ellipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-6);
}

// Round cone from a (radius r1) to b (radius r2), after Inigo Quilez.
fn sd_round_cone(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r1: f32, r2: f32) -> f32 {
    let ba = b - a;
    let l2 = dot(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = dot(pa, ba);
    let z = y - l2;
    let x2 = dot2(pa * l2 - ba * y);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) {
        return sqrt(x2 + z2) * il2 - r2;
    }
    if (sign(y) * a2 * y2 < k) {
        return sqrt(x2 + y2) * il2 - r1;
    }
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

struct Frame {
    f: vec3<f32>,
    up: vec3<f32>,
    rt: vec3<f32>,
};

fn creature_frame(c: Creature) -> Frame {
    var fr: Frame;
    var f = c.forward;
    if (length(f) < 1e-4) {
        f = vec3<f32>(1.0, 0.0, 0.0);
    }
    f = normalize(f);
    var up0 = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(f, up0)) > 0.95) {
        up0 = vec3<f32>(0.0, 0.0, 1.0);
    }
    fr.f = f;
    fr.rt = normalize(cross(f, up0));
    fr.up = cross(fr.rt, f);
    return fr;
}

fn eye_center(c: Creature, fr: Frame, side: f32) -> vec3<f32> {
    return c.center + c.radius * (0.74 * fr.f + 0.40 * fr.up + 0.34 * side * fr.rt);
}

// Distance to the organs of one creature; .y = material
// (1 eye, 2 mouth, 3 claw, 4 ear).
fn organ_sdf(p: vec3<f32>, c: Creature, fr: Frame) -> vec2<f32> {
    let r = c.radius;
    var d = 1e9;
    var mat = 0.0;
    if (c.eye > 0.08) {
        let re = r * (0.10 + 0.16 * c.eye);
        for (var k = 0; k < 2; k++) {
            let side = select(-1.0, 1.0, k == 1);
            let dd = length(p - eye_center(c, fr, side)) - re;
            if (dd < d) {
                d = dd;
                mat = 1.0;
            }
        }
    }
    if (c.mouth > 0.08) {
        let open = 0.03 + 0.06 * c.mouth + 0.10 * c.eat * (0.5 + 0.5 * sin(u.cam_pos.w * 12.0));
        let mc = c.center + r * (0.93 * fr.f - 0.24 * fr.up);
        let q = p - mc;
        let lq = vec3<f32>(dot(q, fr.rt), dot(q, fr.up), dot(q, fr.f));
        let dd = sd_ellipsoid(lq, r * vec3<f32>(0.16 + 0.22 * c.mouth, open, 0.10));
        if (dd < d) {
            d = dd;
            mat = 2.0;
        }
    }
    if (c.claw > 0.08) {
        let len = r * (0.3 + 0.7 * c.claw);
        let pairs = select(1, 2, c.claw > 0.6);
        for (var j = 0; j < pairs; j++) {
            let low = f32(j) * 0.4;
            for (var k = 0; k < 2; k++) {
                let side = select(-1.0, 1.0, k == 1);
                let base = c.center + r * (0.45 * fr.f + 0.80 * side * fr.rt - (0.15 + low) * fr.up);
                let dir = normalize(0.75 * fr.f + 0.55 * side * fr.rt - (0.1 + low * 0.5) * fr.up);
                let dd = sd_round_cone(p, base, base + dir * len, r * 0.13, r * 0.025);
                if (dd < d) {
                    d = dd;
                    mat = 3.0;
                }
            }
        }
    }
    if (abs(c.ear) > 0.08) {
        let len = r * (0.25 + 0.45 * abs(c.ear));
        for (var k = 0; k < 2; k++) {
            let side = select(-1.0, 1.0, k == 1);
            let base = c.center + r * (0.78 * fr.up + 0.38 * side * fr.rt - 0.15 * fr.f);
            let dir = normalize(0.85 * fr.up + 0.45 * side * fr.rt - 0.2 * fr.f);
            let dd = sd_round_cone(p, base, base + dir * len, r * 0.11, r * 0.035);
            if (dd < d) {
                d = dd;
                mat = 4.0;
            }
        }
    }
    return vec2<f32>(d, mat);
}

struct OrganHit {
    t: f32,
    idx: i32,
};

fn trace_organs(ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> OrganHit {
    var hit: OrganHit;
    hit.t = t_max;
    hit.idx = -1;
    let count = min(u.ip2.x, 256u);
    for (var i = 0u; i < count; i++) {
        let c = creatures[i];
        if (c.eye + c.mouth + c.claw + abs(c.ear) < 0.1) {
            continue;
        }
        let bound = c.radius * 2.2;
        let oc = ro - c.center;
        let b = dot(oc, rd);
        let h = b * b - (dot(oc, oc) - bound * bound);
        if (h < 0.0) {
            continue;
        }
        let sq = sqrt(h);
        var t = max(-b - sq, 0.0);
        let t1 = min(-b + sq, hit.t);
        if (t >= t1) {
            continue;
        }
        let fr = creature_frame(c);
        let eps = max(c.radius * 0.004, 1e-4);
        for (var k = 0; k < 48; k++) {
            let sd = organ_sdf(ro + rd * t, c, fr).x;
            if (sd < eps) {
                hit.t = t;
                hit.idx = i32(i);
                break;
            }
            t += sd;
            if (t > t1) {
                break;
            }
        }
    }
    return hit;
}

fn shade_organ(ro: vec3<f32>, rd: vec3<f32>, hit: OrganHit) -> vec3<f32> {
    let c = creatures[hit.idx];
    let fr = creature_frame(c);
    let p = ro + rd * hit.t;
    let e = c.radius * 0.01;
    let n = normalize(vec3<f32>(
        organ_sdf(p + vec3<f32>(e, 0.0, 0.0), c, fr).x - organ_sdf(p - vec3<f32>(e, 0.0, 0.0), c, fr).x,
        organ_sdf(p + vec3<f32>(0.0, e, 0.0), c, fr).x - organ_sdf(p - vec3<f32>(0.0, e, 0.0), c, fr).x,
        organ_sdf(p + vec3<f32>(0.0, 0.0, e), c, fr).x - organ_sdf(p - vec3<f32>(0.0, 0.0, e), c, fr).x));
    let mat = organ_sdf(p, c, fr).y;
    var base = vec3<f32>(0.8);
    var gloss = 0.3;
    if (mat < 1.5) {
        // eye: white with an iris and a pupil looking between ahead and the viewer
        let side = select(-1.0, 1.0, dot(p - c.center, fr.rt) > 0.0);
        let ec = eye_center(c, fr, side);
        let look = normalize(fr.f * 0.6 + normalize(u.cam_pos.xyz - ec) * 0.8);
        let a = dot(normalize(p - ec), look);
        base = vec3<f32>(0.85);
        if (a > 0.80) {
            base = hsv(c.hue + 0.5, 0.7, 0.55);
        }
        if (a > 0.90) {
            base = vec3<f32>(0.01);
        }
        gloss = 1.0;
    } else if (mat < 2.5) {
        base = vec3<f32>(0.18, 0.01, 0.03);
        gloss = 0.6;
    } else if (mat < 3.5) {
        base = vec3<f32>(0.80, 0.74, 0.62);
        gloss = 0.8;
    } else {
        base = hsv(c.hue, 0.55, 0.85);
        gloss = 0.2;
    }
    let L = u.light_dir.xyz;
    let li = vec3<f32>(1.0, 0.95, 0.88) * u.light_dir.w;
    let v = -rd;
    let diff = max(dot(n, L), 0.0);
    let hv = normalize(L + v);
    let spec = pow(max(dot(n, hv), 0.0), 80.0) * gloss;
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.25;
    return base * (sky(n) * 0.5 + diff * li * 0.9) + spec * li + rim * sky(n);
}

// Expanding rings around calling creatures and a halo around the selection.
fn creature_halos(ro: vec3<f32>, rd: vec3<f32>, t_limit: f32) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    let count = min(u.ip2.x, 256u);
    for (var i = 0u; i < count; i++) {
        let c = creatures[i];
        let oc = ro - c.center;
        let tc = -dot(oc, rd);
        if (tc < 0.0) {
            continue;
        }
        let dc = length(oc + rd * tc);
        let px = tc * u.screen.w;
        if (has2(F2_SOUND) && c.level > 0.03 && tc < t_limit + c.radius * 3.0) {
            for (var k = 0; k < 2; k++) {
                let ph = fract(u.cam_pos.w * 0.7 + f32(i) * 0.37 + f32(k) * 0.5);
                let rr = c.radius * (1.1 + 2.4 * ph);
                let w = c.radius * 0.05 + px;
                let x = (dc - rr) / w;
                col += exp(-x * x) * (1.0 - ph) * c.level * hsv(c.hue, 0.35, 1.0) * 0.5 * u.p4.z;
            }
        }
        if (c.selected > 0.5) {
            let x = (dc - c.radius * 1.45) / (c.radius * 0.04 + px * 1.5);
            col += exp(-x * x) * vec3<f32>(1.0, 0.85, 0.4) * 0.7;
        }
    }
    return col;
}

// Faint green haze where there is food.
fn food_haze(ro: vec3<f32>, rd: vec3<f32>, t_limit: f32) -> f32 {
    let th = box_hit(ro, rd, vec3<f32>(-1.0), vec3<f32>(1.0));
    if (th.x >= th.y || th.y <= 0.0) {
        return 0.0;
    }
    let t0 = max(th.x, 0.0);
    let t1 = min(th.y, t_limit);
    if (t1 <= t0) {
        return 0.0;
    }
    let steps = 32;
    let dt = (t1 - t0) / f32(steps);
    var acc = 0.0;
    var t = t0 + dt * ign(vec2<f32>(rd.xy * 997.0));
    for (var k = 0; k < steps; k++) {
        acc += textureSampleLevel(aux, samp, (ro + rd * t) * 0.5 + 0.5, 0.0).x * dt;
        t += dt;
    }
    return 1.0 - exp(-acc * u.p4.x);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let near = u.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = u.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let ro = u.cam_pos.xyz;
    let rd = normalize(far.xyz / far.w - near.xyz / near.w);

    var col = vec3<f32>(0.0);
    var trans = 1.0;
    var t_hit = 1e9;
    let bmin = clip_min();
    let bmax = clip_max();
    let th = box_hit(ro, rd, bmin, bmax);
    if (th.x < th.y && th.y > 0.0) {
        let t0 = max(th.x, 0.0);
        // did the ray enter through the clip plane?
        var entered_clip = false;
        let a = clip_axis();
        if (a != 0u && th.x > 0.0) {
            let pe = ro + rd * th.x;
            entered_clip = abs(pe[a - 1u] - u.p1.w) < 1e-4;
        }
        let jitter = ign(in.pos.xy);
        var m: March;
        if (u.ip.x == 0u) {
            m = march_organic(ro, rd, t0, th.y, jitter, entered_clip);
        } else {
            m = march_glow(ro, rd, t0, th.y, jitter, u.ip.x == 2u);
        }
        col = m.col;
        trans = m.trans;
        t_hit = m.t_hit;
    }
    col += trans * background(ro, rd);
    if (has2(F2_FOOD) && u.p4.x > 0.0) {
        let haze = food_haze(ro, rd, t_hit);
        col = mix(col, vec3<f32>(0.05, 0.16, 0.03), haze * 0.7);
    }
    if (has2(F2_ORGANS) && u.ip2.x > 0u) {
        let oh = trace_organs(ro, rd, t_hit + 0.05);
        if (oh.idx >= 0) {
            col = shade_organ(ro, rd, oh);
            t_hit = min(t_hit, oh.t);
        }
    }
    if (u.ip2.x > 0u) {
        col += creature_halos(ro, rd, t_hit);
    }
    if (has(FLAG_BOX)) {
        col = box_edges(col, ro, rd, t_hit);
    }
    col = overlays(col, ro, rd, t_hit);
    return vec4<f32>(col, 1.0);
}
