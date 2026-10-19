//! Flow mode: Flow Lenia with per-cell genomes, simple organs and an ecology.
//! The math is specified in docs/flow_lenia.md.

use super::fft::{Fft, FftBindings};
use super::params::LeniaParams;
use crate::gpu::linear_groups;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

pub const GENES: usize = 16;
pub type Genome = [f32; GENES];

/// Gene indices (genes 0..5 are the kernel weights h_k).
pub mod gene {
    /// Kernel weights h_0..h_5 are genes 0..5.
    pub const KERNELS: usize = 6;
    pub const TUNE: usize = 6;
    pub const ALARM: usize = 7;
    pub const EYE: usize = 8;
    pub const EAR: usize = 9;
    pub const VOICE: usize = 10;
    pub const MOUTH: usize = 11;
    pub const CLAW: usize = 12;
    pub const PITCH: usize = 13;
    pub const HUE: usize = 14;
    pub const TIMBRE: usize = 15;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FlowParams {
    /// Critical mass for crowding.
    pub theta: f32,
    /// Crowding exponent.
    pub n: f32,
    /// Reintegration half-size (cells).
    pub s: f32,
    /// Sensing radius in units of R.
    pub far: f32,
    pub w_eye: f32,
    pub w_ear: f32,
    pub claw_bias: f32,
    /// Genetic distance above which contact means combat instead of mixing.
    pub compat: f32,
    /// Fraction of foreign matter a clawed resident destroys per unit time.
    pub combat: f32,
}

impl Default for FlowParams {
    fn default() -> Self {
        Self { theta: 2.0, n: 2.0, s: 0.65, far: 2.5, w_eye: 6.0, w_ear: 2.0, claw_bias: 4.0, compat: 0.6, combat: 0.5 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EcologyParams {
    pub enabled: bool,
    pub eat: f32,
    /// Cells thinner than this cannot eat (keeps diffuse dust from growing).
    pub eat_min: f32,
    pub social: f32,
    pub meta_base: f32,
    pub meta_trait: f32,
    pub toxicity: f32,
    pub recycle: f32,
    pub regrow: f32,
    /// Default fertility of the ground (food level food regrows to).
    pub fertility: f32,
    pub mutation_rate: f32,
    pub mutation_size: f32,
    /// Creatures heavier than this bud off a mutated offspring.
    pub bud_mass: f32,
    /// Minimum age / cooldown (steps) between buddings.
    pub bud_age: f32,
    pub bud_mutation: f32,
}

impl Default for EcologyParams {
    fn default() -> Self {
        Self {
            enabled: true,
            eat: 1.0,
            eat_min: 0.3,
            social: 0.5,
            meta_base: 0.01,
            meta_trait: 0.01,
            toxicity: 2.0,
            recycle: 0.5,
            regrow: 0.05,
            fertility: 0.3,
            mutation_rate: 0.0,
            mutation_size: 0.05,
            bud_mass: 150.0,
            bud_age: 300.0,
            bud_mutation: 0.05,
        }
    }
}

/// Ranges for the organ genes of freshly seeded creatures.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TraitRanges {
    pub eye: [f32; 2],
    pub ear: [f32; 2],
    pub voice: [f32; 2],
    pub mouth: [f32; 2],
    pub claw: [f32; 2],
    pub alarm: [f32; 2],
    /// Number of founding "languages": founders share one of this many pitches
    /// (0 = every founder gets a random pitch).
    pub languages: u32,
}

impl Default for TraitRanges {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.3],
            ear: [-0.2, 0.2],
            voice: [0.0, 0.4],
            mouth: [0.3, 0.7],
            claw: [0.0, 0.2],
            alarm: [0.0, 0.3],
            languages: 3,
        }
    }
}

/// Organ genes of a genome (everything except the kernel weights).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Traits {
    pub eye: f32,
    pub ear: f32,
    pub voice: f32,
    pub mouth: f32,
    pub claw: f32,
    pub tune: f32,
    pub alarm: f32,
    pub pitch: f32,
    pub hue: f32,
    pub timbre: f32,
}

impl Traits {
    pub fn of(g: &Genome) -> Self {
        Self {
            eye: g[gene::EYE],
            ear: g[gene::EAR],
            voice: g[gene::VOICE],
            mouth: g[gene::MOUTH],
            claw: g[gene::CLAW],
            tune: g[gene::TUNE],
            alarm: g[gene::ALARM],
            pitch: g[gene::PITCH],
            hue: g[gene::HUE],
            timbre: g[gene::TIMBRE],
        }
    }
}

/// Genome with the rule's kernel weights and the given organ genes.
pub fn make_genome(rule: &LeniaParams, t: &Traits) -> Genome {
    let mut g = [0.0; GENES];
    for (k, kp) in rule.kernels.iter().take(gene::KERNELS).enumerate() {
        g[k] = kp.h.clamp(0.0, 1.0);
    }
    g[gene::EYE] = t.eye;
    g[gene::EAR] = t.ear;
    g[gene::VOICE] = t.voice;
    g[gene::MOUTH] = t.mouth;
    g[gene::CLAW] = t.claw;
    g[gene::TUNE] = t.tune;
    g[gene::ALARM] = t.alarm;
    g[gene::PITCH] = t.pitch;
    g[gene::HUE] = t.hue;
    g[gene::TIMBRE] = t.timbre;
    clamp_genome(&mut g);
    g
}

/// Random founder genome within `t`; `r` yields uniform numbers in [0, 1).
/// Founders of the same language share pitch, ear tuning and (roughly) hue.
pub fn random_genome(rule: &LeniaParams, t: &TraitRanges, mut r: impl FnMut() -> f32) -> Genome {
    let pick = |v: [f32; 2], r: &mut dyn FnMut() -> f32| v[0] + (v[1] - v[0]) * r();
    let eye = pick(t.eye, &mut r);
    let ear = pick(t.ear, &mut r);
    let voice = pick(t.voice, &mut r);
    let mouth = pick(t.mouth, &mut r);
    let claw = pick(t.claw, &mut r);
    let alarm = pick(t.alarm, &mut r);
    let (pitch, hue) = if t.languages > 0 {
        let lang = ((r() * t.languages as f32) as u32).min(t.languages - 1);
        let base = (lang as f32 + 0.5) / t.languages as f32;
        ((base + (r() - 0.5) * 0.04).clamp(0.0, 1.0), (base * 0.85 + (r() - 0.5) * 0.06).rem_euclid(1.0))
    } else {
        (r(), r())
    };
    let traits = Traits { eye, ear, voice, mouth, claw, tune: pitch, alarm, pitch, hue, timbre: r() };
    make_genome(rule, &traits)
}

/// Genetic distance used by the compatibility gate (mirrors the shader).
pub fn genetic_distance(a: &Genome, b: &Genome) -> f32 {
    (0..4).map(|k| (a[k] - b[k]).abs()).sum::<f32>()
        + 3.0 * (a[gene::PITCH] - b[gene::PITCH]).abs()
        + 0.5 * (a[gene::TIMBRE] - b[gene::TIMBRE]).abs()
}

/// Clamp genes to their valid ranges (hue wraps).
pub fn clamp_genome(g: &mut Genome) {
    for (j, v) in g.iter_mut().enumerate() {
        *v = match j {
            gene::EAR => v.clamp(-1.0, 1.0),
            gene::HUE => v.rem_euclid(1.0),
            _ => v.clamp(0.0, 1.0),
        };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FlowUniform {
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
    scale: f32,
    eat: f32,
    social: f32,
    meta_base: f32,
    meta_trait: f32,
    toxicity: f32,
    recycle: f32,
    regrow: f32,
    mutation_rate: f32,
    mutation_size: f32,
    compat: f32,
    combat: f32,
    eat_min: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AffinityParams {
    total: u32,
    part: u32,
    gn: u32,
    first: u32,
    m: f32,
    s: f32,
    h: f32,
    scale: f32,
    k: u32,
    _p: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GaussParams {
    n: u32,
    total: u32,
    _p0: [u32; 2],
    sigma: f32,
    rmax: f32,
    inv_sum: f32,
    _p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushParams {
    n: u32,
    mode: u32,
    seed: u32,
    extent: u32,
    origin: [i32; 3],
    radius: f32,
    center: [f32; 3],
    strength: f32,
    density: f32,
    nch: u32,
    default_fert: f32,
    _p0: f32,
    genes: Genome,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BudParams {
    n: u32,
    nch: u32,
    extent: u32,
    _p0: u32,
    origin: [i32; 3],
    radius: f32,
    center: [f32; 3],
    _p1: f32,
    offset: [i32; 3],
    _p2: u32,
    delta: Genome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowBrush {
    /// Seed a blob of matter carrying a genome.
    Seed = 0,
    /// Raise fertility (food source).
    Food = 1,
    /// Lower fertility (toxic zone).
    Toxic = 2,
    /// Reset fertility to the default.
    Neutral = 3,
    /// Overwrite the genome of existing matter.
    Paint = 4,
    /// Overwrite the genome everywhere (no matter check).
    Fill = 5,
}

fn flow_shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let full = format!("{}\n{}", include_str!("../shaders/flow/common.wgsl"), src);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(full.into()) });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub struct FlowPipelines {
    tick: wgpu::ComputePipeline,
    affinity: wgpu::ComputePipeline,
    sense_pack: wgpu::ComputePipeline,
    scale_real: wgpu::ComputePipeline,
    gauss: wgpu::ComputePipeline,
    flow_field: wgpu::ComputePipeline,
    reintegrate: wgpu::ComputePipeline,
    ecology: wgpu::ComputePipeline,
    brush: wgpu::ComputePipeline,
    bud: wgpu::ComputePipeline,
}

impl FlowPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            tick: flow_shader(device, "flow tick", include_str!("../shaders/flow/tick.wgsl")),
            affinity: flow_shader(device, "flow affinity", include_str!("../shaders/flow/affinity.wgsl")),
            sense_pack: flow_shader(device, "flow sense pack", include_str!("../shaders/flow/sense_pack.wgsl")),
            scale_real: flow_shader(device, "flow scale", include_str!("../shaders/flow/scale_real.wgsl")),
            gauss: flow_shader(device, "flow gauss", include_str!("../shaders/flow/gauss_gen.wgsl")),
            flow_field: flow_shader(device, "flow field", include_str!("../shaders/flow/flow_field.wgsl")),
            reintegrate: flow_shader(device, "flow reintegrate", include_str!("../shaders/flow/reintegrate.wgsl")),
            ecology: flow_shader(device, "flow ecology", include_str!("../shaders/flow/ecology.wgsl")),
            brush: flow_shader(device, "flow brush", include_str!("../shaders/flow/brush.wgsl")),
            bud: flow_shader(device, "flow bud", include_str!("../shaders/flow/bud.wgsl")),
        }
    }
}

fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn uniform<T: Pod>(device: &wgpu::Device, v: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("flow uniform"),
        contents: bytemuck::bytes_of(v),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn bind(device: &wgpu::Device, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries })
}

fn dispatch_linear(pass: &mut wgpu::ComputePass<'_>, total: u32) {
    let (x, y) = linear_groups(total);
    pass.dispatch_workgroups(x, y, 1);
}

/// References to the simulation buffers the flow passes read and write.
pub struct SimBuffers<'a> {
    pub n: u32,
    pub channels: &'a [wgpu::Buffer],
    pub accum: &'a [wgpu::Buffer],
    pub work: &'a [wgpu::Buffer],
    pub wall: &'a wgpu::Buffer,
    pub rule: &'a LeniaParams,
}

struct Plan {
    affinity: Vec<wgpu::BindGroup>,
    sense_pack: wgpu::BindGroup,
    scale_real: wgpu::BindGroup,
    scale_real_b: wgpu::BindGroup,
    gauss: wgpu::BindGroup,
    flow_field: Vec<wgpu::BindGroup>,
    reintegrate: wgpu::BindGroup,
    ecology: wgpu::BindGroup,
    tick: wgpu::BindGroup,
    sense_fft: FftBindings,
    sense_b_fft: FftBindings,
    kfar_fft: FftBindings,
}

pub struct FlowState {
    pub params: FlowParams,
    pub eco: EcologyParams,
    pub genome: wgpu::Buffer,
    genome_next: wgpu::Buffer,
    pub food: wgpu::Buffer,
    pub fert: wgpu::Buffer,
    pub sense: wgpu::Buffer,
    pub sense_b: wgpu::Buffer,
    kfar: wgpu::Buffer,
    next: Vec<wgpu::Buffer>,
    disp: Vec<wgpu::Buffer>,
    dummies: Vec<wgpu::Buffer>,
    state: wgpu::Buffer,
    uniform: wgpu::Buffer,
    pipes: FlowPipelines,
    plan: Option<Plan>,
    kfar_dirty: bool,
    kfar_key: (f32, f32),
    nch: usize,
}

impl FlowState {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, n: u32, nch: usize, params: FlowParams, eco: EcologyParams) -> Self {
        let cells = (n as u64).pow(3);
        let fert = storage(device, "fertility", cells * 4);
        let food = storage(device, "food", cells * 4);
        let ground = vec![eco.fertility; cells as usize];
        queue.write_buffer(&fert, 0, bytemuck::cast_slice(&ground));
        queue.write_buffer(&food, 0, bytemuck::cast_slice(&ground));
        Self {
            genome: storage(device, "genome", cells * 64),
            genome_next: storage(device, "genome next", cells * 64),
            food,
            fert,
            sense: storage(device, "senses a", cells * 8),
            sense_b: storage(device, "senses b", cells * 8),
            kfar: storage(device, "sense kernel", cells * 8),
            next: (0..nch).map(|_| storage(device, "mass next", cells * 4)).collect(),
            disp: (0..nch).map(|_| storage(device, "displacement", cells * 16)).collect(),
            dummies: (0..8).map(|_| storage(device, "dummy", 16)).collect(),
            state: storage(device, "flow state", 16),
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flow params"),
                size: std::mem::size_of::<FlowUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            pipes: FlowPipelines::new(device),
            plan: None,
            kfar_dirty: true,
            kfar_key: (0.0, 0.0),
            nch,
            params,
            eco,
        }
    }

    /// GPU memory used by the flow-specific buffers, in bytes.
    pub fn memory_estimate(n: u32, nch: usize) -> u64 {
        let cells = (n as u64).pow(3);
        cells * (64 * 2 + 4 * 2 + 8 * 3 + nch as u64 * (4 + 16))
    }

    /// Reallocate per-channel buffers after a channel count change.
    pub fn set_channels(&mut self, device: &wgpu::Device, n: u32, nch: usize) {
        if nch == self.nch {
            return;
        }
        let cells = (n as u64).pow(3);
        self.next = (0..nch).map(|_| storage(device, "mass next", cells * 4)).collect();
        self.disp = (0..nch).map(|_| storage(device, "displacement", cells * 16)).collect();
        self.nch = nch;
        self.plan = None;
    }

    pub fn write_uniform(&self, queue: &wgpu::Queue, n: u32, nch: usize, dt: f32) {
        let p = &self.params;
        let e = &self.eco;
        let u = FlowUniform {
            n,
            total: n * n * n,
            nch: nch as u32,
            eco: e.enabled as u32,
            dt,
            theta: p.theta.max(1e-3),
            nexp: p.n,
            s: p.s.clamp(0.3, 0.9),
            w_eye: p.w_eye,
            w_ear: p.w_ear,
            claw_bias: p.claw_bias,
            scale: 1.0 / (n as f32).powi(3),
            eat: e.eat,
            social: e.social,
            meta_base: e.meta_base,
            meta_trait: e.meta_trait,
            toxicity: e.toxicity,
            recycle: e.recycle,
            regrow: e.regrow,
            mutation_rate: e.mutation_rate,
            mutation_size: e.mutation_size,
            compat: p.compat,
            combat: p.combat,
            eat_min: e.eat_min.clamp(0.0, 0.95),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    /// Rebuild bind groups; call whenever the simulation's buffers or rule change.
    pub fn rebuild(&mut self, device: &wgpu::Device, fft: &Fft, sb: &SimBuffers<'_>) {
        let n = sb.n;
        let total = n * n * n;
        let rule = sb.rule;
        let nch = sb.channels.len();
        let ch = |c: usize| sb.channels.get(c).unwrap_or(&sb.channels[0]);
        let pipes = &self.pipes;
        let scale = 1.0 / total as f32;

        // affinity: one dispatch per kernel, grouped by target channel; channels
        // without kernels get a zero-fill dispatch (h = 0)
        let mut affinity = Vec::new();
        for c in 0..nch {
            let ks: Vec<usize> = (0..rule.kernels.len()).filter(|&k| rule.kernels[k].dst == c).collect();
            if ks.is_empty() {
                let u = uniform(
                    device,
                    &AffinityParams { total, part: 0, gn: 1, first: 1, m: 0.0, s: 1.0, h: 0.0, scale, k: 99, _p: [0; 3] },
                );
                affinity.push(bind(device, &pipes.affinity, &[&sb.work[0], &sb.accum[c], &self.genome, &u]));
            }
            for (j, &k) in ks.iter().enumerate() {
                let kp = &rule.kernels[k];
                let u = uniform(
                    device,
                    &AffinityParams {
                        total,
                        part: (k % 2) as u32,
                        gn: kp.gn,
                        first: (j == 0) as u32,
                        m: kp.m,
                        s: kp.s,
                        h: kp.h,
                        scale,
                        k: k as u32,
                        _p: [0; 3],
                    },
                );
                affinity.push(bind(device, &pipes.affinity, &[&sb.work[k / 2], &sb.accum[c], &self.genome, &u]));
            }
        }
        let sense_pack = bind(
            device,
            &pipes.sense_pack,
            &[ch(0), ch(1), ch(2), &self.genome, &self.food, &self.sense, &self.sense_b, &self.uniform],
        );
        let su = uniform(device, &[total, 0u32, 0, 0]);
        let scale_real = bind(device, &pipes.scale_real, &[&self.sense, &self.kfar, &su]);
        let scale_real_b = bind(device, &pipes.scale_real, &[&self.sense_b, &self.kfar, &su]);
        let far = (self.params.far * rule.R).clamp(2.0, n as f32 / 2.0 - 1.0);
        let sigma = far / 3.0;
        let mut sum = 0.0f64;
        let r = far.ceil() as i32;
        for z in -r..=r {
            for y in -r..=r {
                for x in -r..=r {
                    let d2 = (x * x + y * y + z * z) as f32;
                    if d2 < far * far {
                        sum += (-(d2 as f64) / (2.0 * (sigma as f64).powi(2))).exp();
                    }
                }
            }
        }
        let gu = uniform(device, &GaussParams { n, total, _p0: [0; 2], sigma, rmax: far, inv_sum: (1.0 / sum) as f32, _p1: 0.0 });
        let gauss = bind(device, &pipes.gauss, &[&self.kfar, &gu]);
        if self.kfar_key != (far, n as f32) {
            self.kfar_key = (far, n as f32);
            self.kfar_dirty = true;
        }

        let flow_field = (0..nch)
            .map(|c| {
                let cu = uniform(device, &[c as u32, 0, 0, 0]);
                bind(
                    device,
                    &pipes.flow_field,
                    &[&sb.accum[c], ch(0), ch(1), ch(2), &self.sense, &self.genome, sb.wall, &self.disp[c], &self.uniform, &cu, &self.sense_b],
                )
            })
            .collect();
        let d = |c: usize| self.disp.get(c).unwrap_or(&self.disp[0]);
        let nx = |c: usize| self.next.get(c).unwrap_or(&self.dummies[c]);
        let reintegrate = bind(
            device,
            &pipes.reintegrate,
            &[d(0), d(1), d(2), &self.genome, &self.genome_next, nx(0), nx(1), nx(2), sb.wall, &self.uniform, &self.state, &self.food],
        );
        let nr = |c: usize| self.next.get(c).unwrap_or(&self.next[0]);
        let aw = |c: usize| sb.channels.get(c).unwrap_or(&self.dummies[3 + c]);
        let ecology = bind(
            device,
            &pipes.ecology,
            &[
                nr(0),
                nr(1),
                nr(2),
                aw(0),
                aw(1),
                aw(2),
                &self.genome_next,
                &self.genome,
                &self.food,
                &self.fert,
                &self.sense,
                sb.wall,
                &self.uniform,
                &self.state,
                &self.sense_b,
            ],
        );
        let tick = bind(device, &pipes.tick, &[&self.state]);
        self.plan = Some(Plan {
            affinity,
            sense_pack,
            scale_real,
            scale_real_b,
            gauss,
            flow_field,
            reintegrate,
            ecology,
            tick,
            sense_fft: fft.bindings(device, &self.sense),
            sense_b_fft: fft.bindings(device, &self.sense_b),
            kfar_fft: fft.bindings(device, &self.kfar),
        });
    }

    pub fn needs_rebuild(&self) -> bool {
        self.plan.is_none()
    }

    pub fn invalidate(&mut self) {
        self.plan = None;
    }

    /// Everything after the potentials: affinity, senses, flow, reintegration, ecology.
    pub fn encode_after_potentials(&mut self, pass: &mut wgpu::ComputePass<'_>, fft: &Fft, total: u32) {
        let Some(plan) = &self.plan else { return };
        pass.set_pipeline(&self.pipes.tick);
        pass.set_bind_group(0, &plan.tick, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        if self.kfar_dirty {
            pass.set_pipeline(&self.pipes.gauss);
            pass.set_bind_group(0, &plan.gauss, &[]);
            dispatch_linear(pass, total);
            fft.run(pass, &plan.kfar_fft, false);
            self.kfar_dirty = false;
        }
        pass.set_pipeline(&self.pipes.affinity);
        for bg in &plan.affinity {
            pass.set_bind_group(0, bg, &[]);
            dispatch_linear(pass, total);
        }
        pass.set_pipeline(&self.pipes.sense_pack);
        pass.set_bind_group(0, &plan.sense_pack, &[]);
        dispatch_linear(pass, total);
        fft.run(pass, &plan.sense_fft, false);
        fft.run(pass, &plan.sense_b_fft, false);
        pass.set_pipeline(&self.pipes.scale_real);
        pass.set_bind_group(0, &plan.scale_real, &[]);
        dispatch_linear(pass, total);
        pass.set_bind_group(0, &plan.scale_real_b, &[]);
        dispatch_linear(pass, total);
        fft.run(pass, &plan.sense_fft, true);
        fft.run(pass, &plan.sense_b_fft, true);
        pass.set_pipeline(&self.pipes.flow_field);
        for bg in &plan.flow_field {
            pass.set_bind_group(0, bg, &[]);
            dispatch_linear(pass, total);
        }
        pass.set_pipeline(&self.pipes.reintegrate);
        pass.set_bind_group(0, &plan.reintegrate, &[]);
        dispatch_linear(pass, total);
        pass.set_pipeline(&self.pipes.ecology);
        pass.set_bind_group(0, &plan.ecology, &[]);
        dispatch_linear(pass, total);
    }

    /// Apply a flow brush (seed / food / toxic / neutral / paint / fill).
    #[allow(clippy::too_many_arguments)]
    pub fn brush(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        sb: &SimBuffers<'_>,
        mode: FlowBrush,
        center: [f32; 3],
        radius: f32,
        strength: f32,
        density: f32,
        genes: &Genome,
        seed: u32,
    ) {
        let n = sb.n;
        let (radius, center) = if mode == FlowBrush::Fill {
            // a sphere covering the whole cube
            (n as f32 * 0.87 + 1.0, [n as f32 / 2.0; 3])
        } else {
            (radius.max(0.5), center)
        };
        let extent = (2.0 * radius).ceil() as u32 + 2;
        let origin = center.map(|c| (c - radius).floor() as i32 - 1);
        let u = uniform(
            device,
            &BrushParams {
                n,
                mode: mode as u32,
                seed,
                extent,
                origin,
                radius,
                center,
                strength,
                density,
                nch: sb.channels.len() as u32,
                default_fert: self.eco.fertility,
                _p0: 0.0,
                genes: *genes,
            },
        );
        let aw = |c: usize| sb.channels.get(c).unwrap_or(&self.dummies[c]);
        let bg = bind(device, &self.pipes.brush, &[aw(0), aw(1), aw(2), &self.genome, &self.food, &self.fert, sb.wall, &u]);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipes.brush);
        pass.set_bind_group(0, &bg, &[]);
        let g = extent.div_ceil(4);
        pass.dispatch_workgroups(g, g, g);
    }

    /// Budding: move half the matter within `radius` of `center` by `offset`,
    /// giving the moved half the parent's genome plus `delta`.
    #[allow(clippy::too_many_arguments)]
    pub fn bud(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        sb: &SimBuffers<'_>,
        center: [f32; 3],
        radius: f32,
        offset: [i32; 3],
        delta: &Genome,
    ) {
        let extent = (2.0 * radius).ceil() as u32 + 2;
        let origin = center.map(|c| (c - radius).floor() as i32 - 1);
        let u = uniform(
            device,
            &BudParams {
                n: sb.n,
                nch: sb.channels.len() as u32,
                extent,
                _p0: 0,
                origin,
                radius,
                center,
                _p1: 0.0,
                offset,
                _p2: 0,
                delta: *delta,
            },
        );
        let aw = |c: usize| sb.channels.get(c).unwrap_or(&self.dummies[c]);
        let bg = bind(device, &self.pipes.bud, &[aw(0), aw(1), aw(2), &self.genome, sb.wall, &u]);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipes.bud);
        pass.set_bind_group(0, &bg, &[]);
        let g = extent.div_ceil(4);
        pass.dispatch_workgroups(g, g, g);
    }

    /// Reset fertility to the default everywhere.
    pub fn reset_fertility(&self, queue: &wgpu::Queue, n: u32) {
        let ground = vec![self.eco.fertility; (n as usize).pow(3)];
        queue.write_buffer(&self.fert, 0, bytemuck::cast_slice(&ground));
    }

    /// Refill food up to the fertility map.
    pub fn reset_food(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.fert, 0, &self.food, 0, self.fert.size());
    }

    pub fn genome_next_scratch(&self) -> &wgpu::Buffer {
        &self.genome_next
    }
}
