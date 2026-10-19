//! GPU Lenia simulation: multi-channel, multi-kernel, FFT-based.
//!
//! Per step:
//! 1. pack channel pairs into complex fields and FFT them,
//! 2. for every pair of kernels multiply with the precomputed kernel spectra
//!    (two kernels share one complex inverse FFT),
//! 3. inverse FFT, then apply growth and the Euler update.

pub mod chan;
pub mod fft;
pub mod flow;
pub mod params;
pub mod preset;

use crate::gpu::{linear_groups, read_buffer_f32};
use bytemuck::{Pod, Zeroable};
use fft::{Fft, FftBindings};
pub use params::{KernelParams, LeniaParams};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    total: u32,
    has_b: u32,
    _p: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct KernelGenParams {
    n: u32,
    has2: u32,
    nb1: u32,
    nb2: u32,
    kn1: u32,
    kn2: u32,
    total: u32,
    _p0: u32,
    rad1: f32,
    rad2: f32,
    inv_sum1: f32,
    inv_sum2: f32,
    b1: [f32; 4],
    b2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MultiplyParams {
    n: u32,
    total: u32,
    part1: u32,
    part2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrowthParams {
    total: u32,
    part: u32,
    gn: u32,
    flags: u32,
    m: f32,
    s: f32,
    h: f32,
    scale: f32,
    dt: f32,
    _p: [f32; 3],
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
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StampParams {
    n: u32,
    _p: [u32; 3],
    dims: [u32; 3],
    _p3: u32,
    origin: [i32; 3],
    _p4: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceParams {
    total: u32,
    out_offset: u32,
    n: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RollParams {
    n: u32,
    stride: u32,
    _p: [u32; 2],
    shift: [i32; 3],
    _p3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TrackParams {
    n: u32,
    m: u32,
    f: u32,
    nch: u32,
    has_genome: u32,
    _p: [u32; 3],
    scale: f32,
    _q: [f32; 3],
}

/// Floats per coarse block in the tracking buffer.
pub const TRACK_FLOATS: usize = 12;

/// Per-channel statistics from the moments reduction.
#[derive(Clone, Debug, Default)]
pub struct ChannelStats {
    pub mass: f64,
    /// Periodic center of mass in cell coordinates (x, y, z), if any mass.
    pub center: Option<[f64; 3]>,
}

/// Combine the 8-float partial sums of each workgroup into per-channel stats.
pub fn stats_from_partials(vals: &[f32], groups: u32, n: u32) -> Vec<ChannelStats> {
    vals.chunks(groups as usize * 8)
        .map(|ch| {
            let mut s = [0f64; 7];
            for g in ch.chunks_exact(8) {
                for k in 0..7 {
                    s[k] += g[k] as f64;
                }
            }
            let center = (s[0] > 1e-6).then(|| {
                let ang = |c: f64, si: f64| {
                    let a = si.atan2(c) / std::f64::consts::TAU * n as f64;
                    a.rem_euclid(n as f64)
                };
                [ang(s[1], s[2]), ang(s[3], s[4]), ang(s[5], s[6])]
            });
            ChannelStats { mass: s[0], center }
        })
        .collect()
}

/// Mass-weighted periodic center over all channels.
pub fn combined_center(stats: &[ChannelStats], n: u32) -> Option<[f64; 3]> {
    let mut acc = [(0f64, 0f64); 3];
    let mut total = 0.0;
    for s in stats {
        if let Some(c) = s.center {
            for k in 0..3 {
                let a = c[k] / n as f64 * std::f64::consts::TAU;
                acc[k].0 += s.mass * a.cos();
                acc[k].1 += s.mass * a.sin();
            }
            total += s.mass;
        }
    }
    (total > 1e-6).then(|| acc.map(|(c, s)| (s.atan2(c) / std::f64::consts::TAU * n as f64).rem_euclid(n as f64)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushMode {
    Noise = 0,
    Add = 1,
    Erase = 2,
    Wall = 3,
    Unwall = 4,
    /// Soup seeding blob (see `World::seed_soup`); strength = fill probability.
    Seed = 5,
}

struct Pipelines {
    pack: wgpu::ComputePipeline,
    kernel_gen: wgpu::ComputePipeline,
    multiply: wgpu::ComputePipeline,
    growth: wgpu::ComputePipeline,
    brush: wgpu::ComputePipeline,
    stamp: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    roll: wgpu::ComputePipeline,
    track: wgpu::ComputePipeline,
}

fn compute_pipeline(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            pack: compute_pipeline(device, "pack", include_str!("../shaders/pack.wgsl")),
            kernel_gen: compute_pipeline(device, "kernel_gen", include_str!("../shaders/kernel_gen.wgsl")),
            multiply: compute_pipeline(device, "multiply", include_str!("../shaders/multiply.wgsl")),
            growth: compute_pipeline(device, "growth", include_str!("../shaders/growth.wgsl")),
            brush: compute_pipeline(device, "brush", include_str!("../shaders/brush.wgsl")),
            stamp: compute_pipeline(device, "stamp", include_str!("../shaders/stamp.wgsl")),
            reduce: compute_pipeline(device, "reduce", include_str!("../shaders/reduce.wgsl")),
            roll: compute_pipeline(device, "roll", include_str!("../shaders/roll.wgsl")),
            track: compute_pipeline(device, "track", include_str!("../shaders/flow/track.wgsl")),
        }
    }
}

/// Bind groups and uniforms for one simulation step, rebuilt when params change.
struct StepPlan {
    pack: Vec<wgpu::BindGroup>,
    packed_fft: Vec<FftBindings>,
    multiply: Vec<wgpu::BindGroup>,
    work_fft: Vec<FftBindings>,
    growth: Vec<wgpu::BindGroup>,
    kernel_gen: Vec<wgpu::BindGroup>,
    kspec_fft: Vec<FftBindings>,
    reduce: Vec<wgpu::BindGroup>,
}

pub struct Simulation {
    pub n: u32,
    params: LeniaParams,
    pub channels: Vec<wgpu::Buffer>,
    accum: Vec<wgpu::Buffer>,
    pub wall: wgpu::Buffer,
    packed: Vec<wgpu::Buffer>,
    kspec: Vec<wgpu::Buffer>,
    work: Vec<wgpu::Buffer>,
    reduce_out: wgpu::Buffer,
    reduce_groups: u32,
    fft: Fft,
    pipes: Pipelines,
    plan: StepPlan,
    kernels_dirty: bool,
    pub generation: u64,
    pub time: f64,
    layout_id: u64,
    /// Present in Flow mode (Flow Lenia with genomes and ecology).
    pub flow: Option<flow::FlowState>,
    track_out: wgpu::Buffer,
    track_dummy: wgpu::Buffer,
}

fn next_layout_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn field_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn uniform<T: Pod>(device: &wgpu::Device, v: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniform"),
        contents: bytemuck::bytes_of(v),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn bind_group(device: &wgpu::Device, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
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

/// Estimated GPU memory use in bytes for a grid size and rule.
pub fn memory_estimate(n: u32, params: &LeniaParams, flow: bool) -> u64 {
    let cells = (n as u64).pow(3);
    let c = params.channels as u64;
    let pairs = params.kernels.len().div_ceil(2) as u64;
    let packed = (params.channels as u64).div_ceil(2);
    // channels + accumulators + wall + complex buffers + render volumes
    let base = cells * 4 * (2 * c + 1) + cells * 8 * (packed + 2 * pairs) + cells * 16;
    base + if flow { flow::FlowState::memory_estimate(n, params.channels) } else { 0 }
}

/// Coarse tracking grid size for a world size.
pub fn track_grid(n: u32) -> (u32, u32) {
    let m = n.min(32);
    (m, n / m)
}

impl Simulation {
    pub fn new(device: &wgpu::Device, n: u32, mut params: LeniaParams) -> Self {
        params.sanitize(n);
        let cells = (n as u64).pow(3);
        let channels: Vec<_> = (0..params.channels).map(|_| field_buffer(device, "channel", cells * 4)).collect();
        let accum: Vec<_> = (0..params.channels).map(|_| field_buffer(device, "accum", cells * 4)).collect();
        let wall = field_buffer(device, "wall", cells * 4);
        let reduce_groups = (cells as u32).div_ceil(4096);
        let reduce_out = field_buffer(device, "reduce", (reduce_groups as u64 * 32 * 3).max(32));
        let fft = Fft::new(device, n);
        let pipes = Pipelines::new(device);
        let mut sim = Self {
            n,
            params,
            channels,
            accum,
            wall,
            packed: Vec::new(),
            kspec: Vec::new(),
            work: Vec::new(),
            reduce_out,
            reduce_groups,
            fft,
            pipes,
            plan: StepPlan {
                pack: vec![],
                packed_fft: vec![],
                multiply: vec![],
                work_fft: vec![],
                growth: vec![],
                kernel_gen: vec![],
                kspec_fft: vec![],
                reduce: vec![],
            },
            kernels_dirty: true,
            generation: 0,
            time: 0.0,
            layout_id: next_layout_id(),
            flow: None,
            track_out: field_buffer(device, "track", (track_grid(n).0 as u64).pow(3) * TRACK_FLOATS as u64 * 4),
            track_dummy: field_buffer(device, "track dummy", 64),
        };
        sim.allocate_complex(device);
        sim.rebuild_plan(device);
        sim
    }

    pub fn params(&self) -> &LeniaParams {
        &self.params
    }

    /// Changes whenever the channel buffers are reallocated (renderers rebind).
    pub fn layout_id(&self) -> u64 {
        self.layout_id
    }

    pub fn cells(&self) -> u32 {
        self.n * self.n * self.n
    }

    fn allocate_complex(&mut self, device: &wgpu::Device) {
        let bytes = self.cells() as u64 * 8;
        let packed = self.params.channels.div_ceil(2);
        let pairs = self.params.kernels.len().div_ceil(2);
        self.packed = (0..packed).map(|_| field_buffer(device, "packed spectrum", bytes)).collect();
        self.kspec = (0..pairs).map(|_| field_buffer(device, "kernel spectrum", bytes)).collect();
        self.work = (0..pairs).map(|_| field_buffer(device, "potential", bytes)).collect();
    }

    /// Change the rule. Keeps the world when the channel count is unchanged.
    pub fn set_params(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mut params: LeniaParams) {
        params.sanitize(self.n);
        if params == self.params {
            return;
        }
        let shape_changed = params.kernels_shape_changed(&self.params);
        let pairs_changed = params.kernels.len().div_ceil(2) != self.params.kernels.len().div_ceil(2);
        let channels_changed = params.channels != self.params.channels;
        if channels_changed {
            let cells = self.cells() as u64;
            let old = std::mem::take(&mut self.channels);
            self.channels = (0..params.channels).map(|_| field_buffer(device, "channel", cells * 4)).collect();
            self.accum = (0..params.channels).map(|_| field_buffer(device, "accum", cells * 4)).collect();
            let mut enc = device.create_command_encoder(&Default::default());
            for (dst, src) in self.channels.iter().zip(&old) {
                enc.copy_buffer_to_buffer(src, 0, dst, 0, cells * 4);
            }
            queue.submit([enc.finish()]);
            self.layout_id = next_layout_id();
            if let Some(f) = self.flow.as_mut() {
                f.set_channels(device, self.n, params.channels);
            }
        }
        self.params = params;
        if channels_changed || pairs_changed {
            self.allocate_complex(device);
            self.kernels_dirty = true;
        }
        if shape_changed {
            self.kernels_dirty = true;
        }
        self.rebuild_plan(device);
    }

    fn rebuild_plan(&mut self, device: &wgpu::Device) {
        let n = self.n;
        let total = self.cells();
        let p = &self.params;
        let pipes = &self.pipes;

        let pack = (0..self.packed.len())
            .map(|i| {
                let a = &self.channels[2 * i];
                let b = self.channels.get(2 * i + 1);
                let u = uniform(device, &PackParams { total, has_b: b.is_some() as u32, _p: [0; 2] });
                bind_group(device, &pipes.pack, &[a, b.unwrap_or(a), &self.packed[i], &u])
            })
            .collect();
        let packed_fft = self.packed.iter().map(|b| self.fft.bindings(device, b)).collect();

        let mut multiply = Vec::new();
        let mut kernel_gen = Vec::new();
        for q in 0..self.work.len() {
            let k1 = &p.kernels[2 * q];
            let k2 = p.kernels.get(2 * q + 1);
            let k2s = k2.unwrap_or(k1);
            let u = uniform(
                device,
                &MultiplyParams { n, total, part1: (k1.src % 2) as u32, part2: (k2s.src % 2) as u32 },
            );
            multiply.push(bind_group(
                device,
                &pipes.multiply,
                &[&self.packed[k1.src / 2], &self.packed[k2s.src / 2], &self.kspec[q], &self.work[q], &u],
            ));

            let r1 = p.R * k1.r;
            let r2 = p.R * k2s.r;
            let sum1 = params::kernel_sum(r1 as f64, &k1.b, k1.kn).max(1e-12);
            let sum2 = params::kernel_sum(r2 as f64, &k2s.b, k2s.kn).max(1e-12);
            let mut b1 = [0.0; 4];
            let mut b2 = [0.0; 4];
            b1[..k1.b.len()].copy_from_slice(&k1.b);
            b2[..k2s.b.len()].copy_from_slice(&k2s.b);
            let u = uniform(
                device,
                &KernelGenParams {
                    n,
                    has2: k2.is_some() as u32,
                    nb1: k1.b.len() as u32,
                    nb2: k2s.b.len() as u32,
                    kn1: k1.kn,
                    kn2: k2s.kn,
                    total,
                    _p0: 0,
                    rad1: r1,
                    rad2: r2,
                    inv_sum1: (1.0 / sum1) as f32,
                    inv_sum2: (1.0 / sum2) as f32,
                    b1,
                    b2,
                },
            );
            kernel_gen.push(bind_group(device, &pipes.kernel_gen, &[&self.kspec[q], &u]));
        }
        let work_fft = self.work.iter().map(|b| self.fft.bindings(device, b)).collect();
        let kspec_fft = self.kspec.iter().map(|b| self.fft.bindings(device, b)).collect();

        // growth: one dispatch per kernel, grouped by target channel so that
        // the last kernel of a channel performs the update
        let mut growth = Vec::new();
        let dt = p.dt();
        let scale = 1.0 / total as f32;
        for c in 0..p.channels {
            let ks: Vec<usize> = (0..p.kernels.len()).filter(|&k| p.kernels[k].dst == c).collect();
            for (j, &k) in ks.iter().enumerate() {
                let kp = &p.kernels[k];
                let flags = (j == 0) as u32 | (((j + 1 == ks.len()) as u32) << 1);
                let u = uniform(
                    device,
                    &GrowthParams {
                        total,
                        part: (k % 2) as u32,
                        gn: kp.gn,
                        flags,
                        m: kp.m,
                        s: kp.s,
                        h: kp.h,
                        scale,
                        dt,
                        _p: [0.0; 3],
                    },
                );
                growth.push(bind_group(
                    device,
                    &pipes.growth,
                    &[&self.work[k / 2], &self.accum[c], &self.channels[c], &self.wall, &u],
                ));
            }
        }

        let reduce = self
            .channels
            .iter()
            .enumerate()
            .map(|(c, buf)| {
                let u = uniform(
                    device,
                    &ReduceParams { total, out_offset: c as u32 * self.reduce_groups, n, _p: 0 },
                );
                bind_group(device, &pipes.reduce, &[buf, &self.reduce_out, &u])
            })
            .collect();

        self.plan = StepPlan { pack, packed_fft, multiply, work_fft, growth, kernel_gen, kspec_fft, reduce };
        if let Some(f) = self.flow.as_mut() {
            let sb = flow::SimBuffers {
                n: self.n,
                channels: &self.channels,
                accum: &self.accum,
                work: &self.work,
                wall: &self.wall,
                rule: &self.params,
            };
            f.rebuild(device, &self.fft, &sb);
        }
    }

    /// Record `steps` simulation steps into `encoder`.
    pub fn encode_steps(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32) {
        let total = self.cells();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lenia step"), timestamp_writes: None });
        if self.kernels_dirty {
            for (bg, fb) in self.plan.kernel_gen.iter().zip(&self.plan.kspec_fft) {
                pass.set_pipeline(&self.pipes.kernel_gen);
                pass.set_bind_group(0, bg, &[]);
                dispatch_linear(&mut pass, total);
                self.fft.run(&mut pass, fb, false);
            }
            self.kernels_dirty = false;
        }
        for _ in 0..steps {
            for (bg, fb) in self.plan.pack.iter().zip(&self.plan.packed_fft) {
                pass.set_pipeline(&self.pipes.pack);
                pass.set_bind_group(0, bg, &[]);
                dispatch_linear(&mut pass, total);
                self.fft.run(&mut pass, fb, false);
            }
            for (bg, fb) in self.plan.multiply.iter().zip(&self.plan.work_fft) {
                pass.set_pipeline(&self.pipes.multiply);
                pass.set_bind_group(0, bg, &[]);
                dispatch_linear(&mut pass, total);
                self.fft.run(&mut pass, fb, true);
            }
            match self.flow.as_mut() {
                Some(f) => f.encode_after_potentials(&mut pass, &self.fft, total),
                None => {
                    pass.set_pipeline(&self.pipes.growth);
                    for bg in &self.plan.growth {
                        pass.set_bind_group(0, bg, &[]);
                        dispatch_linear(&mut pass, total);
                    }
                }
            }
        }
        self.generation += steps as u64;
        self.time += steps as f64 * self.params.dt() as f64;
    }

    /// Periodically translate the whole world (channels, walls, and in Flow
    /// mode genomes, food and fertility) by `shift` cells.
    pub fn encode_roll(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, shift: [i32; 3]) {
        let n = self.n;
        let bytes = self.cells() as u64 * 4;
        let groups = n.div_ceil(4);
        let u1 = uniform(device, &RollParams { n, stride: 1, _p: [0; 2], shift, _p3: 0 });
        let mut fields: Vec<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u64)> = Vec::new();
        let tmp = &self.accum[0];
        for c in &self.channels {
            fields.push((c, tmp, &u1, bytes));
        }
        fields.push((&self.wall, tmp, &u1, bytes));
        let u16 = uniform(device, &RollParams { n, stride: 16, _p: [0; 2], shift, _p3: 0 });
        if let Some(f) = &self.flow {
            fields.push((&f.food, tmp, &u1, bytes));
            fields.push((&f.fert, tmp, &u1, bytes));
            fields.push((&f.genome, f.genome_next_scratch(), &u16, bytes * 16));
        }
        for (f, tmp, u, size) in fields {
            encoder.copy_buffer_to_buffer(f, 0, tmp, 0, size);
            let bg = bind_group(device, &self.pipes.roll, &[tmp, f, u]);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipes.roll);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups, groups, groups);
        }
    }

    /// Record the coarse tracking pass; returns the buffer, grid size and bytes.
    pub fn encode_track(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) -> (&wgpu::Buffer, u32, u64) {
        let n = self.n;
        let (m, f) = track_grid(n);
        let ch = |c: usize| self.channels.get(c).unwrap_or(&self.channels[0]);
        let (genome, food, sense) = match &self.flow {
            Some(fl) => (&fl.genome, &fl.food, &fl.sense),
            None => (&self.track_dummy, &self.track_dummy, &self.track_dummy),
        };
        let u = uniform(
            device,
            &TrackParams {
                n,
                m,
                f,
                nch: self.channels.len() as u32,
                has_genome: self.flow.is_some() as u32,
                _p: [0; 3],
                scale: 1.0 / (self.cells() as f32),
                _q: [0.0; 3],
            },
        );
        let bg = bind_group(device, &self.pipes.track, &[ch(0), ch(1), ch(2), genome, food, sense, &self.track_out, &u]);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipes.track);
        pass.set_bind_group(0, &bg, &[]);
        let g = m.div_ceil(4);
        pass.dispatch_workgroups(g, g, g);
        drop(pass);
        (&self.track_out, m, (m as u64).pow(3) * TRACK_FLOATS as u64 * 4)
    }

    /// Switch to Flow mode (allocates genome, food and flow buffers).
    pub fn enable_flow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, params: flow::FlowParams, eco: flow::EcologyParams) {
        match self.flow.as_mut() {
            Some(f) => {
                f.params = params;
                f.eco = eco;
                f.invalidate();
            }
            None => {
                self.flow = Some(flow::FlowState::new(device, queue, self.n, self.channels.len(), params, eco));
                self.layout_id = next_layout_id();
            }
        }
        self.rebuild_plan(device);
    }

    /// Back to classic Lenia.
    pub fn disable_flow(&mut self, device: &wgpu::Device) {
        if self.flow.take().is_some() {
            self.layout_id = next_layout_id();
            self.rebuild_plan(device);
        }
    }

    /// Upload Flow-mode parameters (call once per frame before stepping).
    pub fn update_flow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (n, nch, dt) = (self.n, self.channels.len(), self.params.dt());
        if self.flow.as_ref().is_some_and(|f| f.needs_rebuild()) {
            self.rebuild_plan(device);
        }
        if let Some(f) = &self.flow {
            f.write_uniform(queue, n, nch, dt);
        }
    }

    /// Buffers needed by flow brushes and budding.
    pub fn flow_buffers(&self) -> flow::SimBuffers<'_> {
        flow::SimBuffers { n: self.n, channels: &self.channels, accum: &self.accum, work: &self.work, wall: &self.wall, rule: &self.params }
    }

    /// Record a pass computing per-channel partial sums (8 floats per
    /// workgroup: mass and periodic moments) into `reduce_out`.
    pub fn encode_mass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mass"), timestamp_writes: None });
        pass.set_pipeline(&self.pipes.reduce);
        for bg in &self.plan.reduce {
            pass.set_bind_group(0, bg, &[]);
            let x = self.reduce_groups.min(32768);
            pass.dispatch_workgroups(x, self.reduce_groups.div_ceil(x), 1);
        }
    }

    /// The partial-sums buffer, workgroups per channel, and bytes used.
    pub fn mass_buffer(&self) -> (&wgpu::Buffer, u32, u64) {
        (&self.reduce_out, self.reduce_groups, self.reduce_groups as u64 * 32 * self.channels.len() as u64)
    }

    pub fn clear(&mut self, encoder: &mut wgpu::CommandEncoder, walls_too: bool) {
        for c in &self.channels {
            encoder.clear_buffer(c, 0, None);
        }
        if walls_too {
            encoder.clear_buffer(&self.wall, 0, None);
        }
        self.generation = 0;
        self.time = 0.0;
    }

    /// Apply a spherical brush centered at `center` (cell coordinates).
    #[allow(clippy::too_many_arguments)]
    pub fn brush(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mode: BrushMode,
        channel: Option<usize>,
        center: [f32; 3],
        radius: f32,
        strength: f32,
        seed: u32,
    ) {
        let radius = radius.max(0.5);
        let extent = (2.0 * radius).ceil() as u32 + 2;
        let origin = center.map(|c| (c - radius).floor() as i32 - 1);
        let u = uniform(
            device,
            &BrushParams { n: self.n, mode: mode as u32, seed, extent, origin, radius, center, strength },
        );
        let targets: Vec<&wgpu::Buffer> = match (mode, channel) {
            (BrushMode::Wall | BrushMode::Unwall, _) => vec![&self.channels[0]],
            (_, Some(c)) => vec![&self.channels[c.min(self.channels.len() - 1)]],
            (_, None) => self.channels.iter().collect(),
        };
        let groups = extent.div_ceil(4);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipes.brush);
        for (i, t) in targets.iter().enumerate() {
            // different noise per channel
            let u = if i == 0 {
                u.clone()
            } else {
                uniform(
                    device,
                    &BrushParams {
                        n: self.n,
                        mode: mode as u32,
                        seed: seed.wrapping_add(i as u32 * 7919),
                        extent,
                        origin,
                        radius,
                        center,
                        strength,
                    },
                )
            };
            let bg = bind_group(device, &self.pipes.brush, &[t, &self.wall, &u]);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups, groups, groups);
        }
    }

    /// Write a pattern (`dims` = [x, y, z], one Vec per channel, x fastest)
    /// with its center at `center` (cell coordinates).
    pub fn stamp(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dims: [u32; 3],
        data: &[Vec<f32>],
        center: [f32; 3],
    ) {
        let origin = [0, 1, 2].map(|i| (center[i] - dims[i] as f32 / 2.0).round() as i32);
        let u = uniform(device, &StampParams { n: self.n, _p: [0; 3], dims, _p3: 0, origin, _p4: 0 });
        let mut pass_data = Vec::new();
        for (c, chan) in data.iter().enumerate().take(self.channels.len()) {
            let pat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pattern"),
                contents: bytemuck::cast_slice(chan),
                usage: wgpu::BufferUsages::STORAGE,
            });
            pass_data.push(bind_group(device, &self.pipes.stamp, &[&self.channels[c], &pat, &self.wall, &u]));
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipes.stamp);
        for bg in &pass_data {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(dims[0].div_ceil(4), dims[1].div_ceil(4), dims[2].div_ceil(4));
        }
    }

    /// Upload full channel data (len = n^3 each).
    pub fn upload(&self, queue: &wgpu::Queue, channel: usize, data: &[f32]) {
        queue.write_buffer(&self.channels[channel], 0, bytemuck::cast_slice(data));
    }

    pub fn upload_wall(&self, queue: &wgpu::Queue, data: &[f32]) {
        queue.write_buffer(&self.wall, 0, bytemuck::cast_slice(data));
    }

    /// Blocking readback of one channel.
    pub fn download(&self, device: &wgpu::Device, queue: &wgpu::Queue, channel: usize) -> Vec<f32> {
        read_buffer_f32(device, queue, &self.channels[channel])
    }

    pub fn download_wall(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        read_buffer_f32(device, queue, &self.wall)
    }
}
