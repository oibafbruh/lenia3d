//! World management on top of the simulation: loading presets, seeding,
//! stamping creatures, the life loop (tracking and reproduction in Flow
//! mode) and save/load.

use crate::life::{plan_reproduction, ReproKind, ReproOp};
use crate::render::GpuCreature;
use crate::sim::flow::{gene, make_genome, random_genome, EcologyParams, FlowBrush, FlowParams, Genome, TraitRanges, Traits, GENES};
use crate::sim::preset::{Pattern, Preset, SoupInit};
use crate::sim::{BrushMode, LeniaParams, Simulation};
use crate::tracker::Tracker;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub struct World {
    pub sim: Simulation,
    /// Preset most recently loaded (used by "Reset").
    pub current: Option<Preset>,
    /// Scale applied to the current preset (pattern size and R).
    pub scale: f32,
    /// Set when walls may exist (follow mode is disabled then).
    pub has_walls: bool,
    pub tracker: Tracker,
    /// Founder trait ranges for Flow-mode seeding.
    pub traits: TraitRanges,
    pub rng: Rng,
    seed: u32,
}

/// Small deterministic RNG for seeding.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
}

/// Loudness of a creature's call at `time` (seconds): rhythmic chirps whose
/// rate grows with the voice gene. Shared by the audio and the visuals.
pub fn call_envelope(time: f64, id: u64, voice: f32) -> f32 {
    if voice < 0.05 {
        return 0.0;
    }
    let rate = 0.4 + 2.2 * voice as f64;
    let offset = ((id.wrapping_mul(2654435761) >> 7) % 1000) as f64 / 1000.0;
    let ph = (time * rate + offset).fract();
    if ph < 0.28 {
        (std::f64::consts::PI * ph / 0.28).sin().powi(2) as f32 * voice.min(1.0)
    } else {
        0.0
    }
}

impl World {
    pub fn new(device: &wgpu::Device, n: u32, params: LeniaParams) -> Self {
        Self {
            sim: Simulation::new(device, n, params),
            current: None,
            scale: 1.0,
            has_walls: false,
            tracker: Tracker::default(),
            traits: TraitRanges::default(),
            rng: Rng::new(0x5EED),
            seed: 1,
        }
    }

    pub fn n(&self) -> u32 {
        self.sim.n
    }

    pub fn is_flow(&self) -> bool {
        self.sim.flow.is_some()
    }

    pub fn next_seed(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
        self.seed
    }

    /// Recreate the simulation at a new grid size, keeping the rule and mode.
    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, n: u32) {
        let params = self.sim.params().clone();
        let flow = self.sim.flow.as_ref().map(|f| (f.params.clone(), f.eco.clone()));
        self.sim = Simulation::new(device, n, params);
        if let Some((p, e)) = flow {
            self.sim.enable_flow(device, queue, p, e);
        }
        self.has_walls = false;
        self.tracker.reset();
    }

    /// Rule scaled by `scale` (radius grows with the pattern).
    pub fn scaled_rule(rule: &LeniaParams, scale: f32) -> LeniaParams {
        let mut r = rule.clone();
        r.R *= scale;
        r
    }

    /// Load a preset: set its rule and mode and seed the world.
    pub fn load_preset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, preset: &Preset, scale: f32) {
        self.current = Some(preset.clone());
        self.scale = scale;
        self.sim.set_params(device, queue, Self::scaled_rule(&preset.rule, scale));
        match &preset.flow {
            Some(fp) => {
                self.sim.enable_flow(device, queue, fp.clone(), preset.ecology.clone().unwrap_or_default());
                if let Some(t) = preset.init.as_ref().and_then(|i| i.traits.clone()) {
                    self.traits = t;
                }
            }
            None => self.sim.disable_flow(device),
        }
        self.reset(device, queue);
    }

    /// Switch between classic and Flow mode, keeping the rule.
    pub fn set_flow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cfg: Option<(FlowParams, EcologyParams)>) {
        match cfg {
            Some((p, e)) => self.sim.enable_flow(device, queue, p, e),
            None => self.sim.disable_flow(device),
        }
        self.tracker.reset();
    }

    /// Re-seed the world from the current preset (keeps walls).
    pub fn reset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut enc = device.create_command_encoder(&Default::default());
        self.sim.clear(&mut enc, false);
        if let Some(f) = &self.sim.flow {
            f.reset_food(&mut enc);
        }
        let n = self.n() as f32;
        let default_init = SoupInit { blobs: 12, blob_radius: 1.0, density: 0.5, traits: None };
        match self.current.clone() {
            Some(p) => {
                if let Some(pat) = p.pattern() {
                    let pat = pat.scaled(self.scale);
                    self.stamp(device, &mut enc, &pat, [n / 2.0; 3]);
                } else {
                    let init = p.init.clone().unwrap_or(default_init);
                    self.seed_soup(device, &mut enc, &init);
                }
            }
            None => self.seed_soup(device, &mut enc, &default_init),
        }
        queue.submit([enc.finish()]);
        self.tracker.reset();
    }

    /// A founder genome drawn from the current trait ranges.
    pub fn random_genome(&mut self) -> Genome {
        let rule = self.sim.params().clone();
        let traits = self.traits.clone();
        random_genome(&rule, &traits, || self.rng.f32())
    }

    /// The genome given to matter painted or stamped without one.
    pub fn default_genome(&self) -> Genome {
        let t = &self.traits;
        let mid = |v: [f32; 2]| (v[0] + v[1]) * 0.5;
        let traits = Traits {
            eye: mid(t.eye),
            ear: mid(t.ear),
            voice: mid(t.voice),
            mouth: mid(t.mouth),
            claw: mid(t.claw),
            alarm: mid(t.alarm),
            tune: 0.5,
            pitch: 0.5,
            hue: 0.45,
            timbre: 0.5,
        };
        make_genome(self.sim.params(), &traits)
    }

    /// Random blobs scattered over the world, matching `seed_blobs` in
    /// tools/lenia_ref.py. The blob count is given for a 64^3 world and is
    /// scaled with the world volume. In Flow mode every blob is a founder
    /// with its own random genome.
    pub fn seed_soup(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, init: &SoupInit) {
        let n = self.n() as f32;
        let r = self.sim.params().R * init.blob_radius;
        let count = ((init.blobs as f32) * (n / 64.0).powi(3)).round().clamp(1.0, 4000.0) as u32;
        let mut rng = Rng::new(self.next_seed() as u64);
        for _ in 0..count {
            let c = [rng.f32() * n, rng.f32() * n, rng.f32() * n];
            let seed = rng.next_u32();
            if self.sim.flow.is_some() {
                let genes = self.random_genome();
                self.seed_founder(device, enc, c, r, init.density, &genes, seed);
            } else {
                self.sim.brush(device, enc, BrushMode::Seed, None, c, r, init.density, seed);
            }
        }
    }

    /// Flow mode: a blob of matter carrying `genes`.
    #[allow(clippy::too_many_arguments)]
    pub fn seed_founder(
        &self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        center: [f32; 3],
        radius: f32,
        density: f32,
        genes: &Genome,
        seed: u32,
    ) {
        if let Some(f) = &self.sim.flow {
            f.brush(device, enc, &self.sim.flow_buffers(), FlowBrush::Seed, center, radius, 0.0, density, genes, seed);
        }
    }

    /// Stamp a pattern centered at `center` (cell coordinates). In Flow mode
    /// the stamped matter gets the default genome.
    pub fn stamp(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, pat: &Pattern, center: [f32; 3]) {
        self.sim.stamp(device, enc, pat.dims, &pat.channels, center);
        if let Some(f) = &self.sim.flow {
            let r = pat.dims.iter().copied().max().unwrap_or(1) as f32 * 0.9;
            let genes = self.default_genome();
            f.brush(device, enc, &self.sim.flow_buffers(), FlowBrush::Paint, center, r, 0.0, 0.0, &genes, 0);
        }
    }

    /// Feed one tracking readback (taken at `generation`) to the tracker and,
    /// in Flow mode, plan budding and mating. Returns the operations to apply.
    pub fn life_update(&mut self, data: &[f32], m: u32, f: u32, generation: u64) -> Vec<ReproOp> {
        let n = self.n();
        let flow = self.sim.flow.is_some();
        self.tracker.update(data, m, f, n, generation, flow);
        match &self.sim.flow {
            Some(fl) => {
                let (eco, params) = (fl.eco.clone(), fl.params.clone());
                plan_reproduction(&mut self.tracker, &eco, &params, n, generation, &mut self.rng, 4)
            }
            None => vec![],
        }
    }

    /// Run reproduction operations on the GPU.
    pub fn apply_ops(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, ops: &[ReproOp]) {
        let Some(f) = &self.sim.flow else { return };
        let sb = self.sim.flow_buffers();
        for op in ops {
            f.bud(device, enc, &sb, op.center, op.radius, op.offset, &op.delta);
        }
    }

    /// Tracked creatures in renderer form (world units).
    pub fn gpu_creatures(&self, time: f64, selected: Option<u64>) -> Vec<GpuCreature> {
        let n = self.n() as f32;
        let to_world = |c: f32| (c + 0.5) / n * 2.0 - 1.0;
        let flow = self.is_flow();
        self.tracker
            .creatures
            .iter()
            // skip faint wisps that are tracked but too thin to see
            .filter(|c| c.radius >= 2.0 && c.mass / (4.19 * c.radius.powi(3)) > 0.08)
            .map(|c| {
                let g = &c.genome;
                let speed = c.speed();
                let forward = if speed > 1e-3 {
                    c.velocity.map(|v| v / speed)
                } else {
                    // stable pseudo-random facing for creatures at rest
                    let a = (c.id as f32 * 2.399963) % std::f32::consts::TAU;
                    [a.cos(), 0.0, a.sin()]
                };
                let (eye, mouth, claw, ear, voice) = if flow {
                    (g[gene::EYE], g[gene::MOUTH], g[gene::CLAW], g[gene::EAR], g[gene::VOICE])
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0)
                };
                GpuCreature {
                    center: c.center.map(to_world),
                    radius: c.radius * 2.0 / n,
                    forward,
                    eye,
                    mouth,
                    claw,
                    ear,
                    voice,
                    hue: g[gene::HUE],
                    level: call_envelope(time, c.id, voice),
                    eat: (c.growth * 2.0).clamp(0.0, 1.0),
                    selected: (Some(c.id) == selected) as u32 as f32,
                }
            })
            .collect()
    }

    pub fn save(&self, device: &wgpu::Device, queue: &wgpu::Queue, path: &std::path::Path) -> Result<()> {
        let header = SaveHeader {
            version: 2,
            n: self.n(),
            rule: self.sim.params().clone(),
            generation: self.sim.generation,
            time: self.sim.time,
            flow: self.sim.flow.as_ref().map(|f| f.params.clone()),
            ecology: self.sim.flow.as_ref().map(|f| f.eco.clone()),
        };
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        let h = serde_json::to_vec(&header)?;
        f.write_all(&(h.len() as u32).to_le_bytes())?;
        f.write_all(&h)?;
        for c in 0..self.sim.channels.len() {
            f.write_all(bytemuck::cast_slice(&self.sim.download(device, queue, c)))?;
        }
        f.write_all(bytemuck::cast_slice(&self.sim.download_wall(device, queue)))?;
        if let Some(fl) = &self.sim.flow {
            let genome = crate::gpu::read_buffer_f32(device, queue, &fl.genome);
            // genes as bytes; ear is in [-1, 1], everything else in [0, 1]
            let bytes: Vec<u8> = genome
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let v = if i % GENES == gene::EAR { (v + 1.0) * 0.5 } else { v };
                    (v.clamp(0.0, 1.0) * 255.0).round() as u8
                })
                .collect();
            f.write_all(&bytes)?;
            f.write_all(bytemuck::cast_slice(&crate::gpu::read_buffer_f32(device, queue, &fl.food)))?;
            f.write_all(bytemuck::cast_slice(&crate::gpu::read_buffer_f32(device, queue, &fl.fert)))?;
        }
        Ok(())
    }

    pub fn load(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, path: &std::path::Path) -> Result<()> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(anyhow!("not a Lenia3D world file"));
        }
        let mut len = [0u8; 4];
        f.read_exact(&mut len)?;
        let mut h = vec![0u8; u32::from_le_bytes(len) as usize];
        f.read_exact(&mut h)?;
        let header: SaveHeader = serde_json::from_slice(&h)?;
        let cells = (header.n as usize).pow(3);
        if header.n != self.n() {
            self.sim = Simulation::new(device, header.n, header.rule.clone());
        } else {
            self.sim.set_params(device, queue, header.rule.clone());
        }
        match &header.flow {
            Some(p) => self.sim.enable_flow(device, queue, p.clone(), header.ecology.clone().unwrap_or_default()),
            None => self.sim.disable_flow(device),
        }
        let read_f32 = |f: &mut std::io::BufReader<std::fs::File>, version: u32| -> Result<Vec<f32>> {
            if version >= 2 {
                let mut buf = vec![0u8; cells * 4];
                f.read_exact(&mut buf)?;
                Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            } else {
                let mut buf = vec![0u8; cells * 2];
                f.read_exact(&mut buf)?;
                Ok(dequantize(&buf))
            }
        };
        for c in 0..header.rule.channels {
            let data = read_f32(&mut f, header.version)?;
            self.sim.upload(queue, c, &data);
        }
        let wall = read_f32(&mut f, header.version)?;
        self.sim.upload_wall(queue, &wall);
        if let Some(fl) = &self.sim.flow {
            let mut bytes = vec![0u8; cells * GENES];
            f.read_exact(&mut bytes)?;
            let genome: Vec<f32> = bytes
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    let v = b as f32 / 255.0;
                    if i % GENES == gene::EAR {
                        v * 2.0 - 1.0
                    } else {
                        v
                    }
                })
                .collect();
            queue.write_buffer(&fl.genome, 0, bytemuck::cast_slice(&genome));
            let food = read_f32(&mut f, 2)?;
            queue.write_buffer(&fl.food, 0, bytemuck::cast_slice(&food));
            let fert = read_f32(&mut f, 2)?;
            queue.write_buffer(&fl.fert, 0, bytemuck::cast_slice(&fert));
        }
        self.has_walls = wall.iter().any(|&w| w > 0.5);
        self.sim.generation = header.generation;
        self.sim.time = header.time;
        self.tracker.reset();
        Ok(())
    }
}

/// Number of offspring events in a list of operations, for audio and stats.
pub fn births(ops: &[ReproOp]) -> (usize, usize) {
    (ops.iter().filter(|o| o.kind == ReproKind::Bud).count(), ops.iter().filter(|o| o.kind == ReproKind::Mate).count())
}

const MAGIC: &[u8; 8] = b"LENIA3D\x01";

#[derive(Serialize, Deserialize)]
struct SaveHeader {
    version: u32,
    n: u32,
    rule: LeniaParams,
    generation: u64,
    time: f64,
    #[serde(default)]
    flow: Option<FlowParams>,
    #[serde(default)]
    ecology: Option<EcologyParams>,
}

fn dequantize(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0).collect()
}
