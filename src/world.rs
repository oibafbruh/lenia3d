//! World management on top of the simulation: loading presets, seeding,
//! stamping creatures and save/load.

use crate::sim::preset::{Pattern, Preset, SoupInit};
use crate::sim::{BrushMode, LeniaParams, Simulation};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub struct World {
    pub sim: Simulation,
    /// Preset most recently loaded (used by "Reset").
    pub current: Option<Preset>,
    /// Scale applied to the current preset (pattern size and R).
    pub scale: f32,
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

impl World {
    pub fn new(device: &wgpu::Device, n: u32, params: LeniaParams) -> Self {
        Self { sim: Simulation::new(device, n, params), current: None, scale: 1.0, seed: 1 }
    }

    pub fn n(&self) -> u32 {
        self.sim.n
    }

    pub fn next_seed(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
        self.seed
    }

    /// Recreate the simulation at a new grid size, keeping the rule.
    pub fn resize(&mut self, device: &wgpu::Device, n: u32) {
        let params = self.sim.params().clone();
        self.sim = Simulation::new(device, n, params);
    }

    /// Rule scaled by `scale` (radius grows with the pattern).
    pub fn scaled_rule(rule: &LeniaParams, scale: f32) -> LeniaParams {
        let mut r = rule.clone();
        r.R *= scale;
        r
    }

    /// Load a preset: set its rule and seed the world (creature at the center or a soup).
    pub fn load_preset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, preset: &Preset, scale: f32) {
        self.current = Some(preset.clone());
        self.scale = scale;
        self.sim.set_params(device, queue, Self::scaled_rule(&preset.rule, scale));
        self.reset(device, queue);
    }

    /// Re-seed the world from the current preset (keeps walls).
    pub fn reset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut enc = device.create_command_encoder(&Default::default());
        self.sim.clear(&mut enc, false);
        let n = self.n() as f32;
        match self.current.clone() {
            Some(p) => {
                if let Some(pat) = p.pattern() {
                    let pat = pat.scaled(self.scale);
                    self.sim.stamp(device, &mut enc, pat.dims, &pat.channels, [n / 2.0; 3]);
                } else {
                    let init = p.init.clone().unwrap_or(SoupInit { blobs: 12, blob_radius: 1.0, density: 0.5 });
                    self.seed_soup(device, &mut enc, &init);
                }
            }
            None => {
                let init = SoupInit { blobs: 12, blob_radius: 1.0, density: 0.5 };
                self.seed_soup(device, &mut enc, &init);
            }
        }
        queue.submit([enc.finish()]);
    }

    /// Random blobs scattered over the world. The blob count is given for a
    /// 64^3 world and scaled with the world volume.
    pub fn seed_soup(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, init: &SoupInit) {
        let n = self.n() as f32;
        let r = self.sim.params().R * init.blob_radius;
        let count = ((init.blobs as f32) * (n / 64.0).powi(3)).round().clamp(1.0, 4000.0) as u32;
        let mut rng = Rng::new(self.next_seed() as u64);
        for _ in 0..count {
            let c = [rng.f32() * n, rng.f32() * n, rng.f32() * n];
            let seed = rng.next_u32();
            self.sim.brush(device, enc, BrushMode::Noise, None, c, r, init.density * 2.0, seed);
        }
    }

    /// Stamp a pattern centered at `center` (cell coordinates).
    pub fn stamp(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, pat: &Pattern, center: [f32; 3]) {
        self.sim.stamp(device, enc, pat.dims, &pat.channels, center);
    }

    pub fn save(&self, device: &wgpu::Device, queue: &wgpu::Queue, path: &std::path::Path) -> Result<()> {
        let header = SaveHeader {
            version: 1,
            n: self.n(),
            rule: self.sim.params().clone(),
            generation: self.sim.generation,
            time: self.sim.time,
        };
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        let h = serde_json::to_vec(&header)?;
        f.write_all(&(h.len() as u32).to_le_bytes())?;
        f.write_all(&h)?;
        for c in 0..self.sim.channels.len() {
            let data = self.sim.download(device, queue, c);
            f.write_all(&quantize(&data))?;
        }
        f.write_all(&quantize(&self.sim.download_wall(device, queue)))?;
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
        let mut buf = vec![0u8; cells * 2];
        for c in 0..header.rule.channels {
            f.read_exact(&mut buf)?;
            self.sim.upload(queue, c, &dequantize(&buf));
        }
        f.read_exact(&mut buf)?;
        self.sim.upload_wall(queue, &dequantize(&buf));
        self.sim.generation = header.generation;
        self.sim.time = header.time;
        Ok(())
    }
}

const MAGIC: &[u8; 8] = b"LENIA3D\x01";

#[derive(Serialize, Deserialize)]
struct SaveHeader {
    version: u32,
    n: u32,
    rule: LeniaParams,
    generation: u64,
    time: f64,
}

/// 16-bit fixed point keeps saves small and exact enough (1/65535).
fn quantize(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| ((x.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes()).collect()
}

fn dequantize(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 65535.0).collect()
}
