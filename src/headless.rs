//! Command-line modes without a window: screenshots, benchmarks, species checks.

use crate::gpu::{read_buffer_bytes, Gpu};
use crate::render::camera::OrbitCamera;
use crate::render::{Overlay, RenderMode, RenderSettings, Renderer};
use crate::sim::preset::Preset;
use crate::sim::{chan, preset, LeniaParams};
use crate::world::World;
use anyhow::{anyhow, Result};
use std::time::Instant;

pub struct Options {
    pub size: u32,
    pub steps: u32,
    pub preset: Option<String>,
    pub width: u32,
    pub height: u32,
    pub mode: RenderMode,
    pub scale: f32,
    pub chan_file: Option<String>,
    pub yaw: f32,
    pub dist: f32,
    pub render: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            size: 128,
            steps: 200,
            preset: None,
            width: 1280,
            height: 720,
            mode: RenderMode::Organic,
            scale: 1.0,
            chan_file: None,
            yaw: 0.0,
            dist: 0.0,
            render: String::new(),
        }
    }
}

/// All presets: built-in plus Chan's species if a file was given.
pub fn all_presets(chan_file: Option<&str>) -> Result<Vec<Preset>> {
    let mut v = preset::builtin_presets();
    if let Some(f) = chan_file {
        v.extend(chan::parse_animals(&std::fs::read_to_string(f)?)?);
    }
    Ok(v)
}

fn find_preset(list: &[Preset], name: &str) -> Result<Preset> {
    let lname = name.to_lowercase();
    list.iter()
        .find(|p| p.name.to_lowercase() == lname)
        .or_else(|| list.iter().find(|p| p.name.to_lowercase().contains(&lname)))
        .cloned()
        .ok_or_else(|| anyhow!("no preset matching '{name}'"))
}

fn run_steps(g: &Gpu, world: &mut World, steps: u32) {
    let mut done = 0;
    while done < steps {
        let k = (steps - done).min(16);
        let mut enc = g.device.create_command_encoder(&Default::default());
        world.sim.encode_steps(&mut enc, k);
        g.queue.submit([enc.finish()]);
        let _ = g.device.poll(wgpu::PollType::wait_indefinitely());
        done += k;
    }
}

pub fn mass(g: &Gpu, world: &World) -> Vec<f64> {
    let mut enc = g.device.create_command_encoder(&Default::default());
    world.sim.encode_mass(&mut enc);
    g.queue.submit([enc.finish()]);
    let (buf, groups, size) = world.sim.mass_buffer();
    let bytes = read_buffer_bytes(&g.device, &g.queue, buf, 0, size);
    let vals: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    crate::sim::stats_from_partials(&vals, groups, world.n()).iter().map(|s| s.mass).collect()
}

pub fn screenshot(opts: &Options, out: &str) -> Result<()> {
    let g = Gpu::headless()?;
    let presets = all_presets(opts.chan_file.as_deref())?;
    let mut world = World::new(&g.device, opts.size, LeniaParams::default());
    let p = match &opts.preset {
        Some(name) => find_preset(&presets, name)?,
        None => presets.first().cloned().ok_or_else(|| anyhow!("no presets available"))?,
    };
    world.load_preset(&g.device, &g.queue, &p, opts.scale);
    let t = Instant::now();
    run_steps(&g, &mut world, opts.steps);
    eprintln!("{}: {} steps in {:.2}s, mass {:?}", p.name, opts.steps, t.elapsed().as_secs_f64(), mass(&g, &world));

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut renderer = Renderer::new(&g.device, format);
    let target = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot"),
        size: wgpu::Extent3d { width: opts.width, height: opts.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut cam = OrbitCamera::default();
    cam.orbit(-opts.yaw / 0.008, 0.0);
    if opts.dist > 0.0 {
        cam.frame_box(glam::Vec3::ZERO, opts.dist);
    }
    cam.snap();
    let mut settings = RenderSettings { mode: opts.mode, ..Default::default() };
    settings.apply_overrides(&opts.render).map_err(|e| anyhow!(e))?;
    let mut enc = g.device.create_command_encoder(&Default::default());
    renderer.update_volume(&g.device, &g.queue, &mut enc, &world.sim, settings.smooth);
    renderer.render(&g.device, &g.queue, &mut enc, &view, (opts.width, opts.height), &cam, &settings, &Overlay::default(), 0.0);
    g.queue.submit([enc.finish()]);
    let pixels = read_texture(&g.device, &g.queue, &target, opts.width, opts.height);
    save_png(out, opts.width, opts.height, &pixels)?;
    eprintln!("wrote {out}");
    Ok(())
}

pub fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
    let row = (w * 4).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixels"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let raw = read_buffer_bytes(device, queue, &buf, 0, (row * h) as u64);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        out.extend_from_slice(&raw[(y * row) as usize..(y * row + w * 4) as usize]);
    }
    out
}

pub fn save_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}

/// Throughput benchmark of the simulation step for each grid size.
pub fn bench(opts: &Options) -> Result<()> {
    let g = Gpu::headless()?;
    println!("Lenia3D benchmark on {} ({:?})", g.info.name, g.info.backend);
    let presets = all_presets(opts.chan_file.as_deref())?;
    let rules: Vec<(String, LeniaParams)> = match &opts.preset {
        Some(name) => {
            let p = find_preset(&presets, name)?;
            vec![(p.name.clone(), p.rule.clone())]
        }
        None => {
            let mut v = vec![("1 channel, 1 kernel".to_string(), LeniaParams::default())];
            if let Some(p) = presets.iter().max_by_key(|p| p.rule.kernels.len()) {
                v.push((format!("{} ({} ch, {} kernels)", p.name, p.rule.channels, p.rule.kernels.len()), p.rule.clone()));
            }
            v
        }
    };
    let max_n = Gpu::max_grid_size(&g.device);
    for (name, rule) in rules {
        for n in [64u32, 128, 256, 512] {
            if n > max_n {
                println!("  {n}^3: skipped (exceeds device buffer limits)");
                continue;
            }
            let mut world = World::new(&g.device, n, rule.clone());
            world.reset(&g.device, &g.queue);
            run_steps(&g, &mut world, 4); // warm up, builds kernels
            let steps = (opts.steps * (128 * 128 * 128) / (n * n * n)).clamp(5, opts.steps.max(5));
            let t = Instant::now();
            run_steps(&g, &mut world, steps);
            let dt = t.elapsed().as_secs_f64();
            println!(
                "  {name:<40} {n:>3}^3: {:8.1} steps/s ({:6.2} ms/step)",
                steps as f64 / dt,
                dt * 1000.0 / steps as f64
            );
        }
    }
    Ok(())
}

/// Run every preset (built-in, plus Chan's species if a file is given) and
/// report whether it stays alive and bounded.
pub fn check_presets(opts: &Options) -> Result<()> {
    let g = Gpu::headless()?;
    let presets = all_presets(opts.chan_file.as_deref())?;
    let mut ok_count = 0;
    for p in &presets {
        let need = p.pattern().map(|q| q.dims.iter().copied().max().unwrap_or(0) as f32 + 16.0).unwrap_or(0.0).max(2.0 * p.rule.R + 8.0);
        let n = p.required_world_size().unwrap_or(if need <= 64.0 { 64 } else { 128 });
        let mut world = World::new(&g.device, n, p.rule.clone());
        world.load_preset(&g.device, &g.queue, p, 1.0);
        let m0: f64 = mass(&g, &world).iter().sum();
        let t = Instant::now();
        run_steps(&g, &mut world, opts.steps);
        let ms = mass(&g, &world);
        let m1: f64 = ms.iter().sum();
        let cells = (n as f64).powi(3) * p.rule.channels as f64;
        let ok = if p.pattern().is_some() {
            let r = m1 / m0.max(1e-9);
            r > 0.5 && r < 2.0
        } else {
            m1 > 0.001 * cells && m1 < 0.6 * cells
        };
        ok_count += ok as usize;
        println!(
            "{:<4} {:<34} n={n:<3} mass {m0:9.1} -> {m1:9.1} {:?} ({:.1}s)",
            if ok { "ok" } else { "FAIL" },
            p.name,
            ms.iter().map(|m| m.round()).collect::<Vec<_>>(),
            t.elapsed().as_secs_f64()
        );
    }
    println!("{ok_count}/{} presets stable after {} steps", presets.len(), opts.steps);
    Ok(())
}
