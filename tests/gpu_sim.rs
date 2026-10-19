//! GPU correctness tests. Run on any Vulkan/DX12/Metal device, including
//! software rasterizers (lavapipe in CI).

use lenia3d::gpu::{read_buffer_f32, Gpu};
use lenia3d::sim::fft::Fft;
use lenia3d::sim::params::{growth, kernel_shell, kernel_sum, KernelParams, LeniaParams};
use lenia3d::sim::Simulation;
use rustfft::num_complex::Complex;
use wgpu::util::DeviceExt;

fn gpu() -> Option<Gpu> {
    match Gpu::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping GPU test, no adapter: {e}");
            None
        }
    }
}

/// Deterministic pseudo-random numbers in [0, 1).
fn rng(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
}

fn cpu_fft3(data: &mut [Complex<f64>], n: usize, inverse: bool) {
    let mut planner = rustfft::FftPlanner::new();
    let f = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
    let mut line = vec![Complex::new(0.0, 0.0); n];
    for axis in 0..3 {
        let stride = n.pow(axis as u32);
        for a in 0..n {
            for b in 0..n {
                let base = match axis {
                    0 => (a * n + b) * n,
                    1 => a * n * n + b,
                    _ => a * n + b,
                };
                for j in 0..n {
                    line[j] = data[base + j * stride];
                }
                f.process(&mut line);
                for j in 0..n {
                    data[base + j * stride] = line[j];
                }
            }
        }
    }
}

#[test]
fn fft_matches_cpu() {
    let Some(g) = gpu() else { return };
    for &n in &[8u32, 16, 32] {
        let cells = (n as usize).pow(3);
        let mut seed = 42u64 + n as u64;
        let input: Vec<[f32; 2]> = (0..cells).map(|_| [rng(&mut seed) as f32 - 0.5, rng(&mut seed) as f32 - 0.5]).collect();
        let buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let fft = Fft::new(&g.device, n);
        let b = fft.bindings(&g.device, &buf);
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            fft.run(&mut pass, &b, false);
        }
        g.queue.submit([enc.finish()]);
        let out = read_buffer_f32(&g.device, &g.queue, &buf);

        let mut expect: Vec<Complex<f64>> = input.iter().map(|v| Complex::new(v[0] as f64, v[1] as f64)).collect();
        cpu_fft3(&mut expect, n as usize, false);
        let mut max_err = 0.0f64;
        let mut max_val = 0.0f64;
        for i in 0..cells {
            let e = expect[i];
            max_err = max_err.max((out[2 * i] as f64 - e.re).abs()).max((out[2 * i + 1] as f64 - e.im).abs());
            max_val = max_val.max(e.norm());
        }
        assert!(max_err / max_val < 1e-5, "n={n}: relative error {}", max_err / max_val);

        // inverse brings it back (times n^3)
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            fft.run(&mut pass, &b, true);
        }
        g.queue.submit([enc.finish()]);
        let back = read_buffer_f32(&g.device, &g.queue, &buf);
        let scale = 1.0 / cells as f64;
        for i in 0..cells {
            assert!((back[2 * i] as f64 * scale - input[i][0] as f64).abs() < 1e-5);
            assert!((back[2 * i + 1] as f64 * scale - input[i][1] as f64).abs() < 1e-5);
        }
    }
}

/// Straightforward CPU Lenia step with direct periodic convolution in f64.
fn cpu_step(params: &LeniaParams, n: usize, world: &mut [Vec<f64>]) {
    let mut delta = vec![vec![0.0f64; n * n * n]; params.channels];
    let mut hit = vec![false; params.channels];
    for k in &params.kernels {
        let radius = (params.R * k.r) as f64;
        let sum = kernel_sum(radius, &k.b, k.kn);
        let r = radius.ceil() as i64;
        let mut taps = Vec::new();
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let d = ((dx * dx + dy * dy + dz * dz) as f64).sqrt() / radius;
                    let w = kernel_shell(d, &k.b, k.kn) / sum;
                    if w != 0.0 {
                        taps.push((dx, dy, dz, w));
                    }
                }
            }
        }
        let src = &world[k.src];
        let ni = n as i64;
        for z in 0..ni {
            for y in 0..ni {
                for x in 0..ni {
                    let mut u = 0.0;
                    for &(dx, dy, dz, w) in &taps {
                        let xx = (x + dx).rem_euclid(ni);
                        let yy = (y + dy).rem_euclid(ni);
                        let zz = (z + dz).rem_euclid(ni);
                        u += w * src[((zz * ni + yy) * ni + xx) as usize];
                    }
                    let i = ((z * ni + y) * ni + x) as usize;
                    delta[k.dst][i] += k.h as f64 * growth(u, k.m as f64, k.s as f64, k.gn);
                }
            }
        }
        hit[k.dst] = true;
    }
    let dt = params.dt() as f64;
    for c in 0..params.channels {
        if hit[c] {
            for i in 0..world[c].len() {
                world[c][i] = (world[c][i] + dt * delta[c][i]).clamp(0.0, 1.0);
            }
        }
    }
}

#[test]
fn lenia_step_matches_cpu() {
    let Some(g) = gpu() else { return };
    let n = 32u32;
    let cells = (n as usize).pow(3);
    let params = LeniaParams {
        R: 5.0,
        T: 5.0,
        channels: 2,
        kernels: vec![
            KernelParams { src: 0, dst: 0, r: 1.0, b: vec![1.0], m: 0.15, s: 0.03, h: 1.0, kn: 1, gn: 1 },
            KernelParams { src: 1, dst: 0, r: 0.8, b: vec![0.5, 1.0], m: 0.2, s: 0.05, h: 0.6, kn: 2, gn: 2 },
            KernelParams { src: 0, dst: 1, r: 1.2, b: vec![1.0, 0.3, 0.7], m: 0.25, s: 0.06, h: 0.8, kn: 1, gn: 2 },
        ],
    };
    // smooth-ish random world so the growth function is exercised in its active range
    let mut seed = 7u64;
    let mut world: Vec<Vec<f64>> = (0..2)
        .map(|_| {
            (0..cells)
                .map(|i| {
                    let x = (i % 32) as f64;
                    let y = ((i / 32) % 32) as f64;
                    let z = (i / 1024) as f64;
                    let blob = (-((x - 16.0).powi(2) + (y - 15.0).powi(2) + (z - 17.0).powi(2)) / 40.0).exp();
                    (blob * (0.4 + 0.6 * rng(&mut seed))).min(1.0)
                })
                .collect()
        })
        .collect();

    let mut sim = Simulation::new(&g.device, n, params.clone());
    for c in 0..2 {
        let data: Vec<f32> = world[c].iter().map(|&v| v as f32).collect();
        sim.upload(&g.queue, c, &data);
    }
    // compare against f32 inputs exactly as uploaded
    for c in 0..2 {
        for v in world[c].iter_mut() {
            *v = *v as f32 as f64;
        }
    }
    for step in 0..3 {
        let mut enc = g.device.create_command_encoder(&Default::default());
        sim.encode_steps(&mut enc, 1);
        g.queue.submit([enc.finish()]);
        cpu_step(&params, n as usize, &mut world);
        for c in 0..2 {
            let got = sim.download(&g.device, &g.queue, c);
            let max_err = got.iter().zip(&world[c]).map(|(&a, &b)| (a as f64 - b).abs()).fold(0.0, f64::max);
            let mass: f64 = world[c].iter().sum();
            assert!(mass > 1.0, "test world died, adjust parameters");
            assert!(max_err < 2e-4, "step {step} channel {c}: max error {max_err}");
        }
    }
}

#[test]
fn center_of_mass_and_roll() {
    let Some(g) = gpu() else { return };
    let n = 32u32;
    let cells = (n as usize).pow(3);
    let sim = Simulation::new(&g.device, n, LeniaParams::default());
    // a small cube straddling the x/z boundary: cells x in {30,31,0,1}, y in 10..12, z in {31,0}
    let mut data = vec![0.0f32; cells];
    for &x in &[30usize, 31, 0, 1] {
        for y in 10..12 {
            for &z in &[31usize, 0] {
                data[(z * 32 + y) * 32 + x] = 1.0;
            }
        }
    }
    sim.upload(&g.queue, 0, &data);
    let stats = |sim: &Simulation| {
        let mut enc = g.device.create_command_encoder(&Default::default());
        sim.encode_mass(&mut enc);
        g.queue.submit([enc.finish()]);
        let (buf, groups, size) = sim.mass_buffer();
        let bytes = lenia3d::gpu::read_buffer_bytes(&g.device, &g.queue, buf, 0, size);
        lenia3d::sim::stats_from_partials(bytemuck::cast_slice(&bytes), groups, n)
    };
    let s = stats(&sim);
    assert!((s[0].mass - 16.0).abs() < 1e-3);
    let c = s[0].center.unwrap();
    // periodic centers: x = 31.5, y = 10.5, z = 31.5 (i.e. -0.5)
    let wrap_err = |a: f64, b: f64| ((a - b + 16.0).rem_euclid(32.0) - 16.0).abs();
    assert!(wrap_err(c[0], 31.5) < 0.05, "x center {}", c[0]);
    assert!(wrap_err(c[1], 10.5) < 0.05, "y center {}", c[1]);
    assert!(wrap_err(c[2], 31.5) < 0.05, "z center {}", c[2]);

    // roll by (+2, -3, +5) and check where a cell went
    let mut enc = g.device.create_command_encoder(&Default::default());
    sim.encode_roll(&g.device, &mut enc, [2, -3, 5]);
    g.queue.submit([enc.finish()]);
    let out = sim.download(&g.device, &g.queue, 0);
    // old (x=1, y=10, z=0) -> new (3, 7, 5)
    assert_eq!(out[(5 * 32 + 7) * 32 + 3], 1.0);
    assert_eq!(out.iter().sum::<f32>(), 16.0);
    let c2 = stats(&sim)[0].center.unwrap();
    assert!(wrap_err(c2[0], 1.5) < 0.05 && wrap_err(c2[1], 7.5) < 0.05 && wrap_err(c2[2], 4.5) < 0.05, "{c2:?}");
}

#[test]
fn save_load_roundtrip_and_channel_change() {
    let Some(g) = gpu() else { return };
    use lenia3d::world::World;
    let presets = lenia3d::sim::preset::builtin_presets();
    let p = presets.iter().find(|p| p.rule.channels == 2 && p.pattern().is_some()).unwrap_or(&presets[0]).clone();
    let mut w = World::new(&g.device, 32, p.rule.clone());
    w.load_preset(&g.device, &g.queue, &p, 0.5);
    let mut enc = g.device.create_command_encoder(&Default::default());
    w.sim.encode_steps(&mut enc, 3);
    g.queue.submit([enc.finish()]);
    let before: Vec<Vec<f32>> = (0..w.sim.channels.len()).map(|c| w.sim.download(&g.device, &g.queue, c)).collect();
    let dir = std::env::temp_dir().join(format!("lenia3d_test_{}.l3d", std::process::id()));
    w.save(&g.device, &g.queue, &dir).unwrap();

    let mut w2 = World::new(&g.device, 64, LeniaParams::default());
    w2.load(&g.device, &g.queue, &dir).unwrap();
    let _ = std::fs::remove_file(&dir);
    assert_eq!(w2.n(), 32);
    assert_eq!(w2.sim.params(), w.sim.params());
    for (c, b) in before.iter().enumerate() {
        let a = w2.sim.download(&g.device, &g.queue, c);
        let err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-4, "channel {c} differs by {err}");
    }

    // grow to 3 channels and back to 1; data in kept channels must survive
    let mut rule = w2.sim.params().clone();
    rule.channels = 3;
    w2.sim.set_params(&g.device, &g.queue, rule.clone());
    assert_eq!(w2.sim.channels.len(), 3);
    let c0 = w2.sim.download(&g.device, &g.queue, 0);
    assert_eq!(c0, w2.sim.download(&g.device, &g.queue, 0));
    let err0 = c0.iter().zip(&before[0]).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(err0 < 1e-4);
    let mut enc = g.device.create_command_encoder(&Default::default());
    w2.sim.encode_steps(&mut enc, 2);
    g.queue.submit([enc.finish()]);
    rule.channels = 1;
    rule.kernels.truncate(1);
    w2.sim.set_params(&g.device, &g.queue, rule);
    let mut enc = g.device.create_command_encoder(&Default::default());
    w2.sim.encode_steps(&mut enc, 2);
    g.queue.submit([enc.finish()]);
    let out = w2.sim.download(&g.device, &g.queue, 0);
    assert!(out.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
}
