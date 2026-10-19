//! Flow-mode GPU tests against a straightforward CPU implementation of
//! docs/flow_lenia.md (senses and ecology off unless noted).

use lenia3d::gpu::{read_buffer_f32, Gpu};
use lenia3d::sim::flow::{make_genome, EcologyParams, FlowBrush, FlowParams, Traits};
use lenia3d::sim::params::{growth, kernel_shell, kernel_sum, KernelParams, LeniaParams};
use lenia3d::sim::Simulation;

fn gpu() -> Option<Gpu> {
    Gpu::headless().map_err(|e| eprintln!("skipping GPU test: {e}")).ok()
}

fn rng(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
}

fn rule() -> LeniaParams {
    LeniaParams {
        R: 4.0,
        T: 5.0,
        channels: 1,
        kernels: vec![
            KernelParams { src: 0, dst: 0, r: 1.0, b: vec![1.0], m: 0.2, s: 0.06, h: 0.7, kn: 1, gn: 2 },
            KernelParams { src: 0, dst: 0, r: 0.6, b: vec![0.5, 1.0], m: 0.3, s: 0.1, h: 0.4, kn: 2, gn: 2 },
        ],
    }
}

fn eco_off() -> EcologyParams {
    EcologyParams { enabled: false, ..Default::default() }
}

fn idx(n: i64, x: i64, y: i64, z: i64) -> usize {
    ((z.rem_euclid(n) * n + y.rem_euclid(n)) * n + x.rem_euclid(n)) as usize
}

/// One Flow Lenia step on the CPU (single channel, uniform genome, no senses).
fn cpu_flow_step(rule: &LeniaParams, fp: &FlowParams, n: usize, a: &[f64]) -> Vec<f64> {
    let ni = n as i64;
    let cells = n * n * n;
    // affinity
    let mut aff = vec![0.0f64; cells];
    for k in &rule.kernels {
        let radius = (rule.R * k.r) as f64;
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
        for z in 0..ni {
            for y in 0..ni {
                for x in 0..ni {
                    let u: f64 = taps.iter().map(|&(dx, dy, dz, w)| w * a[idx(ni, x + dx, y + dy, z + dz)]).sum();
                    aff[idx(ni, x, y, z)] += k.h as f64 * growth(u, k.m as f64, k.s as f64, k.gn);
                }
            }
        }
    }
    // displacement
    let dt = rule.dt() as f64;
    let s = fp.s as f64;
    let lim = 1.0 - s;
    let mut disp = vec![[0.0f64; 3]; cells];
    for z in 0..ni {
        for y in 0..ni {
            for x in 0..ni {
                let i = idx(ni, x, y, z);
                let grad = |f: &[f64]| {
                    [
                        (f[idx(ni, x + 1, y, z)] - f[idx(ni, x - 1, y, z)]) / 2.0,
                        (f[idx(ni, x, y + 1, z)] - f[idx(ni, x, y - 1, z)]) / 2.0,
                        (f[idx(ni, x, y, z + 1)] - f[idx(ni, x, y, z - 1)]) / 2.0,
                    ]
                };
                let alpha = (a[i].max(0.0) / fp.theta as f64).powf(fp.n as f64).clamp(0.0, 1.0);
                let ga = grad(&aff);
                let gm = grad(a);
                for k in 0..3 {
                    disp[i][k] = (dt * ((1.0 - alpha) * ga[k] - alpha * gm[k])).clamp(-lim, lim);
                }
            }
        }
    }
    // reintegration
    let mut out = vec![0.0f64; cells];
    for z in 0..ni {
        for y in 0..ni {
            for x in 0..ni {
                let mut acc = 0.0;
                for dz in -1..=1i64 {
                    for dy in -1..=1i64 {
                        for dx in -1..=1i64 {
                            let j = idx(ni, x + dx, y + dy, z + dz);
                            let p = [dx as f64 + disp[j][0], dy as f64 + disp[j][1], dz as f64 + disp[j][2]];
                            let ov: f64 = p.iter().map(|&c| ((c + s).min(0.5) - (c - s).max(-0.5)).max(0.0)).product();
                            acc += a[j] * ov / (8.0 * s * s * s);
                        }
                    }
                }
                out[idx(ni, x, y, z)] = acc;
            }
        }
    }
    out
}

fn setup(g: &Gpu, n: u32, eco: EcologyParams, a: &[f32]) -> Simulation {
    let rule = rule();
    let mut sim = Simulation::new(&g.device, n, rule.clone());
    sim.enable_flow(&g.device, &g.queue, FlowParams::default(), eco);
    sim.upload(&g.queue, 0, a);
    let genes = make_genome(&rule, &Traits { mouth: 0.5, pitch: 0.5, hue: 0.3, timbre: 0.5, ..Default::default() });
    let mut enc = g.device.create_command_encoder(&Default::default());
    sim.flow.as_ref().unwrap().brush(&g.device, &mut enc, &sim.flow_buffers(), FlowBrush::Fill, [0.0; 3], 1.0, 0.0, 0.0, &genes, 0);
    g.queue.submit([enc.finish()]);
    sim.update_flow(&g.device, &g.queue);
    sim
}

fn blob_world(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n * n * n)
        .map(|i| {
            let x = (i % n) as f64;
            let y = ((i / n) % n) as f64;
            let z = (i / (n * n)) as f64;
            let c = n as f64 / 2.0;
            let b = (-((x - c).powi(2) + (y - c + 1.0).powi(2) + (z - c - 1.0).powi(2)) / 12.0).exp();
            (b * (0.5 + rng(&mut s))) as f32
        })
        .collect()
}

#[test]
fn flow_step_matches_cpu() {
    let Some(g) = gpu() else { return };
    let n = 16u32;
    let a = blob_world(16, 3);
    let mut sim = setup(&g, n, eco_off(), &a);
    let mut enc = g.device.create_command_encoder(&Default::default());
    sim.encode_steps(&mut enc, 1);
    g.queue.submit([enc.finish()]);
    let got = sim.download(&g.device, &g.queue, 0);
    let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let want = cpu_flow_step(&rule(), &FlowParams::default(), 16, &a64);
    let err = got.iter().zip(&want).map(|(&x, &y)| (x as f64 - y).abs()).fold(0.0, f64::max);
    let moved = want.iter().zip(&a64).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(moved > 1e-3, "test world barely moves ({moved}); adjust parameters");
    assert!(err < 1e-4, "max error {err} (max change {moved})");
}

#[test]
fn flow_conserves_mass() {
    let Some(g) = gpu() else { return };
    let n = 32u32;
    let a = blob_world(32, 9);
    let m0: f64 = a.iter().map(|&v| v as f64).sum();
    let mut sim = setup(&g, n, eco_off(), &a);
    let mut enc = g.device.create_command_encoder(&Default::default());
    sim.encode_steps(&mut enc, 40);
    g.queue.submit([enc.finish()]);
    let got = sim.download(&g.device, &g.queue, 0);
    let m1: f64 = got.iter().map(|&v| v as f64).sum();
    assert!(((m1 - m0) / m0).abs() < 1e-4, "mass {m0} -> {m1}");
    let changed = got.iter().zip(&a).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(changed > 1e-3, "nothing moved");
}

#[test]
fn ecology_conserves_matter_with_full_recycling() {
    let Some(g) = gpu() else { return };
    let n = 32u32;
    let a = blob_world(32, 5);
    let eco = EcologyParams { enabled: true, recycle: 1.0, regrow: 0.0, eat: 2.0, eat_min: 0.0, meta_base: 0.05, mutation_rate: 0.0, ..Default::default() };
    let fert = eco.fertility as f64;
    let mut sim = setup(&g, n, eco, &a);
    let m0: f64 = a.iter().map(|&v| v as f64).sum::<f64>() + fert * (n as f64).powi(3);
    let mut enc = g.device.create_command_encoder(&Default::default());
    sim.encode_steps(&mut enc, 30);
    g.queue.submit([enc.finish()]);
    let mass: f64 = sim.download(&g.device, &g.queue, 0).iter().map(|&v| v as f64).sum();
    let food: f64 = read_buffer_f32(&g.device, &g.queue, &sim.flow.as_ref().unwrap().food).iter().map(|&v| v as f64).sum();
    let m1 = mass + food;
    assert!(((m1 - m0) / m0).abs() < 1e-4, "matter {m0} -> {m1} (mass {mass}, food {food})");
    let a_sum: f64 = a.iter().map(|&v| v as f64).sum();
    assert!((mass - a_sum).abs() > 1e-2, "ecology had no effect");
}

#[test]
fn budding_moves_half_with_mutation() {
    let Some(g) = gpu() else { return };
    let n = 32u32;
    let mut a = vec![0.0f32; 32 * 32 * 32];
    for z in 6..10 {
        for y in 6..10 {
            for x in 6..10 {
                a[(z * 32 + y) * 32 + x] = 1.0;
            }
        }
    }
    let sim = setup(&g, n, eco_off(), &a);
    let mut delta = [0.0f32; 16];
    delta[lenia3d::sim::flow::gene::CLAW] = 0.25;
    let mut enc = g.device.create_command_encoder(&Default::default());
    sim.flow.as_ref().unwrap().bud(&g.device, &mut enc, &sim.flow_buffers(), [7.5, 7.5, 7.5], 4.0, [12, 0, 0], &delta);
    g.queue.submit([enc.finish()]);
    let got = sim.download(&g.device, &g.queue, 0);
    assert!((got.iter().sum::<f32>() - 64.0).abs() < 1e-3);
    assert_eq!(got[(7 * 32 + 7) * 32 + 7], 0.5);
    assert_eq!(got[(7 * 32 + 7) * 32 + 19], 0.5);
    let genome = read_buffer_f32(&g.device, &g.queue, &sim.flow.as_ref().unwrap().genome);
    let child = (7 * 32 + 7) * 32 + 19;
    let parent = (7 * 32 + 7) * 32 + 7;
    assert!((genome[child * 16 + 12] - 0.25).abs() < 1e-6, "child claw {}", genome[child * 16 + 12]);
    assert_eq!(genome[parent * 16 + 12], 0.0);
}

/// Two touching blobs: with different "languages" (pitch) and claws they
/// fight (matter turns into food, mass + food conserved); with the same
/// language nothing is destroyed.
#[test]
fn compatibility_gate_combat() {
    let Some(g) = gpu() else { return };
    let n = 32u32;
    let rule = rule();
    let run = |pitch_b: f32| -> (f64, f64, f64) {
        let a = blob_world(32, 11);
        let mut sim = Simulation::new(&g.device, n, rule.clone());
        sim.enable_flow(&g.device, &g.queue, FlowParams { combat: 2.0, ..Default::default() }, eco_off());
        sim.upload(&g.queue, 0, &a);
        let ga = make_genome(&rule, &Traits { claw: 1.0, pitch: 0.1, ..Default::default() });
        let gb = make_genome(&rule, &Traits { claw: 1.0, pitch: pitch_b, ..Default::default() });
        let fl = sim.flow.as_ref().unwrap();
        let mut enc = g.device.create_command_encoder(&Default::default());
        fl.brush(&g.device, &mut enc, &sim.flow_buffers(), FlowBrush::Fill, [0.0; 3], 1.0, 0.0, 0.0, &ga, 0);
        // the right half of the blob gets genome B
        for z in 0..32 {
            for y in 0..32 {
                fl.brush(&g.device, &mut enc, &sim.flow_buffers(), FlowBrush::Paint, [24.0, y as f32, z as f32], 8.0, 0.0, 0.0, &gb, 0);
            }
        }
        g.queue.submit([enc.finish()]);
        let food0: f64 = read_buffer_f32(&g.device, &g.queue, &fl.food).iter().map(|&v| v as f64).sum();
        let m0: f64 = a.iter().map(|&v| v as f64).sum();
        sim.update_flow(&g.device, &g.queue);
        let mut enc = g.device.create_command_encoder(&Default::default());
        sim.encode_steps(&mut enc, 20);
        g.queue.submit([enc.finish()]);
        let m1: f64 = sim.download(&g.device, &g.queue, 0).iter().map(|&v| v as f64).sum();
        let food1: f64 = read_buffer_f32(&g.device, &g.queue, &sim.flow.as_ref().unwrap().food).iter().map(|&v| v as f64).sum();
        (m0, m1, food1 - food0)
    };
    let (m0, m1, dfood) = run(0.9);
    assert!(m1 < m0 * 0.995, "foreign contact should destroy matter: {m0} -> {m1}");
    assert!(((m1 + dfood) - m0).abs() / m0 < 1e-4, "destroyed matter must become food: {m0} -> {m1} + {dfood}");
    let (m0, m1, dfood) = run(0.1);
    assert!(((m1 - m0) / m0).abs() < 1e-4 && dfood.abs() < 1e-3, "same language must not fight: {m0} -> {m1}, food {dfood}");
}
