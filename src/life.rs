//! Reproduction decisions from creature tracking: asexual budding of
//! well-fed creatures and mating of compatible creatures that touch.

use crate::sim::flow::{clamp_genome, gene, genetic_distance, EcologyParams, FlowParams, Genome, GENES};
use crate::tracker::{periodic_distance, wrap_delta, Tracker};
use crate::world::Rng;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReproKind {
    Bud,
    Mate,
}

/// One reproduction operation to run on the GPU (cell coordinates).
#[derive(Clone, Debug)]
pub struct ReproOp {
    pub kind: ReproKind,
    pub parent: u64,
    pub partner: Option<u64>,
    pub center: [f32; 3],
    /// Radius of the parent region that gets split.
    pub radius: f32,
    pub offset: [i32; 3],
    /// Added to every moved cell's genome.
    pub delta: Genome,
    /// Where the child appears.
    pub child_center: [f32; 3],
    pub child_pitch: f32,
}

fn gauss(rng: &mut Rng) -> f32 {
    let u1 = rng.f32().max(1e-7);
    let u2 = rng.f32();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn unit(rng: &mut Rng) -> [f32; 3] {
    loop {
        let v = [rng.f32() * 2.0 - 1.0, rng.f32() * 2.0 - 1.0, rng.f32() * 2.0 - 1.0];
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if l > 0.1 && l <= 1.0 {
            return v.map(|c| c / l);
        }
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    v.map(|c| c / l)
}

/// Offset for a child of a creature at `center` with split radius `rbox`,
/// choosing among `dirs` the direction with the most free space.
fn place(tracker: &Tracker, center: [f32; 3], rbox: f32, dirs: &[[f32; 3]], n: f32) -> ([i32; 3], [f32; 3]) {
    let len = (2.0 * rbox + 2.0).min(n / 2.0 - 1.0);
    let mut best = (f32::MIN, [0i32; 3], center);
    for d in dirs {
        let off = d.map(|c| (c * len).round() as i32);
        let target = [0, 1, 2].map(|k| (center[k] + off[k] as f32).rem_euclid(n));
        let space = tracker
            .creatures
            .iter()
            .map(|c| periodic_distance(c.center, target, n) - c.radius)
            .fold(f32::MAX, f32::min);
        if space > best.0 {
            best = (space, off, target);
        }
    }
    (best.1, best.2)
}

/// Decide this round's buddings and matings and mark the creatures involved.
pub fn plan_reproduction(
    tracker: &mut Tracker,
    eco: &EcologyParams,
    flow: &FlowParams,
    n: u32,
    generation: u64,
    rng: &mut Rng,
    max_ops: usize,
) -> Vec<ReproOp> {
    let mut ops = Vec::new();
    if !eco.enabled || eco.bud_mass <= 0.0 {
        return ops;
    }
    let nf = n as f32;
    let ready = |c: &crate::tracker::Creature| {
        c.age(generation) as f32 > eco.bud_age && generation.saturating_sub(c.last_repro) as f32 > eco.bud_age
    };
    // mating first: compatible creatures in contact
    let cs = tracker.creatures.clone();
    let mut used = std::collections::HashSet::new();
    for i in 0..cs.len() {
        for j in i + 1..cs.len() {
            if ops.len() >= max_ops {
                break;
            }
            let (a, b) = (&cs[i], &cs[j]);
            if used.contains(&a.id) || used.contains(&b.id) || !ready(a) || !ready(b) {
                continue;
            }
            if a.mass < eco.bud_mass * 0.5 || b.mass < eco.bud_mass * 0.5 {
                continue;
            }
            if periodic_distance(a.center, b.center, nf) > (a.radius + b.radius) * 1.15 + 2.0 {
                continue;
            }
            if genetic_distance(&a.genome, &b.genome) > flow.compat {
                continue;
            }
            let (p, q) = if a.mass >= b.mass { (a, b) } else { (b, a) };
            let mut child = [0.0f32; GENES];
            for (k, c) in child.iter_mut().enumerate() {
                *c = if rng.f32() < 0.5 { p.genome[k] } else { q.genome[k] };
                *c += eco.bud_mutation * gauss(rng);
            }
            // hue: average of the parents on the color circle, slightly mutated
            let dh = wrap_delta(p.genome[gene::HUE], q.genome[gene::HUE], 1.0);
            child[gene::HUE] = p.genome[gene::HUE] + dh * 0.5 + eco.bud_mutation * gauss(rng);
            clamp_genome(&mut child);
            let axis = normalize([0, 1, 2].map(|k| wrap_delta(p.center[k], q.center[k], nf)));
            let dirs: Vec<[f32; 3]> = (0..8).map(|_| normalize(cross(axis, unit(rng)))).collect();
            let rbox = p.radius * 1.3 + 1.0;
            let (offset, child_center) = place(tracker, p.center, rbox, &dirs, nf);
            let mut delta = [0.0f32; GENES];
            for k in 0..GENES {
                delta[k] = child[k] - p.genome[k];
            }
            ops.push(ReproOp {
                kind: ReproKind::Mate,
                parent: p.id,
                partner: Some(q.id),
                center: p.center,
                radius: rbox,
                offset,
                delta,
                child_center,
                child_pitch: child[gene::PITCH],
            });
            used.insert(p.id);
            used.insert(q.id);
        }
    }
    // budding: heavy, well-fed creatures
    for c in &cs {
        if ops.len() >= max_ops {
            break;
        }
        if used.contains(&c.id) || c.mass < eco.bud_mass || !ready(c) {
            continue;
        }
        let mut delta = [0.0f32; GENES];
        for d in delta.iter_mut() {
            *d = eco.bud_mutation * gauss(rng);
        }
        let mut child = c.genome;
        for k in 0..GENES {
            child[k] += delta[k];
        }
        clamp_genome(&mut child);
        let dirs: Vec<[f32; 3]> = (0..8).map(|_| unit(rng)).collect();
        let rbox = c.radius * 1.3 + 1.0;
        let (offset, child_center) = place(tracker, c.center, rbox, &dirs, nf);
        ops.push(ReproOp {
            kind: ReproKind::Bud,
            parent: c.id,
            partner: None,
            center: c.center,
            radius: rbox,
            offset,
            delta,
            child_center,
            child_pitch: child[gene::PITCH],
        });
        used.insert(c.id);
    }
    for op in &ops {
        for id in std::iter::once(op.parent).chain(op.partner) {
            if let Some(c) = tracker.get_mut(id) {
                c.last_repro = generation;
            }
        }
        tracker.expect_offspring(op.child_center, op.parent, generation);
    }
    ops
}
