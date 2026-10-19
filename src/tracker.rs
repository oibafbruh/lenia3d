//! Creature tracking on the coarse statistics grid (see docs/flow_lenia.md):
//! periodic connected components, frame-to-frame matching, life events and
//! population history.

use crate::sim::flow::{gene, Genome, GENES};
use crate::sim::TRACK_FLOATS;

#[derive(Clone, Debug)]
pub struct Creature {
    pub id: u64,
    /// Center of mass in cell coordinates (periodic, in [0, n)).
    pub center: [f32; 3],
    /// Smoothed velocity in cells per step.
    pub velocity: [f32; 3],
    pub mass: f32,
    /// Effective radius in cells.
    pub radius: f32,
    /// Mass-weighted mean genome (hue from the circular mean).
    pub genome: Genome,
    /// Generation when first seen.
    pub born: u64,
    /// Generation of the last budding or mating.
    pub last_repro: u64,
    /// Smoothed mass change per step.
    pub growth: f32,
    /// Parent id if this creature was born from a tracked creature.
    pub parent: Option<u64>,
    pub generation_depth: u32,
}

impl Creature {
    pub fn age(&self, now: u64) -> u64 {
        now.saturating_sub(self.born)
    }
    pub fn speed(&self) -> f32 {
        let v = self.velocity;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }
}

#[derive(Clone, Debug)]
pub enum LifeEvent {
    /// A new creature appeared (natural fission, seeding, or offspring).
    Born { id: u64, center: [f32; 3], pitch: f32 },
    /// A creature vanished (starved, eaten or merged).
    Died { id: u64, center: [f32; 3], pitch: f32 },
}

#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub generation: u64,
    pub count: f32,
    pub mass: f32,
    pub food: f32,
    pub eye: f32,
    pub ear: f32,
    pub voice: f32,
    pub mouth: f32,
    pub claw: f32,
    pub alarm: f32,
    pub languages: f32,
}

pub struct Tracker {
    pub creatures: Vec<Creature>,
    pub events: Vec<LifeEvent>,
    pub history: Vec<Sample>,
    next_id: u64,
    last_gen: Option<u64>,
    n: u32,
    /// Offspring placed by budding/mating: (position, parent id, generation).
    expected: Vec<([f32; 3], u64, u64)>,
}

impl Default for Tracker {
    fn default() -> Self {
        Self { creatures: vec![], events: vec![], history: vec![], next_id: 1, last_gen: None, n: 0, expected: vec![] }
    }
}

/// Periodic difference b - a in a world of size n.
pub fn wrap_delta(a: f32, b: f32, n: f32) -> f32 {
    let mut d = b - a;
    if d > n / 2.0 {
        d -= n;
    } else if d < -n / 2.0 {
        d += n;
    }
    d
}

pub fn periodic_distance(a: [f32; 3], b: [f32; 3], n: f32) -> f32 {
    let d = [0, 1, 2].map(|k| wrap_delta(a[k], b[k], n));
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

struct Dsu(Vec<u32>);

impl Dsu {
    fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let p = self.0[self.0[x as usize] as usize];
            self.0[x as usize] = p;
            x = p;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a.max(b) as usize] = a.min(b);
        }
    }
}

/// A connected blob found in one frame.
#[derive(Clone, Debug)]
pub struct Blob {
    pub center: [f32; 3],
    pub mass: f32,
    pub radius: f32,
    pub genome: Genome,
    pub blocks: u32,
}

/// Find connected blobs in the coarse grid (`m`³ blocks of `f`³ cells each).
pub fn find_blobs(data: &[f32], m: u32, f: u32, has_genome: bool) -> Vec<Blob> {
    let mu = m as usize;
    let count = mu * mu * mu;
    let thr = 0.03 * (f * f * f) as f32;
    let on: Vec<bool> = (0..count).map(|i| data[i * TRACK_FLOATS] > thr).collect();
    let mut dsu = Dsu((0..count as u32).collect());
    let idx = |x: i32, y: i32, z: i32| -> usize {
        let w = |v: i32| v.rem_euclid(m as i32) as usize;
        (w(z) * mu + w(y)) * mu + w(x)
    };
    for z in 0..m as i32 {
        for y in 0..m as i32 {
            for x in 0..m as i32 {
                let i = idx(x, y, z);
                if !on[i] {
                    continue;
                }
                // half of the 26 neighbors is enough for union-find
                for (dx, dy, dz) in [
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    (1, 1, 0),
                    (1, -1, 0),
                    (1, 0, 1),
                    (1, 0, -1),
                    (0, 1, 1),
                    (0, 1, -1),
                    (1, 1, 1),
                    (1, 1, -1),
                    (1, -1, 1),
                    (1, -1, -1),
                ] {
                    let j = idx(x + dx, y + dy, z + dz);
                    if on[j] {
                        dsu.union(i as u32, j as u32);
                    }
                }
            }
        }
    }
    // accumulate per root
    struct Acc {
        mass: f64,
        blocks: u32,
        genes: [f64; GENES],
        hue: (f64, f64),
        circ: [(f64, f64); 3],
    }
    let mut accs: std::collections::HashMap<u32, Acc> = std::collections::HashMap::new();
    let tau = std::f64::consts::TAU;
    for (i, &is_on) in on.iter().enumerate() {
        if !is_on {
            continue;
        }
        let r = dsu.find(i as u32);
        let d = &data[i * TRACK_FLOATS..(i + 1) * TRACK_FLOATS];
        let a = accs.entry(r).or_insert(Acc { mass: 0.0, blocks: 0, genes: [0.0; GENES], hue: (0.0, 0.0), circ: [(0.0, 0.0); 3] });
        let w = d[0] as f64;
        a.mass += w;
        a.blocks += 1;
        for k in 0..GENES {
            a.genes[k] += d[1 + k] as f64;
        }
        a.hue.0 += d[17] as f64;
        a.hue.1 += d[18] as f64;
        let coords = [i % mu, (i / mu) % mu, i / (mu * mu)];
        for k in 0..3 {
            let ang = coords[k] as f64 / m as f64 * tau;
            a.circ[k].0 += w * ang.cos();
            a.circ[k].1 += w * ang.sin();
        }
    }
    let mut blobs: Vec<Blob> = accs
        .into_values()
        .filter(|a| a.mass > 1.0)
        .map(|a| {
            let center = a.circ.map(|(c, s)| {
                let b = (s.atan2(c) / tau * m as f64).rem_euclid(m as f64);
                (b * f as f64 + (f as f64 - 1.0) / 2.0) as f32
            });
            let mut genome = [0.0f32; GENES];
            if has_genome && a.mass > 0.0 {
                for k in 0..GENES {
                    genome[k] = (a.genes[k] / a.mass) as f32;
                }
                genome[gene::HUE] = (a.hue.1.atan2(a.hue.0) / tau).rem_euclid(1.0) as f32;
            }
            let cells = a.blocks as f64 * (f as f64).powi(3);
            let radius = (3.0 * cells / (4.0 * std::f64::consts::PI)).cbrt() as f32;
            Blob { center, mass: a.mass as f32, radius, genome, blocks: a.blocks }
        })
        .collect();
    blobs.sort_by(|a, b| b.mass.total_cmp(&a.mass));
    blobs
}

/// Number of distinct pitch clusters ("languages") among creatures.
pub fn count_languages(creatures: &[Creature]) -> usize {
    let mut p: Vec<f32> = creatures.iter().filter(|c| c.genome[gene::VOICE] > 0.05).map(|c| c.genome[gene::PITCH]).collect();
    if p.is_empty() {
        return 0;
    }
    p.sort_by(|a, b| a.total_cmp(b));
    1 + p.windows(2).filter(|w| w[1] - w[0] > 0.08).count()
}

impl Tracker {
    pub fn reset(&mut self) {
        let next_id = self.next_id;
        *self = Self::default();
        self.next_id = next_id;
    }

    /// The world was translated by `shift` cells (follow mode).
    pub fn shift(&mut self, shift: [i32; 3]) {
        let n = self.n as f32;
        if n <= 0.0 {
            return;
        }
        let mv = |c: &mut [f32; 3]| {
            for k in 0..3 {
                c[k] = (c[k] + shift[k] as f32).rem_euclid(n);
            }
        };
        for c in &mut self.creatures {
            mv(&mut c.center);
        }
        for e in &mut self.expected {
            mv(&mut e.0);
        }
    }

    /// Record that an offspring was just placed at `center` (cell coords).
    pub fn expect_offspring(&mut self, center: [f32; 3], parent: u64, generation: u64) {
        self.expected.push((center, parent, generation));
    }

    pub fn get(&self, id: u64) -> Option<&Creature> {
        self.creatures.iter().find(|c| c.id == id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut Creature> {
        self.creatures.iter_mut().find(|c| c.id == id)
    }

    /// Update from a coarse statistics readback taken at `generation`.
    pub fn update(&mut self, data: &[f32], m: u32, f: u32, n: u32, generation: u64, has_genome: bool) {
        if n != self.n {
            self.reset();
            self.n = n;
        }
        let nf = n as f32;
        let blobs = find_blobs(data, m, f, has_genome);
        let dt_steps = self.last_gen.map(|g| generation.saturating_sub(g)).unwrap_or(0).max(1) as f32;
        if self.last_gen.is_some_and(|g| generation < g) {
            // world was reset or reloaded
            self.creatures.clear();
        }
        self.last_gen = Some(generation);

        // greedy one-to-one matching by distance
        let mut pairs = Vec::new();
        for (bi, b) in self.creatures.iter().enumerate() {
            let predicted = [0, 1, 2].map(|k| b.center[k] + b.velocity[k] * dt_steps);
            for (ni, nb) in blobs.iter().enumerate() {
                let d = periodic_distance(predicted, nb.center, nf);
                let limit = b.radius.max(nb.radius) + 2.0 + f as f32;
                if d < limit {
                    pairs.push((d, bi, ni));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut old_taken = vec![false; self.creatures.len()];
        let mut new_taken: Vec<Option<usize>> = vec![None; blobs.len()];
        for (_, bi, ni) in pairs {
            if !old_taken[bi] && new_taken[ni].is_none() {
                old_taken[bi] = true;
                new_taken[ni] = Some(bi);
            }
        }

        let mut next = Vec::with_capacity(blobs.len());
        for (ni, nb) in blobs.iter().enumerate() {
            match new_taken[ni] {
                Some(bi) => {
                    let old = &self.creatures[bi];
                    let v = [0, 1, 2].map(|k| wrap_delta(old.center[k], nb.center[k], nf) / dt_steps);
                    let velocity = [0, 1, 2].map(|k| old.velocity[k] * 0.6 + v[k] * 0.4);
                    let growth = old.growth * 0.7 + (nb.mass - old.mass) / dt_steps * 0.3;
                    next.push(Creature {
                        center: nb.center,
                        velocity,
                        mass: nb.mass,
                        radius: nb.radius,
                        genome: nb.genome,
                        growth,
                        ..old.clone()
                    });
                }
                None => {
                    let id = self.next_id;
                    self.next_id += 1;
                    // offspring we placed ourselves, or a fission of a nearby creature
                    let mut parent = None;
                    let mut depth = 0;
                    if let Some(pos) =
                        self.expected.iter().position(|(c, _, g)| periodic_distance(*c, nb.center, nf) < nb.radius + 4.0 && generation >= *g)
                    {
                        let (_, p, _) = self.expected.swap_remove(pos);
                        parent = Some(p);
                    } else if let Some(near) = self
                        .creatures
                        .iter()
                        .filter(|c| periodic_distance(c.center, nb.center, nf) < c.radius + nb.radius + 3.0)
                        .min_by(|a, b| periodic_distance(a.center, nb.center, nf).total_cmp(&periodic_distance(b.center, nb.center, nf)))
                    {
                        parent = Some(near.id);
                    }
                    if let Some(p) = parent.and_then(|p| self.creatures.iter().find(|c| c.id == p)) {
                        depth = p.generation_depth + 1;
                    }
                    self.events.push(LifeEvent::Born { id, center: nb.center, pitch: nb.genome[gene::PITCH] });
                    next.push(Creature {
                        id,
                        center: nb.center,
                        velocity: [0.0; 3],
                        mass: nb.mass,
                        radius: nb.radius,
                        genome: nb.genome,
                        born: generation,
                        last_repro: generation,
                        growth: 0.0,
                        parent,
                        generation_depth: depth,
                    });
                }
            }
        }
        for (bi, taken) in old_taken.iter().enumerate() {
            if !taken {
                let c = &self.creatures[bi];
                self.events.push(LifeEvent::Died { id: c.id, center: c.center, pitch: c.genome[gene::PITCH] });
            }
        }
        self.creatures = next;
        self.expected.retain(|(_, _, g)| generation < g + 600);
        // keep the event queue bounded if nobody drains it
        if self.events.len() > 256 {
            let drop = self.events.len() - 256;
            self.events.drain(..drop);
        }

        // population sample
        let total: f32 = self.creatures.iter().map(|c| c.mass).sum();
        let food: f32 = (0..(m as usize).pow(3)).map(|i| data[i * TRACK_FLOATS + 19]).sum();
        let mean = |g: usize| {
            if total > 0.0 {
                self.creatures.iter().map(|c| c.genome[g] * c.mass).sum::<f32>() / total
            } else {
                0.0
            }
        };
        let s = Sample {
            generation,
            count: self.creatures.len() as f32,
            mass: total,
            food,
            eye: mean(gene::EYE),
            ear: mean(gene::EAR),
            voice: mean(gene::VOICE),
            mouth: mean(gene::MOUTH),
            claw: mean(gene::CLAW),
            alarm: mean(gene::ALARM),
            languages: count_languages(&self.creatures) as f32,
        };
        self.history.push(s);
        if self.history.len() > 4000 {
            // thin out the older half
            let half = self.history.len() / 2;
            let mut thinned: Vec<Sample> = self.history[..half].iter().step_by(2).cloned().collect();
            thinned.extend_from_slice(&self.history[half..]);
            self.history = thinned;
        }
    }

    pub fn drain_events(&mut self) -> Vec<LifeEvent> {
        std::mem::take(&mut self.events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(m: usize, blocks: &[(usize, usize, usize, f32)]) -> Vec<f32> {
        let mut d = vec![0.0f32; m * m * m * TRACK_FLOATS];
        for &(x, y, z, mass) in blocks {
            let i = (z * m + y) * m + x;
            d[i * TRACK_FLOATS] = mass;
            d[i * TRACK_FLOATS + 1 + gene::PITCH] = mass * 0.5;
        }
        d
    }

    #[test]
    fn blobs_wrap_around() {
        // one blob straddling the x boundary, one separate blob
        let d = grid(8, &[(7, 3, 3, 10.0), (0, 3, 3, 10.0), (4, 6, 6, 5.0)]);
        let b = find_blobs(&d, 8, 1, true);
        assert_eq!(b.len(), 2);
        assert!((b[0].mass - 20.0).abs() < 1e-4);
        // center between 7 and 0 (wrapped): about -0.5 / 7.5
        assert!((b[0].center[0] - 7.5).abs() < 0.01 || b[0].center[0].abs() < 0.51, "{:?}", b[0].center);
        assert!((b[0].genome[gene::PITCH] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn tracking_keeps_ids_and_reports_events() {
        let mut t = Tracker::default();
        t.update(&grid(8, &[(2, 2, 2, 10.0)]), 8, 1, 8, 0, true);
        let id = t.creatures[0].id;
        t.update(&grid(8, &[(3, 2, 2, 10.0)]), 8, 1, 8, 10, true);
        assert_eq!(t.creatures[0].id, id);
        assert!(t.creatures[0].velocity[0] > 0.0);
        t.update(&grid(8, &[(3, 2, 2, 10.0), (6, 6, 6, 4.0)]), 8, 1, 8, 20, true);
        assert_eq!(t.creatures.len(), 2);
        t.update(&grid(8, &[(6, 6, 6, 4.0)]), 8, 1, 8, 30, true);
        let ev = t.drain_events();
        assert!(ev.iter().any(|e| matches!(e, LifeEvent::Died { id: d, .. } if *d == id)));
        assert_eq!(count_languages(&t.creatures), 0);
    }
}
