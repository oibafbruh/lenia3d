//! Lenia rule parameters and the kernel/growth functions shared by CPU and GPU.
//!
//! The math follows Bert Chan's Lenia (arXiv:1812.05433), extended to multiple
//! channels and kernels as in "Lenia and Expanded Universe" (2020):
//!
//! ```text
//! U_k   = K_k * A[src_k]                         (periodic convolution)
//! D_c   = sum_{k : dst_k = c} h_k * G_k(U_k)
//! A_c  <- clip(A_c + dt * D_c, 0, 1),  dt = 1/T
//! ```

use serde::{Deserialize, Serialize};

pub const MAX_CHANNELS: usize = 3;
pub const MAX_PEAKS: usize = 4;
pub const MAX_KERNELS: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KernelParams {
    /// Source channel the kernel reads from.
    #[serde(default)]
    pub src: usize,
    /// Target channel the growth is applied to.
    #[serde(default)]
    pub dst: usize,
    /// Relative radius; the kernel radius in cells is `R * r`.
    #[serde(default = "one")]
    pub r: f32,
    /// Peak heights of the concentric shells (1..=4).
    pub b: Vec<f32>,
    /// Growth center (mu).
    pub m: f32,
    /// Growth width (sigma).
    pub s: f32,
    /// Weight of this kernel's growth.
    #[serde(default = "one")]
    pub h: f32,
    /// Kernel core: 1 polynomial, 2 exponential, 3 step, 4 staircase.
    #[serde(default = "one_u")]
    pub kn: u32,
    /// Growth function: 1 polynomial, 2 gaussian, 3 step.
    #[serde(default = "one_u")]
    pub gn: u32,
}

fn one() -> f32 {
    1.0
}
fn one_u() -> u32 {
    1
}

impl Default for KernelParams {
    fn default() -> Self {
        Self { src: 0, dst: 0, r: 1.0, b: vec![1.0], m: 0.15, s: 0.016, h: 1.0, kn: 1, gn: 1 }
    }
}

impl KernelParams {
    /// True if two kernels produce the same kernel field (shape-only comparison).
    pub fn same_shape(&self, other: &Self) -> bool {
        self.r == other.r && self.b == other.b && self.kn == other.kn
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(non_snake_case)]
pub struct LeniaParams {
    /// Base kernel radius in cells.
    pub R: f32,
    /// Time resolution; dt = 1/T.
    pub T: f32,
    pub channels: usize,
    pub kernels: Vec<KernelParams>,
}

impl Default for LeniaParams {
    fn default() -> Self {
        Self { R: 10.0, T: 10.0, channels: 1, kernels: vec![KernelParams::default()] }
    }
}

impl LeniaParams {
    pub fn dt(&self) -> f32 {
        1.0 / self.T.max(1e-3)
    }

    /// Clamp everything into the ranges the GPU code supports.
    pub fn sanitize(&mut self, n: u32) {
        self.channels = self.channels.clamp(1, MAX_CHANNELS);
        self.kernels.truncate(MAX_KERNELS);
        if self.kernels.is_empty() {
            self.kernels.push(KernelParams::default());
        }
        let max_radius = (n as f32 / 2.0 - 1.0).max(1.0);
        self.R = self.R.clamp(1.0, max_radius);
        self.T = self.T.clamp(1.0, 1000.0);
        for k in &mut self.kernels {
            k.src = k.src.min(self.channels - 1);
            k.dst = k.dst.min(self.channels - 1);
            k.r = k.r.clamp(0.05, max_radius / self.R);
            k.b.truncate(MAX_PEAKS);
            if k.b.is_empty() {
                k.b.push(1.0);
            }
            for b in &mut k.b {
                *b = b.clamp(0.0, 1.0);
            }
            k.s = k.s.max(1e-4);
            k.kn = k.kn.clamp(1, 4);
            k.gn = k.gn.clamp(1, 3);
        }
    }

    /// Kernel radius in cells.
    pub fn radius(&self, k: usize) -> f32 {
        self.R * self.kernels[k].r
    }

    /// Whether changing from `old` to `self` requires regenerating kernel spectra.
    pub fn kernels_shape_changed(&self, old: &Self) -> bool {
        self.R != old.R
            || self.kernels.len() != old.kernels.len()
            || self.kernels.iter().zip(&old.kernels).any(|(a, b)| !a.same_shape(b))
    }
}

/// Kernel core function on `[0, 1]`.
pub fn kernel_core(f: f64, kn: u32) -> f64 {
    match kn {
        2 => {
            if f > 0.0 && f < 1.0 {
                (4.0 - 1.0 / (f * (1.0 - f))).exp()
            } else {
                0.0
            }
        }
        3 => {
            if (0.25..=0.75).contains(&f) {
                1.0
            } else {
                0.0
            }
        }
        4 => {
            if (0.25..=0.75).contains(&f) {
                1.0
            } else if f < 0.25 {
                0.5
            } else {
                0.0
            }
        }
        _ => (4.0 * f * (1.0 - f)).powi(4),
    }
}

/// Kernel shell value at normalized distance `d` (distance / radius).
pub fn kernel_shell(d: f64, b: &[f32], kn: u32) -> f64 {
    if d >= 1.0 || b.is_empty() {
        return 0.0;
    }
    let nb = b.len();
    let br = nb as f64 * d;
    let fi = br.floor();
    let idx = (fi as usize).min(nb - 1);
    b[idx] as f64 * kernel_core(br - fi, kn)
}

/// Growth function.
pub fn growth(u: f64, m: f64, s: f64, gn: u32) -> f64 {
    match gn {
        2 => (-(u - m).powi(2) / (2.0 * s * s)).exp() * 2.0 - 1.0,
        3 => {
            if (u - m).abs() <= s {
                1.0
            } else {
                -1.0
            }
        }
        _ => (1.0 - (u - m).powi(2) / (9.0 * s * s)).max(0.0).powi(4) * 2.0 - 1.0,
    }
}

/// Sum of the discrete (unnormalized) kernel over the integer lattice,
/// matching exactly what the GPU kernel generator produces.
pub fn kernel_sum(radius: f64, b: &[f32], kn: u32) -> f64 {
    let r = radius.ceil() as i64 + 1;
    let mut sum = 0.0;
    for z in -r..=r {
        for y in -r..=r {
            for x in -r..=r {
                let d = ((x * x + y * y + z * z) as f64).sqrt() / radius;
                if d < 1.0 {
                    sum += kernel_shell(d, b, kn);
                }
            }
        }
    }
    sum
}

/// Parse Chan-style fraction lists like "1,5/12,1/6".
pub fn parse_fractions(s: &str) -> Vec<f32> {
    s.split(',')
        .filter_map(|t| {
            let t = t.trim();
            if t.is_empty() {
                return None;
            }
            if let Some((a, b)) = t.split_once('/') {
                let a: f32 = a.trim().parse().ok()?;
                let b: f32 = b.trim().parse().ok()?;
                Some(if b != 0.0 { a / b } else { 0.0 })
            } else {
                t.parse().ok()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions() {
        assert_eq!(parse_fractions("1,1/2, 3/4"), vec![1.0, 0.5, 0.75]);
    }

    #[test]
    fn growth_peak() {
        assert!((growth(0.15, 0.15, 0.02, 1) - 1.0).abs() < 1e-12);
        assert!((growth(0.15, 0.15, 0.02, 2) - 1.0).abs() < 1e-12);
        assert_eq!(growth(0.9, 0.15, 0.02, 1), -1.0);
    }

    #[test]
    fn shell_shape() {
        // polynomial core peaks at the middle of the shell
        assert!((kernel_shell(0.5, &[1.0], 1) - 1.0).abs() < 1e-12);
        assert_eq!(kernel_shell(1.0, &[1.0], 1), 0.0);
        // second shell scaled by its peak height
        assert!((kernel_shell(0.75, &[1.0, 0.5], 1) - 0.5).abs() < 1e-12);
    }
}
