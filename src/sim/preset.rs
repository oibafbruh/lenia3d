//! Presets (rule + optional creature pattern) and 3D patterns.

use super::params::LeniaParams;
use serde::{Deserialize, Serialize};

/// A dense 3D pattern, one `Vec` per channel, x fastest then y then z.
#[derive(Clone, Debug, Default)]
pub struct Pattern {
    pub dims: [u32; 3],
    pub channels: Vec<Vec<f32>>,
}

impl Pattern {
    pub fn len(&self) -> usize {
        (self.dims[0] * self.dims[1] * self.dims[2]) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Trilinear resample by `scale` (used to enlarge creatures together with R).
    pub fn scaled(&self, scale: f32) -> Pattern {
        if (scale - 1.0).abs() < 1e-3 || self.is_empty() {
            return self.clone();
        }
        let nd = self.dims.map(|d| ((d as f32 * scale).round() as u32).max(1));
        let [sx, sy, sz] = self.dims.map(|d| d as i64);
        let channels = self
            .channels
            .iter()
            .map(|src| {
                let at = |x: i64, y: i64, z: i64| -> f32 {
                    if x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz {
                        0.0
                    } else {
                        src[((z * sy + y) * sx + x) as usize]
                    }
                };
                let mut out = Vec::with_capacity((nd[0] * nd[1] * nd[2]) as usize);
                for z in 0..nd[2] {
                    for y in 0..nd[1] {
                        for x in 0..nd[0] {
                            let f = [x, y, z].map(|v| (v as f32 + 0.5) / scale - 0.5);
                            let i = f.map(|v| v.floor() as i64);
                            let t = [0, 1, 2].map(|k| f[k] - i[k] as f32);
                            let mut v = 0.0;
                            for dz in 0..2 {
                                for dy in 0..2 {
                                    for dx in 0..2 {
                                        let w = (if dx == 1 { t[0] } else { 1.0 - t[0] })
                                            * (if dy == 1 { t[1] } else { 1.0 - t[1] })
                                            * (if dz == 1 { t[2] } else { 1.0 - t[2] });
                                        v += w * at(i[0] + dx, i[1] + dy, i[2] + dz);
                                    }
                                }
                            }
                            out.push(v.clamp(0.0, 1.0));
                        }
                    }
                }
                out
            })
            .collect();
        Pattern { dims: nd, channels }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cells {
    /// [Z, Y, X]
    pub shape: [u32; 3],
    /// base64 of u8 values (A * 255), one string per channel, C-order [z][y][x].
    pub data: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoupInit {
    pub blobs: u32,
    #[serde(default = "one")]
    pub blob_radius: f32,
    #[serde(default = "half")]
    pub density: f32,
}

fn one() -> f32 {
    1.0
}
fn half() -> f32 {
    0.5
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(flatten)]
    pub rule: LeniaParams,
    #[serde(default)]
    pub cells: Option<Cells>,
    #[serde(default)]
    pub init: Option<SoupInit>,
    #[serde(default)]
    pub world_size: Option<u32>,
    #[serde(default)]
    pub kind: String,
}

impl Preset {
    /// World size this preset must run in (periodic structures spanning the world).
    pub fn required_world_size(&self) -> Option<u32> {
        let w = self.world_size?;
        let pat = self.cells.as_ref()?;
        pat.shape.contains(&w).then_some(w)
    }

    pub fn pattern(&self) -> Option<Pattern> {
        let cells = self.cells.as_ref()?;
        let [z, y, x] = cells.shape;
        let len = (x * y * z) as usize;
        let channels: Vec<Vec<f32>> = cells
            .data
            .iter()
            .map(|s| {
                let mut v: Vec<f32> = base64_decode(s).into_iter().map(|b| b as f32 / 255.0).collect();
                v.resize(len, 0.0);
                v
            })
            .collect();
        Some(Pattern { dims: [x, y, z], channels })
    }

    pub fn from_pattern(name: &str, description: &str, rule: &LeniaParams, pat: &Pattern) -> Preset {
        let [x, y, z] = pat.dims;
        Preset {
            name: name.to_string(),
            description: description.to_string(),
            rule: rule.clone(),
            cells: Some(Cells {
                shape: [z, y, x],
                data: pat
                    .channels
                    .iter()
                    .map(|c| base64_encode(&c.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect::<Vec<_>>()))
                    .collect(),
            }),
            init: None,
            world_size: None,
            kind: "creature".into(),
        }
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_decode(s: &str) -> Vec<u8> {
    let mut lut = [255u8; 256];
    for (i, &c) in B64.iter().enumerate() {
        lut[c as usize] = i as u8;
    }
    lut[b'-' as usize] = 62;
    lut[b'_' as usize] = 63;
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0;
    for &c in s.as_bytes() {
        let v = lut[c as usize];
        if v == 255 {
            continue; // padding, whitespace
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let v = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[(v >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

include!(concat!(env!("OUT_DIR"), "/builtin_presets.rs"));

/// Presets embedded in the binary from the `presets/` folder (never empty).
pub fn builtin_presets() -> Vec<Preset> {
    let mut v: Vec<Preset> = BUILTIN_PRESETS
        .iter()
        .filter_map(|(file, src)| match serde_json::from_str::<Preset>(src) {
            Ok(p) => Some(p),
            Err(e) => {
                log::warn!("bad builtin preset {file}: {e}");
                None
            }
        })
        .collect();
    if v.is_empty() {
        v.push(fallback_preset());
    }
    v
}

/// Simple single-kernel soup, used if no presets are embedded.
pub fn fallback_preset() -> Preset {
    Preset {
        name: "Primordial soup".into(),
        description: "Random blobs under a classic single-kernel rule.".into(),
        rule: super::params::LeniaParams::default(),
        cells: None,
        init: Some(SoupInit { blobs: 12, blob_radius: 1.0, density: 0.5 }),
        world_size: None,
        kind: "soup".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip() {
        for len in 0..10 {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 5) as u8).collect();
            assert_eq!(base64_decode(&base64_encode(&data)), data);
        }
    }

    #[test]
    fn builtin_presets_parse() {
        for (file, src) in BUILTIN_PRESETS {
            let p: Preset = serde_json::from_str(src).unwrap_or_else(|e| panic!("{file}: {e}"));
            if let Some(pat) = p.pattern() {
                assert_eq!(pat.channels.len(), p.rule.channels, "{file}");
            }
        }
    }

    #[test]
    fn scale_pattern() {
        let p = Pattern { dims: [2, 2, 2], channels: vec![vec![1.0; 8]] };
        let s = p.scaled(2.0);
        assert_eq!(s.dims, [4, 4, 4]);
    }
}
