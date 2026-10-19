//! Import of Bert Chan's published 3D Lenia species (`animals3D.json`).
//!
//! His repository has no license file, so the species are not bundled; the app
//! downloads the file on request or loads a copy the user provides.

use super::params::{parse_fractions, KernelParams, LeniaParams};
use super::preset::{Pattern, Preset};
use anyhow::{anyhow, Result};

pub const ANIMALS3D_URL: &str = "https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals3D.json";

/// Convert Chan's species list into presets. Header entries (no params) are skipped.
pub fn parse_animals(json: &str) -> Result<Vec<Preset>> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let arr = v.as_array().ok_or_else(|| anyhow!("expected a JSON array"))?;
    let mut out = Vec::new();
    let mut family = String::new();
    for e in arr {
        let Some(p) = e.get("params") else {
            if let Some(name) = e.get("name").and_then(|n| n.as_str()) {
                family = name.trim().to_string();
            }
            continue;
        };
        let get = |k: &str| p.get(k).and_then(|v| v.as_f64());
        let b = match p.get("b") {
            Some(serde_json::Value::String(s)) => parse_fractions(s),
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect(),
            _ => vec![1.0],
        };
        let rule = LeniaParams {
            R: get("R").unwrap_or(10.0) as f32,
            T: get("T").unwrap_or(10.0) as f32,
            channels: 1,
            kernels: vec![KernelParams {
                src: 0,
                dst: 0,
                r: 1.0,
                b,
                m: get("m").unwrap_or(0.15) as f32,
                s: get("s").unwrap_or(0.015) as f32,
                h: 1.0,
                kn: get("kn").unwrap_or(1.0) as u32,
                gn: get("gn").unwrap_or(1.0) as u32,
            }],
        };
        let cells = match e.get("cells") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(""),
            _ => continue,
        };
        let pat = rle_decode_3d(&cells);
        if pat.is_empty() {
            continue;
        }
        let code = e.get("code").and_then(|c| c.as_str()).unwrap_or("").trim();
        let name = e.get("name").and_then(|c| c.as_str()).unwrap_or("").trim();
        let title = match (code.is_empty(), name.is_empty()) {
            (false, false) => format!("{code} {name}"),
            (true, false) => name.to_string(),
            (false, true) => code.to_string(),
            _ => format!("Species {}", out.len() + 1),
        };
        let mut preset = Preset::from_pattern(&title, &format!("Bert Chan, {family}"), &rule, &pat);
        preset.kind = "chan".into();
        // Sheets and rods span the whole periodic world along some axis: they
        // only survive in a world of exactly that size.
        let max_dim = pat.dims.iter().copied().max().unwrap_or(0);
        if max_dim >= 64 && max_dim.is_power_of_two() {
            preset.world_size = Some(max_dim);
            preset.description = format!("{} (spans the periodic world: needs exactly {max_dim}³)", preset.description);
        }
        out.push(preset);
    }
    Ok(out)
}

fn ch2val(token: &str) -> f32 {
    let b = token.as_bytes();
    let v = match b {
        [b'.'] | [b'b'] => 0,
        [b'o'] => 255,
        [c] => (*c as i32 - b'A' as i32 + 1).clamp(0, 255),
        [c0, c1] => ((*c0 as i32 - b'p' as i32) * 24 + (*c1 as i32 - b'A' as i32 + 25)).clamp(0, 255),
        _ => 0,
    };
    v as f32 / 255.0
}

/// Decode Chan's N-dimensional RLE for 3D: '$' ends a row, '%' ends a layer.
/// Mirrors `Board.rle2arr` in his LeniaND.py, including its repeat-count quirks.
pub fn rle_decode_3d(st: &str) -> Pattern {
    let mut row: Vec<f32> = Vec::new();
    let mut layer: Vec<Vec<f32>> = Vec::new();
    let mut volume: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut last = String::new();
    let mut count = String::new();
    let s = format!("{}%", st.trim().trim_end_matches('!'));
    for ch in s.chars() {
        if ch.is_whitespace() {
            continue;
        }
        if ch.is_ascii_digit() {
            count.push(ch);
        } else if "pqrstuvwxy@".contains(ch) {
            last = ch.to_string();
        } else {
            let token = format!("{last}{ch}");
            let n: usize = count.parse().unwrap_or(1).max(1);
            match token.as_str() {
                "$" => {
                    layer.push(std::mem::take(&mut row));
                    layer.extend(std::iter::repeat_n(Vec::new(), n - 1));
                }
                "%" => {
                    layer.push(std::mem::take(&mut row));
                    layer.extend(std::iter::repeat_n(Vec::new(), n - 1));
                    volume.push(std::mem::take(&mut layer));
                    volume.extend(std::iter::repeat_n(Vec::new(), n - 1));
                }
                _ => {
                    let v = ch2val(&token);
                    row.extend(std::iter::repeat_n(v, n));
                }
            }
            last.clear();
            count.clear();
        }
    }
    let nz = volume.len();
    let ny = volume.iter().map(|l| l.len()).max().unwrap_or(0);
    let nx = volume.iter().flat_map(|l| l.iter().map(|r| r.len())).max().unwrap_or(0);
    let mut data = vec![0.0f32; nx * ny * nz];
    for (z, l) in volume.iter().enumerate() {
        for (y, r) in l.iter().enumerate() {
            for (x, &v) in r.iter().enumerate() {
                data[(z * ny + y) * nx + x] = v;
            }
        }
    }
    Pattern { dims: [nx as u32, ny as u32, nz as u32], channels: vec![data] }
}

/// Download the species file with the system's curl (ships with Windows 10+).
pub fn download_animals() -> Result<String> {
    let out = std::process::Command::new("curl")
        .args(["-fsSL", "--max-time", "60", ANIMALS3D_URL])
        .output()
        .map_err(|e| anyhow!("could not run curl: {e}"))?;
    if !out.status.success() {
        return Err(anyhow!("download failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8(out.stdout)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_simple() {
        // 2 layers of 2x3
        let p = rle_decode_3d("o2.$3o%A.o$2.B!");
        assert_eq!(p.dims, [3, 2, 2]);
        let d = &p.channels[0];
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 0.0);
        assert_eq!(d[3..6], [1.0, 1.0, 1.0]);
        assert!((d[6] - 1.0 / 255.0).abs() < 1e-6);
        assert!((d[11] - 2.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn two_char_values() {
        let p = rle_decode_3d("yO!");
        assert!((p.channels[0][0] - 1.0).abs() < 1e-6); // (24-15)*24... = 255
    }
}
