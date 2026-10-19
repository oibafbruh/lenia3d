//! Sonification: every tracked creature that calls is an FM voice (pitch from
//! its pitch gene on a pentatonic scale, chirp rhythm shared with the visual
//! sound rings), plus chimes for births and matings, a soft tone for deaths,
//! an ambient pad and a small reverb.

use crate::world::call_envelope;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const MAX_VOICES: usize = 24;

/// Parameters of one creature voice (set by the main thread).
#[derive(Clone, Copy, Debug)]
pub struct VoiceParams {
    pub id: u64,
    pub freq: f32,
    /// Loudness after distance attenuation (0..1).
    pub amp: f32,
    /// -1 left .. 1 right.
    pub pan: f32,
    pub timbre: f32,
    /// Voice gene (drives the chirp rhythm).
    pub voice: f32,
    /// 0..1: screaming (alarm call).
    pub scream: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum Sfx {
    Bud { freq: f32, pan: f32 },
    Mate { freq: f32, pan: f32 },
    Birth { freq: f32, pan: f32 },
    Death { freq: f32, pan: f32 },
}

#[derive(Clone, Debug)]
pub struct AudioSettings {
    pub master: f32,
    pub reverb: f32,
    pub muted: bool,
    pub voices: bool,
    pub events: bool,
    /// Ambient pad level (0 = off).
    pub ambient: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self { master: 0.6, reverb: 0.35, muted: false, voices: true, events: true, ambient: 0.25 }
    }
}

#[derive(Default)]
struct Scene {
    voices: Vec<VoiceParams>,
    sfx: Vec<Sfx>,
    settings: Option<AudioSettings>,
    app_time: f64,
    /// Ambient intensity from the amount of life (0..1).
    life: f32,
    pad_root: f32,
}

/// Pitch gene (0..1) to a frequency on an A minor pentatonic scale (3 octaves from A2).
pub fn pentatonic(p: f32) -> f32 {
    const SCALE: [f32; 5] = [0.0, 3.0, 5.0, 7.0, 10.0];
    let idx = ((p.clamp(0.0, 0.999) * 15.0) as usize).min(14);
    let semis = 12.0 * (idx / 5) as f32 + SCALE[idx % 5];
    110.0 * 2f32.powf(semis / 12.0)
}

pub struct Audio {
    _stream: Option<cpal::Stream>,
    shared: Arc<Mutex<Scene>>,
    pub available: bool,
    pub status: String,
}

impl Audio {
    /// Open the default output device; on failure the app keeps running silently.
    pub fn new() -> Self {
        let shared = Arc::new(Mutex::new(Scene::default()));
        match Self::open(shared.clone()) {
            Ok((stream, status)) => Self { _stream: Some(stream), shared, available: true, status },
            Err(e) => {
                log::warn!("audio disabled: {e}");
                Self { _stream: None, shared, available: false, status: format!("no audio: {e}") }
            }
        }
    }

    fn open(shared: Arc<Mutex<Scene>>) -> anyhow::Result<(cpal::Stream, String)> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or_else(|| anyhow::anyhow!("no output device"))?;
        let supported = device.default_output_config()?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let sr = config.sample_rate as f32;
        let ch = config.channels as usize;
        let status = format!("{} Hz, {} ch, {:?}", config.sample_rate, ch, format);
        let mut synth = Synth::new(sr, shared);
        let err = |e| log::warn!("audio stream error: {e}");
        let stream = match format {
            cpal::SampleFormat::F32 => device.build_output_stream(
                config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| synth.fill(data, ch, |v| v),
                err,
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_output_stream(
                config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| synth.fill(data, ch, |v| (v * 32767.0) as i16),
                err,
                None,
            )?,
            cpal::SampleFormat::U16 => device.build_output_stream(
                config,
                move |data: &mut [u16], _: &cpal::OutputCallbackInfo| synth.fill(data, ch, |v| ((v * 0.5 + 0.5) * 65535.0) as u16),
                err,
                None,
            )?,
            other => anyhow::bail!("unsupported sample format {other:?}"),
        };
        stream.play()?;
        Ok((stream, status))
    }

    /// Send the current voices (called a few times per second).
    pub fn update(&self, voices: Vec<VoiceParams>, app_time: f64, settings: &AudioSettings, life: f32, pad_root: f32) {
        if let Ok(mut s) = self.shared.lock() {
            s.voices = voices;
            s.app_time = app_time;
            s.settings = Some(settings.clone());
            s.life = life;
            s.pad_root = pad_root;
        }
    }

    pub fn play(&self, sfx: Sfx) {
        if let Ok(mut s) = self.shared.lock() {
            if s.sfx.len() < 32 {
                s.sfx.push(sfx);
            }
        }
    }
}

impl Default for Audio {
    fn default() -> Self {
        Self::new()
    }
}

struct VoiceState {
    params: VoiceParams,
    phase: f64,
    mod_phase: f64,
    amp: f32,
    alive: bool,
}

struct SfxState {
    freqs: [f32; 3],
    start: [f32; 3],
    t: f32,
    decay: f32,
    pan: f32,
    gain: f32,
    phases: [f64; 3],
    falling: bool,
}

struct Comb {
    buf: Vec<f32>,
    i: usize,
    store: f32,
}

impl Comb {
    fn new(len: usize) -> Self {
        Self { buf: vec![0.0; len.max(1)], i: 0, store: 0.0 }
    }
    fn tick(&mut self, x: f32, fb: f32, damp: f32) -> f32 {
        let y = self.buf[self.i];
        self.store = y * (1.0 - damp) + self.store * damp;
        self.buf[self.i] = x + self.store * fb;
        self.i = (self.i + 1) % self.buf.len();
        y
    }
}

struct Allpass {
    buf: Vec<f32>,
    i: usize,
}

impl Allpass {
    fn new(len: usize) -> Self {
        Self { buf: vec![0.0; len.max(1)], i: 0 }
    }
    fn tick(&mut self, x: f32) -> f32 {
        let b = self.buf[self.i];
        let y = -x + b;
        self.buf[self.i] = x + b * 0.5;
        self.i = (self.i + 1) % self.buf.len();
        y
    }
}

struct Reverb {
    combs: [Vec<Comb>; 2],
    aps: [Vec<Allpass>; 2],
}

impl Reverb {
    fn new(sr: f32) -> Self {
        let s = sr / 44100.0;
        let mk = |spread: usize| {
            [1557usize, 1617, 1491, 1422].iter().map(|&d| Comb::new(((d + spread) as f32 * s) as usize)).collect::<Vec<_>>()
        };
        let ap = |spread: usize| [556usize, 441].iter().map(|&d| Allpass::new(((d + spread) as f32 * s) as usize)).collect::<Vec<_>>();
        Self { combs: [mk(0), mk(23)], aps: [ap(0), ap(23)] }
    }
    fn tick(&mut self, x: f32) -> (f32, f32) {
        let mut out = [0.0f32; 2];
        for c in 0..2 {
            let mut y: f32 = self.combs[c].iter_mut().map(|cb| cb.tick(x, 0.82, 0.3)).sum();
            for a in &mut self.aps[c] {
                y = a.tick(y);
            }
            out[c] = y * 0.25;
        }
        (out[0], out[1])
    }
}

struct Synth {
    sr: f32,
    shared: Arc<Mutex<Scene>>,
    voices: HashMap<u64, VoiceState>,
    sfx: Vec<SfxState>,
    settings: AudioSettings,
    time: f64,
    reverb: Reverb,
    life: f32,
    life_target: f32,
    pad_root: f32,
    pad_phase: [f64; 3],
    lp: (f32, f32),
    master: f32,
}

fn pan_gains(pan: f32) -> (f32, f32) {
    let a = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (a.cos(), a.sin())
}

impl Synth {
    fn new(sr: f32, shared: Arc<Mutex<Scene>>) -> Self {
        Self {
            sr,
            shared,
            voices: HashMap::new(),
            sfx: Vec::new(),
            settings: AudioSettings::default(),
            time: 0.0,
            reverb: Reverb::new(sr),
            life: 0.0,
            life_target: 0.0,
            pad_root: 55.0,
            pad_phase: [0.0; 3],
            lp: (0.0, 0.0),
            master: 0.0,
        }
    }

    /// Pull new parameters from the main thread (never blocks the audio thread).
    fn sync(&mut self) {
        let Ok(mut s) = self.shared.try_lock() else { return };
        if let Some(st) = s.settings.take() {
            self.settings = st;
        }
        if (s.app_time - self.time).abs() > 0.08 {
            self.time = s.app_time;
        }
        self.life_target = s.life;
        self.pad_root = s.pad_root.max(30.0);
        for v in self.voices.values_mut() {
            v.alive = false;
        }
        for p in &s.voices {
            let e = self.voices.entry(p.id).or_insert(VoiceState { params: *p, phase: 0.0, mod_phase: 0.0, amp: 0.0, alive: true });
            e.params = *p;
            e.alive = true;
        }
        for sfx in s.sfx.drain(..) {
            if self.sfx.len() >= 24 {
                break;
            }
            let (freqs, decay, pan, gain, falling) = match sfx {
                Sfx::Bud { freq, pan } => ([freq * 2.0, freq * 3.0, freq * 4.0], 1.2, pan, 0.18, false),
                Sfx::Mate { freq, pan } => ([freq * 2.0, freq * 2.52, freq * 3.0], 1.6, pan, 0.2, false),
                Sfx::Birth { freq, pan } => ([freq * 2.0, freq * 3.0, freq * 4.0], 0.9, pan, 0.08, false),
                Sfx::Death { freq, pan } => ([freq * 0.5, freq * 0.75, freq * 0.5], 0.9, pan, 0.12, true),
            };
            self.sfx.push(SfxState { freqs, start: [0.0, 0.09, 0.18], t: 0.0, decay, pan, gain, phases: [0.0; 3], falling });
        }
    }

    fn fill<T>(&mut self, data: &mut [T], channels: usize, conv: impl Fn(f32) -> T) {
        self.sync();
        let dt = 1.0 / self.sr;
        let tau = std::f64::consts::TAU;
        let target_master = if self.settings.muted { 0.0 } else { self.settings.master };
        for frame in data.chunks_mut(channels.max(1)) {
            self.time += dt as f64;
            self.master += (target_master - self.master) * 0.0005;
            self.life += (self.life_target - self.life) * 0.00005;
            let mut l = 0.0f32;
            let mut r = 0.0f32;
            // creature voices
            if self.settings.voices {
                for v in self.voices.values_mut() {
                    let p = v.params;
                    let target = if v.alive { p.amp } else { 0.0 };
                    v.amp += (target - v.amp) * 0.0008;
                    if v.amp < 1e-4 {
                        continue;
                    }
                    let env = call_envelope(self.time * (1.0 + 2.0 * p.scream as f64), p.id, p.voice.max(p.scream));
                    let breath = 0.08;
                    let gain = v.amp * (breath + env);
                    let freq = p.freq * (1.0 + 0.06 * env + 0.5 * p.scream) as f32;
                    v.phase = (v.phase + freq as f64 / self.sr as f64).fract();
                    v.mod_phase = (v.mod_phase + freq as f64 * 2.0 / self.sr as f64).fract();
                    let index = 0.4 + 2.2 * p.timbre + 1.5 * p.scream;
                    let s = ((v.phase * tau) + index as f64 * env as f64 * (v.mod_phase * tau).sin()).sin() as f32;
                    let (gl, gr) = pan_gains(p.pan);
                    l += s * gain * gl * 0.22;
                    r += s * gain * gr * 0.22;
                }
            }
            // one-shots
            if self.settings.events {
                for e in &mut self.sfx {
                    e.t += dt;
                    let mut s = 0.0f32;
                    for k in 0..3 {
                        let tk = e.t - e.start[k];
                        if tk < 0.0 {
                            continue;
                        }
                        let f = if e.falling { e.freqs[k] * (1.0 - 0.3 * (tk / e.decay).min(1.0)) } else { e.freqs[k] };
                        e.phases[k] = (e.phases[k] + f as f64 / self.sr as f64).fract();
                        let ph = e.phases[k] * tau;
                        let attack = (tk / 0.006).min(1.0);
                        let env = attack * (-tk * 4.0 / e.decay).exp();
                        // bell: fundamental plus an inharmonic partial
                        s += ((ph.sin() + 0.3 * (ph * 2.76).sin()) as f32) * env;
                    }
                    let (gl, gr) = pan_gains(e.pan);
                    l += s * e.gain * gl;
                    r += s * e.gain * gr;
                }
                self.sfx.retain(|e| e.t < e.decay * 1.6 + 0.2);
            }
            // ambient pad: three soft detuned tones
            let amb = self.settings.ambient * (0.15 + 0.85 * self.life.clamp(0.0, 1.0));
            if amb > 0.0 {
                let ratios = [1.0f64, 1.5, 2.003];
                let mut pad = 0.0f32;
                for (k, ratio) in ratios.iter().enumerate() {
                    self.pad_phase[k] = (self.pad_phase[k] + self.pad_root as f64 * ratio / self.sr as f64).fract();
                    let ph = self.pad_phase[k] * tau;
                    pad += (ph.sin() + 0.25 * (2.0 * ph).sin()) as f32;
                }
                let swell = 0.6 + 0.4 * ((self.time * 0.13).sin() as f32);
                self.lp.0 += (pad * amb * swell * 0.05 - self.lp.0) * 0.05;
                l += self.lp.0;
                r += self.lp.0;
            }
            // reverb send
            let (rl, rr) = self.reverb.tick((l + r) * 0.5);
            let wet = self.settings.reverb;
            let out_l = ((l + rl * wet * 1.6) * self.master * 1.2).tanh();
            let out_r = ((r + rr * wet * 1.6) * self.master * 1.2).tanh();
            for (c, sample) in frame.iter_mut().enumerate() {
                *sample = conv(if c % 2 == 0 { out_l } else { out_r });
            }
        }
        self.voices.retain(|_, v| v.alive || v.amp > 1e-4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_is_pentatonic() {
        assert!((pentatonic(0.0) - 110.0).abs() < 1e-3);
        assert!((pentatonic(5.0 / 15.0 + 0.01) - 220.0).abs() < 1e-3);
        assert!(pentatonic(1.0) < 880.0);
    }

    #[test]
    fn synth_renders_bounded_audio() {
        let shared = Arc::new(Mutex::new(Scene::default()));
        let mut s = Synth::new(48000.0, shared.clone());
        {
            let mut sc = shared.lock().unwrap();
            sc.voices = (0..MAX_VOICES as u64)
                .map(|id| VoiceParams { id, freq: 220.0, amp: 1.0, pan: 0.3, timbre: 0.5, voice: 1.0, scream: 0.5 })
                .collect();
            sc.sfx.push(Sfx::Mate { freq: 330.0, pan: -0.5 });
            sc.settings = Some(AudioSettings { master: 1.0, ..Default::default() });
            sc.life = 1.0;
        }
        let mut buf = vec![0.0f32; 2 * 48000];
        for chunk in buf.chunks_mut(1024) {
            s.fill(chunk, 2, |v| v);
        }
        assert!(buf.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        let energy: f32 = buf.iter().map(|v| v * v).sum();
        assert!(energy > 1.0, "silent output");
    }
}
