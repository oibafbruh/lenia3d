//! Interactive application: window, input handling, UI and the frame loop.

use crate::audio::{pentatonic, Audio, Sfx, VoiceParams, MAX_VOICES};
use crate::gpu;
use crate::headless::{save_png, Options};
use crate::life::{ReproKind, ReproOp};
use crate::sim::flow::{gene, EcologyParams, FlowBrush, FlowParams};
use crate::tracker::LifeEvent;
use crate::render::camera::OrbitCamera;
use crate::render::{Overlay, RenderMode, Renderer};
use crate::sim::preset::{builtin_presets, Preset};
use crate::sim::{chan, memory_estimate, BrushMode, LeniaParams};
use crate::ui::{self, Action, FrameInfo, Tool, UiState};
use crate::world::{Rng, World};
use anyhow::{anyhow, Result};
use glam::{Vec2, Vec3};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

pub fn run(opts: Options, size_given: bool) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { opts, size_given, state: None, error: None };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

struct App {
    opts: Options,
    size_given: bool,
    state: Option<State>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(el, &self.opts, self.size_given) {
            Ok(s) => self.state = Some(s),
            Err(e) => {
                self.error = Some(e);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(s) = self.state.as_mut() {
            s.window_event(el, event);
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

/// Asynchronous readback of per-channel mass (partial sums).
struct MassReader {
    staging: wgpu::Buffer,
    groups: u32,
    channels: usize,
    layout: u64,
    /// `roll_count` when the request was recorded (stale centers are ignored).
    roll_at_request: u64,
    state: Arc<AtomicU8>, // 0 idle, 1 in flight, 2 ready
}

/// Asynchronous readback of the coarse tracking grid.
struct TrackReader {
    staging: wgpu::Buffer,
    m: u32,
    layout: u64,
    generation: u64,
    roll_at_request: u64,
    state: Arc<AtomicU8>, // 0 idle, 1 in flight, 2 ready
}

#[derive(Default)]
struct Input {
    left: bool,
    right: bool,
    middle: bool,
    cursor: Vec2,
    drag_from_ui: bool,
    modifiers: ModifiersState,
    keys: std::collections::HashSet<KeyCode>,
    last_paint: Option<Vec3>,
    stamped: bool,
    press_pos: Option<Vec2>,
    stroke_genome: Option<crate::sim::flow::Genome>,
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    world: World,
    camera: OrbitCamera,
    ui: UiState,
    input: Input,
    start: Instant,
    last_frame: Instant,
    fps: f32,
    steps_counter: (Instant, u64, f32),
    mass: Vec<f64>,
    mass_reader: Option<MassReader>,
    frame_index: u64,
    volume_dirty: bool,
    max_n: u32,
    gpu_errors: Arc<Mutex<Vec<String>>>,
    chan_rx: Option<mpsc::Receiver<Result<String>>>,
    seed_rng: Rng,
    roll_count: u64,
    pending_roll: Option<[i32; 3]>,
    floor_offset: Vec2,
    audio: Audio,
    track_reader: Option<TrackReader>,
    pending_ops: Vec<ReproOp>,
    last_audio: f64,
    sfx_budget: f32,
}

fn data_dir() -> PathBuf {
    let base = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_else(|| PathBuf::from("."));
    base.join("lenia3d_data")
}

fn chan_cache() -> PathBuf {
    data_dir().join("animals3D.json")
}

impl State {
    fn new(el: &ActiveEventLoop, opts: &Options, size_given: bool) -> Result<Self> {
        let attrs = Window::default_attributes()
            .with_title("Lenia3D")
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 940.0))
            .with_min_inner_size(winit::dpi::LogicalSize::new(800.0, 500.0));
        let window = Arc::new(el.create_window(attrs)?);
        let instance = gpu::create_instance();
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(gpu::pick_adapter(&instance, Some(&surface)))?;
        let info = adapter.get_info();
        log::info!("using {} ({:?}, driver {})", info.name, info.backend, info.driver_info);
        let (device, queue) = pollster::block_on(gpu::request_device(&adapter))?;
        let gpu_errors: Arc<Mutex<Vec<String>>> = Arc::default();
        {
            let errs = gpu_errors.clone();
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                log::error!("GPU error: {e}");
                if let Ok(mut v) = errs.lock() {
                    v.push(e.to_string().lines().next().unwrap_or("GPU error").to_string());
                }
            }));
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let renderer = Renderer::new(&device, format);
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        let mut presets = builtin_presets();
        let mut chan_loaded = false;
        let chan_file = opts.chan_file.clone().map(PathBuf::from).unwrap_or_else(chan_cache);
        if let Ok(src) = std::fs::read_to_string(&chan_file) {
            if let Ok(list) = chan::parse_animals(&src) {
                presets.extend(list);
                chan_loaded = true;
            }
        }
        if let Some(file) = &opts.preset_file {
            match std::fs::read_to_string(file).map_err(anyhow::Error::from).and_then(|src| {
                Ok(match serde_json::from_str::<Preset>(&src) {
                    Ok(p) => vec![p],
                    Err(_) => serde_json::from_str::<Vec<Preset>>(&src)?,
                })
            }) {
                Ok(mut extra) => {
                    extra.extend(presets);
                    presets = extra;
                }
                Err(e) => log::warn!("could not load {file}: {e}"),
            }
        }
        let max_n = gpu::Gpu::max_grid_size(&device);
        let mut ui = UiState::new(presets);
        ui.chan_loaded = chan_loaded;
        let start_idx = match &opts.preset {
            Some(name) => {
                let l = name.to_lowercase();
                ui.presets.iter().position(|p| p.name.to_lowercase().contains(&l)).unwrap_or(0)
            }
            None => ui
                .presets
                .iter()
                .position(|p| p.name == "Tadpole Sea")
                .or_else(|| ui.presets.iter().position(|p| p.kind == "soup"))
                .unwrap_or(0),
        };
        ui.selected = start_idx;
        let n = if size_given { opts.size.min(max_n) } else { 128.min(max_n) };
        let mut world = World::new(&device, n, LeniaParams::default());
        let p = ui.presets[start_idx].clone();
        world.load_preset(&device, &queue, &p, 1.0);
        ui.rule = world.sim.params().clone();
        ui.follow = p.pattern().is_some();
        ui.set_status(format!("Loaded \"{}\"", p.name), 0.0);
        if let Some(f) = &world.sim.flow {
            ui.flow_params = f.params.clone();
            ui.eco = f.eco.clone();
        }
        ui.traits = world.traits.clone();
        let audio = Audio::new();

        let mut camera = OrbitCamera::default();
        if let Some(pat) = p.pattern() {
            let r = pat.dims.iter().copied().max().unwrap_or(1) as f32 / n as f32 * 2.0;
            camera.frame_box(Vec3::ZERO, r.max(0.25));
        }
        camera.snap();
        let now = Instant::now();
        Ok(Self {
            window,
            surface,
            config,
            device,
            queue,
            adapter_name: format!("{} ({:?})", info.name, info.backend),
            renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
            world,
            camera,
            ui,
            input: Input::default(),
            start: now,
            last_frame: now,
            fps: 60.0,
            steps_counter: (now, 0, 0.0),
            mass: vec![],
            mass_reader: None,
            frame_index: 0,
            volume_dirty: true,
            max_n,
            gpu_errors,
            chan_rx: None,
            seed_rng: Rng::new(12345),
            roll_count: 0,
            pending_roll: None,
            floor_offset: Vec2::ZERO,
            audio,
            track_reader: None,
            pending_ops: Vec::new(),
            last_audio: 0.0,
            sfx_budget: 3.0,
        })
    }

    /// Copy the world's Flow settings into the editable UI state.
    fn pull_flow_ui(&mut self) {
        if let Some(f) = &self.world.sim.flow {
            self.ui.flow_params = f.params.clone();
            self.ui.eco = f.eco.clone();
        }
        self.ui.traits = self.world.traits.clone();
    }

    /// World-space center and radius of a tracked creature.
    fn creature_world(&self, c: &crate::tracker::Creature) -> (Vec3, f32) {
        let n = self.world.n() as f32;
        let p = c.center.map(|v| (v + 0.5) / n * 2.0 - 1.0);
        (Vec3::new(p[0], p[1], p[2]), c.radius * 2.0 / n)
    }

    /// Stereo pan and distance attenuation for a world position.
    fn listen(&self, p: Vec3) -> (f32, f32) {
        let eye = self.camera.eye();
        let (right, _, _) = self.camera.basis();
        let d = (p - eye).length().max(1e-3);
        let pan = ((p - eye).dot(right) / d * 1.6).clamp(-1.0, 1.0);
        (pan, 1.0 / (1.0 + (d / 1.8).powi(2)))
    }

    fn cell_to_world(&self, c: [f32; 3]) -> Vec3 {
        let n = self.world.n() as f32;
        let p = c.map(|v| (v + 0.5) / n * 2.0 - 1.0);
        Vec3::new(p[0], p[1], p[2])
    }

    fn pick_creature(&mut self) {
        let (o, d) = self.camera.ray(self.cursor_ndc(), self.aspect());
        let mut best: Option<(f32, u64)> = None;
        for c in &self.world.tracker.creatures {
            let (center, r) = self.creature_world(c);
            let r = r * 1.15 + 0.01;
            let oc = o - center;
            let b = oc.dot(d);
            let h = b * b - (oc.dot(oc) - r * r);
            if h >= 0.0 {
                let t = -b - h.sqrt();
                if t > 0.0 && best.is_none_or(|(bt, _)| t < bt) {
                    best = Some((t, c.id));
                }
            }
        }
        let now = self.now();
        self.ui.selected_creature = best.map(|b| b.1);
        if let Some(id) = self.ui.selected_creature {
            self.ui.set_status(format!("Selected creature #{id} (Follow keeps it centered)"), now);
        }
    }

    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }

    fn cursor_ndc(&self) -> Vec2 {
        let w = self.config.width.max(1) as f32;
        let h = self.config.height.max(1) as f32;
        Vec2::new(self.input.cursor.x / w * 2.0 - 1.0, 1.0 - self.input.cursor.y / h * 2.0)
    }

    /// Where the mouse ray hits the work plane inside the world box.
    fn brush_point(&self) -> Option<Vec3> {
        let (o, d) = self.camera.ray(self.cursor_ndc(), self.aspect());
        if d.y.abs() < 1e-5 {
            return None;
        }
        let t = (self.ui.plane_y - o.y) / d.y;
        if t <= 0.0 {
            return None;
        }
        let p = o + d * t;
        (p.x.abs() <= 1.0 && p.z.abs() <= 1.0).then_some(p)
    }

    fn world_to_cell(&self, p: Vec3) -> [f32; 3] {
        let n = self.world.n() as f32;
        [(p.x + 1.0) * 0.5 * n - 0.5, (p.y + 1.0) * 0.5 * n - 0.5, (p.z + 1.0) * 0.5 * n - 0.5]
    }

    fn brush_world_radius(&self) -> f32 {
        self.ui.brush_radius * 2.0 / self.world.n() as f32
    }

    fn window_event(&mut self, el: &ActiveEventLoop, event: WindowEvent) {
        let resp = self.egui_state.on_window_event(&self.window, &event);
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.config.width = size.width;
                    self.config.height = size.height;
                    self.surface.configure(&self.device, &self.config);
                }
            }
            WindowEvent::ModifiersChanged(m) => self.input.modifiers = m.state(),
            WindowEvent::DroppedFile(path) => self.open_file(&path),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == ElementState::Released {
                        self.input.keys.remove(&code);
                    } else if !self.egui_ctx.egui_wants_keyboard_input() {
                        let repeatable = matches!(
                            code,
                            KeyCode::BracketLeft | KeyCode::BracketRight | KeyCode::PageUp | KeyCode::PageDown | KeyCode::Period
                        );
                        if !event.repeat || repeatable {
                            self.on_key(el, code);
                        }
                        self.input.keys.insert(code);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                log::debug!(
                    "mouse {button:?} pressed={pressed} consumed={} over_egui={}",
                    resp.consumed,
                    self.egui_ctx.is_pointer_over_egui()
                );
                if pressed && (resp.consumed || self.egui_ctx.is_pointer_over_egui()) {
                    self.input.drag_from_ui = true;
                    return;
                }
                if !pressed {
                    self.input.drag_from_ui = false;
                }
                match button {
                    MouseButton::Left => {
                        self.input.left = pressed;
                        if pressed {
                            self.input.last_paint = None;
                            self.input.stamped = false;
                            self.input.press_pos = Some(self.input.cursor);
                            self.input.stroke_genome = self.world.is_flow().then(|| self.world.random_genome());
                        } else {
                            let clicked = self.input.press_pos.take().is_some_and(|p| (p - self.input.cursor).length() < 4.0);
                            if clicked && self.ui.tool == Tool::View {
                                self.pick_creature();
                            }
                        }
                    }
                    MouseButton::Right => self.input.right = pressed,
                    MouseButton::Middle => self.input.middle = pressed,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = Vec2::new(position.x as f32, position.y as f32);
                let delta = p - self.input.cursor;
                self.input.cursor = p;
                if self.input.drag_from_ui {
                    return;
                }
                let shift = self.input.modifiers.shift_key();
                let view_tool = self.ui.tool == Tool::View;
                if self.input.middle || ((self.input.right || (self.input.left && view_tool)) && shift) {
                    self.camera.pan(delta.x, delta.y, self.config.height as f32);
                } else if self.input.right || (self.input.left && view_tool) {
                    self.camera.orbit(delta.x, delta.y);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.egui_ctx.is_pointer_over_egui() {
                    return;
                }
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                if self.input.modifiers.control_key() {
                    self.ui.brush_radius = (self.ui.brush_radius * (1.0 + steps * 0.1)).clamp(1.0, 64.0);
                } else if self.input.modifiers.alt_key() {
                    self.ui.plane_y = (self.ui.plane_y + steps * 0.03).clamp(-1.0, 1.0);
                } else {
                    let ndc = self.cursor_ndc();
                    let aspect = self.aspect();
                    self.camera.zoom(steps, ndc, aspect);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.frame() {
                    log::error!("frame error: {e:#}");
                }
            }
            _ => {}
        }
    }

    fn on_key(&mut self, el: &ActiveEventLoop, code: KeyCode) {
        let now = self.now();
        match code {
            KeyCode::Space => self.ui.running = !self.ui.running,
            KeyCode::Period => self.apply(Action::Step),
            KeyCode::KeyR => self.apply(Action::Reset),
            KeyCode::KeyC => self.apply(Action::Clear),
            KeyCode::KeyH | KeyCode::F1 => self.ui.show_help = !self.ui.show_help,
            KeyCode::Tab => self.ui.show_ui = !self.ui.show_ui,
            KeyCode::KeyF => self.apply(Action::FrameView),
            KeyCode::KeyM => {
                self.ui.settings.mode = match self.ui.settings.mode {
                    RenderMode::Organic => RenderMode::Glow,
                    RenderMode::Glow => RenderMode::Xray,
                    RenderMode::Xray => RenderMode::Organic,
                }
            }
            KeyCode::BracketLeft => self.ui.brush_radius = (self.ui.brush_radius / 1.15).max(1.0),
            KeyCode::BracketRight => self.ui.brush_radius = (self.ui.brush_radius * 1.15).min(64.0),
            KeyCode::PageUp => self.ui.plane_y = (self.ui.plane_y + 0.03).min(1.0),
            KeyCode::PageDown => self.ui.plane_y = (self.ui.plane_y - 0.03).max(-1.0),
            KeyCode::Escape => self.ui.tool = Tool::View,
            KeyCode::F11 => {
                let fs = if self.window.fullscreen().is_some() { None } else { Some(Fullscreen::Borderless(None)) };
                self.window.set_fullscreen(fs);
            }
            KeyCode::F12 => self.apply(Action::Screenshot),
            KeyCode::KeyP => self.ui.show_population = !self.ui.show_population,
            KeyCode::KeyV => {
                self.ui.audio.muted = !self.ui.audio.muted;
                let msg = if self.ui.audio.muted { "Sound off" } else { "Sound on" };
                self.ui.set_status(msg, now);
            }
            KeyCode::Digit8 | KeyCode::Digit9 | KeyCode::Digit0 if self.world.is_flow() => {
                self.ui.tool = match code {
                    KeyCode::Digit8 => Tool::Food,
                    KeyCode::Digit9 => Tool::Toxic,
                    _ => Tool::Founder,
                };
                self.ui.set_status(format!("Tool: {}", self.ui.tool.label()), now);
            }
            KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3 | KeyCode::Digit4 | KeyCode::Digit5 | KeyCode::Digit6 | KeyCode::Digit7 => {
                let i = match code {
                    KeyCode::Digit1 => 0,
                    KeyCode::Digit2 => 1,
                    KeyCode::Digit3 => 2,
                    KeyCode::Digit4 => 3,
                    KeyCode::Digit5 => 4,
                    KeyCode::Digit6 => 5,
                    _ => 6,
                };
                self.ui.tool = Tool::ALL[i];
                self.ui.set_status(format!("Tool: {}", self.ui.tool.label()), now);
            }
            _ => {}
        }
        let _ = el;
    }

    fn open_file(&mut self, path: &std::path::Path) {
        let now = self.now();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if ext == "json" {
            match std::fs::read_to_string(path).map_err(anyhow::Error::from).and_then(|s| self.import_presets(&s)) {
                Ok(k) => self.ui.set_status(format!("Imported {k} presets from {}", path.display()), now),
                Err(e) => self.ui.set_status(format!("Import failed: {e}"), now),
            }
        } else {
            match self.world.load(&self.device, &self.queue, path) {
                Ok(()) => {
                    self.ui.rule = self.world.sim.params().clone();
                    self.volume_dirty = true;
                    self.ui.set_status(format!("Opened {}", path.display()), now);
                }
                Err(e) => self.ui.set_status(format!("Open failed: {e}"), now),
            }
        }
    }

    /// Accept either Chan's species list or a single preset / list of presets.
    fn import_presets(&mut self, src: &str) -> Result<usize> {
        if let Ok(p) = serde_json::from_str::<Preset>(src) {
            self.ui.presets.push(p);
            return Ok(1);
        }
        if let Ok(list) = serde_json::from_str::<Vec<Preset>>(src) {
            let k = list.len();
            self.ui.presets.extend(list);
            return Ok(k);
        }
        let list = chan::parse_animals(src)?;
        if list.is_empty() {
            return Err(anyhow!("no species found"));
        }
        let k = list.len();
        self.ui.presets.retain(|p| p.kind != "chan");
        self.ui.presets.extend(list);
        self.ui.chan_loaded = true;
        let _ = std::fs::create_dir_all(data_dir());
        let _ = std::fs::write(chan_cache(), src);
        Ok(k)
    }

    fn apply(&mut self, action: Action) {
        let now = self.now();
        let (device, queue) = (&self.device, &self.queue);
        match action {
            Action::LoadPreset(i) => {
                if let Some(p) = self.ui.presets.get(i).cloned() {
                    self.ui.selected = i;
                    if let Some(w) = p.required_world_size() {
                        // periodic sheet/rod: exact world size, no scaling
                        self.ui.scale = 1.0;
                        if self.world.n() != w && w <= self.max_n {
                            self.world.resize(device, queue, w);
                        }
                    }
                    // creatures need room: grow the world if it is too small for the pattern
                    let need = p.pattern().map(|q| q.dims.iter().copied().max().unwrap_or(0) as f32 * self.ui.scale).unwrap_or(0.0);
                    let need = need.max(p.rule.R * self.ui.scale * 2.0 + 8.0);
                    if p.required_world_size().is_none() && (self.world.n() as f32) < need {
                        let mut n = self.world.n();
                        while (n as f32) < need && n < self.max_n {
                            n *= 2;
                        }
                        self.world.resize(device, queue, n);
                    }
                    self.world.load_preset(device, queue, &p, self.ui.scale);
                    self.ui.rule = self.world.sim.params().clone();
                    self.ui.running = true;
                    self.ui.follow = p.pattern().is_some();
                    self.ui.selected_creature = None;
                    self.frame_preset(&p);
                    self.pull_flow_ui();
                    self.ui.set_status(format!("Loaded \"{}\"", p.name), now);
                }
            }
            Action::SetMode(flow) => {
                // Evolution mode opens with the grazing ecosystem; Classic with the first soup.
                let found = self
                    .ui
                    .presets
                    .iter()
                    .position(|p| flow && p.name == "Grazing Drift")
                    .or_else(|| self.ui.presets.iter().position(|p| p.is_flow() == flow && (flow || p.kind == "soup")));
                match found {
                    Some(i) => return self.apply(Action::LoadPreset(i)),
                    None => {
                        let cfg = flow.then(|| (FlowParams::default(), EcologyParams::default()));
                        self.world.set_flow(device, queue, cfg);
                        self.world.reset(device, queue);
                        self.pull_flow_ui();
                    }
                }
            }
            Action::SeedFounders => {
                let n = self.world.n() as f32;
                let r = self.world.sim.params().R;
                let mut enc = device.create_command_encoder(&Default::default());
                for _ in 0..4 {
                    let c = [self.seed_rng.f32() * n, self.seed_rng.f32() * n, self.seed_rng.f32() * n];
                    let genes = self.world.random_genome();
                    let seed = self.seed_rng.next_u32();
                    self.world.seed_founder(device, &mut enc, c, r, 0.7, &genes, seed);
                }
                queue.submit([enc.finish()]);
                self.ui.set_status("Added 4 founders", now);
            }
            Action::RefillFood | Action::ResetGround => {
                if let Some(f) = &self.world.sim.flow {
                    if matches!(action, Action::ResetGround) {
                        f.reset_fertility(queue, self.world.n());
                    }
                    let mut enc = device.create_command_encoder(&Default::default());
                    f.reset_food(&mut enc);
                    queue.submit([enc.finish()]);
                }
            }
            Action::KillSelected => {
                if let Some(c) = self.ui.selected_creature.and_then(|id| self.world.tracker.get(id)).cloned() {
                    let mut enc = device.create_command_encoder(&Default::default());
                    self.world.sim.brush(device, &mut enc, BrushMode::Erase, None, c.center, c.radius * 1.4 + 2.0, 1.0, 0);
                    queue.submit([enc.finish()]);
                    self.ui.selected_creature = None;
                }
            }
            Action::CloneSelected => {
                if let Some(c) = self.ui.selected_creature.and_then(|id| self.world.tracker.get(id)).cloned() {
                    let mut delta = [0.0f32; 16];
                    let size = self.ui.eco.bud_mutation;
                    for d in delta.iter_mut() {
                        *d = (self.seed_rng.f32() * 2.0 - 1.0) * size * 1.7;
                    }
                    let rbox = c.radius * 1.3 + 1.0;
                    let len = (2.0 * rbox + 2.0).min(self.world.n() as f32 / 2.0 - 1.0);
                    let a = self.seed_rng.f32() * std::f32::consts::TAU;
                    let offset = [(a.cos() * len).round() as i32, 0, (a.sin() * len).round() as i32];
                    let child = [0, 1, 2].map(|k| c.center[k] + offset[k] as f32);
                    let op = ReproOp {
                        kind: ReproKind::Bud,
                        parent: c.id,
                        partner: None,
                        center: c.center,
                        radius: rbox,
                        offset,
                        delta,
                        child_center: child,
                        child_pitch: c.genome[gene::PITCH],
                    };
                    self.world.tracker.expect_offspring(child, c.id, self.world.sim.generation);
                    self.pending_ops.push(op);
                }
            }
            Action::Islands => {
                let n = self.world.n() as usize;
                let mut w = self.world.sim.download_wall(device, queue);
                for z in 0..n {
                    for y in 0..n {
                        for x in 0..n {
                            let wall = |v: usize| v < 2 || (v >= n / 2 && v < n / 2 + 2);
                            if wall(x) || wall(z) {
                                w[(z * n + y) * n + x] = 1.0;
                            }
                        }
                    }
                }
                self.world.sim.upload_wall(queue, &w);
                self.world.has_walls = true;
                self.ui.set_status("Four islands: separated populations will drift apart (watch their languages)", now);
            }
            Action::Reset => {
                self.world.reset(device, queue);
                self.ui.set_status("Reset", now);
            }
            Action::Clear => {
                let mut enc = device.create_command_encoder(&Default::default());
                self.world.sim.clear(&mut enc, false);
                queue.submit([enc.finish()]);
            }
            Action::Soup => {
                self.ui.follow = false;
                let mut enc = device.create_command_encoder(&Default::default());
                self.world.sim.clear(&mut enc, false);
                let init = crate::sim::preset::SoupInit { blobs: 10, blob_radius: 1.0, density: 0.5, traits: None };
                self.world.seed_soup(device, &mut enc, &init);
                queue.submit([enc.finish()]);
                self.ui.running = true;
            }
            Action::Resize(n) => {
                self.world.resize(device, queue, n);
                self.world.reset(device, queue);
                self.ui.rule = self.world.sim.params().clone();
                self.ui.set_status(format!("World is now {n}³"), now);
            }
            Action::Step => {
                let mut enc = device.create_command_encoder(&Default::default());
                self.world.sim.encode_steps(&mut enc, 1);
                queue.submit([enc.finish()]);
                self.ui.running = false;
            }
            Action::SaveWorld => {
                if let Some(path) = rfd::FileDialog::new().add_filter("Lenia3D world", &["l3d"]).set_file_name("world.l3d").save_file() {
                    match self.world.save(device, queue, &path) {
                        Ok(()) => self.ui.set_status(format!("Saved {}", path.display()), now),
                        Err(e) => self.ui.set_status(format!("Save failed: {e}"), now),
                    }
                }
            }
            Action::LoadWorld => {
                if let Some(path) = rfd::FileDialog::new().add_filter("Lenia3D world or presets", &["l3d", "json"]).pick_file() {
                    self.open_file(&path);
                }
            }
            Action::ImportChan => {
                if let Some(path) = rfd::FileDialog::new().add_filter("animals3D.json", &["json"]).pick_file() {
                    self.open_file(&path);
                }
            }
            Action::DownloadChan => {
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(chan::download_animals());
                });
                self.chan_rx = Some(rx);
                self.ui.downloading = true;
                self.ui.set_status("Downloading Bert Chan's 3D species…", now);
            }
            Action::Screenshot => {
                if let Err(e) = self.screenshot() {
                    self.ui.set_status(format!("Screenshot failed: {e}"), now);
                }
            }
            Action::FrameView => {
                self.camera.reset();
            }
            Action::ClearWalls => {
                let mut enc = device.create_command_encoder(&Default::default());
                enc.clear_buffer(&self.world.sim.wall, 0, None);
                queue.submit([enc.finish()]);
                self.world.has_walls = false;
            }
            Action::EncloseWalls => {
                let n = self.world.n() as usize;
                let t = (n / 64).max(1) + 1;
                let mut w = self.world.sim.download_wall(device, queue);
                for z in 0..n {
                    for y in 0..n {
                        for x in 0..n {
                            if x < t || y < t || z < t || x >= n - t || y >= n - t || z >= n - t {
                                w[(z * n + y) * n + x] = 1.0;
                            }
                        }
                    }
                }
                self.world.sim.upload_wall(queue, &w);
                self.world.has_walls = true;
                self.ui.set_status("World enclosed by walls (follow mode is paused while walls exist)", now);
            }
            Action::Pillars => {
                let n = self.world.n() as f32;
                let mut enc = device.create_command_encoder(&Default::default());
                for _ in 0..6 {
                    let x = self.seed_rng.f32() * n;
                    let z = self.seed_rng.f32() * n;
                    let r = n * (0.02 + 0.03 * self.seed_rng.f32());
                    let mut y = 0.0;
                    while y < n {
                        self.world.sim.brush(device, &mut enc, BrushMode::Wall, None, [x, y, z], r, 1.0, 0);
                        y += r * 0.7;
                    }
                }
                queue.submit([enc.finish()]);
                self.world.has_walls = true;
            }
        }
        self.volume_dirty = true;
    }

    /// Point the camera at a freshly loaded creature (or the whole world for soups).
    fn frame_preset(&mut self, p: &Preset) {
        match p.pattern() {
            Some(pat) => {
                let cells = pat.dims.iter().copied().max().unwrap_or(1) as f32 * self.ui.scale;
                let r = cells / self.world.n() as f32 * 2.0;
                self.camera.frame_box(Vec3::ZERO, r.max(0.25));
            }
            None => self.camera.reset(),
        }
    }

    fn screenshot(&mut self) -> Result<()> {
        let (w, h) = (self.config.width, self.config.height);
        let format = self.config.format;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        let overlay = Overlay::default();
        let t = self.now() as f32;
        self.renderer.render(&self.device, &self.queue, &mut enc, &view, (w, h), &self.camera, &self.ui.settings, &overlay, t);
        self.queue.submit([enc.finish()]);
        let mut px = crate::headless::read_texture(&self.device, &self.queue, &tex, w, h);
        if matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb) {
            for p in px.chunks_exact_mut(4) {
                p.swap(0, 2);
            }
        }
        let dir = std::path::Path::new("screenshots");
        std::fs::create_dir_all(dir)?;
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let path = dir.join(format!("lenia3d_{secs}.png"));
        save_png(path.to_str().unwrap_or("screenshot.png"), w, h, &px)?;
        self.ui.set_status(format!("Saved {}", path.display()), self.now());
        Ok(())
    }

    fn handle_tools(&mut self, enc: &mut wgpu::CommandEncoder) {
        if !self.input.left || self.input.drag_from_ui || self.ui.tool == Tool::View {
            return;
        }
        let Some(p) = self.brush_point() else { return };
        let r = self.ui.brush_radius;
        let center = self.world_to_cell(p);
        let tool = self.ui.tool;
        if self.world.is_flow() && matches!(tool, Tool::Food | Tool::Toxic | Tool::Founder | Tool::Noise | Tool::Paint) {
            let genes = self.input.stroke_genome.unwrap_or_else(|| self.world.default_genome());
            let Some(f) = &self.world.sim.flow else { return };
            let sb = self.world.sim.flow_buffers();
            match tool {
                Tool::Founder => {
                    if !self.input.stamped {
                        self.input.stamped = true;
                        let r = self.world.sim.params().R;
                        f.brush(&self.device, enc, &sb, FlowBrush::Seed, center, r, 0.0, 0.7, &genes, self.seed_rng.next_u32());
                    }
                }
                Tool::Food | Tool::Toxic => {
                    let mode = if tool == Tool::Food { FlowBrush::Food } else { FlowBrush::Toxic };
                    f.brush(&self.device, enc, &sb, mode, center, r, self.ui.brush_strength, 0.0, &genes, 0);
                }
                _ => {
                    if let Some(last) = self.input.last_paint {
                        if (last - p).length() < self.brush_world_radius() * 0.35 {
                            return;
                        }
                    }
                    self.input.last_paint = Some(p);
                    f.brush(&self.device, enc, &sb, FlowBrush::Seed, center, r, 0.0, 0.6, &genes, self.seed_rng.next_u32());
                }
            }
            self.volume_dirty = true;
            return;
        }
        if tool == Tool::Creature {
            if !self.input.stamped {
                self.input.stamped = true;
                if let Some(pat) = self.ui.presets.get(self.ui.selected).and_then(|p| p.pattern()) {
                    let pat = pat.scaled(self.ui.scale);
                    self.world.stamp(&self.device, enc, &pat, center);
                    self.volume_dirty = true;
                } else {
                    self.ui.set_status("The selected preset has no creature pattern (it is a soup rule)", self.now());
                }
            }
            return;
        }
        // space out continuous strokes
        if let Some(last) = self.input.last_paint {
            if (last - p).length() < self.brush_world_radius() * 0.35 && matches!(tool, Tool::Noise | Tool::Paint) {
                return;
            }
        }
        self.input.last_paint = Some(p);
        let mode = match tool {
            Tool::Noise => BrushMode::Noise,
            Tool::Paint => BrushMode::Add,
            Tool::Erase => BrushMode::Erase,
            Tool::Wall => BrushMode::Wall,
            Tool::Unwall => BrushMode::Unwall,
            _ => return,
        };
        if tool == Tool::Wall {
            self.world.has_walls = true;
        }
        let seed = self.seed_rng.next_u32();
        self.world.sim.brush(&self.device, enc, mode, self.ui.brush_channel, center, r, self.ui.brush_strength, seed);
        self.volume_dirty = true;
    }

    fn poll_track(&mut self) {
        let Some(r) = &self.track_reader else { return };
        if r.state.load(Ordering::Acquire) != 2 {
            return;
        }
        let vals: Vec<f32> = {
            let view = r.staging.slice(..).get_mapped_range().expect("mapped");
            bytemuck::cast_slice(&view).to_vec()
        };
        let (m, generation, stale) = (r.m, r.generation, r.roll_at_request != self.roll_count);
        r.staging.unmap();
        r.state.store(0, Ordering::Release);
        let layout = self.world.sim.layout_id() ^ ((self.world.n() as u64) << 48);
        if stale || r.layout != layout {
            return;
        }
        let f = self.world.n() / m;
        let ops = self.world.life_update(&vals, m, f, generation);
        for op in &ops {
            let (pan, att) = self.listen(self.cell_to_world(op.child_center));
            let freq = pentatonic(op.child_pitch);
            let sfx = match op.kind {
                ReproKind::Bud => Sfx::Bud { freq, pan },
                ReproKind::Mate => Sfx::Mate { freq, pan },
            };
            if att > 0.05 {
                self.audio.play(sfx);
            }
        }
        if !ops.is_empty() {
            let (buds, mates) = crate::world::births(&ops);
            log::debug!("reproduction: {buds} buds, {mates} matings");
        }
        self.pending_ops.extend(ops);
        let fresh_world = self.world.tracker.history.len() < 4;
        for e in self.world.tracker.drain_events() {
            if fresh_world || self.sfx_budget < 1.0 {
                continue;
            }
            let (center, pitch, death) = match e {
                LifeEvent::Born { center, pitch, .. } => (center, pitch, false),
                LifeEvent::Died { center, pitch, .. } => (center, pitch, true),
            };
            let (pan, att) = self.listen(self.cell_to_world(center));
            if att < 0.08 {
                continue;
            }
            self.sfx_budget -= 1.0;
            let freq = pentatonic(pitch);
            self.audio.play(if death { Sfx::Death { freq, pan } } else { Sfx::Birth { freq, pan } });
        }
    }

    /// Send the loudest creature voices to the synthesizer.
    fn update_audio(&mut self, now: f64) {
        if !self.audio.available {
            return;
        }
        let flow = self.world.is_flow();
        let mut voices: Vec<VoiceParams> = self
            .world
            .tracker
            .creatures
            .iter()
            .filter_map(|c| {
                let g = &c.genome;
                let (voice, pitch, timbre, scream) = if flow {
                    let scream = (-c.growth * 0.4).clamp(0.0, 1.0) * g[gene::ALARM];
                    (g[gene::VOICE], g[gene::PITCH], g[gene::TIMBRE], scream)
                } else {
                    // classic Lenia: bigger creatures sing lower, moving ones louder
                    let pitch = (1.0 - c.radius / 24.0).clamp(0.0, 1.0);
                    (0.25 + (c.speed() * 6.0).min(0.6), pitch, 0.3, 0.0)
                };
                if voice < 0.05 && scream < 0.05 {
                    return None;
                }
                let (center, _) = self.creature_world(c);
                let (pan, att) = self.listen(center);
                let size = (c.mass.max(1.0)).powf(0.2) / 4.0;
                Some(VoiceParams {
                    id: c.id,
                    freq: pentatonic(pitch),
                    amp: (voice.max(scream) * att * size.clamp(0.4, 1.5)).min(1.0),
                    pan,
                    timbre,
                    voice,
                    scream,
                })
            })
            .collect();
        voices.sort_by(|a, b| b.amp.total_cmp(&a.amp));
        voices.truncate(MAX_VOICES);
        let life = (self.world.tracker.creatures.len() as f32 / 20.0).min(1.0);
        self.audio.update(voices, now, &self.ui.audio, life, 55.0);
    }

    fn poll_mass(&mut self) {
        let Some(r) = &self.mass_reader else { return };
        if r.state.load(Ordering::Acquire) != 2 {
            return;
        }
        let n = self.world.n();
        let stats = {
            let view = r.staging.slice(..).get_mapped_range().expect("mapped");
            let vals: &[f32] = bytemuck::cast_slice(&view);
            crate::sim::stats_from_partials(vals, r.groups, n)
        };
        let layout = self.world.sim.layout_id() ^ ((n as u64) << 48);
        let fresh = r.roll_at_request == self.roll_count && r.layout == layout;
        r.staging.unmap();
        r.state.store(0, Ordering::Release);
        if r.layout != layout {
            return;
        }
        self.mass = stats.iter().map(|s| s.mass).collect();
        // follow: roll the periodic world so the center of mass sits in the middle
        if self.ui.follow && fresh && !self.world.has_walls && self.pending_roll.is_none() {
            let selected = self.ui.selected_creature.and_then(|id| self.world.tracker.get(id)).map(|c| c.center.map(|v| v as f64));
            if let Some(c) = selected.or_else(|| crate::sim::combined_center(&stats, n)) {
                let half = n as f64 / 2.0;
                let shift = c.map(|v| {
                    let d = (half - v).rem_euclid(n as f64);
                    let d = if d >= half { d - n as f64 } else { d };
                    d.round() as i32
                });
                if shift.iter().any(|d| d.abs() >= 1) {
                    self.pending_roll = Some(shift);
                }
            }
        }
    }

    fn frame(&mut self) -> Result<()> {
        let now_i = Instant::now();
        let dt = (now_i - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now_i;
        self.fps = self.fps * 0.95 + (1.0 / dt.max(1e-4)) * 0.05;
        self.frame_index += 1;
        let now = self.now();

        // results from background work
        if let Some(rx) = &self.chan_rx {
            if let Ok(res) = rx.try_recv() {
                self.chan_rx = None;
                self.ui.downloading = false;
                match res.and_then(|s| self.import_presets(&s)) {
                    Ok(k) => self.ui.set_status(format!("Added {k} species by Bert Chan"), now),
                    Err(e) => self.ui.set_status(
                        format!("Download failed ({e}). Download {} manually and drop it onto the window.", chan::ANIMALS3D_URL),
                        now,
                    ),
                }
            }
        }
        if let Ok(mut errs) = self.gpu_errors.lock() {
            if let Some(e) = errs.pop() {
                self.ui.set_status(format!("GPU error: {e}"), now);
                errs.clear();
            }
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.poll_mass();
        self.poll_track();
        self.sfx_budget = (self.sfx_budget + dt * 3.0).min(4.0);

        // camera fly keys
        let k = &self.input.keys;
        let mut fly = Vec3::ZERO;
        if k.contains(&KeyCode::KeyW) {
            fly.z += 1.0;
        }
        if k.contains(&KeyCode::KeyS) {
            fly.z -= 1.0;
        }
        if k.contains(&KeyCode::KeyD) {
            fly.x += 1.0;
        }
        if k.contains(&KeyCode::KeyA) {
            fly.x -= 1.0;
        }
        if k.contains(&KeyCode::KeyE) {
            fly.y += 1.0;
        }
        if k.contains(&KeyCode::KeyQ) {
            fly.y -= 1.0;
        }
        if fly != Vec3::ZERO {
            self.camera.fly(fly, dt);
        }
        self.camera.update(dt);

        // UI
        let selected = self.ui.selected_creature.and_then(|id| self.world.tracker.get(id)).cloned();
        if self.ui.selected_creature.is_some() && selected.is_none() && self.world.tracker.history.len() > 2 {
            // the selected creature died or merged
            self.ui.selected_creature = None;
        }
        let info = FrameInfo {
            flow: self.world.is_flow(),
            tracker: &self.world.tracker,
            selected,
            audio_status: self.audio.status.clone(),
            fps: self.fps,
            steps_per_sec: self.steps_counter.2,
            generation: self.world.sim.generation,
            time: self.world.sim.time,
            mass: self.mass.clone(),
            n: self.world.n(),
            max_n: self.max_n,
            adapter: self.adapter_name.clone(),
            mem_mb: memory_estimate(self.world.n(), self.world.sim.params(), self.world.sim.flow.is_some()) as f64 / 1e6,
        };
        let raw = self.egui_state.take_egui_input(&self.window);
        let mut actions = Vec::new();
        let ui_state = &mut self.ui;
        let full = self.egui_ctx.run_ui(raw, |ui| ui::draw(ui, ui_state, &info, &mut actions, now));
        self.egui_state.handle_platform_output(&self.window, full.platform_output);
        for a in actions {
            self.apply(a);
        }
        if self.ui.rule != *self.world.sim.params() {
            self.world.sim.set_params(&self.device, &self.queue, self.ui.rule.clone());
            self.ui.rule = self.world.sim.params().clone();
        }
        // Flow-mode settings edited in the UI
        if let Some(f) = self.world.sim.flow.as_mut() {
            if f.params != self.ui.flow_params {
                let rebuild = f.params.far != self.ui.flow_params.far;
                f.params = self.ui.flow_params.clone();
                if rebuild {
                    f.invalidate();
                }
            }
            if f.eco != self.ui.eco {
                f.eco = self.ui.eco.clone();
            }
        }
        self.world.traits = self.ui.traits.clone();
        self.world.sim.update_flow(&self.device, &self.queue);
        if now - self.last_audio > 0.1 {
            self.last_audio = now;
            self.update_audio(now);
        }
        let want_present = if self.ui.vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        if self.config.present_mode != want_present {
            self.config.present_mode = want_present;
            self.surface.configure(&self.device, &self.config);
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            _ => return Ok(()),
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        if let Some(shift) = self.pending_roll.take() {
            self.world.sim.encode_roll(&self.device, &mut enc, shift);
            self.world.tracker.shift(shift);
            self.roll_count += 1;
            let cell = 2.0 / self.world.n() as f32;
            let off = self.floor_offset + Vec2::new(shift[0] as f32, shift[2] as f32) * cell;
            self.floor_offset = Vec2::new(off.x.rem_euclid(0.25), off.y.rem_euclid(0.25));
            self.volume_dirty = true;
        }
        self.handle_tools(&mut enc);
        if self.ui.running {
            let steps = self.ui.steps_per_frame;
            // Big worlds: separate submissions keep each GPU packet short
            // (Windows resets the driver if one runs for ~2 s).
            let per_submit = match self.world.n() {
                n if n >= 512 => 1,
                n if n >= 256 => 4,
                _ => steps,
            };
            let mut left = steps;
            while left > 0 {
                let k = left.min(per_submit);
                let mut step_enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("steps") });
                self.world.sim.encode_steps(&mut step_enc, k);
                self.queue.submit([step_enc.finish()]);
                left -= k;
            }
            self.steps_counter.1 += steps as u64;
            self.volume_dirty = true;
        }
        let el = now_i - self.steps_counter.0;
        if el.as_secs_f32() >= 0.5 {
            self.steps_counter.2 = self.steps_counter.1 as f32 / el.as_secs_f32();
            self.steps_counter = (now_i, 0, self.steps_counter.2);
        }

        if !self.pending_ops.is_empty() {
            let ops = std::mem::take(&mut self.pending_ops);
            self.world.apply_ops(&self.device, &mut enc, &ops);
            self.volume_dirty = true;
        }

        // creature tracking readback
        let mut track_submitted = false;
        if self.frame_index % 6 == 3 {
            let (m, _) = crate::sim::track_grid(self.world.n());
            let layout = self.world.sim.layout_id() ^ ((self.world.n() as u64) << 48);
            let bytes = (m as u64).pow(3) * crate::sim::TRACK_FLOATS as u64 * 4;
            let idle = self.track_reader.as_ref().is_none_or(|r| r.state.load(Ordering::Acquire) == 0);
            if idle && self.track_reader.as_ref().is_none_or(|r| r.layout != layout || r.m != m) {
                self.track_reader = Some(TrackReader {
                    staging: self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("track staging"),
                        size: bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    m,
                    layout,
                    generation: 0,
                    roll_at_request: 0,
                    state: Arc::new(AtomicU8::new(0)),
                });
            }
            let generation = self.world.sim.generation;
            let roll_count = self.roll_count;
            if let Some(r) = &mut self.track_reader {
                if r.layout == layout && r.state.load(Ordering::Acquire) == 0 {
                    let (buf, _, _) = self.world.sim.encode_track(&self.device, &mut enc);
                    enc.copy_buffer_to_buffer(buf, 0, &r.staging, 0, bytes);
                    r.generation = generation;
                    r.roll_at_request = roll_count;
                    r.state.store(1, Ordering::Release);
                    track_submitted = true;
                }
            }
        }

        // statistics (mass, center of mass) every few frames
        let mut mass_submitted = false;
        if self.frame_index.is_multiple_of(3) {
            let (buf, groups, bytes) = self.world.sim.mass_buffer();
            let channels = self.world.sim.channels.len();
            let layout = self.world.sim.layout_id() ^ ((self.world.n() as u64) << 48);
            let need_new = self.mass_reader.as_ref().is_none_or(|r| r.layout != layout || r.groups != groups || r.channels != channels);
            if need_new && self.mass_reader.as_ref().is_none_or(|r| r.state.load(Ordering::Acquire) == 0) {
                self.mass_reader = Some(MassReader {
                    staging: self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("mass staging"),
                        size: bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    groups,
                    channels,
                    layout,
                    roll_at_request: 0,
                    state: Arc::new(AtomicU8::new(0)),
                });
            }
            let roll_count = self.roll_count;
            if let Some(r) = &mut self.mass_reader {
                if r.layout == layout && r.state.load(Ordering::Acquire) == 0 {
                    self.world.sim.encode_mass(&mut enc);
                    enc.copy_buffer_to_buffer(buf, 0, &r.staging, 0, r.staging.size());
                    r.state.store(1, Ordering::Release);
                    r.roll_at_request = roll_count;
                    mass_submitted = true;
                }
            }
        }

        if self.volume_dirty {
            self.renderer.update_volume(&self.device, &self.queue, &mut enc, &self.world.sim, self.ui.settings.smooth);
            self.volume_dirty = false;
        }
        let painting = self.ui.tool != Tool::View;
        let brush = if painting && !self.egui_ctx.is_pointer_over_egui() {
            self.brush_point().map(|p| (p, self.brush_world_radius()))
        } else {
            None
        };
        let overlay = Overlay { plane_y: painting.then_some(self.ui.plane_y), brush, floor_offset: self.floor_offset.into() };
        let creatures = self.world.gpu_creatures(now, self.ui.selected_creature);
        self.renderer.set_creatures(&self.queue, &creatures);
        let size = (self.config.width, self.config.height);
        self.renderer.render(&self.device, &self.queue, &mut enc, &view, size, &self.camera, &self.ui.settings, &overlay, now as f32);

        // egui on top
        let ppp = full.pixels_per_point;
        let jobs = self.egui_ctx.tessellate(full.shapes, ppp);
        for (id, deltas) in &full.textures_delta.set {
            for delta in deltas {
                self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.0, size.1], pixels_per_point: ppp };
        let extra = self.egui_renderer.update_buffers(&self.device, &self.queue, &mut enc, &jobs, &screen);
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            self.egui_renderer.render(&mut pass, &jobs, &screen);
        }
        for id in &full.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.queue.submit(extra.into_iter().chain([enc.finish()]));
        if track_submitted {
            if let Some(r) = &self.track_reader {
                let state = r.state.clone();
                r.staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                    state.store(if res.is_ok() { 2 } else { 0 }, Ordering::Release);
                });
            }
        }
        if mass_submitted {
            if let Some(r) = &self.mass_reader {
                let state = r.state.clone();
                r.staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                    state.store(if res.is_ok() { 2 } else { 0 }, Ordering::Release);
                });
            }
        }
        self.window.pre_present_notify();
        self.queue.present(frame);
        Ok(())
    }
}
