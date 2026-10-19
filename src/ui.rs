//! egui panels: transport bar, presets, rule editor, tools and render settings.

use crate::render::{ClipAxis, RenderMode, RenderSettings};
use crate::sim::params::{growth, kernel_shell, KernelParams, LeniaParams, MAX_CHANNELS, MAX_KERNELS, MAX_PEAKS};
use crate::sim::preset::Preset;
use egui::{Color32, RichText};
use egui_plot::{Line, Plot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    View,
    Creature,
    Noise,
    Paint,
    Erase,
    Wall,
    Unwall,
}

impl Tool {
    pub const ALL: [Tool; 7] = [Tool::View, Tool::Creature, Tool::Noise, Tool::Paint, Tool::Erase, Tool::Wall, Tool::Unwall];

    pub fn label(self) -> &'static str {
        match self {
            Tool::View => "View",
            Tool::Creature => "Stamp creature",
            Tool::Noise => "Random blob",
            Tool::Paint => "Paint density",
            Tool::Erase => "Erase",
            Tool::Wall => "Build wall",
            Tool::Unwall => "Remove wall",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Tool::View => "Left-drag orbits the camera.",
            Tool::Creature => "Click to place the selected creature on the work plane.",
            Tool::Noise => "Drag to spray random matter; good for starting new life.",
            Tool::Paint => "Drag to add smooth density.",
            Tool::Erase => "Drag to remove matter.",
            Tool::Wall => "Drag to build solid walls. Matter cannot exist inside walls.",
            Tool::Unwall => "Drag to remove walls.",
        }
    }
}

pub enum Action {
    LoadPreset(usize),
    Reset,
    Clear,
    Soup,
    Resize(u32),
    Step,
    SaveWorld,
    LoadWorld,
    DownloadChan,
    ImportChan,
    Screenshot,
    FrameView,
    ClearWalls,
    EncloseWalls,
    Pillars,
}

pub struct FrameInfo {
    pub fps: f32,
    pub steps_per_sec: f32,
    pub generation: u64,
    pub time: f64,
    pub mass: Vec<f64>,
    pub n: u32,
    pub max_n: u32,
    pub adapter: String,
    pub mem_mb: f64,
}

pub struct UiState {
    pub running: bool,
    pub steps_per_frame: u32,
    pub settings: RenderSettings,
    pub rule: LeniaParams,
    pub presets: Vec<Preset>,
    pub selected: usize,
    pub filter: String,
    pub scale: f32,
    pub tool: Tool,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub brush_channel: Option<usize>,
    pub plane_y: f32,
    pub show_ui: bool,
    pub show_help: bool,
    pub vsync: bool,
    /// Keep the center of mass in the middle by rolling the periodic world.
    pub follow: bool,
    pub chan_loaded: bool,
    pub downloading: bool,
    pub status: String,
    pub status_time: f64,
    pub rng_state: u64,
}

impl UiState {
    pub fn new(presets: Vec<Preset>) -> Self {
        Self {
            running: true,
            steps_per_frame: 2,
            settings: RenderSettings::default(),
            rule: LeniaParams::default(),
            presets,
            selected: 0,
            filter: String::new(),
            scale: 1.0,
            tool: Tool::View,
            brush_radius: 10.0,
            brush_strength: 1.0,
            brush_channel: None,
            plane_y: 0.0,
            show_ui: true,
            show_help: true,
            vsync: true,
            follow: false,
            chan_loaded: false,
            downloading: false,
            status: String::new(),
            status_time: 0.0,
            rng_state: 0x2545F4914F6CDD1D,
        }
    }

    pub fn set_status(&mut self, s: impl Into<String>, now: f64) {
        self.status = s.into();
        self.status_time = now;
    }
}

fn panel_frame(ctx: &egui::Context) -> egui::Frame {
    let mut f = egui::Frame::side_top_panel(&ctx.global_style());
    f.fill = Color32::from_rgba_unmultiplied(16, 18, 24, 236);
    f
}

pub fn draw(ui: &mut egui::Ui, st: &mut UiState, info: &FrameInfo, actions: &mut Vec<Action>, now: f64) {
    let ctx = ui.ctx().clone();
    if !st.show_ui {
        egui::Area::new(egui::Id::unique("hidden hint")).fixed_pos(egui::pos2(10.0, 8.0)).show(&ctx, |ui| {
            ui.label(RichText::new("Tab: show panels").weak());
        });
        return;
    }
    top_bar(ui, st, info, actions);
    egui::Panel::left("controls").resizable(true).default_size(340.0).frame(panel_frame(&ctx)).show(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            presets_section(ui, st, actions);
            rule_section(ui, st, info);
            tools_section(ui, st, actions);
        });
    });
    egui::Panel::right("render").resizable(true).default_size(270.0).frame(panel_frame(&ctx)).show(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            render_section(ui, st, actions);
        });
    });
    if st.show_help {
        let mut open = true;
        egui::Window::new("Controls")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-290.0, -12.0))
            .show(&ctx, help_contents);
        if !open {
            st.show_help = false;
        }
    }
    if !st.status.is_empty() && now - st.status_time < 6.0 {
        egui::Area::new(egui::Id::unique("status"))
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -16.0))
            .show(&ctx, |ui| {
                egui::Frame::popup(&ctx.global_style()).show(ui, |ui| {
                    ui.label(&st.status);
                });
            });
    }
}

fn help_contents(ui: &mut egui::Ui) {
    egui::Grid::new("help").num_columns(2).spacing([24.0, 4.0]).show(ui, |ui| {
        let row = |ui: &mut egui::Ui, k: &str, v: &str| {
            ui.label(RichText::new(k).strong());
            ui.label(v);
            ui.end_row();
        };
        row(ui, "Right-drag", "orbit (left-drag too, in View tool)");
        row(ui, "Middle-drag / Shift+drag", "pan");
        row(ui, "Wheel", "zoom toward cursor");
        row(ui, "W A S D, Q E", "fly");
        row(ui, "F", "frame the world");
        row(ui, "Space", "play / pause");
        row(ui, ". (period)", "single step");
        row(ui, "R / C", "reset / clear world");
        row(ui, "1 - 7", "tools (view, creature, blob, paint, erase, wall, unwall)");
        row(ui, "[ ]  or Ctrl+wheel", "brush size");
        row(ui, "PgUp/PgDn or Alt+wheel", "work plane height");
        row(ui, "M", "cycle render mode");
        row(ui, "Tab", "hide/show panels");
        row(ui, "F11 / F12", "fullscreen / screenshot");
        row(ui, "H", "this help");
    });
    ui.add_space(6.0);
    ui.label(RichText::new("Pick a creature on the left, press Load, and watch it live. Edit the rule live with the sliders.").weak());
}

fn top_bar(ui: &mut egui::Ui, st: &mut UiState, info: &FrameInfo, actions: &mut Vec<Action>) {
    let ctx = ui.ctx().clone();
    egui::Panel::top("top").frame(panel_frame(&ctx)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Lenia3D").strong().color(Color32::from_rgb(120, 220, 200)));
            ui.separator();
            let play = if st.running { "Pause" } else { "Play" };
            if ui.button(play).on_hover_text("Space").clicked() {
                st.running = !st.running;
            }
            if ui.button("Step").on_hover_text(". (period)").clicked() {
                actions.push(Action::Step);
            }
            ui.add(egui::Slider::new(&mut st.steps_per_frame, 1..=32).text("steps/frame"));
            ui.checkbox(&mut st.follow, "Follow").on_hover_text(
                "Keep the creature centered: the periodic world is shifted so the center of mass stays in the middle. Paused while walls exist.",
            );
            ui.separator();
            egui::ComboBox::from_id_salt("worldsize")
                .selected_text(format!("World {}³", info.n))
                .show_ui(ui, |ui| {
                    for n in [64u32, 128, 256, 512] {
                        let ok = n <= info.max_n;
                        let label = format!("{n}³  ({:.1} M cells)", (n as f64).powi(3) / 1e6);
                        if ui.add_enabled(ok, egui::Button::selectable(n == info.n, label)).clicked() && n != info.n {
                            actions.push(Action::Resize(n));
                        }
                    }
                })
                .response
                .on_hover_text(format!("GPU memory for this world: ~{:.0} MB", info.mem_mb));
            if ui.button("Reset").on_hover_text("R: re-seed the current creature / soup").clicked() {
                actions.push(Action::Reset);
            }
            if ui.button("Soup").on_hover_text("Fill the world with random blobs under the current rule").clicked() {
                actions.push(Action::Soup);
            }
            if ui.button("Clear").on_hover_text("C").clicked() {
                actions.push(Action::Clear);
            }
            ui.separator();
            if ui.button("Save…").clicked() {
                actions.push(Action::SaveWorld);
            }
            if ui.button("Open…").clicked() {
                actions.push(Action::LoadWorld);
            }
            ui.separator();
            let mass: f64 = info.mass.iter().sum();
            ui.label(format!(
                "gen {}  t={:.1}  mass {:.0}  |  {:.0} fps  {:.0} steps/s",
                info.generation, info.time, mass, info.fps, info.steps_per_sec
            ));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Help").clicked() {
                    st.show_help = !st.show_help;
                }
                ui.label(RichText::new(&info.adapter).weak().small());
            });
        });
    });
}

fn presets_section(ui: &mut egui::Ui, st: &mut UiState, actions: &mut Vec<Action>) {
    egui::CollapsingHeader::new(RichText::new("Creatures").strong()).default_open(true).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Search");
            ui.text_edit_singleline(&mut st.filter);
        });
        let filter = st.filter.to_lowercase();
        egui::ScrollArea::vertical().id_salt("presetlist").max_height(230.0).show(ui, |ui| {
            let mut last_group = "";
            for (i, p) in st.presets.iter().enumerate() {
                if !filter.is_empty() && !p.name.to_lowercase().contains(&filter) && !p.description.to_lowercase().contains(&filter) {
                    continue;
                }
                let group = if p.kind == "chan" { "Bert Chan's 3D species" } else { "Built-in" };
                if group != last_group {
                    ui.label(RichText::new(group).small().weak());
                    last_group = group;
                }
                let tag = match p.kind.as_str() {
                    "soup" => "  [soup]".to_string(),
                    _ if p.rule.channels > 1 => format!("  [{}ch]", p.rule.channels),
                    _ => String::new(),
                };
                let r = ui.add(egui::Button::selectable(st.selected == i, format!("{}{}", p.name, tag)));
                let r = if p.description.is_empty() { r } else { r.on_hover_text(&p.description) };
                if r.clicked() {
                    st.selected = i;
                }
                if r.double_clicked() {
                    actions.push(Action::LoadPreset(i));
                }
            }
        });
        ui.add(egui::Slider::new(&mut st.scale, 0.5..=4.0).text("size ×").step_by(0.25))
            .on_hover_text("Scales the creature and its kernel radius R. Bigger looks smoother but is slower to simulate.");
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Load into world").strong()).on_hover_text("Double-click a name to load it too").clicked() {
                actions.push(Action::LoadPreset(st.selected));
            }
            if ui.button("Stamp with mouse").clicked() {
                st.tool = Tool::Creature;
            }
        });
        if let Some(p) = st.presets.get(st.selected) {
            if !p.description.is_empty() {
                ui.label(RichText::new(&p.description).small().weak());
            }
        }
        if !st.chan_loaded {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let label = if st.downloading { "Downloading…" } else { "Get Bert Chan's 36 3D species" };
                if ui.add_enabled(!st.downloading, egui::Button::new(label)).on_hover_text(
                    "Downloads animals3D.json from github.com/Chakazul/Lenia (not bundled: that repository has no license file).",
                ).clicked() {
                    actions.push(Action::DownloadChan);
                }
                if ui.button("Import file…").clicked() {
                    actions.push(Action::ImportChan);
                }
            });
        }
    });
}

fn log_slider(ui: &mut egui::Ui, v: &mut f32, range: std::ops::RangeInclusive<f32>, text: &str) -> egui::Response {
    ui.add(egui::Slider::new(v, range).logarithmic(true).text(text))
}

fn rule_section(ui: &mut egui::Ui, st: &mut UiState, info: &FrameInfo) {
    egui::CollapsingHeader::new(RichText::new("Rule (live)").strong()).default_open(true).show(ui, |ui| {
        let rule = &mut st.rule;
        let max_r = (info.n as f32 / 2.0 - 1.0).min(80.0);
        ui.add(egui::Slider::new(&mut rule.R, 2.0..=max_r).text("R  radius")).on_hover_text("Kernel radius in cells: the 'size' of the physics.");
        ui.add(egui::Slider::new(&mut rule.T, 1.0..=50.0).text("T  time res.")).on_hover_text("Steps per unit time (dt = 1/T). Higher = smoother, slower.");
        ui.horizontal(|ui| {
            ui.label("Channels");
            for c in 1..=MAX_CHANNELS {
                if ui.add(egui::Button::selectable(rule.channels == c, c.to_string())).clicked() {
                    rule.channels = c;
                }
            }
        });
        let nch = rule.channels;
        let mut remove = None;
        for (i, k) in rule.kernels.iter_mut().enumerate() {
            let title = if nch > 1 { format!("Kernel {}   ch{} -> ch{}", i + 1, k.src, k.dst) } else { format!("Kernel {}", i + 1) };
            egui::CollapsingHeader::new(title).id_salt(("kernel", i)).default_open(i == 0).show(ui, |ui| {
                kernel_editor(ui, k, nch, i);
                if ui.small_button("Remove kernel").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            if rule.kernels.len() > 1 {
                rule.kernels.remove(i);
            }
        }
        ui.horizontal(|ui| {
            if rule.kernels.len() < MAX_KERNELS && ui.button("+ Add kernel").clicked() {
                let mut k = rule.kernels.last().cloned().unwrap_or_default();
                k.m = (k.m * 1.1).min(1.0);
                rule.kernels.push(k);
            }
            if ui.button("Mutate").on_hover_text("Nudge every growth parameter by a few percent").clicked() {
                st.rng_state = mutate_rule(rule, st.rng_state, 0.04);
            }
            if ui.button("Random rule").on_hover_text("A new random rule with the same channels (then press Soup)").clicked() {
                st.rng_state = randomize_rule(rule, st.rng_state);
            }
        });
    });
}

fn next_rand(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32) / (1u64 << 24) as f32
}

fn mutate_rule(rule: &mut LeniaParams, mut state: u64, amount: f32) -> u64 {
    for k in &mut rule.kernels {
        let mut j = |v: &mut f32, lo: f32, hi: f32| {
            *v = (*v * (1.0 + (next_rand(&mut state) * 2.0 - 1.0) * amount)).clamp(lo, hi);
        };
        j(&mut k.m, 0.01, 0.6);
        j(&mut k.s, 0.001, 0.2);
        j(&mut k.h, 0.0, 2.0);
        j(&mut k.r, 0.1, 2.0);
        for b in &mut k.b {
            j(b, 0.0, 1.0);
        }
    }
    state
}

fn randomize_rule(rule: &mut LeniaParams, mut state: u64) -> u64 {
    let nch = rule.channels;
    let count = (nch * nch).clamp(1, 6) + (next_rand(&mut state) * 2.0) as usize;
    rule.kernels = (0..count)
        .map(|i| {
            let mut r = || next_rand(&mut state);
            let nb = 1 + (r() * 3.0) as usize;
            let m = 0.1 + r() * 0.25;
            KernelParams {
                src: if nch > 1 { i % nch } else { 0 },
                dst: if nch > 1 { (i / nch) % nch } else { 0 },
                r: 0.4 + r() * 0.6,
                b: (0..nb).map(|j| if j == 0 { 1.0 } else { (r() * 100.0).round() / 100.0 }).collect(),
                m,
                s: m * (0.08 + r() * 0.12),
                h: 0.3 + r() * 0.7,
                kn: 1,
                gn: 1,
            }
        })
        .collect();
    state
}

fn kernel_editor(ui: &mut egui::Ui, k: &mut KernelParams, nch: usize, id: usize) {
    if nch > 1 {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt(("src", id)).selected_text(format!("from ch{}", k.src)).show_ui(ui, |ui| {
                for c in 0..nch {
                    ui.selectable_value(&mut k.src, c, format!("ch{c}"));
                }
            });
            egui::ComboBox::from_id_salt(("dst", id)).selected_text(format!("to ch{}", k.dst)).show_ui(ui, |ui| {
                for c in 0..nch {
                    ui.selectable_value(&mut k.dst, c, format!("ch{c}"));
                }
            });
        });
    }
    ui.add(egui::Slider::new(&mut k.m, 0.01..=0.6).text("μ  growth center")).on_hover_text("Neighborhood density that makes cells grow.");
    log_slider(ui, &mut k.s, 0.001..=0.2, "σ  growth width").on_hover_text("How tolerant growth is around μ.");
    ui.add(egui::Slider::new(&mut k.h, 0.0..=2.0).text("h  weight"));
    ui.add(egui::Slider::new(&mut k.r, 0.1..=2.0).text("r  relative radius"));
    ui.horizontal(|ui| {
        ui.label("rings");
        let mut del = false;
        let nb = k.b.len();
        for (j, b) in k.b.iter_mut().enumerate() {
            ui.add(egui::DragValue::new(b).range(0.0..=1.0).speed(0.01).max_decimals(2)).on_hover_text(format!("Height of ring {}", j + 1));
        }
        if nb < MAX_PEAKS && ui.small_button("+").clicked() {
            k.b.push(0.5);
        }
        if nb > 1 && ui.small_button("−").clicked() {
            del = true;
        }
        if del {
            k.b.pop();
        }
    });
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt(("kn", id))
            .selected_text(["polynomial", "exponential", "step", "staircase"][(k.kn as usize - 1).min(3)])
            .show_ui(ui, |ui| {
                for (i, n) in ["polynomial", "exponential", "step", "staircase"].iter().enumerate() {
                    ui.selectable_value(&mut k.kn, i as u32 + 1, *n);
                }
            });
        egui::ComboBox::from_id_salt(("gn", id))
            .selected_text(["polynomial", "gaussian", "step"][(k.gn as usize - 1).min(2)])
            .show_ui(ui, |ui| {
                for (i, n) in ["polynomial", "gaussian", "step"].iter().enumerate() {
                    ui.selectable_value(&mut k.gn, i as u32 + 1, *n);
                }
            });
    });
    // live plots of kernel profile and growth function
    let kpts: Vec<[f64; 2]> = (0..=200).map(|i| {
        let r = i as f64 / 200.0;
        [r, kernel_shell(r, &k.b, k.kn)]
    }).collect();
    let umax = ((k.m + 4.0 * k.s) as f64).max(0.05).min(1.0);
    let gpts: Vec<[f64; 2]> = (0..=200).map(|i| {
        let u = i as f64 / 200.0 * umax;
        [u, growth(u, k.m as f64, k.s as f64, k.gn)]
    }).collect();
    ui.columns(2, |cols| {
        cols[0].label(RichText::new("kernel shells").small().weak());
        Plot::new(("kplot", id))
            .height(70.0)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_axes([false, false])
            .include_y(0.0)
            .include_y(1.0)
            .show(&mut cols[0], |p| p.line(Line::new("K", kpts).color(Color32::from_rgb(90, 210, 190))));
        cols[1].label(RichText::new("growth G(u)").small().weak());
        Plot::new(("gplot", id))
            .height(70.0)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_axes([false, false])
            .include_y(-1.0)
            .include_y(1.0)
            .show(&mut cols[1], |p| p.line(Line::new("G", gpts).color(Color32::from_rgb(240, 160, 90))));
    });
}

fn tools_section(ui: &mut egui::Ui, st: &mut UiState, actions: &mut Vec<Action>) {
    egui::CollapsingHeader::new(RichText::new("Tools & environment").strong()).default_open(true).show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            for (i, t) in Tool::ALL.iter().enumerate() {
                ui.add(egui::Button::selectable(st.tool == *t, format!("{} {}", i + 1, t.label())))
                    .clicked()
                    .then(|| st.tool = *t);
            }
        });
        ui.label(RichText::new(st.tool.hint()).small().weak());
        ui.add(egui::Slider::new(&mut st.brush_radius, 1.0..=64.0).text("brush radius")).on_hover_text("[ ] or Ctrl+wheel");
        ui.add(egui::Slider::new(&mut st.brush_strength, 0.05..=1.0).text("strength"));
        ui.add(egui::Slider::new(&mut st.plane_y, -1.0..=1.0).text("work plane height")).on_hover_text("PgUp/PgDn or Alt+wheel. Brushes act where the mouse ray hits this plane.");
        let nch = st.rule.channels;
        if nch > 1 {
            egui::ComboBox::from_id_salt("brushch")
                .selected_text(match st.brush_channel {
                    None => "all channels".to_string(),
                    Some(c) => format!("channel {c}"),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut st.brush_channel, None, "all channels");
                    for c in 0..nch {
                        ui.selectable_value(&mut st.brush_channel, Some(c), format!("channel {c}"));
                    }
                });
        }
        ui.horizontal(|ui| {
            if ui.button("Enclose world").on_hover_text("Walls on all six faces: turns the periodic torus into a closed box.").clicked() {
                actions.push(Action::EncloseWalls);
            }
            if ui.button("Pillars").on_hover_text("Scatter some pillars as obstacles").clicked() {
                actions.push(Action::Pillars);
            }
            if ui.button("Clear walls").clicked() {
                actions.push(Action::ClearWalls);
            }
        });
    });
}

fn render_section(ui: &mut egui::Ui, st: &mut UiState, actions: &mut Vec<Action>) {
    let s = &mut st.settings;
    ui.heading("View");
    ui.horizontal(|ui| {
        for (m, name) in [(RenderMode::Organic, "Organic"), (RenderMode::Glow, "Glow"), (RenderMode::Xray, "X-ray")] {
            if ui.add(egui::Button::selectable(s.mode == m, name)).clicked() {
                s.mode = m;
            }
        }
    });
    match s.mode {
        RenderMode::Organic => {
            ui.add(egui::Slider::new(&mut s.iso, 0.01..=0.9).text("skin level"));
            ui.add(egui::Slider::new(&mut s.core_iso, 0.02..=1.5).text("core level"));
            ui.add(egui::Slider::new(&mut s.membrane_opacity, 0.0..=1.0).text("skin opacity"));
            ui.add(egui::Slider::new(&mut s.density_gain, 0.0..=30.0).text("jelly density"));
            ui.add(egui::Slider::new(&mut s.sss, 0.0..=2.0).text("translucency"));
        }
        _ => {
            ui.add(egui::Slider::new(&mut s.emission, 0.1..=10.0).logarithmic(true).text("glow"));
            ui.add(egui::Slider::new(&mut s.density_gain, 0.0..=30.0).text("opacity"));
        }
    }
    ui.checkbox(&mut s.colormap, "Heat colormap");
    ui.horizontal(|ui| {
        ui.label("Colors");
        for c in 0..st.rule.channels {
            ui.color_edit_button_rgb(&mut s.channel_colors[c]);
        }
        if s.mode == RenderMode::Organic {
            ui.label("core");
            ui.color_edit_button_rgb(&mut s.core_color);
        }
    });
    ui.separator();
    ui.heading("Cut");
    ui.horizontal(|ui| {
        for (a, name) in [(ClipAxis::None, "off"), (ClipAxis::X, "X"), (ClipAxis::Y, "Y"), (ClipAxis::Z, "Z")] {
            if ui.add(egui::Button::selectable(s.clip_axis == a, name)).clicked() {
                s.clip_axis = a;
            }
        }
        ui.checkbox(&mut s.clip_flip, "flip");
    });
    if s.clip_axis != ClipAxis::None {
        ui.add(egui::Slider::new(&mut s.clip_pos, -1.0..=1.0).text("position"));
    }
    ui.separator();
    ui.heading("Light");
    ui.add(egui::Slider::new(&mut s.light_azimuth, -3.14..=3.14).text("direction"));
    ui.add(egui::Slider::new(&mut s.light_elevation, 0.05..=1.55).text("height"));
    ui.add(egui::Slider::new(&mut s.light_intensity, 0.0..=4.0).text("intensity"));
    ui.add(egui::Slider::new(&mut s.shadow, 0.0..=1.0).text("shadows"));
    ui.add(egui::Slider::new(&mut s.ao, 0.0..=2.0).text("occlusion"));
    ui.separator();
    ui.heading("Camera & image");
    ui.add(egui::Slider::new(&mut s.exposure, 0.2..=4.0).logarithmic(true).text("exposure"));
    ui.add(egui::Slider::new(&mut s.bloom, 0.0..=0.5).text("bloom"));
    ui.add(egui::Slider::new(&mut s.vignette, 0.0..=1.0).text("vignette"));
    ui.add(egui::Slider::new(&mut s.step_scale, 0.25..=1.5).text("ray step (quality)")).on_hover_text("Smaller = finer, slower.");
    ui.add(egui::Slider::new(&mut s.render_scale, 0.25..=1.0).text("render scale")).on_hover_text("Render the 3D view at lower resolution for speed (useful on 4K screens).");
    ui.horizontal(|ui| {
        ui.checkbox(&mut s.floor, "floor");
        ui.checkbox(&mut s.show_box, "box");
        ui.checkbox(&mut s.show_walls, "walls");
        ui.checkbox(&mut s.smooth, "smooth");
    });
    ui.checkbox(&mut st.vsync, "VSync");
    ui.horizontal(|ui| {
        if ui.button("Screenshot (F12)").clicked() {
            actions.push(Action::Screenshot);
        }
        if ui.button("Frame (F)").clicked() {
            actions.push(Action::FrameView);
        }
    });
    if ui.button("Reset view settings").clicked() {
        st.settings = RenderSettings::default();
    }
}
