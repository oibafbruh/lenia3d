//! Volumetric renderer: volume texture + brick map, raymarching into an HDR
//! target, bloom and tonemapping.

pub mod camera;

use crate::sim::Simulation;
use bytemuck::{Pod, Zeroable};
use camera::OrbitCamera;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const BLOOM_LEVELS: u32 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    Organic = 0,
    Glow = 1,
    Xray = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipAxis {
    None = 0,
    X = 1,
    Y = 2,
    Z = 3,
}

#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub mode: RenderMode,
    /// Membrane (outer surface) density level.
    pub iso: f32,
    /// Core (inner, opaque) density level.
    pub core_iso: f32,
    pub membrane_opacity: f32,
    pub density_gain: f32,
    pub emission: f32,
    pub step_scale: f32,
    pub channel_colors: [[f32; 3]; 3],
    pub core_color: [f32; 3],
    pub colormap: bool,
    pub exposure: f32,
    pub bloom: f32,
    pub vignette: f32,
    pub light_azimuth: f32,
    pub light_elevation: f32,
    pub light_intensity: f32,
    pub sss: f32,
    pub shadow: f32,
    pub ao: f32,
    pub floor: bool,
    pub show_box: bool,
    pub show_walls: bool,
    pub clip_axis: ClipAxis,
    pub clip_pos: f32,
    pub clip_flip: bool,
    pub bg_top: [f32; 3],
    pub bg_bottom: [f32; 3],
    /// Smooth the volume (3x3x3 filter) before rendering.
    pub smooth: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            mode: RenderMode::Organic,
            iso: 0.12,
            core_iso: 0.55,
            membrane_opacity: 0.45,
            density_gain: 6.0,
            emission: 1.5,
            step_scale: 0.5,
            channel_colors: [[0.25, 0.85, 0.75], [1.0, 0.45, 0.35], [0.45, 0.55, 1.0]],
            core_color: [1.0, 0.62, 0.38],
            colormap: false,
            exposure: 1.1,
            bloom: 0.06,
            vignette: 0.35,
            light_azimuth: 0.9,
            light_elevation: 0.85,
            light_intensity: 1.2,
            sss: 0.6,
            shadow: 0.9,
            ao: 1.0,
            floor: true,
            show_box: true,
            show_walls: true,
            clip_axis: ClipAxis::None,
            clip_pos: 0.0,
            clip_flip: false,
            bg_top: [0.035, 0.045, 0.07],
            bg_bottom: [0.012, 0.012, 0.018],
            smooth: true,
        }
    }
}

impl RenderSettings {
    /// Apply "key=value,key=value" overrides (used by the command line).
    pub fn apply_overrides(&mut self, spec: &str) -> Result<(), String> {
        for kv in spec.split(',').filter(|s| !s.trim().is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad override '{kv}'"))?;
            let f = || v.trim().parse::<f32>().map_err(|_| format!("bad number in '{kv}'"));
            match k.trim() {
                "iso" => self.iso = f()?,
                "core" => self.core_iso = f()?,
                "membrane" => self.membrane_opacity = f()?,
                "gain" => self.density_gain = f()?,
                "emission" => self.emission = f()?,
                "exposure" => self.exposure = f()?,
                "bloom" => self.bloom = f()?,
                "light" => self.light_intensity = f()?,
                "sss" => self.sss = f()?,
                "shadow" => self.shadow = f()?,
                "ao" => self.ao = f()?,
                "colormap" => self.colormap = f()? > 0.5,
                "smooth" => self.smooth = f()? > 0.5,
                "floor" => self.floor = f()? > 0.5,
                "box" => self.show_box = f()? > 0.5,
                "clip_x" | "clip_y" | "clip_z" => {
                    self.clip_axis = match k.trim() {
                        "clip_x" => ClipAxis::X,
                        "clip_y" => ClipAxis::Y,
                        _ => ClipAxis::Z,
                    };
                    self.clip_pos = f()?;
                }
                other => return Err(format!("unknown render setting '{other}'")),
            }
        }
        Ok(())
    }

    pub fn light_dir(&self) -> Vec3 {
        let (sa, ca) = self.light_azimuth.sin_cos();
        let (se, ce) = self.light_elevation.sin_cos();
        Vec3::new(ce * sa, se, ce * ca).normalize()
    }
}

/// Things drawn on top of the volume for editing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Overlay {
    /// World-space height of the work plane, if shown.
    pub plane_y: Option<f32>,
    /// Brush sphere (world center, world radius), if shown.
    pub brush: Option<(Vec3, f32)>,
    /// Floor grid offset (x, z) so the ground scrolls in follow mode.
    pub floor_offset: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    light_dir: [f32; 4],
    col0: [f32; 4],
    col1: [f32; 4],
    col2: [f32; 4],
    core_col: [f32; 4],
    bg_top: [f32; 4],
    bg_bot: [f32; 4],
    brush: [f32; 4],
    p0: [f32; 4],
    p1: [f32; 4],
    p2: [f32; 4],
    ip: [u32; 4],
    screen: [f32; 4],
    p3: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomParams {
    texel: [f32; 2],
    first: f32,
    _p: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeParams {
    exposure: f32,
    bloom: f32,
    encode_srgb: f32,
    vignette: f32,
}

fn ext3(c: [f32; 3]) -> [f32; 4] {
    [c[0], c[1], c[2], 1.0]
}

struct Targets {
    width: u32,
    height: u32,
    hdr_view: wgpu::TextureView,
    bloom_views: Vec<wgpu::TextureView>,
    bloom_sizes: Vec<(u32, u32)>,
    down_bgs: Vec<wgpu::BindGroup>,
    up_bgs: Vec<wgpu::BindGroup>,
    composite_bg: wgpu::BindGroup,
}

struct VolumeBinding {
    n: u32,
    nch: u32,
    sim_id: u64,
    view: wgpu::TextureView,
    bricks: wgpu::Buffer,
    prep_params: wgpu::Buffer,
    prep_bg: wgpu::BindGroup,
    brick_bg: wgpu::BindGroup,
    raymarch_bg: wgpu::BindGroup,
}

pub struct Renderer {
    output_format: wgpu::TextureFormat,
    linear_sampler: wgpu::Sampler,
    prep_pipeline: wgpu::ComputePipeline,
    brick_pipeline: wgpu::ComputePipeline,
    raymarch_pipeline: wgpu::RenderPipeline,
    down_pipeline: wgpu::RenderPipeline,
    up_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    composite_params: wgpu::Buffer,
    bloom_params: Vec<wgpu::Buffer>,
    targets: Option<Targets>,
    volume: Option<VolumeBinding>,
    frame: u32,
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    fs: &str,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
}

fn compute(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let m = shader(device, label, src);
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &m,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl Renderer {
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let prep_pipeline = compute(device, "volume prep", include_str!("../shaders/volume_prep.wgsl"));
        let brick_pipeline = compute(device, "bricks", include_str!("../shaders/bricks.wgsl"));
        let rm = shader(device, "raymarch", include_str!("../shaders/raymarch.wgsl"));
        let raymarch_pipeline = fullscreen_pipeline(device, "raymarch", &rm, "fs_main", HDR_FORMAT, None);
        let bm = shader(device, "bloom", include_str!("../shaders/bloom.wgsl"));
        let down_pipeline = fullscreen_pipeline(device, "bloom down", &bm, "fs_down", HDR_FORMAT, None);
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let up_pipeline = fullscreen_pipeline(device, "bloom up", &bm, "fs_up", HDR_FORMAT, Some(additive));
        let cm = shader(device, "composite", include_str!("../shaders/composite.wgsl"));
        let composite_pipeline = fullscreen_pipeline(device, "composite", &cm, "fs_main", output_format, None);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raymarch uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite params"),
            size: std::mem::size_of::<CompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bloom_params = (0..2 * BLOOM_LEVELS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bloom params"),
                    size: std::mem::size_of::<BloomParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self {
            output_format,
            linear_sampler,
            prep_pipeline,
            brick_pipeline,
            raymarch_pipeline,
            down_pipeline,
            up_pipeline,
            composite_pipeline,
            uniforms,
            composite_params,
            bloom_params,
            targets: None,
            volume: None,
            frame: 0,
        }
    }

    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.targets.as_ref().is_some_and(|t| t.width == width && t.height == height) {
            return;
        }
        let tex = |label: &str, w: u32, h: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let hdr_view = tex("hdr", width, height).create_view(&Default::default());
        let mut bloom_views = Vec::new();
        let mut bloom_sizes = Vec::new();
        let (mut w, mut h) = (width, height);
        for _ in 0..BLOOM_LEVELS {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            bloom_views.push(tex("bloom", w, h).create_view(&Default::default()));
            bloom_sizes.push((w, h));
        }
        let bloom_bg = |pipeline: &wgpu::RenderPipeline, src: &wgpu::TextureView, params: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                ],
            })
        };
        let mut down_bgs = Vec::new();
        let mut up_bgs = Vec::new();
        for i in 0..BLOOM_LEVELS as usize {
            let (src, size) = if i == 0 { (&hdr_view, (width, height)) } else { (&bloom_views[i - 1], bloom_sizes[i - 1]) };
            let p = &self.bloom_params[i];
            queue.write_buffer(
                p,
                0,
                bytemuck::bytes_of(&BloomParams {
                    texel: [1.0 / size.0 as f32, 1.0 / size.1 as f32],
                    first: (i == 0) as u32 as f32,
                    _p: 0.0,
                }),
            );
            down_bgs.push(bloom_bg(&self.down_pipeline, src, p));
        }
        for i in 0..BLOOM_LEVELS as usize - 1 {
            // upsample from level i+1 into level i
            let p = &self.bloom_params[BLOOM_LEVELS as usize + i];
            let size = bloom_sizes[i + 1];
            queue.write_buffer(
                p,
                0,
                bytemuck::bytes_of(&BloomParams { texel: [1.0 / size.0 as f32, 1.0 / size.1 as f32], first: 0.0, _p: 0.0 }),
            );
            up_bgs.push(bloom_bg(&self.up_pipeline, &bloom_views[i + 1], p));
        }
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite"),
            layout: &self.composite_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&bloom_views[0]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.composite_params.as_entire_binding() },
            ],
        });
        self.targets = Some(Targets { width, height, hdr_view, bloom_views, bloom_sizes, down_bgs, up_bgs, composite_bg });
    }

    fn ensure_volume(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, sim: &Simulation) {
        if self.volume.as_ref().is_some_and(|v| v.n == sim.n && v.sim_id == sim.layout_id()) {
            return;
        }
        let n = sim.n;
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume"),
            size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let nb = (n / 8).max(1);
        let bricks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bricks"),
            size: (nb as u64).pow(3) * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let ch = &sim.channels;
        let prep_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[n, ch.len() as u32, 1, 0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume prep"),
            layout: &self.prep_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ch[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ch.get(1).unwrap_or(&ch[0]).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ch.get(2).unwrap_or(&ch[0]).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sim.wall.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 5, resource: prep_params.as_entire_binding() },
            ],
        });
        let brick_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[n, nb, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let brick_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bricks"),
            layout: &self.brick_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: bricks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: brick_params.as_entire_binding() },
            ],
        });
        let raymarch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raymarch"),
            layout: &self.raymarch_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: bricks.as_entire_binding() },
            ],
        });
        let _ = queue;
        self.volume = Some(VolumeBinding {
            n,
            nch: ch.len() as u32,
            sim_id: sim.layout_id(),
            view,
            bricks,
            prep_params,
            prep_bg,
            brick_bg,
            raymarch_bg,
        });
    }

    /// Copy the simulation state into the render volume and rebuild the brick map.
    pub fn update_volume(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sim: &Simulation,
        smooth: bool,
    ) {
        self.ensure_volume(device, queue, sim);
        let v = self.volume.as_ref().unwrap();
        queue.write_buffer(&v.prep_params, 8, bytemuck::bytes_of(&(smooth as u32)));
        let n = v.n;
        let nb = (n / 8).max(1);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("volume"), timestamp_writes: None });
        pass.set_pipeline(&self.prep_pipeline);
        pass.set_bind_group(0, &v.prep_bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(4), n.div_ceil(4), n.div_ceil(4));
        pass.set_pipeline(&self.brick_pipeline);
        pass.set_bind_group(0, &v.brick_bg, &[]);
        pass.dispatch_workgroups(nb.div_ceil(4), nb.div_ceil(4), nb.div_ceil(4));
        let _ = &v.view;
        let _ = &v.bricks;
    }

    /// Render the scene into `target` (a view of a texture in `output_format`).
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: (u32, u32),
        camera: &OrbitCamera,
        settings: &RenderSettings,
        overlay: &Overlay,
        time: f32,
    ) {
        self.ensure_targets(device, queue, size.0, size.1);
        let Some(vol) = self.volume.as_ref() else { return };
        let t = self.targets.as_ref().unwrap();
        let aspect = t.width as f32 / t.height as f32;
        let vp: Mat4 = camera.view_proj(aspect);
        let eye = camera.eye();
        let l = settings.light_dir();
        let clip_world = settings.clip_pos.clamp(-1.0, 1.0);
        let mut flags = 0u32;
        if settings.floor {
            flags |= 1;
        }
        if settings.show_box {
            flags |= 2;
        }
        if settings.colormap {
            flags |= 4;
        }
        flags |= (settings.clip_axis as u32) << 3;
        if settings.clip_flip {
            flags |= 32;
        }
        if settings.show_walls {
            flags |= 64;
        }
        if overlay.plane_y.is_some() {
            flags |= 128;
        }
        let pixel_angle = 2.0 * (camera.fov_y * 0.5).tan() / t.height as f32;
        let brush = overlay.brush.map(|(c, r)| [c.x, c.y, c.z, r]).unwrap_or([0.0, 0.0, 0.0, -1.0]);
        let u = Uniforms {
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, time],
            light_dir: [l.x, l.y, l.z, settings.light_intensity],
            col0: ext3(settings.channel_colors[0]),
            col1: ext3(settings.channel_colors[1]),
            col2: ext3(settings.channel_colors[2]),
            core_col: ext3(settings.core_color),
            bg_top: ext3(settings.bg_top),
            bg_bot: ext3(settings.bg_bottom),
            brush,
            p0: [settings.iso, settings.core_iso, settings.membrane_opacity, settings.density_gain],
            p1: [settings.emission, settings.step_scale.clamp(0.2, 2.0), overlay.plane_y.unwrap_or(0.0), clip_world],
            p2: [settings.sss, settings.shadow, settings.ao, -1.02],
            ip: [settings.mode as u32, vol.n, vol.nch, flags],
            screen: [t.width as f32, t.height as f32, (self.frame % 64) as f32, pixel_angle],
            p3: [overlay.floor_offset[0], overlay.floor_offset[1], 0.0, 0.0],
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        queue.write_buffer(
            &self.composite_params,
            0,
            bytemuck::bytes_of(&CompositeParams {
                exposure: settings.exposure,
                bloom: settings.bloom,
                encode_srgb: (!self.output_format.is_srgb()) as u32 as f32,
                vignette: settings.vignette,
            }),
        );
        self.frame = self.frame.wrapping_add(1);

        fn color_pass<'e>(
            encoder: &'e mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            load: wgpu::LoadOp<wgpu::Color>,
        ) -> wgpu::RenderPass<'e> {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
        }
        {
            let mut pass = color_pass(encoder, &t.hdr_view, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            pass.set_pipeline(&self.raymarch_pipeline);
            pass.set_bind_group(0, &vol.raymarch_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        for i in 0..BLOOM_LEVELS as usize {
            let mut pass = color_pass(encoder, &t.bloom_views[i], wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            pass.set_pipeline(&self.down_pipeline);
            pass.set_bind_group(0, &t.down_bgs[i], &[]);
            pass.draw(0..3, 0..1);
        }
        for i in (0..BLOOM_LEVELS as usize - 1).rev() {
            let mut pass = color_pass(encoder, &t.bloom_views[i], wgpu::LoadOp::Load);
            pass.set_pipeline(&self.up_pipeline);
            pass.set_bind_group(0, &t.up_bgs[i], &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = color_pass(encoder, target, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &t.composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        let _ = &t.bloom_sizes;
    }
}
