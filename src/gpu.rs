//! GPU device creation shared by the app, headless mode and tests.

use anyhow::{anyhow, Result};

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

pub fn create_instance() -> wgpu::Instance {
    // WGPU_BACKEND (e.g. "vulkan", "dx12") overrides the default choice.
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    if std::env::var_os("WGPU_BACKEND").is_none() {
        desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
    }
    wgpu::Instance::new(desc)
}

fn adapter_score(info: &wgpu::AdapterInfo) -> i32 {
    let ty = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 1000,
        wgpu::DeviceType::IntegratedGpu => 500,
        wgpu::DeviceType::VirtualGpu => 200,
        wgpu::DeviceType::Other => 100,
        wgpu::DeviceType::Cpu => 50,
    };
    let backend = match info.backend {
        wgpu::Backend::Vulkan => 20,
        wgpu::Backend::Dx12 => 10,
        wgpu::Backend::Metal => 20,
        _ => 0,
    };
    ty + backend
}

/// Pick the best adapter: discrete GPU first, Vulkan preferred over DX12.
pub async fn pick_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<wgpu::Adapter> {
    let mut adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    adapters.retain(|a| surface.map(|s| a.is_surface_supported(s)).unwrap_or(true));
    for a in &adapters {
        let i = a.get_info();
        log::info!("adapter candidate: {} ({:?}, {:?})", i.name, i.backend, i.device_type);
    }
    if let Some(name) = std::env::var("LENIA_ADAPTER").ok().map(|s| s.to_lowercase()) {
        if let Some(pos) = adapters.iter().position(|a| a.get_info().name.to_lowercase().contains(&name)) {
            return Ok(adapters.swap_remove(pos));
        }
    }
    adapters.sort_by_key(|a| -adapter_score(&a.get_info()));
    if let Some(a) = adapters.into_iter().next() {
        return Ok(a);
    }
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| anyhow!("no suitable GPU adapter found: {e}"))
}

pub async fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let al = adapter.limits();
    let mut limits = wgpu::Limits::default();
    // Large grids need big storage buffers; take whatever the adapter offers.
    limits.max_buffer_size = al.max_buffer_size;
    limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
    limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage.min(16);
    limits.max_texture_dimension_3d = al.max_texture_dimension_3d;
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("lenia3d"),
            required_features: features,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await?;
    Ok((device, queue))
}

impl Gpu {
    /// Create a device without any surface (tests, benchmarks, headless rendering).
    pub fn headless() -> Result<Self> {
        pollster::block_on(async {
            let instance = create_instance();
            let adapter = pick_adapter(&instance, None).await?;
            let info = adapter.get_info();
            log::info!("using adapter {} ({:?})", info.name, info.backend);
            let (device, queue) = request_device(&adapter).await?;
            Ok(Self { instance, adapter, device, queue, info })
        })
    }

    /// Largest power-of-two grid size whose complex FFT buffers fit the device limits.
    pub fn max_grid_size(device: &wgpu::Device) -> u32 {
        let l = device.limits();
        let max_bind = (l.max_storage_buffer_binding_size as u64).min(l.max_buffer_size);
        let mut n = 32u32;
        while n < 512 && (2 * n as u64).pow(3) * 8 <= max_bind {
            n *= 2;
        }
        n
    }
}

/// Blocking readback of a buffer (used by tests, save, and headless mode).
pub fn read_buffer_f32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer) -> Vec<f32> {
    read_buffer_bytes(device, queue, buf, 0, buf.size()).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn read_buffer_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, offset, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("buffer mapping failed").to_vec();
    staging.unmap();
    data
}

/// Dispatch helper for 1D work spread over a 2D grid of 256-thread workgroups
/// (keeps every dimension under the 65535 workgroup limit).
pub fn linear_groups(total: u32) -> (u32, u32) {
    let groups = total.div_ceil(256);
    let x = groups.min(32768);
    (x, groups.div_ceil(x))
}
