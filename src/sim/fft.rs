//! GPU 3D FFT over power-of-two cubes of complex `vec2<f32>` values.

use wgpu::util::DeviceExt;

const WORKGROUP_ELEMS: u32 = 2048; // 16 KiB of vec2<f32> workgroup memory
const WORKGROUP_THREADS: u32 = 256;

pub struct Fft {
    pub n: u32,
    lines: u32,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    twiddle: wgpu::Buffer,
    /// Uniforms for (axis, inverse) at index `axis * 2 + inverse`.
    params: Vec<wgpu::Buffer>,
}

/// Bind groups for one buffer, indexed by `axis * 2 + inverse`.
pub struct FftBindings(Vec<wgpu::BindGroup>);

impl Fft {
    pub fn new(device: &wgpu::Device, n: u32) -> Self {
        assert!(n.is_power_of_two() && n >= 4, "FFT size must be a power of two");
        let lines = (WORKGROUP_ELEMS / n).clamp(1, n);
        let src = include_str!("../shaders/fft.wgsl")
            .replace("{{N}}", &n.to_string())
            .replace("{{LOG2N}}", &n.trailing_zeros().to_string())
            .replace("{{LINES}}", &lines.to_string())
            .replace("{{WG}}", &WORKGROUP_THREADS.to_string());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let tw: Vec<[f32; 2]> = (0..n / 2)
            .map(|k| {
                let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [a.cos() as f32, a.sin() as f32]
            })
            .collect();
        let twiddle = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fft twiddle"),
            contents: bytemuck::cast_slice(&tw),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = (0..6u32)
            .map(|i| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("fft params"),
                    contents: bytemuck::cast_slice(&[i / 2, i % 2, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();
        Self { n, lines, pipeline, layout, twiddle, params }
    }

    pub fn bindings(&self, device: &wgpu::Device, data: &wgpu::Buffer) -> FftBindings {
        FftBindings(
            self.params
                .iter()
                .map(|p| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("fft bg"),
                        layout: &self.layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: self.twiddle.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: p.as_entire_binding() },
                        ],
                    })
                })
                .collect(),
        )
    }

    /// Unnormalized transform: the inverse is scaled by N^3 relative to the true inverse.
    pub fn run(&self, pass: &mut wgpu::ComputePass<'_>, b: &FftBindings, inverse: bool) {
        pass.set_pipeline(&self.pipeline);
        for axis in 0..3 {
            pass.set_bind_group(0, &b.0[axis * 2 + inverse as usize], &[]);
            pass.dispatch_workgroups(self.n / self.lines, self.n, 1);
        }
    }
}
