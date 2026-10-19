# Lenia3D

Real-time **3D Lenia** artificial life on the GPU, written in Rust with wgpu
(Vulkan / DX12) and egui. It is based on Bert Chan's
[Lenia](https://arxiv.org/abs/1812.05433) and supports multiple channels and
multiple kernels.

* GPU FFT convolution (custom WGSL compute shaders): 64³ to 512³ worlds.
* Volumetric renderer with three modes:
  * *Organic*: translucent skin and core, soft shadows, translucency.
  * *Glow*: emissive volume with bloom.
  * *X-ray*: maximum intensity.
* Cross-section cuts through the volume.
* A live rule editor with kernel and growth plots, where every slider changes
  the physics immediately.
* Brushes to spray matter, paint, erase, build and remove walls, and stamp
  creatures.
* Built-in creatures, plus one-click download of Bert Chan's 36 published
  3D species.

## Run it

**Windows:**

1. Download the `lenia3d-windows` artifact from the latest
   [Actions run](../../actions).
2. Unzip it and run `lenia3d.exe`.

You need an up-to-date GPU driver, but no other installs.

**From source (Windows / Linux / macOS):**

1. Install Rust from https://rustup.rs.
2. Run:

   ```
   cargo run --release
   ```

The app picks the discrete GPU and prefers Vulkan. To force a backend, set the
`WGPU_BACKEND=dx12` or `WGPU_BACKEND=vulkan` environment variable.

## Controls

| Input | Action |
|---|---|
| Right-drag (or left-drag in View tool) | orbit |
| Middle-drag / Shift+drag | pan |
| Wheel | zoom toward cursor |
| W A S D, Q E | fly |
| F | frame the world |
| Space / . | play-pause / single step |
| R / C | reset / clear |
| 1-7 | tools: view, stamp creature, random blob, paint, erase, wall, remove wall |
| [ ] or Ctrl+wheel | brush size |
| PgUp/PgDn or Alt+wheel | work-plane height (brushes act on this plane) |
| M | cycle render mode |
| Tab | hide panels |
| F11 / F12 | fullscreen / screenshot (saved to `screenshots/`) |
| H | help |

Drop a `.l3d` world file or a preset / `animals3D.json` file onto the window
to open it.

## Command line

```
lenia3d --bench                         # simulation speed at 64³..512³
lenia3d --size 256 --preset Comet       # start with a given world size / preset
lenia3d --screenshot out.png --preset Comet --steps 300 --mode glow
lenia3d --list-presets
```

## How it works

Each simulation step does the following:

1. Packs channel pairs into complex fields and FFTs them (radix-2 Stockham-style
   passes in workgroup memory).
2. Multiplies by the precomputed kernel spectra. Two kernels share one complex
   inverse FFT, because Lenia kernels are radially symmetric and so have real
   spectra.
3. Applies the growth function with an Euler update.

```
U_k = K_k * A[src_k]
A_c <- clip(A_c + dt * sum_k h_k G_k(U_k), 0, 1)
```

For rendering, the state is copied (with optional 3×3×3 smoothing) into an
rgba16f 3D texture. An 8³ brick max-map lets the raymarcher skip empty space.

Tests in `tests/gpu_sim.rs` check the GPU FFT against rustfft. They also check
full multi-channel, multi-kernel Lenia steps against a direct CPU convolution.

See `PLAN.md` for the roadmap: Flow Lenia, evolution, ecology, creature
tracking, and automated discovery.

## Credits

Lenia was created by Bert Chan. His species are not bundled: his
[repository](https://github.com/Chakazul/Lenia) has no license file, so the
app downloads them on request.
