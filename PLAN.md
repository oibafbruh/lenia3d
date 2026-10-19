# Lenia3D — Plan

A real-time, GPU-native 3D artificial-life sandbox based on Lenia (Chan, arXiv:1812.05433),
extended with Flow Lenia for multi-species evolution, an editable environment, and a
volumetric renderer. Target machine: RTX 3090 (24 GB), 64 GB RAM.

## 1. Technology: our own engine in Rust + wgpu (Vulkan / DX12) + egui

**Why.** The cost of Lenia is convolution with large kernels (radius 10–30 cells in 3D).
The fast way to do that is FFT on the GPU, and the renderer has to raymarch the same GPU
volume without copying it back to the CPU. That is a custom compute pipeline in any engine,
so a game engine adds weight without helping with the hard part.

- wgpu runs natively on Vulkan or DX12, so the 3090 is fully available for compute.
- One self-contained binary, with no CUDA toolkit, Visual Studio, or Unity install. GitHub
  Actions builds a Windows `.exe`, or you can run `cargo run --release`.
- I can compile and test it in my environment: the GPU shaders run on software Vulkan
  (lavapipe) and are compared against a NumPy reference.

**Rejected alternatives**

| Option | Why not |
|---|---|
| Unity / Unreal | FFT and volume raymarching would still be hand-written compute shaders. Heavy install, and I can't build or test them here. |
| C++ / CUDA + cuFFT | About 2× faster FFT, but NVIDIA-only and a painful Windows build. Kept as an optional later backend if FFT becomes the bottleneck. |
| Browser WebGPU | No install, but buffer-size limits and less headroom at 256³ and above. |
| Python (JAX / PyTorch) | Great for research, weak for an interactive 3D tool. |

## 2. Simulation core (three layers, each building on the last)

1. **Classic Lenia (the paper).** Multi-channel and multi-kernel. Each kernel is a set of
   concentric shells (β peaks) using the exponential core function, and growth is a Gaussian
   bump: `A ← clip(A + dt · Σ h·G(K * A), 0, 1)`. It imports Bert Chan's 36 published 3D
   species (`animals3D.json`) as a correctness check. His repo has no license file, so the
   app downloads or imports that file instead of shipping a copy.
2. **Flow Lenia (Plantec et al., 2022).** Mass-conserving: growth becomes a flow field, and
   mass is moved by reintegration tracking. Parameters are stored per cell and move with the
   mass ("parameter localization"), so each creature carries its own genome (per-kernel
   μ, σ, h) inside the grid. This lets many species live and evolve in one world.
3. **Ecology layer**
   - A nutrient field that regrows. Creatures turn it into mass where they overlap, and
     metabolism slowly turns mass back into nutrients. Total matter is conserved, so
     creatures really compete.
   - Mutation when mass is created, and mass-weighted genome mixing when creatures collide.
   - Environment fields: walls (block flow), nutrient sources, toxic zones, currents, and
     "temperature" zones that shift growth parameters.

**Performance design**

- A custom FFT written as WGSL compute shaders. It is shared-memory Stockham, one workgroup
  per line, with coalesced tiles for the y and z axes.
- Two real fields are packed into one complex FFT, both forward and inverse. Lenia kernels
  are radially symmetric, so their spectra are real, and packing halves the number of FFTs.
- Kernel spectra are precomputed and rebuilt only when a kernel changes.
- All data stays on the GPU. The CPU only reads back small statistics, asynchronously.
- The number of simulation steps per frame is independent of the render frame rate.
- Estimated performance on the 3090 (to be confirmed with a benchmark):

  | Grid | Mode | Estimate |
  |---|---|---|
  | 128³ | any | several hundred steps/s |
  | 256³ | ~10 kernels | 30–60 steps/s |
  | 512³ | Classic | feasible |
  | 384³ | Flow / evolution | experimental |

## 3. Rendering (custom volumetric raymarcher)

- Empty-space skipping uses a brick occupancy map (8³ bricks). Creatures are sparse, so most
  rays skip almost everything.
- Render modes:
  - **Organic:** isosurface with gradient normals, soft shadows, ambient occlusion, and
    subsurface glow, so creatures look like living jelly.
  - **Bioluminescent:** emissive volume with bloom.
  - **X-ray:** max-intensity projection.
  - **Slices:** 2D cross-section planes like the figures in the paper, plus a clip box for
    cutting into the volume.
- Coloring is by channel or by genome. Species get stable hues automatically.
- Walls render as solid voxels and nutrients as a faint mist, with a ground grid and fog.
- HDR, bloom, ACES tonemapping, and blue-noise jitter with temporal accumulation (no banding).
- Camera controls:

  | Input | Action |
  |---|---|
  | Left-drag | Orbit |
  | Right/middle-drag | Pan |
  | Wheel | Zoom toward the cursor |
  | WASD | Fly mode |
  | F | Focus the selection |
  | Double-click | Recenter |

## 4. UI / UX (egui, dockable panels)

- **Top bar:** play/pause/step, speed, world size, reset, FPS, and per-stage timings.
- **Species library:** thumbnails, save/load presets (JSON), import Chan's species.
- **Kernel and growth editor:** live plots of the kernel profile and growth curve, as in
  Chan's tool. Sliders for R, β peaks, μ, σ, h, and dt, plus a channel routing matrix.
  Changes apply live.
- **Tools** (number keys):
  - Inspect.
  - Paint a creature (stamp a species or a random blob).
  - Paint walls or nutrients, and erase.
  - Shapes: box, sphere, tube, maze, noise terrain.

  Brushes work on a movable work plane or on surfaces, and Ctrl+wheel sets brush size.
  Ctrl+Z undoes edits using GPU snapshots.
- **Inspector:** click a creature to see its mass, speed, age, and genome. From there you can
  follow it with the camera, clone, mutate, delete, save it as a preset, or breed two selected
  creatures.
- **Evolution panel:** mutation, regrowth, metabolism, and mixing settings. Live graphs of
  population per species, diversity, and total mass, plus a lineage tree.
- Save/load the full world (zstd-compressed), screenshots, and PNG-sequence recording.

## 5. Creature tracking and discovery

- **Tracking:** GPU connected-component labeling runs every N steps and produces a creature
  list with mass, center, velocity, bounds, and mean genome. Genomes are clustered into
  species. This list feeds the inspector, graphs, and follow-cam.
- **Lab mode (automated search):** many small worlds (for example 64 × 64³) run in parallel
  on the GPU. Candidates are scored on whether they survive, move, stay compact, and show
  complexity. A MAP-Elites archive keeps a range of behaviors, and the best finds go to the
  species library. This is the most reliable way to find complex creatures.

## 6. Code layout

```
crates/core     params, genome, presets, serialization
crates/gpu      FFT, classic step, flow step, ecology, labeling, stats (WGSL)
crates/render   raymarcher, post-processing, camera
crates/app      winit + egui application, tools, panels
tools/reference.py   NumPy reference implementation used by tests
```

## 7. Verification

- NumPy reference implementations of Classic and Flow Lenia. GPU results are compared with
  them on 32³ and 64³ grids, using software Vulkan here and in CI.
- FFT checked against NumPy FFT. Flow Lenia checked for mass conservation.
- Chan's species must survive and keep their shape over N steps.
- Windows and Linux release builds via GitHub Actions.
- **Not verifiable here:** real-GPU speed and visual quality on your screen. A built-in
  benchmark (`--bench`) prints per-stage timings that you can send back.

## 8. Status (M1 delivered)

- Done: classic multi-channel/multi-kernel 3D Lenia on a custom GPU FFT, verified against
  rustfft and a direct CPU convolution (Vulkan/lavapipe and DX12/WARP in CI).
- Done: renderer modes (Organic skin+core, Glow, X-ray), cross-section cuts, shadows,
  bloom, follow mode with a scrolling floor.
- Done: live rule editor with plots, Mutate / Random rule buttons, brushes, walls, save/load,
  and screenshots.
- Built-in presets: 7 original presets (`presets/`), all stable in the GPU check.
- Bert Chan's species: downloaded on demand. 33 of his 36 species are stable in our
  engine; sheets and rods load into their exact 64³ world.
- Not yet: everything in M2 to M4 below.

## 9. Milestones

| Milestone | Contents |
|---|---|
| **M1** — target for tonight | App and camera, classic multi-channel 3D Lenia on the GPU FFT, Organic and Bioluminescent render modes plus slices, kernel/growth editor, paint/erase/wall tools, Chan species import, save/load, Windows `.exe` |
| **M2** | Flow Lenia, localized genomes, multi-species, mutation, nutrients and metabolism, environment builder |
| **M3** | Creature tracking, inspector, follow-cam, breeding, population graphs, lineage |
| **M4** | Lab mode, render polish (shadows, subsurface scattering, temporal AA), recording, 512³ tuning |
