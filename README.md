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
* **Evolution mode** (3D Flow Lenia): every cell carries a genome. Creatures
  grow eyes, mouths, claws and antennae, sing, eat, bud, mate, fight and
  mutate. You can hear them.

## Run it

**Windows:**

1. Download `lenia3d.exe` from the **latest** pre-release on the
   [Releases page](../../releases). CI rebuilds it on every push.
   Each green **build** run under [Actions](../../actions) also has a `lenia3d-windows`
   artifact.
2. Run `lenia3d.exe`.

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
| 8 9 0 | Evolution mode: food, toxin, founder brushes |
| Click a creature | Evolution mode: inspect its genes and organs; follow, clone or remove it |
| P / V | Evolution mode: population window / sound on-off |

Drop a `.l3d` world file or a preset / `animals3D.json` file onto the window
to open it.

## Command line

```
lenia3d --bench                         # simulation speed at 64³..512³
lenia3d --size 256 --preset Comet       # start with a given world size / preset
lenia3d --screenshot out.png --preset Comet --steps 300 --mode glow
lenia3d --list-presets
lenia3d --check --chan-file animals3D.json   # verify every preset/species stays alive
```

When you launch the app it starts with the *Tadpole Sea* soup at 128³.

* Load a creature: pick one in **Creatures** and press **Load into world**, or double-click
  its name.
* Explore new rules: use **Mutate** / **Random rule**, then press **Soup**.

Bert Chan's species download with one click. 33 of his 36 species are stable in this
engine. His sheet/rod "membrane" species span the periodic world, so they load into an
exact 64³ world automatically.

## Evolution mode

Switch to **Evolution** at the top of the left panel. It starts with the *Grazing Drift*
ecosystem. The physics is mass-conserving Flow Lenia, and every cell carries a
16-gene genome. The genes cover the update rule weights, the organs (eye, ear, voice,
mouth, claw), voice pitch and timbre, an alarm call, and color.

* **Senses.** Eyes steer towards food. Ears steer towards or away from voices in the
  pitch band they are tuned to. Claws attack foreign matter.
* **Ecology.** Mouths eat the green food field. Bodies pay metabolism, and lost matter
  becomes food again.
* **Reproduction.** Well-fed creatures bud mutated children. Compatible creatures that
  touch mate and produce a crossover child. Incompatible ones fight: foreign matter
  that flows into a body is destroyed by its claws.
* **Sound.** Each creature sings on a pentatonic scale with its own pitch and timbre.
  Sound is panned by camera position, and you also hear births, buds and deaths.
* **Building.** Build walls with the wall brush, or use **Islands** to split the world.
  Isolated populations drift apart into different "languages" (pitch groups).

See `docs/flow_lenia.md` for the math.

## How it works

Each simulation step does the following:

1. Packs channel pairs into complex fields and FFTs them (radix-2 Cooley-Tukey
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
