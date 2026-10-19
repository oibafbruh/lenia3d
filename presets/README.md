# Built-in presets

These are original 3D Lenia rules and creatures. They were found by random search plus local
tuning with the NumPy reference implementation (`tools/lenia_ref.py`). None of them are
copied from Bert Chan's species: every rule and every cell pattern comes from random noise.
Chan's `animals3D.json` was used only to check that the reference implementation keeps his
species alive. It is not shipped and no data was taken from it.

All runs use `T = 10` (dt = 0.1) and a periodic 64³ world, the smallest size each preset was
verified on. Speeds are in cells per step.

## Creatures

| File | Name | Ch | Kernels | R | Behaviour | Mass mean [min, max] | Radius of gyration | Speed | Steps verified |
|---|---|---|---|---|---|---|---|---|---|
| `comet.json` | Comet | 1 | 3 (kn2/gn2) | 13 | Lopsided three-lobed glider that moves in a straight line | 1439 [1426, 1451] | 7.65 | **0.116** (174 cells in 1500 steps) | 1500 |
| `glass_bubble.json` | Glass Bubble | 1 | 1 (kn1/gn1, 3 shells) | 10 | Motionless hollow spherical shell | 782 [770, 809] | 6.49 | 0.0001 | 1500 |
| `ember.json` | Ember | 1 | 3 (kn2/gn2) | 10 | Small speckled shell that keeps flickering and pulsing (mass ±13 %) | 198 [160, 252] | 3.74 | 0.0004 | 1500 |
| `twin_onion.json` | Twin Onion | 2 | 4 (0→0, 1→1, 0→1, 1→0) | 13 | Static nested sphere: a thin ch0 skin (radius about 7.5) around two concentric ch1 shells (radii about 2 and 6) | 2043 [2025, 2055] (ch0 761, ch1 1282) | 7.20 | 0.0001 | 1500 |
| `pilot_fish.json` | Pilot Fish | 2 | 5 (Comet's 3 + 0→1 + 1→0) | 13 | Two-channel glider: a Comet-like ch0 body inside a larger ch1 sheath that trails slightly behind | 3047 [2944, 3116] (ch0 1297, ch1 1750) | 7.23 | **0.066** (0.08 at steps 300–400) | 1500 |

"Mass" is Σ A over all channels. "Radius of gyration" is measured around the periodic centre
of mass, in cells. "Speed" is the centre-of-mass velocity over the second half of the run.
All five passed these checks for the whole run: never dead, radius of gyration within ±35 %,
mass within [0.4×, 2×] of its mean, and one body (not split into separate pieces). Each preset
was checked starting from the stored `uint8` cells, pasted into the centre of an empty 64³
world.

## Soups

| File | Name | Ch | Kernels | R | Seeding (`init`) | Occupied cells (A>0.1) | Activity | Structures | Steps verified |
|---|---|---|---|---|---|---|---|---|---|
| `tadpole_sea.json` | Tadpole Sea | 2 | 3 (0→0, 0→1, 1→0) | 10 | 12 blobs, radius 1.21 R, density 0.78 | 15–23 % | 0.65–0.74 | 60–90, constantly changing | 1500–2000 × 4 seeds |
| `onion_reef.json` | Onion Reef | 2 | 4 (0→0, 1→1, 0→1, 1→0) | 10 | 10 blobs, radius 1.12 R, density 0.89 | 33–48 % | 0.09–0.12 | 40–85 | 1500–2000 × 4 seeds |

"Activity" is Σ|A(t) − A(t−20)| / Σ A(t), measured at the end of the run. A frozen world
scores about 0, and a fast glider scores about 0.5 per 10 steps. "Structures" is the number of
connected components with A > 0.15. Both soups stayed alive, never filled the world, and never
froze in any seed tried. Onion Reef uses the Twin Onion rule with every growth width `s` about
17 % wider.

## Soup seeding (`init` block)

`lenia_ref.seed_blobs(shape, channels, R, blobs, blob_radius, density, seed)`. All channels
start at 0. For each blob:

1. Pick a centre `c` uniformly at random in `[0, N)³` (continuous coordinates).
2. Set `rb = blob_radius * R`.
3. For every cell whose wrapped distance `d` to `c` is less than `rb` (each axis offset is
   wrapped to `[-N/2, N/2)`), compute `t = 1 - d/rb` and `w = t*t*(3 - 2t)` (smoothstep).
   Then, for each channel independently, set `v = U(0,1) * w` with probability `density`,
   otherwise `v = 0`, and set `A[ch][cell] = max(A[ch][cell], v)`.

Only this distribution matters, so any RNG works.

## How they were found (`tools/search_presets.py`)

1. **`search`**: random rules, 4 worker processes, 48³ worlds, 300 steps per run. Each rule
   writes its growth widths as `s_k = λ · m_k · q_k`. The search seeds one noisy blob of radius
   0.7–1.25 R (half of the blobs are lopsided) and bisects `λ` between "dies" and "fills the
   world". Every run in between that stays alive and localized is scored on survival,
   compactness, speed, activity and mass oscillation. Modes:
   - `single`: one kernel.
   - `multik`: one channel, 2–3 kernels.
   - `multich`: 2–3 channels with cross kernels.
   - `mover`: multi-shell kernels with low `m`.

   About 4,200 rules (about 29,600 runs) were screened, using the `single`, `multik` and
   `multich` modes. The `mover` mode exists but was not used for these presets. Comet came from `multik`, Twin Onion
   from `multich`, Glass Bubble from `single`, and Ember from `multik`.
2. **`graft`**: Pilot Fish. Starting from a settled Comet, the search adds channel 1 with a
   random 0→1 kernel (h = 1) and a weak random 1→0 feedback kernel. That kernel takes weight ε
   from channel 0's own kernels, so the h values still sum to 1. 14 of 24 random grafts stayed
   alive, compact, two-channel and moving after 500 steps. Of the 9 checked at length, 8
   passed 1500 steps. The fastest one exploded later.
3. **`search --mode soup`**: rules sampled fresh or nudged from creature candidates, seeded
   with `seed_blobs` on 64³ for 500 steps, about 490 rules. The best were re-run for
   1500–2000 steps with several seeds.
4. **`export`** (creatures):
   - Settle the candidate in its screening world with the same seed noise.
   - Optionally retune `s` (`--s-scale`; Glass Bubble uses ×0.97).
   - Optionally zoom to a new R and settle again (`--R`; Comet, Pilot Fish and Twin Onion went
     from R = 10 to R = 13).
   - Centre the creature, crop to its bounding box plus a 2-cell margin, and quantize to uint8.
   - Re-verify from exactly those bytes on 64³.

   **`export-soup`** re-runs the soup with new seeds.

`tools/check_presets.py` reloads every JSON exactly as the app would and re-runs it (default:
1000 steps). It exits non-zero on any failure.

## Notes and caveats

- **Keep each preset's R.** Lenia is only approximately scale-free. Comet, Pilot Fish and Twin
  Onion survived zooming from R = 10 to 13. Ember (R = 14) and Glass Bubble (R = 13) blew up
  when rescaled.
- **Comet and Pilot Fish must be started from their stored cells.** Random blobs under the
  Comet rule turn into comets only rarely. Larger seeded worlds under that rule tend to
  crystallize into a static foam of hollow bubbles, which is why there is no "comet swarm"
  soup.
- Soup outcomes depend on the random seed. Both soups held up on every seed tried (4 each),
  but they were checked only on 64³. A 128³ world should keep the same blob density, which
  means about 8× the blob count.
- Several attractive near-critical candidates were rejected because they exploded after
  300–700 steps. These included large pulsating "breathers" and a two-channel glider found
  directly by the `multich` search.
