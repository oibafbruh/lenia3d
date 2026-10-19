# Flow-mode presets

These presets are for Lenia3D's *Flow* mode: 3D Flow Lenia with per-cell genomes, organs and an
ecology. The math is in `docs/flow_lenia.md`. All of them share one original rule, called
"Drifter" here. It was found by random search plus a hill-climb on glider speed using the NumPy
reference `tools/flow_ref.py`. The search tools are `tools/flow_search.py` and
`tools/flow_ref.py`.

**Drifter rule:** 1 channel, 8 kernels (mixed `kn` 1/2, all `gn = 2` Gaussian growth), R = 10,
T = 5 and theta = 2.84. One isolated body of about 600 mass is a slow glider: it moves at about
0.02 cells/step on its own in an empty 64³ world, with a radius of gyration of about 5 cells.
Bodies attract each other over a distance of about 1–2 kernel radii and merge into large,
**static** blobs. Every working preset therefore gives the founders a loud `voice` and a
negative `ear`, so they avoid each other's sound. Kernels 6 and 7 use the rule's global `h`,
following the revised gene layout (genes 0..5 = `h_0..h_5`).

All numbers below come from `tools/flow_search.py eco <preset> --steps S --seeds K` on 64³.
Creature counts are tracked components with mass ≥ 10, using the spec tracker: 32³ blocks,
threshold 0.03, sampled every 10 steps. "Late" means the last two thirds of the run. Speed is
the median net displacement of tracked ids over 100-step windows.

## Presets

| File | Name | Ecology | Seeding (64³) | Measured |
|---|---|---|---|---|
| `primordial_flow.json` | Primordial Flow | off | 10 blobs, radius 1.25 R, density 0.8; ear −1…−0.6, voice 0.6…1, all other organs 0, 3 languages | Seed 5: 7 creatures at every sample from step 10 to 1200 (mass about 630 each), median speed 0.0165 cells/step. The same rule with 8 blobs, seeds 3 and 4: 5–6 and 6 creatures (6 at every sample from step 0 to 1500), speed 0.020 and 0.019. Total mass is conserved (5018 ± 1). |
| `grazing_drift.json` | Grazing Drift | on, capacity 6000 | 8 blobs, radius 1.25 R, density 0.8; eye 0.2…0.6, ear −1…−0.4, voice 0.4…1, mouth 0.4…0.8, claw 0…0.2, alarm 0…0.2 | One 2500-step run in the Lenia3D engine (GPU, lavapipe), with `regrow 0.02`, `capacity 6000` and `combat 1`. Creature count rose from 6 to 16; there were 10 births; mass went 4944 → 6830 (step 800) → 3014 (step 2500) and had flattened out at the end (3036 → 3014 over the last 100 steps). The crowding factor fell from 2.3 to 1.25. Variants with `regrow` 0.03 or 0.04 (capacity 5000–6000) also kept 17–18 creatures, but their mass was still falling at step 2500. The old `regrow 0.01` setting without a capacity starved: mass 6686 → 1794 by step 800. Keep-alive re-seeds founders if everything dies. |
| `claw_hunters.json` | Claw Hunters | on, capacity 6000 | As Grazing Drift but claw 0.2…0.7, `claw_bias` 8 | **Experimental.** Shares Grazing Drift's ecology. With `combat = 1` and claws 0.2–0.7, total mass fell from 4944 to 443 within 100 steps (see caveat below). Shipped with `combat = 0.25`. Keep-alive re-seeds founders if the population dies out. |

Ecology parameters shared by both ecosystem presets: `eat 1.0, eat_min 0.3, social 0.5,
meta_base 0.002, meta_trait 0.0005, toxicity 2, recycle 0.3, regrow 0.02, fertility 0.3,
mutation_rate 0, mutation_size 0.05, bud_mass 900, bud_age 300, bud_mutation 0.05,
capacity 6000, keep_alive true`, with
`w_eye = w_ear = 2`. Primordial Flow carries the same block with `enabled: false`.

How the ecology tuning went, all on 64³ with the Drifter rule and 8 founders:

| fertility | regrow | meta_base / meta_trait | result |
|---|---|---|---|
| 0.3 | 0.05 | 0.01 / 0.01, `eat_min` 0 (spec as first written) | dust explosion: mass 3.4k → 237k in 800 steps, 1 giant component |
| 0.5 | 0.002–0.004 | 0.004–0.007 / 0.001 | boom then bust: mass ×6 (≈30k, one giant blob) by step 300, then a crash; n = 0–3 at step 2000 |
| 0.3 | 0.03 | 0.01 / 0.005 | starvation: mass 4944 → 92 by step 1200 |
| 0.15 | 0.03 | 0.003 / 0.001 | starvation: mass 4944 → 214, extinct by step 1400 |
| 0.3 | 0.01 | 0.002 / 0.0005 | previously shipped: 5–10 creatures and 4 buds in 800 steps, mass declining; on a fast GPU the population vanished within seconds |
| 0.3 | 0.02 + capacity 6000 | 0.002 / 0.0005 | **shipped**: 6 → 16 creatures in 2500 steps, mass 4944 → 6830 → 3014, flattening out at the end |

Trait signal: in the runs with `eat_min = 0` (dust regime), mean `mouth` rose from 0.59 to
0.85 in 1500 steps. With `eat_min` there was no clear trend within 800–2000 steps.


## Caveats and findings

* **Random 1-channel kernels in 3D almost never glide.** 18 random 64³ soups and 610
  single-blob screens on 32³ (3–8 kernels, kn 1/2, gn 2, R 8–12, theta 1–3) all settled into
  static blobs. Only about 4 % moved faster than 0.002 cells/step, and only one moved steadily
  (0.007). A hill-climb on that rule (312 evaluations) raised it to 0.02–0.028 cells/step for a lone body. In a
  soup with no sound organs, bodies still merge into 2–5 static blobs. Sound repulsion
  (`voice` > 0 with `ear` < 0, `w_ear = 2`) keeps them apart, so they keep gliding.
* **Dust problem (now fixed in the spec with `eat_min`).** With `sat = min(A_Σ, 1)`, any thin
  mass eats at the same per-mass rate as a dense body, or a faster one. Reintegration always
  leaks a faint halo, and the affinity is flat far from bodies. With the suggested starting
  values, total mass grew from 3.4k to 237k in 800 steps, ending as one giant component. With
  stronger metabolism the dense creatures starve while the dust survives. `eat_min = 0.3`
  stops this: dust cannot eat, so metabolism removes it.
* **Eye steering is weak.** `Kfar` has σ = far·R/3 ≈ 8 cells, so the depletion hole of a
  single creature changes `∇Sf / Sf` by less than 1 % per cell. Food seeking matters only
  where large areas are grazed.
* **Combat together with claw-biased inheritance can wipe out mixed bodies.** Reintegration
  swaps about 54 % of a cell's mass with its neighbours every step, even without net flow
  (each box overlaps every neighbour by 0.15 per axis). Claw-biased inheritance plants
  foreign genomes inside bodies cell by cell. In such a mixed region every cell destroys a
  large share of its incoming flux at the rate `combat·claw·dt`, so the body burns down within
  about 100 steps. Possible fixes: kill only the *net* inflow, or scale `kill` by the share of
  foreign mass among the neighbours. A dominant-genome (argmax) inheritance would also avoid
  the cell-by-cell mixing.
* **Seeding change in `flow_ref.seed_flow`:** a founder's genome is written only where its
  value sets the maximum (`v > 0` and `v >= previous`). The original rule ("v > 0, later blobs
  overwrite") mixes genomes cell by cell where founders overlap.
* **Ecology timescales are slow** because creatures move at about 0.02 cells/step. Grazing
  (sweeping fresh food) has to beat sitting still, so `regrow` is kept small. Otherwise a
  static body can live on the food that regrows under it, and the world fills up.
