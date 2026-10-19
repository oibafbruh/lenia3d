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

PRESET_TABLE

## Caveats and findings

* **Random 1-channel kernels in 3D almost never glide.** About 85 random 64³ soups and 610
  single-blob screens on 32³ (3–8 kernels, kn 1/2, gn 2, R 8–12, theta 1–3) all settled into
  static blobs. Only about 4 % moved faster than 0.002 cells/step, and only one moved steadily
  (0.007). A hill-climb on that rule raised it to 0.02–0.028 cells/step for a lone body. In a
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
* **Ecology timescales are slow** because creatures move at about 0.02 cells/step. Grazing
  (sweeping fresh food) has to beat sitting still, so `regrow` is kept small. Otherwise a
  static body can live on the food that regrows under it, and the world fills up.
