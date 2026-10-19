# Flow mode: Flow Lenia with genomes, organs and an ecology

This is the exact math of Lenia3D's *Flow* mode. The GPU implementation (`src/sim/flow.rs`,
`src/shaders/flow_*.wgsl`) and the NumPy reference (`tools/flow_ref.py`) both follow it.

Flow mode is based on Flow Lenia (Plantec et al. 2022,
"Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation
and parameter localization"). On top of it we add per-cell genomes with simple organs and
an ecology: food, metabolism, budding, and mutation.

## State

Grid: `N³`, periodic, indexed `[z][y][x]`, with cell centers at integer coordinates.

| Field | Meaning |
|---|---|
| `A_c(x) ≥ 0` | Mass of channel `c` (`C = 1..3` channels). It is **not** clipped to 1. `A_Σ = Σ_c A_c`. |
| `g(x) ∈ R^16` | Genome of the matter in cell `x` (layout below). |
| `Φ(x) ≥ 0` | Food. |
| `Ψ(x) ∈ [-1, 1]` | Fertility. `Ψ > 0` means food regrows towards `Ψ`; `Ψ < 0` is a toxic zone. |
| `W(x) ∈ {0, 1}` | Walls. |

### Genome layout

| Gene | Name | Range | Effect |
|---|---|---|---|
| 0..5 | `h_0..h_5` | [0, 1] | Weight of kernel k in this cell's affinity (body physics) |
| 6 | `tune` | [0, 1] | Which pitch the ears listen to |
| 7 | `alarm` | [0, 1] | Scream when claws are smelled nearby |
| 8 | `eye` | [0, 1] | Steer towards food (follows the gradient of sensed food) |
| 9 | `ear` | [-1, 1] | Steer towards (+) or away from (−) what the ears hear |
| 10 | `voice` | [0, 1] | Calls into the sound field |
| 11 | `mouth` | [0, 1] | Eats food |
| 12 | `claw` | [0, 1] | Fights foreign matter |
| 13 | `pitch` | [0, 1] | Voice frequency; also the creature's "language" |
| 14 | `hue` | [0, 1) | Color (cosmetic, inherited) |
| 15 | `timbre` | [0, 1] | Sound timbre (audio only) |

Kernels `k ≥ 6` use the rule's global `h_k`.

## Parameters

* **Rule:** `R`, `T` (`dt = 1/T`, typically `T = 5`, so `dt = 0.2`), and kernels
  `(src, dst, r, b, μ, σ, kn, gn)` exactly as in classic Lenia (see `src/sim/params.rs`). In
  Flow mode the kernel's `h` field is only the default for gene `h_k`.
* **Flow:**

  | Parameter | Meaning | Default |
  |---|---|---|
  | `theta` | critical mass | 2.0 |
  | `n` | crowding exponent | 2.0 |
  | `s` | reintegration half-size | 0.65 |
  | `far` | sensing radius in units of R | 2.5 |
  | `w_eye` | eye steering gain | 1.0 |
  | `w_ear` | ear steering gain | 1.0 |
  | `claw_bias` | claw advantage in contested cells | 4.0 |

  `dd = 1`: matter moves at most `dd - s = 0.35` cells per step along each axis.
* **Ecology:**

  | Parameter | Meaning |
  |---|---|
  | `eat` | eating rate |
  | `social` | eating boost from local sound |
  | `meta_base` | base metabolism |
  | `meta_trait` | metabolic cost of each organ |
  | `toxicity` | extra metabolism in toxic zones |
  | `recycle` | fraction of metabolized mass that turns back into food |
  | `regrow` | food regrowth rate |
  | `mutation_rate` | per-cell, per-step probability of a mutation |
  | `mutation_size` | size of a per-cell mutation |

  Budding (see below) uses `bud_mass`, `bud_age` and `bud_mutation`.

## One step

### 1. Potentials

`U_k = K_k * A_{src_k}` is a periodic convolution with the normalized Lenia kernel, the same
as in classic mode.

### 2. Affinity

`Aff_c(x) = Σ_{k : dst_k = c} h_k(x) · G_k(U_k(x))`, where `G_k` is the growth function
(`gn`) with the rule's global `μ_k, σ_k`, and `h_k(x)` is gene `k` of `g(x)` for `k < 8`.

### 3. Senses

Two complex fields carry four sensed quantities. Sound is split into a low and a high band by
the caller's pitch, and claws leave a scent that triggers alarm calls:

```
T_prev = claw scent sensed in the previous step (sense_b.y)
call   = min(voice + alarm · min(8·T_prev, 1) · (1 − claw), 1.5)
E      = call · A_Σ
sense_a = (Kfar * Φ,  Kfar * (E · (1 − pitch)))       = (Sf, S_lo)
sense_b = (Kfar * (E · pitch),  Kfar * (claw · A_Σ))  = (S_hi, T)
H      = (1 − tune) · S_lo + tune · S_hi              (what this cell hears)
```

`Kfar` is a Gaussian with `σ_far = far·R/3`, truncated at radius `far·R` (wrapped offsets),
and normalized to sum 1.

### 4. Flow

Gradients are periodic central differences: `∇f(x)_i = (f(x+e_i) − f(x−e_i)) / 2`. The
crowding weight is:

```
α(x) = clamp((A_Σ(x) / theta)^n, 0, 1)
```

The flow for channel `c` is:

```
F_c = (1 − α) ∇Aff_c − α ∇A_Σ
      + w_eye · eye(x) · clip1(∇Sf / (Sf + 0.05))
      + w_ear · ear(x) · clip1(∇H / (H + 0.05))
```

`clip1(v)` scales `v` down so that `|v| ≤ 1`. The displacement is:

```
D_c(x) = clamp(dt · F_c(x), −(1 − s), +(1 − s))
```

The clamp is applied per component. Walls: for each axis `i`, if `D_c(x)_i ≠ 0` and
`W(x + sign(D_c(x)_i) e_i) > 0.5`, then `D_c(x)_i = 0`.

### 5. Reintegration tracking

Reintegration is a gather over the 3×3×3 neighborhood. For destination `x` and each offset
`o ∈ {−1, 0, 1}³`, let `y = x + o` (wrapped). For each channel, the matter of `y` lands as a
box of half-size `s` centered at `p = o + D_c(y)`, relative to `x`. The destination cell is
`[−½, ½]³`.

```
ov_i     = max(0, min(p_i + s, ½) − max(p_i − s, −½))
w_c(y)   = ov_x · ov_y · ov_z / (2s)³
A'_c(x)  = Σ_y A_c(y) · w_c(y)
m(y)     = Σ_c A_c(y) · w_c(y)
```

This conserves mass exactly: each source's box always lies within the neighborhood because
`|p| ≤ 1 − s`.

**Compatibility gate.** This applies when the destination is occupied
(`Σ_c A_c(x) > 0.02`).

1. A source `y` is *foreign* if `dist(g(x), g(y)) > compat`, where:

   ```
   dist = Σ_{k<4} |Δh_k| + 3·|Δpitch| + 0.5·|Δtimbre|
   ```

   Pitch weighs most, so different "languages" are incompatible.
2. Each foreign contribution is scaled by `1 − kill`, with
   `kill = clamp(combat · claw(x) · dt, 0, 1)`: the resident's claws destroy foreign matter.
3. The destroyed mass is added to `Φ(x)`.

**Genome inheritance.** Among the neighbors with `m(y) > 0` (after combat), pick `y*` with
probability proportional to `m(y) · (1 + claw_bias · claw(y) · foreign(y))`, using a per-cell
random number. Foreign claws win contested cells, which amounts to assimilation. Then
`g'(x) = g(y*)`. If no neighbor contributes, `g'(x) = g(x)`.

**Walls.** If `W(x) > 0.5`, then `A'(x) = 0`. A small amount of mass is lost to walls.

### 6. Ecology

This step is pointwise and runs only if ecology is enabled. It uses `A'`, `g'`, `Φ`, `Ψ`,
`Ss`, and `A'_Σ = Σ_c A'_c`.

**Eating:**

```
eat = min(Φ, eat · mouth · Φ · min(A'_Σ, 1) · (1 + social · H/(H + 0.1)) · dt)
Φ  -= eat
A'_c += eat · A'_c / A'_Σ        (or into channel 0 if A'_Σ = 0)
```

**Metabolism:**

```
meta  = (meta_base + meta_trait · (eye + |ear| + voice + mouth + claw)) · (1 + toxicity · max(−Ψ, 0))
dec_c = min(A'_c, meta · A'_c · dt)
A'_c -= dec_c
Φ    += recycle · Σ_c dec_c
```

**Food regrowth:** if `Ψ > Φ`, then `Φ += regrow · (Ψ − Φ) · dt`.

**Per-cell mutation:** only where `A'_Σ > 0.01`, and with probability `mutation_rate`, pick
one gene `j ∈ 0..15` and add `mutation_size · N(0, 1)` to it. Then clamp it to its range
(hue wraps).

**Walls:** `Φ = 0` inside walls.

## Budding (reproduction)

Budding is decided on the CPU from creature tracking and runs as one GPU operation.

**When it happens:** a tracked creature (a connected blob) buds when all of these hold:

* its mass exceeds `bud_mass`,
* it is older than `bud_age` steps,
* it has not budded in the last `bud_age` steps.

**What it does:**

1. Choose a random direction `u`. The offset is `o = u · (2.6 r + 2)`, where `r` is the
   creature's effective radius.
2. Draw one mutation vector `δ` with `δ_j = bud_mutation · N(0, 1)` for every gene.
3. For every cell `x` within `1.3 r` of the creature's center:
   * move half of its mass: `A(x+o) += A(x)/2` and `A(x) -= A(x)/2`;
   * set `g(x+o) = clamp(g(x) + δ)`.

The offspring is a mutated half-copy of the parent. All cells of the offspring share the
same mutation, so it behaves as one organism.

## Mating

Mating is triggered by the CPU from tracking.

**When it happens:** two creatures mate when all of these hold:

* their bounding spheres touch,
* they are *compatible* (`dist` of their mean genomes ≤ `compat`),
* both are heavier than half of `bud_mass`,
* both are past their cooldown.

**What it does:**

1. Build the child genome: each gene comes from parent A or parent B at random (uniform
   crossover), plus the `bud_mutation` noise.
2. The heavier parent buds the child to the side, perpendicular to the line between the
   parents. The bud delta is set so that `g(child) = child genome`.

Because compatible matter mixes peacefully while foreign matter fights, populations that are
separated by walls drift apart in pitch ("language") and later treat each other as hostile.

## Creature tracking

Tracking runs on a coarse grid of 32³ cells (`f = N/32` cells per block).

1. The GPU sums mass and the mass-weighted genome per block.
2. The CPU finds 26-connected periodic components with `block mass / f³ > 0.03`.
3. For each component it computes the mass, the periodic center of mass, the effective radius
   `r = cbrt(3 · cells / 4π)`, and the mean traits.
4. Components are matched to the previous frame by nearest center. This gives stable ids,
   velocity and age, as well as birth and death events.

The tracker drives budding, the organ visuals, the inspector, the population graphs and the
audio.
