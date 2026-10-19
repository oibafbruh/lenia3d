"""NumPy reference implementation of Lenia3D's *Flow* mode (docs/flow_lenia.md).

3D Flow Lenia (Plantec et al. 2022) with per-cell genomes, organs and an ecology.
Kernel construction and growth functions are imported from ``lenia_ref`` so the
conventions match classic mode exactly.

State (a plain dict)::

    A    (C, N, N, N) float32   mass per channel (not clipped)
    G    (16, N, N, N) float32  genome (layout below)
    food (N, N, N) float32      Phi
    fert (N, N, N) float32      Psi in [-1, 1]
    wall (N, N, N) float32      W in {0, 1}

Genome layout: 0..7 h_k, 8 eye, 9 ear, 10 voice, 11 mouth, 12 claw, 13 pitch,
14 hue, 15 timbre.

Main entry points
-----------------
* :class:`FlowSim` - caches kernel spectra for one (rule, flow, N); ``sim.step``.
* :func:`flow_step(state, rule, flow, eco, rng)` - convenience wrapper (cached sim).
* :func:`mass_only_step(A, rule, flow, genome=None)` - deterministic mass update with
  ecology OFF and a spatially uniform genome (used for the GPU comparison fixture).
* :class:`Tracker` - coarse-grid periodic component tracker (spec "Creature tracking").
* :func:`bud` - budding operation (spec "Budding").
* :func:`run_world` - steps + tracker + tracker-driven budding, returns metrics.
* :func:`seed_flow` - per-blob genome soup seeding for flow presets.

Choices where the spec leaves freedom (keep the GPU aligned):
* Kfar truncation is ``d <= far*R`` (d = Euclidean wrapped offset, as in make_kernel).
* Budding offset ``o`` is rounded to whole cells per axis (``np.rint``); "within 1.3 r"
  is ``d <= 1.3 r`` measured from cell centres to the (continuous) creature centre.
  Hue wraps (mod 1) instead of clamping, for the bud mutation too.
* Tracker effective radius uses ``cells = blocks * f^3`` (fine cells).
* Tracker matching: greedy nearest pairs (periodic distance), a pair is only accepted if
  its distance is ``<= max(4, r_prev + r_cur)``.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lenia_ref as L  # noqa: E402
from lenia_ref import _rfftn, _irfftn, _wrapped_offsets, growth, make_kernel, normalize_rule  # noqa: E402

NGENES = 16
EYE, EAR, VOICE, MOUTH, CLAW, PITCH, HUE, TIMBRE = 8, 9, 10, 11, 12, 13, 14, 15
TRAITS = {"eye": EYE, "ear": EAR, "voice": VOICE, "mouth": MOUTH, "claw": CLAW}
GENE_LO = np.zeros(NGENES, np.float32)
GENE_LO[EAR] = -1.0
GENE_HI = np.ones(NGENES, np.float32)

FLOW_DEFAULTS = {"theta": 2.0, "n": 2.0, "s": 0.65, "far": 2.5, "w_eye": 1.0, "w_ear": 1.0,
                 "claw_bias": 4.0}
ECO_DEFAULTS = {"enabled": False, "eat": 1.0, "social": 0.5, "meta_base": 0.01, "meta_trait": 0.01,
                "toxicity": 2.0, "recycle": 0.5, "regrow": 0.05, "fertility": 0.3,
                "mutation_rate": 0.0, "mutation_size": 0.05, "bud_mass": 150.0, "bud_age": 300,
                "bud_mutation": 0.05}


def flow_params(flow: dict | None) -> dict:
    out = dict(FLOW_DEFAULTS)
    out.update(flow or {})
    return {k: float(v) for k, v in out.items()}


def eco_params(eco: dict | None) -> dict:
    out = dict(ECO_DEFAULTS)
    out.update(eco or {})
    return out


def clamp_genes(G: np.ndarray) -> np.ndarray:
    """Clamp a (16, ...) genome array to its ranges in place (hue wraps)."""
    G[HUE] = np.mod(G[HUE], 1.0)
    for j in range(NGENES):
        if j == HUE:
            continue
        np.clip(G[j], GENE_LO[j], GENE_HI[j], out=G[j])
    return G


def make_far_kernel(shape: Sequence[int], R: float, far: float) -> np.ndarray:
    """Gaussian sigma = far*R/3, truncated at d <= far*R (wrapped offsets), sum 1."""
    oz, oy, ox = (_wrapped_offsets(n) for n in shape)
    d2 = oz[:, None, None] ** 2 + oy[None, :, None] ** 2 + ox[None, None, :] ** 2
    rad = far * R
    sig = rad / 3.0
    K = np.where(d2 <= rad * rad, np.exp(-d2 / (2.0 * sig * sig)), 0.0)
    return K / K.sum()


def grad(f: np.ndarray) -> list:
    """Periodic central differences of a 3D field -> [d/dz, d/dy, d/dx]."""
    return [(np.roll(f, -1, axis=ax) - np.roll(f, 1, axis=ax)) * np.float32(0.5) for ax in range(3)]


def clip1(v: list) -> list:
    """Scale the 3-vector field down so that |v| <= 1."""
    n = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    f = np.where(n > 1.0, 1.0 / np.maximum(n, 1e-30), 1.0).astype(np.float32)
    return [c * f for c in v]


def _overlap(p: np.ndarray, s: float) -> np.ndarray:
    return np.maximum(np.float32(0.0), np.minimum(p + np.float32(s), np.float32(0.5))
                      - np.maximum(p - np.float32(s), np.float32(-0.5)))


OFFSETS = [(oz, oy, ox) for oz in (-1, 0, 1) for oy in (-1, 0, 1) for ox in (-1, 0, 1)]


class FlowSim:
    """Stepper for one rule + flow parameters on an N^3 grid."""

    def __init__(self, rule: dict, flow: dict | None, N: int):
        self.rule = normalize_rule(rule)
        self.flow = flow_params(flow)
        self.N = int(N)
        self.shape = (self.N,) * 3
        self.C = self.rule["channels"]
        self.dt = np.float32(1.0 / self.rule["T"])
        self.kernels = self.rule["kernels"]
        self.kfft = []
        cache = {}
        for k in self.kernels:
            key = (k["r"], tuple(k["b"]), k["kn"])
            if key not in cache:
                K = make_kernel(self.shape, self.rule["R"], k["r"], k["b"], k["kn"])
                cache[key] = _rfftn(K.astype(np.float32)).astype(np.complex64)
            self.kfft.append(cache[key])
        self.sources = sorted({k["src"] for k in self.kernels})
        Kf = make_far_kernel(self.shape, self.rule["R"], self.flow["far"])
        self.farfft = _rfftn(Kf.astype(np.float32)).astype(np.complex64)

    # ---------------------------------------------------------------- fields
    def conv_far(self, f: np.ndarray) -> np.ndarray:
        return _irfftn(_rfftn(f) * self.farfft, self.shape).astype(np.float32, copy=False)

    def affinity(self, A: np.ndarray, G: np.ndarray | None) -> np.ndarray:
        """Aff_c = sum_{k: dst=c} h_k(x) G_k(U_k).  G=None -> rule's global h."""
        Fa = {c: _rfftn(A[c]) for c in self.sources}
        Aff = np.zeros((self.C,) + self.shape, np.float32)
        for k_i, (k, kf) in enumerate(zip(self.kernels, self.kfft)):
            U = _irfftn(Fa[k["src"]] * kf, self.shape).astype(np.float32, copy=False)
            g = growth(k["gn"], U, k["m"], k["s"])
            if k_i < 8 and G is not None:
                Aff[k["dst"]] += G[k_i] * g
            else:
                Aff[k["dst"]] += np.float32(k["h"]) * g
        return Aff

    def displacement(self, A, G, food=None, wall=None, need_ss=False):
        """Returns (D (C,3,N,N,N), Ss or None)."""
        fl = self.flow
        AS = A.sum(axis=0)
        Aff = self.affinity(A, G)
        alpha = np.clip((AS / np.float32(fl["theta"])) ** np.float32(fl["n"]), 0.0, 1.0).astype(np.float32)
        gA = grad(AS)
        steer = None
        Ss = None
        if G is not None:
            eye, ear = G[EYE], G[EAR]
            use_eye = fl["w_eye"] != 0.0 and food is not None and bool(np.any(eye != 0))
            use_ear = fl["w_ear"] != 0.0 and bool(np.any(ear != 0))
            if use_ear or need_ss:
                Ss = self.conv_far(G[VOICE] * AS)
            if use_eye or use_ear:
                steer = [np.zeros(self.shape, np.float32) for _ in range(3)]
                if use_eye:
                    Sf = self.conv_far(food)
                    v = clip1([g / (Sf + np.float32(0.05)) for g in grad(Sf)])
                    w = np.float32(fl["w_eye"]) * eye
                    for i in range(3):
                        steer[i] += w * v[i]
                if use_ear:
                    v = clip1([g / (Ss + np.float32(0.05)) for g in grad(Ss)])
                    w = np.float32(fl["w_ear"]) * ear
                    for i in range(3):
                        steer[i] += w * v[i]
        lim = np.float32(1.0 - fl["s"])
        D = np.empty((self.C, 3) + self.shape, np.float32)
        one_m = np.float32(1.0) - alpha
        for c in range(self.C):
            gAff = grad(Aff[c])
            for i in range(3):
                F = one_m * gAff[i] - alpha * gA[i]
                if steer is not None:
                    F = F + steer[i]
                D[c, i] = np.clip(self.dt * F, -lim, lim)
        if wall is not None and np.any(wall > 0.5):
            Wb = wall > 0.5
            for i in range(3):
                wp = np.roll(Wb, -1, axis=i)  # W(x + e_i)
                wm = np.roll(Wb, 1, axis=i)   # W(x - e_i)
                for c in range(self.C):
                    d = D[c, i]
                    d[((d > 0) & wp) | ((d < 0) & wm)] = 0.0
        return D, Ss

    def reintegrate(self, A, D, want_weights=False):
        """Mass gather.  Returns (A', m_o list of 27 (N,N,N) or None)."""
        s = self.flow["s"]
        inv = np.float32(1.0 / (2.0 * s) ** 3)
        Anew = np.zeros_like(A)
        ov = []  # ov[c][axis][o+1] at the source
        for c in range(self.C):
            ov.append([[_overlap(np.float32(o) + D[c, i], s) for o in (-1, 0, 1)] for i in range(3)])
        for c in range(self.C):
            Ac = A[c] * inv
            acc = np.zeros(self.shape, np.float32)
            for oz in (-1, 0, 1):
                Pz = Ac * ov[c][0][oz + 1]
                acc_y = np.zeros(self.shape, np.float32)
                for oy in (-1, 0, 1):
                    Py = Pz * ov[c][1][oy + 1]
                    acc_x = np.zeros(self.shape, np.float32)
                    for ox in (-1, 0, 1):
                        acc_x += np.roll(Py * ov[c][2][ox + 1], -ox, axis=2)
                    acc_y += np.roll(acc_x, -oy, axis=1)
                acc += np.roll(acc_y, -oz, axis=0)
            Anew[c] = acc
        mo = None
        if want_weights:
            mo = []
            for (oz, oy, ox) in OFFSETS:
                src = np.zeros(self.shape, np.float32)
                for c in range(self.C):
                    src += A[c] * ov[c][0][oz + 1] * ov[c][1][oy + 1] * ov[c][2][ox + 1]
                mo.append(np.roll(src * inv, (-oz, -oy, -ox), axis=(0, 1, 2)))
        return Anew, mo

    def inherit(self, G, mo, rng):
        """Stochastic genome inheritance: P(y) ~ m(y)(1 + claw_bias*claw(y)).

        Neighbour ``i`` (in OFFSETS order) is chosen where the running sum of the
        weights first exceeds ``u * total`` (u uniform in [0,1) per cell)."""
        cb = np.float32(self.flow["claw_bias"])
        N = self.N
        one = np.float32(1.0)
        W = []
        tot = np.zeros(self.shape, np.float32)
        for i, (oz, oy, ox) in enumerate(OFFSETS):
            claw_src = np.roll(G[CLAW], (-oz, -oy, -ox), axis=(0, 1, 2))
            w = np.maximum(mo[i], 0.0) * (one + cb * claw_src)
            W.append(w)
            tot += w
        target = rng.random(self.shape, dtype=np.float32) * tot
        cum = np.zeros(self.shape, np.float32)
        choice = np.full(self.shape, -1, np.int32)
        for i, w in enumerate(W):
            cum += w
            np.copyto(choice, np.int32(i), where=(choice < 0) & (cum > target))
        choice[choice < 0] = 13  # no contributor: keep own genome (offset 0,0,0)
        if not hasattr(self, "_zyx"):
            self._zyx = np.indices(self.shape, dtype=np.int32)
        z, y, x = self._zyx
        sz = (z + (choice // 9 - 1)) % N
        sy = (y + ((choice // 3) % 3 - 1)) % N
        sx = (x + (choice % 3 - 1)) % N
        flat = ((sz * N + sy) * N + sx).ravel()
        return np.take(G.reshape(NGENES, -1), flat, axis=1).reshape(G.shape)

    # ---------------------------------------------------------------- step
    def step(self, state: dict, eco: dict | None, rng, inherit: bool = True) -> dict:
        """One full step.  ``state`` is modified in place and returned."""
        A, G = state["A"], state["G"]
        food, fert, wall = state.get("food"), state.get("fert"), state.get("wall")
        eco_on = eco is not None and eco.get("enabled", False)
        need_ss = eco_on and float(eco["social"]) != 0.0
        D, Ss = self.displacement(A, G, food, wall, need_ss=need_ss)
        Anew, mo = self.reintegrate(A, D, want_weights=inherit)
        if inherit:
            Gn = self.inherit(G, mo, rng)
        else:
            Gn = G
        if wall is not None:
            Anew[:, wall > 0.5] = 0.0
        if eco_on:
            self.ecology(Anew, Gn, food, fert, Ss, eco, rng)
            if wall is not None:
                food[wall > 0.5] = 0.0
        state["A"], state["G"] = Anew, Gn
        state["step"] = state.get("step", 0) + 1
        return state

    def ecology(self, A, G, food, fert, Ss, eco, rng):
        dt = self.dt
        AS = A.sum(axis=0)
        boost = np.float32(1.0)
        if Ss is not None:
            boost = np.float32(1.0) + np.float32(eco["social"]) * Ss / (Ss + np.float32(0.1))
        eat = np.minimum(food, np.float32(eco["eat"]) * G[MOUTH] * food * np.minimum(AS, 1.0) * boost * dt)
        food -= eat
        pos = AS > 0
        frac = np.where(pos, eat / np.where(pos, AS, 1.0), 0.0).astype(np.float32)
        for c in range(self.C):
            A[c] += frac * A[c]
        if self.C > 0:
            A[0] += np.where(pos, 0.0, eat).astype(np.float32)
        organs = G[EYE] + np.abs(G[EAR]) + G[VOICE] + G[MOUTH] + G[CLAW]
        meta = (np.float32(eco["meta_base"]) + np.float32(eco["meta_trait"]) * organs) * \
               (np.float32(1.0) + np.float32(eco["toxicity"]) * np.maximum(-fert, 0.0))
        dec_tot = np.zeros(self.shape, np.float32)
        for c in range(self.C):
            dec = np.minimum(A[c], meta * A[c] * dt)
            A[c] -= dec
            dec_tot += dec
        food += np.float32(eco["recycle"]) * dec_tot
        grow = fert > food
        food[grow] += np.float32(eco["regrow"]) * (fert[grow] - food[grow]) * dt
        rate = float(eco.get("mutation_rate", 0.0))
        if rate > 0:
            # A'_Sigma as defined at the start of the ecology step (after reintegration,
            # before eating/metabolism)
            hit = (AS > 0.01) & (rng.random(self.shape, dtype=np.float32) < rate)
            idx = np.nonzero(hit)
            if len(idx[0]):
                j = rng.integers(0, NGENES, size=len(idx[0]))
                vals = G[(j,) + idx] + np.float32(eco["mutation_size"]) * rng.standard_normal(len(j)).astype(np.float32)
                hue = j == HUE
                vals[hue] = np.mod(vals[hue], 1.0)
                vals[~hue] = np.clip(vals[~hue], GENE_LO[j[~hue]], GENE_HI[j[~hue]])
                G[(j,) + idx] = vals


_SIM_CACHE: dict = {}


def _get_sim(rule, flow, N) -> FlowSim:
    key = (json.dumps(normalize_rule(rule), sort_keys=True), json.dumps(flow_params(flow), sort_keys=True), int(N))
    sim = _SIM_CACHE.get(key)
    if sim is None:
        if len(_SIM_CACHE) > 8:
            _SIM_CACHE.clear()
        sim = _SIM_CACHE[key] = FlowSim(rule, flow, N)
    return sim


def flow_step(state: dict, rule: dict, flow: dict | None, eco: dict | None, rng) -> dict:
    """One Flow-mode step (spec sections 1-6).  Modifies/returns ``state``."""
    sim = _get_sim(rule, flow, state["A"].shape[-1])
    return sim.step(state, eco_params(eco) if eco is not None else None, rng, inherit=True)


def mass_only_step(A: np.ndarray, rule: dict, flow: dict | None, genome=None, wall=None) -> np.ndarray:
    """Deterministic mass update, ecology OFF, spatially uniform genome.

    ``genome`` is a length-16 vector (default: h_k = rule h_k, all traits 0).  With a
    uniform genome the inheritance step cannot change anything, so this is exact.
    Senses: food is uniform (no eye steering effect); ear steering uses Ss from voice.
    """
    rule = normalize_rule(rule)
    N = A.shape[-1]
    sim = _get_sim(rule, flow, N)
    if genome is None:
        genome = default_genome(rule)
    genome = np.asarray(genome, np.float32)
    G = np.broadcast_to(genome[:, None, None, None], (NGENES,) + (N,) * 3)
    D, _ = sim.displacement(np.asarray(A, np.float32), G, None, wall)
    Anew, _ = sim.reintegrate(np.asarray(A, np.float32), D)
    if wall is not None:
        Anew[:, wall > 0.5] = 0.0
    return Anew


def default_genome(rule: dict, traits: dict | None = None) -> np.ndarray:
    rule = normalize_rule(rule)
    g = np.zeros(NGENES, np.float32)
    for i, k in enumerate(rule["kernels"][:8]):
        g[i] = k["h"]
    for name, v in (traits or {}).items():
        g[TRAITS[name]] = v
    return g


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def seed_flow(rule: dict, N: int, init: dict, eco: dict | None = None, seed=None) -> dict:
    """Flow preset seeding: lenia_ref.seed_blobs distribution, per-blob random genome.

    genes h_k = rule h_k exactly; traits uniform in init["traits"] ranges (missing -> 0);
    pitch/hue/timbre uniform [0,1].  The genome is written where that blob's v > 0
    (later blobs overwrite).  Food = fertility = eco.fertility everywhere.
    """
    rule = normalize_rule(rule)
    rng = np.random.default_rng(seed)
    shape = (N, N, N)
    C = rule["channels"]
    A = np.zeros((C,) + shape, np.float32)
    G = np.broadcast_to(default_genome(rule)[:, None, None, None], (NGENES,) + shape).copy()
    rb = float(init.get("blob_radius", 1.0)) * rule["R"]
    density = float(init.get("density", 0.5))
    traits = init.get("traits", {}) or {}
    for _ in range(int(init.get("blobs", 12))):
        c = [rng.uniform(0, n) for n in shape]
        d = L._wrapped_dist(shape, c)
        inside = d < rb
        t = 1.0 - d[inside] / rb
        w = t * t * (3.0 - 2.0 * t)
        g = default_genome(rule)
        for name, gi in TRAITS.items():
            lo, hi = traits.get(name, [0.0, 0.0])
            g[gi] = rng.uniform(lo, hi)
        g[PITCH], g[HUE], g[TIMBRE] = rng.random(3)
        anyv = np.zeros(w.shape, bool)
        for ch in range(C):
            v = rng.random(w.shape) * w
            v = v * (rng.random(w.shape) < density)
            A[ch][inside] = np.maximum(A[ch][inside], v.astype(np.float32))
            anyv |= v > 0
        idx = tuple(a[anyv] for a in np.nonzero(inside))
        G[(slice(None),) + idx] = g[:, None]
    fertv = float((eco or {}).get("fertility", ECO_DEFAULTS["fertility"]))
    fert = np.full(shape, fertv, np.float32)
    return {"A": A, "G": G, "food": fert.copy(), "fert": fert, "wall": np.zeros(shape, np.float32), "step": 0}


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------

def _pdiff(a, b, n):
    d = np.asarray(a, float) - np.asarray(b, float)
    return (d + n / 2) % n - n / 2


class Tracker:
    """Coarse-grid periodic component tracker (spec "Creature tracking")."""

    def __init__(self, N: int, coarse: int = 32, thresh: float = 0.03):
        from scipy import ndimage  # noqa: F401
        self.N = N
        self.cn = min(coarse, N)
        self.f = N // self.cn
        self.thresh = thresh
        self.prev = []  # list of dict
        self.next_id = 0
        self.births = 0
        self.deaths = 0

    def blocks(self, A, G):
        f, cn = self.f, self.cn
        M = A.sum(axis=0)
        def bsum(x):
            return x.reshape(cn, f, cn, f, cn, f).sum(axis=(1, 3, 5))
        BM = bsum(M).astype(np.float64)
        BT = {name: bsum(M * G[gi]).astype(np.float64) for name, gi in TRAITS.items()} if G is not None else {}
        return BM, BT

    def components(self, A, G=None):
        from scipy import ndimage
        BM, BT = self.blocks(A, G)
        f, cn = self.f, self.cn
        mask = BM / f ** 3 > self.thresh
        if not mask.any():
            return []
        P = np.pad(mask, 1, mode="wrap")
        lab, nl = ndimage.label(P, structure=np.ones((3, 3, 3), bool))
        parent = np.arange(nl + 1)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        inner = lab[1:-1, 1:-1, 1:-1]
        # union border copies with the interior cells they duplicate
        Pin = np.pad(inner, 1, mode="wrap")
        pairs = np.stack([lab.ravel(), Pin.ravel()], 1)
        pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 0] != pairs[:, 1])]
        if len(pairs):
            pairs = np.unique(pairs, axis=0)
            for a, b in pairs:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        roots = np.array([find(i) for i in range(nl + 1)])
        comp = roots[inner]
        comp[~mask] = 0
        ids = np.unique(comp[comp > 0])
        out = []
        idx = np.indices(mask.shape)
        for cid in ids:
            sel = comp == cid
            m = BM[sel]
            mass = float(m.sum())
            center = []
            for ax in range(3):
                pos = (idx[ax][sel] + 0.5) * f - 0.5
                th = 2 * np.pi * pos / self.N
                cz, sz = (m * np.cos(th)).sum(), (m * np.sin(th)).sum()
                center.append((math.atan2(sz, cz) % (2 * np.pi)) * self.N / (2 * np.pi))
            cells = int(sel.sum()) * f ** 3
            r = (3.0 * cells / (4.0 * math.pi)) ** (1.0 / 3.0)
            traits = {name: float(BT[name][sel].sum() / max(mass, 1e-12)) for name in BT}
            out.append({"mass": mass, "center": np.array(center), "r": r, "blocks": int(sel.sum()),
                        "traits": traits})
        return out

    def update(self, A, G, step: int):
        """Track; returns current component list with id/age/vel/last_bud."""
        cur = self.components(A, G)
        prev = self.prev
        pairs = []
        for i, c in enumerate(cur):
            for j, p in enumerate(prev):
                d = float(np.linalg.norm(_pdiff(c["center"], p["center"], self.N)))
                if d <= max(4.0, c["r"] + p["r"]):
                    pairs.append((d, i, j))
        pairs.sort()
        used_c, used_p = set(), set()
        for d, i, j in pairs:
            if i in used_c or j in used_p:
                continue
            used_c.add(i)
            used_p.add(j)
            p, c = prev[j], cur[i]
            dtsteps = max(1, step - p["step"])
            c["id"] = p["id"]
            c["birth"] = p["birth"]
            c["last_bud"] = p["last_bud"]
            c["vel"] = _pdiff(c["center"], p["center"], self.N) / dtsteps
        for i, c in enumerate(cur):
            if i not in used_c:
                c["id"] = self.next_id
                self.next_id += 1
                c["birth"] = step
                c["last_bud"] = -10 ** 9
                c["vel"] = None
                if prev or step > 0:
                    self.births += 1
            c["step"] = step
            c["age"] = step - c["birth"]
        self.deaths += len(prev) - len(used_p)
        self.prev = cur
        return cur


def bud(state: dict, comp: dict, rng, bud_mutation: float) -> np.ndarray:
    """Budding: move half the mass within 1.3 r of the centre by o = u (2.6 r + 2)."""
    A, G = state["A"], state["G"]
    N = A.shape[-1]
    shape = (N,) * 3
    u = rng.standard_normal(3)
    u /= np.linalg.norm(u) + 1e-12
    r = comp["r"]
    o = np.rint(u * (2.6 * r + 2.0)).astype(int)
    d = L._wrapped_dist(shape, comp["center"])
    mask = d <= 1.3 * r
    H = np.zeros_like(A)
    H[:, mask] = A[:, mask] * np.float32(0.5)
    A -= H
    A += np.roll(H, tuple(o), axis=(1, 2, 3))
    delta = (bud_mutation * rng.standard_normal(NGENES)).astype(np.float32)
    Gm = G[:, mask] + delta[:, None]
    clamp_genes(Gm)
    dst = np.roll(mask, tuple(o), axis=(0, 1, 2))
    # destination ordering: roll preserves C-order of the masked cells only per axis shift;
    # build explicit source->dest index mapping instead
    src_idx = np.nonzero(mask)
    dst_idx = tuple((src_idx[a] + o[a]) % N for a in range(3))
    G[(slice(None),) + dst_idx] = Gm
    del dst
    return o


# --------------------------------------------------------------------------
# Driver with metrics
# --------------------------------------------------------------------------

def run_world(rule, flow, eco, state, steps, seed=0, track_every=10, inherit=None,
              budding=True, log_every=100, verbose=False, min_mass_report=0.0, history=None,
              tracker=None, rng=None):
    """Run ``steps`` steps with tracking every ``track_every`` steps (and budding if
    ecology is enabled).  Returns a list of per-track-sample records."""
    rule = normalize_rule(rule)
    N = state["A"].shape[-1]
    sim = _get_sim(rule, flow, N)
    ecop = eco_params(eco) if eco is not None else None
    eco_on = ecop is not None and ecop.get("enabled", False)
    if inherit is None:
        inherit = True
    rng = np.random.default_rng(seed) if rng is None else rng
    cont = tracker is not None and history is not None
    tr = Tracker(N) if tracker is None else tracker
    recs = [] if history is None else history
    for t in range(steps + 1):
        if t % track_every == 0 and not (cont and t == 0):
            comps = tr.update(state["A"], state["G"], state.get("step", 0))
            vels = [float(np.linalg.norm(c["vel"])) for c in comps if c.get("vel") is not None]
            rec = {"step": state.get("step", 0), "n": len(comps),
                   "mass": float(state["A"].sum()),
                   "food": float(state["food"].sum()) if state.get("food") is not None else 0.0,
                   "masses": sorted([round(c["mass"], 1) for c in comps], reverse=True),
                   "radii": [round(c["r"], 2) for c in comps],
                   "speed": float(np.mean(vels)) if vels else 0.0,
                   "tracked": float(sum(c["mass"] for c in comps)),
                   "births": tr.births, "deaths": tr.deaths,
                   "ids": [int(c["id"]) for c in comps],
                   "centers": [[round(float(v), 2) for v in c["center"]] for c in comps],
                   "cmass": [round(c["mass"], 1) for c in comps]}
            if comps:
                tot = sum(c["mass"] for c in comps)
                rec["traits"] = {k: sum(c["traits"].get(k, 0) * c["mass"] for c in comps) / max(tot, 1e-9)
                                 for k in TRAITS}
            budded = 0
            if eco_on and budding:
                for c in comps:
                    if (c["mass"] > float(ecop["bud_mass"]) and c["age"] > int(ecop["bud_age"])
                            and state.get("step", 0) - c["last_bud"] >= int(ecop["bud_age"])):
                        bud(state, c, rng, float(ecop["bud_mutation"]))
                        c["last_bud"] = state.get("step", 0)
                        budded += 1
            rec["budded"] = budded
            recs.append(rec)
            if verbose and t % log_every == 0:
                print(json.dumps({k: (v if k != "masses" else v[:8]) for k, v in rec.items() if k not in ("radii", "ids", "centers", "cmass")}), flush=True)
        if t == steps:
            break
        sim.step(state, ecop, rng, inherit=inherit)
    return recs


def load_flow_preset(path) -> dict:
    with open(path) as fh:
        p = json.load(fh)
    p["rule"] = normalize_rule({k: p[k] for k in ("R", "T", "channels", "kernels")})
    return p


if __name__ == "__main__":
    import time
    rule = {"R": 10.0, "T": 5.0, "channels": 1,
            "kernels": [{"src": 0, "dst": 0, "r": 1.0, "b": [1.0], "m": 0.15, "s": 0.03, "h": 0.5,
                         "kn": 1, "gn": 2}]}
    st = seed_flow(rule, 64, {"blobs": 8, "blob_radius": 1.0, "density": 0.6}, seed=1)
    m0 = st["A"].sum()
    sim = FlowSim(rule, None, 64)
    rng = np.random.default_rng(0)
    t = time.time()
    for _ in range(10):
        sim.step(st, None, rng, inherit=False)
    t1 = time.time() - t
    t = time.time()
    for _ in range(10):
        sim.step(st, None, rng, inherit=True)
    t2 = time.time() - t
    print("10 steps no-inherit %.2fs, inherit %.2fs, mass %.4f -> %.4f" % (t1, t2, m0, st["A"].sum()))


def window_speeds(recs, N, window=100, t0=0, t1=None):
    """Net displacement speed (cells/step) of ids present at both ends of each
    ``window``-step interval within [t0, t1].  Returns a list of speeds."""
    by_step = {r["step"]: r for r in recs}
    steps = sorted(s for s in by_step if s >= t0 and (t1 is None or s <= t1))
    out = []
    for s in steps:
        e = s + window
        if e not in by_step:
            continue
        a, b = by_step[s], by_step[e]
        ca = dict(zip(a["ids"], a["centers"]))
        for i, c in zip(b["ids"], b["centers"]):
            if i in ca:
                out.append(float(np.linalg.norm(_pdiff(c, ca[i], N))) / window)
    return out
