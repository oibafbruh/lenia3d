#!/usr/bin/env python3
"""Search for original 3D Lenia presets with the NumPy reference (lenia_ref.py).

Strategy
--------
Creature search ("single", "multik", "multich" modes):
  * Sample a random rule (shell heights b, growth centre m, kernel radii, h
    weights, cross-channel routing ...).  Growth widths are written as
    ``s_k = lam * m_k * q_k`` with a shared "permissiveness" factor ``lam``.
  * Seed one noisy (optionally lopsided) blob of radius ~R and bisect ``lam``
    on a log scale: too small -> the blob dies, too large -> it fills the
    world.  Every run in between that stays alive *and* localised is recorded
    as a candidate together with its metrics.
  * Runs are short (screen: 48^3, ~300 steps) and parallel (multiprocessing,
    one candidate rule per worker).

Soup search ("soup" mode): sample rules (fresh or from creature candidates),
seed ``lenia_ref.seed_blobs`` on 64^3 and keep rules whose mass stays in a band
with ongoing change and several separate structures.

``graft`` adds a second channel to a settled creature (random 0->1 kernel plus
a weak 1->0 feedback kernel).  ``refine`` hill-climbs a candidate's parameters
toward speed/activity (experimental; not used for the shipped presets).
``export`` settles a candidate, optionally retunes s / rescales R, crops it and
re-verifies it from the quantised cells for 1000-2000 steps on 64^3 before
writing the preset; ``export-soup`` re-runs a soup rule with fresh seeds.
``tools/check_presets.py`` re-checks the shipped JSON files end to end.

Usage (what produced presets/, see presets/README.md)::

    python3 tools/search_presets.py search --mode multik --minutes 8 --seed 2000000 --out multik.jsonl
    python3 tools/search_presets.py search --mode multich --minutes 9 --seed 3000000 --out multich.jsonl
    python3 tools/search_presets.py search --mode single --minutes 3 --seed 1 --out single0.jsonl
    python3 tools/search_presets.py rank multik.jsonl multich.jsonl single0.jsonl
    cat *.jsonl | (positive-score records) > pos.jsonl
    python3 tools/search_presets.py search --mode soup --N 64 --steps 500 --workers 3 \
        --seed 5000000 --from-file pos.jsonl --minutes 7 --out soup.jsonl
    python3 tools/search_presets.py graft --file multik.jsonl --select seed=2001017,iter=5 \
        --trials 24 --steps 500 --workers 3 --out graft.jsonl
    python3 tools/search_presets.py export --file multik.jsonl --select seed=2001017,iter=5 \
        --R 13 --settle 300 --verify 1500 --name Comet --desc "..." --out presets/comet.json
    python3 tools/search_presets.py export-soup --file soup.jsonl --select seed=5000049 \
        --verify 2000 --seeds 41 42 --name "Tadpole Sea" --desc "..." --out presets/tadpole_sea.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lenia_ref as L  # noqa: E402

try:
    from scipy import ndimage as _ndi
except ImportError:  # pragma: no cover
    _ndi = None


# --------------------------------------------------------------------------
# Rule sampling
# --------------------------------------------------------------------------

def _rand_b(rng, B):
    """Random shell heights, rounded to 0.01, max normalised to 1."""
    b = rng.random(B)
    if B > 1 and rng.random() < 0.3:
        b[rng.integers(B)] = 0.0  # an empty shell now and then
    b[rng.integers(B)] = 1.0 if b.max() < 1e-6 else b.max()
    b = np.round(b / b.max(), 2)
    return [float(x) for x in b]


def _rand_B(rng):
    return int(rng.choice([1, 2, 3, 4], p=[0.2, 0.35, 0.3, 0.15]))


def _rand_core(rng):
    return (1, 1) if rng.random() < 0.65 else (2, 2)


def sample_rule(rng, mode, R=10.0):
    """Return (rule_template, q) where actual s_k = lam * m_k * q_k."""
    kernels = []
    if mode == "single":
        kn, gn = _rand_core(rng)
        kernels.append(dict(src=0, dst=0, r=1.0, b=_rand_b(rng, _rand_B(rng)),
                            m=float(np.round(rng.uniform(0.10, 0.34), 3)), h=1.0, kn=kn, gn=gn))
        C = 1
    elif mode == "multik":  # one channel, 2-3 kernels
        kn, gn = _rand_core(rng)
        nk = int(rng.choice([2, 3], p=[0.6, 0.4]))
        hs = rng.dirichlet(np.ones(nk) * 2.0)
        for i in range(nk):
            r = 1.0 if i == 0 else float(np.round(rng.uniform(0.35, 1.0), 2))
            kernels.append(dict(src=0, dst=0, r=r, b=_rand_b(rng, _rand_B(rng)),
                                m=float(np.round(rng.uniform(0.08, 0.40), 3)),
                                h=float(np.round(hs[i], 3)), kn=kn, gn=gn))
        C = 1
    elif mode == "mover":  # single kernel, multi-shell, low m (the regime of known 3D movers)
        kn, gn = (1, 1) if rng.random() < 0.8 else (2, 2)
        kernels.append(dict(src=0, dst=0, r=1.0, b=_rand_b(rng, int(rng.choice([2, 3, 4]))),
                            m=float(np.round(rng.uniform(0.10, 0.20), 3)), h=1.0, kn=kn, gn=gn))
        C = 1
    elif mode == "multich":  # 2 (sometimes 3) channels, cross-coupled
        kn, gn = _rand_core(rng)
        if rng.random() < 0.5:
            # "driven" layout: 0 is the body, 1 is a halo driven by 0 that feeds back
            C = 2
            routes = [(0, 0), (0, 1), (1, 0)]
            if rng.random() < 0.5:
                routes.append((1, 1))
            hraw = [rng.uniform(0.6, 1.0) if (a, b) == (0, 0) else rng.uniform(0.15, 0.6)
                    for a, b in routes]
            hraw[1] = 1.0  # 0->1 is the main input of channel 1
        else:
            C = 2 if rng.random() < 0.75 else 3
            routes = [(c, c) for c in range(C)]
            # cross routes: a ring plus a few random extras
            for c in range(C):
                routes.append((c, (c + 1) % C))
            for _ in range(int(rng.integers(0, C + 1))):
                a, b = int(rng.integers(C)), int(rng.integers(C))
                routes.append((a, b))
            hraw = [rng.uniform(0.5, 1.0) if a == b else rng.uniform(0.1, 0.8) for a, b in routes]
        tot = {c: sum(h for (a, b), h in zip(routes, hraw) if b == c) for c in range(C)}
        for (a, b), h in zip(routes, hraw):
            r = 1.0 if (a == b and a == 0) else float(np.round(rng.uniform(0.4, 1.0), 2))
            kernels.append(dict(src=a, dst=b, r=r, b=_rand_b(rng, _rand_B(rng)),
                                m=float(np.round(rng.uniform(0.08, 0.36), 3)),
                                h=float(np.round(h / tot[b], 3)), kn=kn, gn=gn))
    else:
        raise ValueError(mode)
    q = [float(np.round(rng.uniform(0.07, 0.16), 3)) for _ in kernels]
    rule = {"R": float(R), "T": 10.0, "channels": C, "kernels": kernels}
    return rule, q


def apply_lam(rule, q, lam):
    r = json.loads(json.dumps(rule))
    for k, qq in zip(r["kernels"], q):
        k["s"] = float(np.round(lam * k["m"] * qq, 5))
    return r


# --------------------------------------------------------------------------
# Running and measuring
# --------------------------------------------------------------------------

def make_seed(shape, C, R, init, rng=None):
    """Single blob for creature search.  init = {seed, radius, skew}."""
    rng = np.random.default_rng(init["seed"])
    center = [n / 2 for n in shape]
    A = np.zeros((C,) + tuple(shape), dtype=np.float32)
    for c in range(C):
        A[c] = L.blob(shape, center, init["radius"] * R, rng, density=1.0,
                      skew=init.get("skew"))
    return A


def n_components(A, thr=0.04):
    if _ndi is None:
        return -1
    M = L.center_world(A).max(axis=0) > thr
    lab, n = _ndi.label(M)
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())[1:]
    return int((sizes >= 8).sum())


def run_creature(rule, N, steps, init, every=10, keep_state=False):
    """Run a single-seed creature trial.  Returns (outcome, features, A)."""
    shape = (N, N, N)
    sim = L.Lenia(rule, shape)
    C = sim.C
    A = make_seed(shape, C, rule["R"], init)
    hist = []  # (step, mass, rg, com, mass_ch)
    prev_snap = A.copy()
    prev_com = None
    disp = np.zeros(3)
    act = []
    outcome = "alive"
    vol = N ** 3
    for t in range(1, steps + 1):
        A = sim.step(A)
        if t % every == 0:
            st = L.stats(A)
            if st["mass"] < 2.0 or st["max"] < 0.02:
                outcome = "dead"
                break
            if st["mass"] > 0.18 * vol or st["rg"] > 0.27 * N:
                outcome = "explode"
                break
            if prev_com is not None:
                d = (st["com"] - prev_com + N / 2) % N - N / 2
                disp = disp + d
            prev_com = st["com"]
            act.append(float(np.abs(A - prev_snap).sum() / max(st["mass"], 1e-6)))
            prev_snap = A.copy()
            hist.append((t, st["mass"], st["rg"], disp.copy(), st["mass_ch"]))
    feats = {"steps_run": t, "outcome": outcome}
    if hist:
        tail = [h for h in hist if h[0] > steps // 2] or hist[-3:]
        ms = np.array([h[1] for h in tail])
        rgs = np.array([h[2] for h in tail])
        d0, d1 = tail[0][3], tail[-1][3]
        span = max(tail[-1][0] - tail[0][0], 1)
        vel = (d1 - d0) / span
        feats.update({
            "mass": float(ms.mean()), "mass_cv": float(ms.std() / max(ms.mean(), 1e-9)),
            "rg": float(rgs.mean()), "rg_max": float(rgs.max()),
            "rg_cv": float(rgs.std() / max(rgs.mean(), 1e-9)),
            "speed": float(np.linalg.norm(vel)), "vel": [float(x) for x in vel],
            "activity": float(np.mean(act[-max(1, len(act) // 3):])) if act else 0.0,
            "mass_ch": [float(x) for x in np.array([h[4] for h in tail]).mean(axis=0)],
        })
        if outcome == "alive":
            feats["components"] = n_components(A)
    return outcome, feats, (A if keep_state else None)


def creature_score(f, C=1):
    """Heuristic interestingness for a surviving localised creature."""
    if f.get("outcome") != "alive":
        return -1.0
    if f.get("components", 1) != 1:
        return -0.5
    if f["mass_cv"] > 0.35 or f["rg"] < 2.0:
        return -0.2
    if C > 1:
        mc = np.array(f["mass_ch"])
        if mc.min() < 0.08 * mc.sum():
            return -0.3
    s = 1.0
    s += 40.0 * min(f["speed"], 0.08)          # movers
    s += 4.0 * min(f["activity"], 0.3)          # pulsing / rotating
    s += 1.0 * min(f["mass_cv"], 0.15) / 0.15   # breathing
    s -= 2.0 * max(f["mass_cv"] - 0.2, 0)
    return float(s)


def bisect_task(task):
    """Worker: one random rule, bisect lam, return list of records."""
    seed, mode, N, steps, iters = task
    rng = np.random.default_rng(seed)
    rule0, q = sample_rule(rng, mode)
    init = {"seed": int(rng.integers(1 << 30)),
            "radius": float(np.round(rng.uniform(0.7, 1.25), 2)),
            "skew": None}
    if rng.random() < (0.9 if mode == "mover" else 0.5):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(0.4, 1.5)
        init["skew"] = [float(np.round(x, 3)) for x in v]
    lo, hi = math.log(0.35), math.log(2.6)
    recs = []
    for it in range(iters):
        lam = math.exp(0.5 * (lo + hi))
        rule = apply_lam(rule0, q, lam)
        try:
            outcome, feats, _ = run_creature(rule, N, steps, init)
        except Exception as e:  # pragma: no cover
            recs.append({"seed": seed, "error": repr(e)})
            break
        rec = {"seed": seed, "mode": mode, "iter": it, "lam": lam, "rule": rule,
               "init": init, "N": N, "steps": steps, "feats": feats,
               "score": creature_score(feats, rule["channels"])}
        recs.append(rec)
        if outcome == "dead":
            lo = math.log(lam)
        elif outcome == "explode":
            hi = math.log(lam)
        else:
            # alive: probe both sides a little, biased toward the
            # explode edge where more dynamic creatures live
            if rng.random() < 0.6:
                lo = math.log(lam)
            else:
                hi = math.log(lam)
    return recs


# --------------------------------------------------------------------------
# Soups
# --------------------------------------------------------------------------

def run_soup(rule, N, steps, init, seed, every=20, return_state=False):
    shape = (N, N, N)
    sim = L.Lenia(rule, shape)
    A = L.seed_blobs(shape, sim.C, rule["R"], init["blobs"], init["blob_radius"],
                     init["density"], seed=seed)
    vol = N ** 3
    prev = A.copy()
    rows = []
    outcome = "alive"
    for t in range(1, steps + 1):
        A = sim.step(A)
        if t % every == 0:
            M = A.sum(axis=0)
            frac = float((M > 0.1).mean())
            mass = float(M.sum())
            act = float(np.abs(A - prev).sum() / max(mass, 1e-6))
            prev = A.copy()
            ncomp = -1
            if _ndi is not None and t % (every * 5) == 0:
                lab, n = _ndi.label(A.max(axis=0) > 0.15)
                ncomp = int((np.bincount(lab.ravel())[1:] >= 8).sum()) if n else 0
            rows.append((t, mass, frac, act, ncomp))
            if mass < 2.0:
                outcome = "dead"
                break
            if frac > 0.6:
                outcome = "full"
                break
    feats = {"outcome": outcome, "steps_run": t}
    if rows:
        tail = [r for r in rows if r[0] > steps // 3] or rows[-3:]
        feats.update({
            "frac_min": float(min(r[2] for r in tail)), "frac_max": float(max(r[2] for r in tail)),
            "frac_end": float(rows[-1][2]), "act_mean": float(np.mean([r[3] for r in tail])),
            "act_end": float(np.mean([r[3] for r in rows[-5:]])),
            "comp": [r[4] for r in rows if r[4] >= 0],
            "mass_end": float(rows[-1][1]),
        })
    return (outcome, feats, A) if return_state else (outcome, feats)


def soup_score(f):
    if f["outcome"] != "alive":
        return -1.0
    if f["frac_max"] > 0.5 or f["frac_min"] < 0.002:
        return -0.5
    if f["act_end"] < 0.02:  # froze into static blobs
        return -0.4
    comp = [c for c in f["comp"][len(f["comp"]) // 3:]]
    if not comp or min(comp) < 2:
        return -0.3
    s = 1.0 + 10 * min(f["act_end"], 0.2) + 0.3 * min(np.std(comp), 5) + 0.02 * min(np.mean(comp), 40)
    return float(s)


def soup_task(task):
    seed, src_rule, N, steps = task
    rng = np.random.default_rng(seed)
    if src_rule is None:
        mode = str(rng.choice(["single", "multik", "multich"], p=[0.45, 0.35, 0.2]))
        rule0, q = sample_rule(rng, mode)
        lam = float(np.exp(rng.uniform(math.log(0.6), math.log(1.8))))
        rule = apply_lam(rule0, q, lam)
    else:
        rule = json.loads(json.dumps(src_rule))
        mode = "from-creature"
        # small random nudge of every s toward the permissive side
        f = float(np.exp(rng.normal(0.05, 0.06)))
        for k in rule["kernels"]:
            k["s"] = float(np.round(k["s"] * f, 5))
    init = {"blobs": int(rng.integers(8, 20)), "blob_radius": float(np.round(rng.uniform(0.8, 1.3), 2)),
            "density": float(np.round(rng.uniform(0.4, 0.9), 2))}
    outcome, feats = run_soup(rule, N, steps, init, seed)
    return {"seed": seed, "mode": mode, "rule": rule, "init": init, "N": N, "steps": steps,
            "feats": feats, "score": soup_score(feats)}


# --------------------------------------------------------------------------
# Local search (hill-climb on a settled creature, mainly for speed)
# --------------------------------------------------------------------------

def settle(rule, init, N, steps):
    """Run a blob seed for ``steps``; return the world (None if not alive)."""
    outcome, feats, A = run_creature(rule, N, steps, init, keep_state=True)
    return (A if outcome == "alive" else None), feats


def track(rule, A, steps, every=10, comp_every=100):
    """Run an existing world; return (outcome, metrics dict, final A)."""
    N = A.shape[-1]
    sim = L.Lenia(rule, A.shape[1:])
    vol = N ** 3
    prev_com = L.stats(A)["com"]
    disp = np.zeros(3)
    rows, comps, acts = [], [], []
    snap = A.copy()
    outcome = "alive"
    for t in range(1, steps + 1):
        A = sim.step(A)
        if t % every == 0:
            st = L.stats(A)
            if st["mass"] < 2.0 or st["max"] < 0.02:
                outcome = "dead"
                break
            if st["mass"] > 0.18 * vol or st["rg"] > 0.27 * N:
                outcome = "explode"
                break
            disp = disp + (st["com"] - prev_com + N / 2) % N - N / 2
            prev_com = st["com"]
            acts.append(float(np.abs(A - snap).sum() / max(st["mass"], 1e-6)))
            snap = A.copy()
            rows.append((t, st["mass"], st["rg"], disp.copy(), st["mass_ch"]))
            if t % comp_every == 0:
                comps.append(n_components(A))
    m = {"outcome": outcome, "steps_run": t}
    if rows:
        ms = np.array([r[1] for r in rows])
        rg = np.array([r[2] for r in rows])
        half = rows[len(rows) // 2:]
        span = max(half[-1][0] - half[0][0], 1)
        vel = (half[-1][3] - half[0][3]) / span
        m.update({
            "mass_mean": float(ms.mean()), "mass_min": float(ms.min()), "mass_max": float(ms.max()),
            "mass_cv": float(ms.std() / ms.mean()),
            "rg_mean": float(rg.mean()), "rg_max": float(rg.max()), "rg_min": float(rg.min()),
            "speed": float(np.linalg.norm(vel)), "vel": [float(x) for x in vel],
            "total_disp": float(np.linalg.norm(rows[-1][3])),
            "activity": float(np.mean(acts)),
            "components": comps,
            "mass_ch": [float(x) for x in np.mean([r[4] for r in rows], axis=0)],
        })
    return outcome, m, A


def perturb(rule, rng, scale=1.0):
    r = json.loads(json.dumps(rule))
    for k in r["kernels"]:
        k["m"] = float(np.round(k["m"] * math.exp(rng.normal(0, 0.03 * scale)), 4))
        k["s"] = float(np.round(k["s"] * math.exp(rng.normal(0, 0.05 * scale)), 5))
        if len(k["b"]) > 1 and rng.random() < 0.5:
            i = int(rng.integers(len(k["b"])))
            b = np.array(k["b"])
            b[i] = np.clip(b[i] + rng.normal(0, 0.08 * scale), 0, 1)
            b = b / b.max()
            k["b"] = [float(x) for x in np.round(b, 3)]
        if k["r"] != 1.0 and rng.random() < 0.3:
            k["r"] = float(np.round(np.clip(k["r"] * math.exp(rng.normal(0, 0.05 * scale)), 0.3, 1.0), 3))
    if len(r["kernels"]) > 1 and rng.random() < 0.4:
        for c in range(r["channels"]):
            ks = [k for k in r["kernels"] if k["dst"] == c]
            if len(ks) > 1:
                hs = np.array([k["h"] for k in ks]) * np.exp(rng.normal(0, 0.1 * scale, len(ks)))
                hs = hs / hs.sum()
                for k, h in zip(ks, hs):
                    k["h"] = float(np.round(h, 4))
    return r


def refine_task(task):
    """Hill-climb a candidate toward higher speed (or activity) while staying a
    single bounded creature.  Works on the settled state so the creature
    morphs continuously with the parameters."""
    rec, N, n_iter, steps, objective, seed = task
    rng = np.random.default_rng(seed)
    rule = rec["rule"]
    A, _ = settle(rule, rec["init"], N, 300)
    if A is None:
        return {"seed": seed, "src_seed": rec["seed"], "error": "did not settle"}
    oc, best_m, A_best = track(rule, A, steps)
    if oc != "alive":
        return {"seed": seed, "src_seed": rec["seed"], "error": "not alive at start"}

    def score(m):
        if m["outcome"] != "alive" or any(c != 1 for c in m["components"]) or m["mass_cv"] > 0.3:
            return -1
        if objective == "speed":
            return m["speed"] + 0.02 * min(m["activity"], 0.3)
        return m["activity"] + 2 * m["speed"]

    best, best_rule = score(best_m), rule
    history = [best]
    for it in range(n_iter):
        cand = perturb(best_rule, rng, scale=1.0 if it < n_iter // 2 else 0.5)
        oc, m, A2 = track(cand, A_best.copy(), steps)
        sc = score(m)
        if sc > best:
            best, best_rule, best_m, A_best = sc, cand, m, A2
        history.append(best)
    return {"seed": seed, "src_seed": rec["seed"], "mode": rec.get("mode"), "rule": best_rule,
            "init": rec["init"], "N": N, "score": best, "metrics": best_m, "history": history,
            "state_b64": L.encode_cells(L.crop_creature(A_best, margin=4))}


# --------------------------------------------------------------------------
# Grafting: add a second channel to an existing creature
# --------------------------------------------------------------------------

def graft_task(task):
    """Add channel 1 to a settled single-channel creature: a kernel 0->1
    (channel 1 grows from channel 0's potential) and a weak feedback kernel
    1->0 that takes weight ``eps`` from channel 0's own kernels.  Random
    parameters; keep results that stay one compact, alive creature with a
    substantial channel 1."""
    base_rule, A0, seed, steps = task
    rng = np.random.default_rng(seed)
    rule = json.loads(json.dumps(base_rule))
    kn, gn = rule["kernels"][0]["kn"], rule["kernels"][0]["gn"]
    eps = float(np.round(rng.uniform(0.05, 0.3), 3))
    for k in rule["kernels"]:
        k["h"] = float(np.round(k["h"] * (1 - eps), 4))
    m01 = float(np.round(rng.uniform(0.08, 0.35), 3))
    rule["kernels"].append(dict(src=0, dst=1, r=float(np.round(rng.uniform(0.3, 1.0), 2)),
                                b=_rand_b(rng, _rand_B(rng)), m=m01,
                                s=float(np.round(m01 * rng.uniform(0.1, 0.35), 5)), h=1.0, kn=kn, gn=gn))
    m10 = float(np.round(rng.uniform(0.05, 0.3), 3))
    rule["kernels"].append(dict(src=1, dst=0, r=float(np.round(rng.uniform(0.3, 1.0), 2)),
                                b=_rand_b(rng, _rand_B(rng)), m=m10,
                                s=float(np.round(m10 * rng.uniform(0.1, 0.35), 5)), h=eps, kn=kn, gn=gn))
    rule["channels"] = 2
    A = np.concatenate([A0, np.zeros_like(A0)], axis=0)
    oc, m, A = track(rule, A, steps)
    rec = {"seed": seed, "mode": "graft", "rule": rule, "N": A.shape[-1], "metrics": m, "score": -1.0}
    if oc == "alive" and creature_ok(m, A.shape[-1]) and min(m["mass_ch"]) > 0.1 * sum(m["mass_ch"]):
        rec["score"] = 1.0 + 10 * m["speed"] + m["activity"]
        rec["state_b64"] = L.encode_cells(L.crop_creature(A, margin=4))
    return rec


def cmd_graft(a):
    from multiprocessing import Pool
    src = select_record(a.file, a.select)
    A, _ = settle(src["rule"], src["init"], src["N"], 300)
    tasks = [(src["rule"], A, a.seed + i, a.steps) for i in range(a.trials)]
    with Pool(a.workers) as pool, open(a.out, "a") as fh:
        for rec in pool.imap_unordered(graft_task, tasks):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            m = rec["metrics"]
            print(f"seed={rec['seed']} score={rec['score']:.2f} {m['outcome']} speed={m.get('speed', 0):.4f} "
                  f"act={m.get('activity', 0):.3f} mass_ch={[round(x) for x in m.get('mass_ch', [])]}", flush=True)


# --------------------------------------------------------------------------
# Verification and export
# --------------------------------------------------------------------------

def select_record(path, sel):
    """``sel`` like 'seed=123,iter=4' or 'line=17'."""
    recs = load_jsonl(path)
    kv = dict(x.split("=") for x in sel.split(","))
    if "line" in kv:
        return recs[int(kv["line"])]
    out = [r for r in recs if all(str(r.get(k)) == v for k, v in kv.items())]
    if not out:
        raise SystemExit(f"no record matches {sel}")
    return max(out, key=lambda r: r.get("score", -1))


def verify_creature(rule, cells, N, steps):
    """Paste ``cells`` (C,z,y,x) in an N^3 world and track it."""
    A = L.place_cells((N, N, N), cells)
    return track(rule, A, steps)


def creature_ok(m, N):
    """Alive for the whole run, one compact body (no splitting: radius of
    gyration stays within a narrow band; a nested shell+core may count as 2
    labelled components), and mass within a band."""
    comps = sorted(m["components"])
    return (m["outcome"] == "alive" and comps[len(comps) // 2] <= 2
            and m["rg_max"] < 0.25 * N and m["rg_max"] < 1.35 * m["rg_min"]
            and m["mass_min"] > 0.4 * m["mass_mean"] and m["mass_max"] < 2.0 * m["mass_mean"])


def cmd_export(a):
    rec = select_record(a.file, a.select)
    rule = rec["rule"]
    N = a.N
    if "state_b64" in rec:
        A = L.place_cells((N, N, N), L.decode_cells(rec["state_b64"]))
        oc, m, A = track(rule, A, a.settle)
    else:
        # settle in the exact world the candidate was screened in (same seed
        # noise), then move the creature into the verification world
        Ns = int(rec.get("N", N))
        A, _ = settle(rule, rec["init"], Ns, a.settle)
        oc = "alive" if A is not None else "dead"
        if A is not None and Ns != N:
            A = L.place_cells((N, N, N), L.crop_creature(A, margin=4))
    if oc != "alive" or A is None:
        raise SystemExit(f"candidate did not survive settling: {oc}")
    if a.s_scale != 1.0:
        # retune: a creature found right at the explode edge is often only
        # transient; nudging every growth width s down while the creature
        # already exists keeps the shape and stabilises it.
        rule = json.loads(json.dumps(rule))
        for k in rule["kernels"]:
            k["s"] = float(np.round(k["s"] * a.s_scale, 5))
        oc, _, A = track(rule, A, a.settle)
        if oc != "alive":
            raise SystemExit(f"did not survive s-scale {a.s_scale}: {oc}")
    if a.R and abs(a.R - rule["R"]) > 1e-9:
        # Lenia is (nearly) scale-free: zoom the settled creature to the new
        # kernel radius, then let it re-settle at the new resolution.
        f = a.R / rule["R"]
        small = L.crop_creature(A, margin=4)
        big = np.stack([_ndi.zoom(c, f, order=1) for c in small]).clip(0, 1).astype(np.float32)
        rule = json.loads(json.dumps(rule))
        rule["R"] = float(a.R)
        A = L.place_cells((N, N, N), big)
        oc, m0, A = track(rule, A, a.settle)
        if oc != "alive":
            raise SystemExit(f"did not survive rescale to R={a.R}: {oc}")
    cells = L.crop_creature(A, margin=2)
    cells = np.rint(cells * 255) / 255  # exactly what the preset stores
    t0 = time.time()
    oc, m, _ = verify_creature(rule, cells.astype(np.float32), N, a.verify)
    m["seconds"] = round(time.time() - t0, 1)
    m["cells_shape"] = list(cells.shape)
    ok = creature_ok(m, N)
    print(json.dumps({k: v for k, v in m.items() if k != "vel"}, default=float))
    print("PASS" if ok else "FAIL")
    if ok or a.force:
        L.save_preset(a.out, a.name, a.desc, rule, cells=cells, world_size=N, kind="creature")
        stats_path = os.path.join(os.path.dirname(os.path.abspath(a.stats)), os.path.basename(a.stats))
        allst = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
        allst[os.path.basename(a.out)] = {"verify_steps": a.verify, "N": N, **m}
        json.dump(allst, open(stats_path, "w"), indent=1, default=float)
        print("wrote", a.out)


def cmd_export_soup(a):
    rec = select_record(a.file, a.select)
    rule = json.loads(json.dumps(rec["rule"]))
    if a.R:
        rule["R"] = float(a.R)
    init = dict(rec["init"])
    if a.blobs:
        init["blobs"] = a.blobs
    N = a.N
    results = []
    for sd in a.seeds:
        t0 = time.time()
        oc, f = run_soup(rule, N, a.verify, init, sd)
        f["score"] = soup_score(f)
        f["seconds"] = round(time.time() - t0, 1)
        results.append(f)
        print(sd, json.dumps({k: (v if k != "comp" else v[-8:]) for k, v in f.items()}, default=float), flush=True)
    ok = all(f["score"] > 0 for f in results)
    print("PASS" if ok else "FAIL")
    if ok or a.force:
        L.save_preset(a.out, a.name, a.desc, rule, cells=None, world_size=N, kind="soup", init=init)
        stats_path = a.stats
        allst = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
        allst[os.path.basename(a.out)] = {"verify_steps": a.verify, "N": N, "runs": results}
        json.dump(allst, open(stats_path, "w"), indent=1, default=float)
        print("wrote", a.out)


def cmd_refine(a):
    from multiprocessing import Pool
    recs = []
    for p in a.files:
        recs += [r for r in load_jsonl(p) if r.get("score", -1) > 0]
    recs.sort(key=lambda r: -(r["feats"]["speed"] if a.objective == "speed" else r["score"]))
    # one record per source rule
    seen, pick = set(), []
    for r in recs:
        if r["seed"] in seen:
            continue
        seen.add(r["seed"])
        pick.append(r)
        if len(pick) >= a.top:
            break
    tasks = [(r, a.N, a.iters, a.steps, a.objective, 7000000 + i) for i, r in enumerate(pick)]
    with Pool(a.workers) as pool, open(a.out, "a") as fh:
        for res in pool.imap_unordered(refine_task, tasks):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if "metrics" in res:
                m = res["metrics"]
                print(f"src={res['src_seed']} score={res['score']:.4f} speed={m['speed']:.4f} "
                      f"act={m['activity']:.3f} mass={m['mass_mean']:.0f} rg={m['rg_mean']:.2f} "
                      f"hist={[round(h, 4) for h in res['history'][::5]]}", flush=True)
            else:
                print(res, flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_search(a):
    from multiprocessing import Pool
    deadline = time.time() + 60 * a.minutes
    base = a.seed if a.seed is not None else int(time.time()) % 100000 * 1000
    n_done = 0
    src_rules = None
    if a.mode == "soup" and a.from_file:
        src = [json.loads(l) for l in open(a.from_file)]
        src = [r for r in src if r.get("score", -1) > 0]
        src_rules = [r["rule"] for r in src]
        print(f"{len(src_rules)} source rules", flush=True)
    fn = soup_task if a.mode == "soup" else bisect_task

    def make_task(i):
        if a.mode == "soup":
            rr = None
            if src_rules and (i % 3 != 0):
                rr = src_rules[(i * 7919) % len(src_rules)]
            return (base + i, rr, a.N, a.steps)
        return (base + i, a.mode, a.N, a.steps, a.iters)

    with Pool(a.workers) as pool, open(a.out, "a") as fh:
        i = 0
        pending = []
        while True:
            # keep a bounded number of tasks in flight until the deadline
            while time.time() < deadline and len(pending) < 2 * a.workers:
                i += 1
                pending.append(pool.apply_async(fn, (make_task(i),)))
            if not pending:
                break
            ready = [p for p in pending if p.ready()]
            if not ready:
                time.sleep(0.05)
                continue
            for p in ready:
                pending.remove(p)
                res = p.get()
                recs = res if isinstance(res, list) else [res]
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                n_done += 1
                best = max((r.get("score", -1) for r in recs), default=-1)
                if best > 0:
                    f = max(recs, key=lambda r: r.get("score", -1))["feats"]
                    print(f"[{n_done}] seed={recs[0]['seed']} score={best:.2f} "
                          + " ".join(f"{k}={v:.3g}" for k, v in f.items() if isinstance(v, float)),
                          flush=True)
    print("done", n_done, "rules")


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def cmd_rank(a):
    recs = []
    for p in a.files:
        recs += [r for r in load_jsonl(p) if r.get("score", -1) > 0]
    recs.sort(key=lambda r: -r["score"])
    for r in recs[: a.top]:
        f = r["feats"]
        ks = r["rule"]["kernels"]
        print(f"{r['score']:.2f} seed={r['seed']} it={r.get('iter')} C={r['rule']['channels']} nk={len(ks)} "
              + " ".join(f"{k}={v:.3g}" for k, v in f.items() if isinstance(v, float))
              + f" comp={f.get('components', f.get('comp'))} "
              + "| " + "; ".join(f"{k['src']}>{k['dst']} b={k['b']} m={k['m']} s={k['s']} h={k['h']} r={k['r']}" for k in ks[:4]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--mode", choices=["single", "mover", "multik", "multich", "soup"], required=True)
    s.add_argument("--minutes", type=float, default=10)
    s.add_argument("--N", type=int, default=48)
    s.add_argument("--steps", type=int, default=300)
    s.add_argument("--iters", type=int, default=7)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int)
    s.add_argument("--from-file", help="soup mode: jsonl of creature candidates to reuse rules from")
    s.add_argument("--out", required=True)
    r = sub.add_parser("rank")
    r.add_argument("files", nargs="+")
    r.add_argument("--top", type=int, default=30)
    f = sub.add_parser("refine")
    f.add_argument("files", nargs="+")
    f.add_argument("--top", type=int, default=8)
    f.add_argument("--N", type=int, default=48)
    f.add_argument("--iters", type=int, default=20)
    f.add_argument("--steps", type=int, default=200)
    f.add_argument("--objective", choices=["speed", "activity"], default="speed")
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--out", required=True)
    g = sub.add_parser("graft")
    g.add_argument("--file", required=True)
    g.add_argument("--select", required=True)
    g.add_argument("--trials", type=int, default=32)
    g.add_argument("--steps", type=int, default=600)
    g.add_argument("--seed", type=int, default=8000000)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--out", required=True)
    for name in ("export", "export-soup"):
        e = sub.add_parser(name)
        e.add_argument("--file", required=True)
        e.add_argument("--select", required=True)
        e.add_argument("--name", required=True)
        e.add_argument("--desc", required=True)
        e.add_argument("--out", required=True)
        e.add_argument("--N", type=int, default=64)
        e.add_argument("--verify", type=int, default=1500)
        e.add_argument("--stats", default="preset_stats.json")
        e.add_argument("--force", action="store_true")
        e.add_argument("--R", type=float, help="use this kernel radius (creatures are zoomed to match)")
        if name == "export":
            e.add_argument("--settle", type=int, default=500)
            e.add_argument("--s-scale", type=float, default=1.0,
                           help="after settling, multiply every s by this and settle again")

        else:
            e.add_argument("--blobs", type=int)
            e.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    {"search": cmd_search, "rank": cmd_rank, "refine": cmd_refine, "graft": cmd_graft,
     "export": cmd_export, "export-soup": cmd_export_soup}[a.cmd](a)
