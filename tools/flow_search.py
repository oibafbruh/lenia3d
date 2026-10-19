#!/usr/bin/env python3
"""Search 3D Flow Lenia rules / ecology parameters with tools/flow_ref.py.

Modes
-----
search : sample random 1-channel kernel sets, seed a 64^3 soup (ecology OFF, uniform
         genome), screen for ``--screen`` steps, extend promising ones to ``--steps``.
         One JSON line per candidate in ``--out``.
eco    : run a preset JSON (ecology ON, budding) for ``--steps`` steps and print the
         population curve summary (or ``--grid`` over a few eco parameters).

Examples::

    python3 tools/flow_search.py search --minutes 25 --workers 3 --out flow_search.jsonl
    python3 tools/flow_search.py eco presets/flow/grazing_drift.json --steps 2000 --seeds 1 2
    python3 tools/flow_search.py eco presets/flow/primordial_flow.json --steps 1500 --seeds 4
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
import flow_ref as F  # noqa: E402


def sample_rule(rng):
    K = int(rng.integers(3, 9))
    ks = []
    for _ in range(K):
        nb = int(rng.integers(1, 4))
        b = [round(float(x), 3) for x in rng.uniform(0.0, 1.0, nb)]
        if max(b) < 0.2:
            b[int(rng.integers(nb))] = 1.0
        ks.append({"src": 0, "dst": 0, "r": round(float(rng.uniform(0.2, 1.0)), 3), "b": b,
                   "m": round(float(rng.uniform(0.05, 0.5)), 4),
                   "s": round(float(math.exp(rng.uniform(math.log(0.005), math.log(0.18)))), 4),
                   "h": round(float(rng.uniform(0.1, 1.0)), 3),
                   "kn": int(rng.integers(1, 3)), "gn": 2})
    R = float(rng.integers(8, 13))
    return {"R": R, "T": 5.0, "channels": 1, "kernels": ks}


def summarize(recs, N, t0, t1, min_mass=10.0):
    sel = [r for r in recs if t0 <= r["step"] <= t1]
    if not sel:
        return {}
    ns = [sum(1 for m in r["cmass"] if m >= min_mass) for r in sel]
    big = [max(r["cmass"]) / max(r["mass"], 1e-9) if r["cmass"] else 0.0 for r in sel]
    trk = [r["tracked"] / max(r["mass"], 1e-9) for r in sel]
    radii = [rr for r in sel for rr, m in zip(r["radii"], r["cmass"]) if m >= min_mass]
    sp = F.window_speeds(recs, N, 100, t0, t1)
    return {"n_min": int(min(ns)), "n_mean": float(np.mean(ns)), "n_max": int(max(ns)),
            "big_max": float(max(big)), "big_mean": float(np.mean(big)),
            "tracked_min": float(min(trk)),
            "r_med": float(np.median(radii)) if radii else 0.0,
            "r_p90": float(np.percentile(radii, 90)) if radii else 0.0,
            "speed_med": float(np.median(sp)) if sp else 0.0,
            "speed_mean": float(np.mean(sp)) if sp else 0.0,
            "mass": float(sel[-1]["mass"])}


def good(s, nmin=4):
    return (s and s["n_mean"] >= nmin and s["n_min"] >= 3 and s["big_max"] < 0.6
            and 2.5 <= s["r_med"] <= 11 and s["speed_med"] > 0.015 and s["tracked_min"] > 0.6)


def score(s):
    if not s:
        return -1e9
    sc = min(s["n_mean"], 15) * 0.2 + 10 * min(s["speed_med"], 0.1) - 3 * max(0, s["big_mean"] - 0.4)
    sc -= 2 * max(0, 0.8 - s["tracked_min"])
    if s["n_min"] < 3:
        sc -= 2
    return sc


def eval_rule(args):
    seed, rule, init, flow, screen, steps = args
    t = time.time()
    try:
        N = 64
        st = F.seed_flow(rule, N, init, None, seed=seed)
        tr = F.Tracker(N)
        recs = F.run_world(rule, flow, None, st, screen, seed=seed, inherit=False, tracker=tr)
        s1 = summarize(recs, N, max(100, screen - 150), screen)
        out = {"seed": seed, "rule": rule, "init": init, "flow": flow, "screen": s1, "score1": score(s1)}
        if s1 and s1["n_mean"] >= 3 and s1["big_max"] < 0.7 and s1["speed_med"] > 0.01 and steps > screen:
            recs = F.run_world(rule, flow, None, st, steps - screen, seed=seed, inherit=False,
                               history=recs, tracker=tr)
            s2 = summarize(recs, N, 300, steps)
            out["full"] = s2
            out["score2"] = score(s2)
            out["good"] = bool(good(s2))
            out["curve_n"] = [sum(1 for m in r["cmass"] if m >= 10) for r in recs[::10]]
        out["secs"] = round(time.time() - t, 1)
        return out
    except Exception as e:  # pragma: no cover
        return {"seed": seed, "error": repr(e)}


def cmd_search(a):
    from multiprocessing import Pool
    rng0 = np.random.default_rng(a.seed)
    deadline = time.time() + a.minutes * 60

    def jobs():
        i = 0
        while time.time() < deadline:
            sd = a.seed * 1000 + i
            rng = np.random.default_rng(sd)
            rule = sample_rule(rng)
            init = {"blobs": int(rng.integers(8, 21)), "blob_radius": 1.0,
                    "density": round(float(rng.uniform(0.5, 0.9)), 2)}
            flow = dict(F.FLOW_DEFAULTS)
            flow["theta"] = round(float(rng.uniform(1.5, 3.0)), 2)
            yield (sd, rule, init, flow, a.screen, a.steps)
            i += 1
    with Pool(a.workers) as pool, open(a.out, "a") as fh:
        for res in pool.imap_unordered(eval_rule, jobs()):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            s = res.get("full") or res.get("screen") or {}
            print(res["seed"], res.get("secs"), "good" if res.get("good") else "",
                  {k: round(v, 3) for k, v in s.items()} if s else res.get("error"), flush=True)
            if time.time() > deadline:
                pool.terminate()
                break
    del rng0


def cmd_rerun(a):
    """Re-run a candidate (by seed from a jsonl) with other seeds / thetas."""
    rec = None
    for line in open(a.file):
        r = json.loads(line)
        if r.get("seed") == a.select:
            rec = r
    rule, init, flow = rec["rule"], dict(rec["init"]), dict(rec["flow"])
    if a.theta:
        flow["theta"] = a.theta
    if a.blobs:
        init["blobs"] = a.blobs
    if a.density:
        init["density"] = a.density
    from multiprocessing import Pool
    jobs = [(sd, rule, init, flow, a.screen, a.steps) for sd in a.seeds]
    with Pool(min(a.workers, len(jobs))) as pool:
        for res in pool.imap_unordered(eval_rule, jobs):
            print(res["seed"], res.get("secs"), res.get("good"), res.get("full") or res.get("screen"),
                  res.get("curve_n"), flush=True)


def eco_run(args):
    preset, seed, steps, over = args
    p = json.load(open(preset)) if isinstance(preset, str) else json.loads(json.dumps(preset))
    for k, v in (over or {}).items():
        sect, key = k.split(".")
        p[sect][key] = v
    rule = F.normalize_rule({k: p[k] for k in ("R", "T", "channels", "kernels")})
    N = int(p.get("world_size", 64))
    st = F.seed_flow(rule, N, p["init"], p["ecology"], seed=seed)
    t = time.time()
    recs = F.run_world(rule, p["flow"], p["ecology"], st, steps, seed=seed, inherit=True)
    ns = [sum(1 for m in r["cmass"] if m >= 10) for r in recs]
    late = [r for r in recs if r["step"] >= steps // 3]
    nl = [sum(1 for m in r["cmass"] if m >= 10) for r in late]
    res = {"over": over, "seed": seed, "secs": round(time.time() - t),
           "n_min": min(nl), "n_mean": round(float(np.mean(nl)), 2), "n_max": max(nl),
           "n_end": ns[-1], "mass_min": round(min(r["mass"] for r in late)),
           "mass_mean": round(float(np.mean([r["mass"] for r in late]))),
           "mass_max": round(max(r["mass"] for r in late)),
           "food_mean": round(float(np.mean([r["food"] for r in late]))),
           "births": recs[-1]["births"], "deaths": recs[-1]["deaths"],
           "buds": int(sum(r["budded"] for r in recs)),
           "speed_med": round(float(np.median(F.window_speeds(recs, N, 100, steps // 3) or [0])), 4),
           "cmass_med": round(float(np.median([m for r in late for m in r["cmass"] if m >= 10] or [0])), 1),
           "traits_start": recs[0].get("traits"), "traits_end": recs[-1].get("traits"),
           "curve_n": ns[::10], "curve_mass": [round(r["mass"]) for r in recs[::10]],
           "curve_food": [round(r["food"]) for r in recs[::10]]}
    return res


def cmd_eco(a):
    from multiprocessing import Pool
    overs = [None]
    if a.grid:
        overs = [json.loads(g) for g in a.grid]
    jobs = [(a.preset, sd, a.steps, o) for o in overs for sd in a.seeds]
    with Pool(min(a.workers, len(jobs))) as pool, open(a.out, "a") as fh:
        for res in pool.imap_unordered(eco_run, jobs):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(json.dumps({k: v for k, v in res.items() if not k.startswith("curve")}), flush=True)
            print("  n:", res["curve_n"], flush=True)
            print("  mass:", res["curve_mass"][::2], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--minutes", type=float, default=20)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--screen", type=int, default=400)
    s.add_argument("--steps", type=int, default=1300)
    s.add_argument("--out", default="flow_search.jsonl")
    r = sub.add_parser("rerun")
    r.add_argument("--file", required=True)
    r.add_argument("--select", type=int, required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    r.add_argument("--theta", type=float)
    r.add_argument("--blobs", type=int)
    r.add_argument("--density", type=float)
    r.add_argument("--screen", type=int, default=400)
    r.add_argument("--steps", type=int, default=1300)
    r.add_argument("--workers", type=int, default=3)
    e = sub.add_parser("eco")
    e.add_argument("preset")
    e.add_argument("--steps", type=int, default=2000)
    e.add_argument("--seeds", type=int, nargs="+", default=[1])
    e.add_argument("--grid", nargs="*", help='JSON overrides, e.g. \'{"ecology.regrow": 0.01}\'')
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--out", default="flow_eco.jsonl")
    a = ap.parse_args()
    {"search": cmd_search, "rerun": cmd_rerun, "eco": cmd_eco}[a.cmd](a)


if __name__ == "__main__":
    main()
