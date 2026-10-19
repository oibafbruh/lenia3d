#!/usr/bin/env python3
"""End-to-end check of the shipped presets with the NumPy reference.

Loads every ``presets/*.json`` exactly as the app would (creature pasted in the
centre of a ``world_size``^3 world, or random-blob soup seeding), runs it and
reports mass / radius of gyration / speed.  Exits non-zero if a creature dies,
explodes or falls apart, or a soup dies, fills the world or freezes.

    python3 tools/check_presets.py                 # all presets, 1000 steps
    python3 tools/check_presets.py --steps 300 presets/comet.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lenia_ref as L  # noqa: E402
import search_presets as S  # noqa: E402

REQUIRED = {"name", "description", "R", "T", "channels", "kernels", "cells", "world_size", "kind"}
KERNEL_KEYS = {"src", "dst", "r", "b", "m", "s", "h", "kn", "gn"}


def check_schema(p):
    missing = REQUIRED - set(p)
    assert not missing, f"missing keys {missing}"
    assert p["kind"] in ("creature", "soup")
    for k in p["kernels"]:
        assert set(k) == KERNEL_KEYS, f"kernel keys {set(k)}"
        assert 0 <= k["src"] < p["channels"] and 0 <= k["dst"] < p["channels"]
        assert 1 <= len(k["b"]) <= 4
    if p["kind"] == "creature":
        assert p["cells"] and len(p["cells"]["data"]) == p["channels"]
    else:
        assert p["cells"] is None and set(p["init"]) == {"blobs", "blob_radius", "density"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presets")
    files = a.files or sorted(glob.glob(os.path.join(root, "*.json")))
    bad = 0
    for path in files:
        with open(path) as fh:
            raw = json.load(fh)
        check_schema(raw)
        p = L.load_preset(path)
        N = p["world_size"]
        if p["kind"] == "creature":
            oc, m, _ = S.track(p["rule"], L.initial_world(p), a.steps)
            ok = S.creature_ok(m, N)
            msg = (f"mass {m.get('mass_mean', 0):.0f} [{m.get('mass_min', 0):.0f},{m.get('mass_max', 0):.0f}] "
                   f"rg {m.get('rg_mean', 0):.2f} speed {m.get('speed', 0):.4f} cells/step")
        else:
            oc, f = S.run_soup(p["rule"], N, a.steps, p["init"], a.seed)
            ok = S.soup_score(f) > 0
            msg = (f"occupied {f.get('frac_min', 0):.3f}-{f.get('frac_max', 0):.3f} "
                   f"activity {f.get('act_end', 0):.3f} structures {f.get('comp', [])[-3:]}")
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {os.path.basename(path):22s} {p['kind']:8s} {oc:8s} {msg}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
