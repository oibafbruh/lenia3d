#!/usr/bin/env python3
"""Generate tests/fixtures/flow_step_16.json: one exact Flow-mode mass step.

N=16, C=1, two kernels, R=4, T=5 (dt=0.2), theta=2, n=2, s=0.65, far=2.5.
A smooth random mass field (values 0..5, so the crowding weight alpha
covers [0, 1] and the displacement clamp +-0.35 is hit in places), a spatially
uniform genome with eye = ear = voice = 0, ecology OFF, no walls.  Exactly one
step of :func:`flow_ref.mass_only_step`.

NOTE: the genome's h genes (``genome[0..1]``) deliberately differ from the rule's
kernel ``h`` values, so the test checks that the per-cell genes (not the rule h)
weight the affinity.

JSON fields (all arrays are flat lists in C order over [z][y][x]):
  N, channels, rule {R, T, channels, kernels}, flow {...}, genome (16 floats),
  A_in, A_out (C*N^3 floats each, channel-major), aff (affinity of channel 0,
  for debugging), stats {mass_in, mass_out, max_abs_D, frac_clamped}.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flow_ref as F  # noqa: E402

N = 16
RULE = {
    "R": 4.0, "T": 5.0, "channels": 1,
    "kernels": [
        {"src": 0, "dst": 0, "r": 1.0, "b": [1.0], "m": 0.25, "s": 0.08, "h": 0.6, "kn": 1, "gn": 2},
        {"src": 0, "dst": 0, "r": 0.6, "b": [1.0, 0.4], "m": 0.6, "s": 0.15, "h": 0.4, "kn": 2, "gn": 1},
    ],
}
FLOW = {"theta": 2.0, "n": 2.0, "s": 0.65, "far": 2.5, "w_eye": 1.0, "w_ear": 1.0, "claw_bias": 4.0}


def main(out=None):
    from scipy import ndimage
    rng = np.random.default_rng(12345)
    noise = rng.random((N, N, N))
    sm = ndimage.gaussian_filter(noise, 1.0, mode="wrap")
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    A = 5.0 * (np.maximum(sm - 0.25, 0.0) / 0.75) ** 1.5  # 0 .. 5, zeros in ~25%
    A = A.astype(np.float32)[None]
    genome = F.default_genome(RULE)
    genome[0], genome[1] = 0.8, 0.3            # differ from rule h (0.6, 0.4)
    genome[F.MOUTH], genome[F.CLAW] = 0.5, 0.2  # irrelevant here (ecology off, uniform genome)
    genome[F.PITCH], genome[F.HUE], genome[F.TIMBRE] = 0.5, 0.25, 0.75
    A_out = F.mass_only_step(A, RULE, FLOW, genome=genome)
    sim = F._get_sim(RULE, FLOW, N)
    G = np.broadcast_to(genome[:, None, None, None], (16, N, N, N))
    aff = sim.affinity(A, G)[0]
    D, _ = sim.displacement(A, G)
    lim = 1.0 - FLOW["s"]

    def fl(a):
        return [float("%.9g" % v) for v in np.asarray(a, np.float32).ravel()]
    data = {
        "description": "One Flow-mode step (ecology off, uniform genome) from tools/flow_fixture.py",
        "N": N, "channels": 1, "steps": 1, "rule": RULE, "flow": FLOW,
        "genome": [float("%.9g" % g) for g in genome],
        "A_in": fl(A), "A_out": fl(A_out), "aff": fl(aff),
        "stats": {"mass_in": float(A.astype(np.float64).sum()), "mass_out": float(A_out.astype(np.float64).sum()),
                  "max_abs_D": float(np.abs(D).max()),
                  "frac_clamped": float((np.abs(D) >= lim - 1e-7).mean()),
                  "max_A_in": float(A.max())},
    }
    out = out or os.path.join(HERE, "..", "tests", "fixtures", "flow_step_16.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(out, os.path.getsize(out), "bytes", data["stats"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
