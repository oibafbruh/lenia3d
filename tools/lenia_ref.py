"""NumPy reference implementation of multi-channel, multi-kernel 3D Lenia.

This is the ground truth the GPU engine is compared against, and the engine the
preset search (``tools/search_presets.py``) runs on.  It depends only on NumPy;
if SciPy is installed its (much faster) ``scipy.fft`` is used, which gives the
same results to float32 rounding.

Model (identical to the engine)
-------------------------------
World: ``C`` channels, each a periodic float32 grid ``A[c]`` of shape
``(N, N, N)`` indexed ``[z, y, x]`` with values in ``[0, 1]``.

Rule (a plain dict, the same shape as the preset JSON)::

    {"R": 10.0, "T": 10.0, "channels": 1,
     "kernels": [{"src": 0, "dst": 0, "r": 1.0, "b": [1.0], "m": 0.15,
                  "s": 0.016, "h": 1.0, "kn": 1, "gn": 1}]}

Kernel ``k`` lives on the periodic grid centred on the origin (index 0) with
wrap-around.  For each cell offset ``(dz, dy, dx)`` (each the signed wrapped
offset in ``[-N/2, N/2)``)::

    D   = sqrt(dx^2 + dy^2 + dz^2) / (R * r)
    K   = 0                                      if D >= 1
    B   = len(b);  Br = B * D
    idx = min(floor(Br), B - 1);  f = Br - floor(Br)
    K   = b[idx] * core_kn(f)
    K  /= sum(K)

    core 1 polynomial  : (4 f (1 - f))^4
    core 2 exponential : exp(4 - 1 / (f (1 - f)))   for 0 < f < 1, else 0
    core 3 step        : 1 if 1/4 <= f <= 3/4 else 0
    core 4 staircase   : step(f) + (0.5 if f < 1/4 else 0)

Growth ``G`` of potential ``u``::

    gn 1 polynomial : max(0, 1 - (u - m)^2 / (9 s^2))^4 * 2 - 1
    gn 2 gaussian   : exp(-(u - m)^2 / (2 s^2)) * 2 - 1
    gn 3 step       : +1 if |u - m| <= s else -1

Update (synchronous; every kernel sees the same old state)::

    U_k = K_k (*) A[src_k]                       periodic convolution (FFT)
    D_c = sum_{k : dst_k == c} h_k * G_k(U_k)
    A[c] <- clip(A[c] + D_c / T, 0, 1)           for every c that is the dst of
                                                 at least one kernel; other
                                                 channels are left unchanged.

Random-blob seeding (used for "soup" presets, ``"init"`` block)
---------------------------------------------------------------
See :func:`seed_blobs`.  For each of ``blobs`` blobs:

1. pick a centre ``c`` uniformly in ``[0, N)^3`` (continuous coordinates);
2. ``rb = blob_radius * R``;
3. for every cell whose *wrapped* distance ``d`` (cell index vs. ``c``, each
   axis wrapped to ``[-N/2, N/2)``) is ``< rb``:
       ``t = 1 - d / rb``;  ``w = t * t * (3 - 2 t)``   (smoothstep falloff)
       for each channel independently:
           with probability ``density``: ``v = U(0, 1) * w``, otherwise ``v = 0``
           ``A[ch][cell] = max(A[ch][cell], v)``
All channels start at 0 before seeding.  The exact random stream does not
matter (the Rust side may use any RNG); only this distribution does.

Preset JSON
-----------
See :func:`save_preset` / :func:`load_preset`.  ``cells.data`` is one base64
string per channel holding ``uint8`` values ``round(A * 255)`` in C order over
``[z][y][x]``; ``cells.shape`` is ``[Z, Y, X]``.  Soup presets have
``"cells": null`` and an ``"init"`` block for :func:`seed_blobs`.
"""
from __future__ import annotations

import base64
import json
import math
from typing import Iterable, Sequence

import numpy as np

try:  # SciPy's pocketfft is ~5-8x faster than numpy.fft for float32 3D.
    import scipy.fft as _sfft

    def _rfftn(a):
        return _sfft.rfftn(a, axes=(-3, -2, -1), workers=1)

    def _irfftn(a, shape):
        return _sfft.irfftn(a, s=shape, axes=(-3, -2, -1), workers=1)

except ImportError:  # pragma: no cover - plain NumPy fallback

    def _rfftn(a):
        return np.fft.rfftn(a, axes=(-3, -2, -1))

    def _irfftn(a, shape):
        return np.fft.irfftn(a, s=shape, axes=(-3, -2, -1))


__all__ = [
    "kernel_core", "growth", "make_kernel", "Lenia", "seed_blobs", "blob",
    "stats", "periodic_com", "center_world", "crop_creature", "place_cells",
    "encode_cells", "decode_cells", "save_preset", "load_preset",
    "decode_chan_rle", "normalize_rule", "parse_fraction_list", "initial_world",
]

# --------------------------------------------------------------------------
# Core functions
# --------------------------------------------------------------------------

def kernel_core(kn: int, f: np.ndarray) -> np.ndarray:
    """Shell profile ``core_kn(f)`` for ``f`` in ``[0, 1)`` (float64)."""
    f = np.asarray(f, dtype=np.float64)
    if kn == 1:
        return (4.0 * f * (1.0 - f)) ** 4
    if kn == 2:
        out = np.zeros_like(f)
        inside = (f > 0.0) & (f < 1.0)
        fi = f[inside]
        out[inside] = np.exp(4.0 - 1.0 / (fi * (1.0 - fi)))
        return out
    if kn == 3:
        return ((f >= 0.25) & (f <= 0.75)).astype(np.float64)
    if kn == 4:
        return ((f >= 0.25) & (f <= 0.75)).astype(np.float64) + 0.5 * (f < 0.25)
    raise ValueError(f"unknown kernel core kn={kn}")


def growth(gn: int, u: np.ndarray, m: float, s: float) -> np.ndarray:
    """Growth mapping ``G(u)`` in ``[-1, 1]`` (keeps the dtype of ``u``)."""
    d = u - np.asarray(m, dtype=u.dtype)
    if gn == 1:
        q = np.maximum(0.0, 1.0 - d * d / np.asarray(9.0 * s * s, dtype=u.dtype))
        q = q * q
        return (q * q * 2.0 - 1.0).astype(u.dtype, copy=False)
    if gn == 2:
        return (np.exp(-d * d / np.asarray(2.0 * s * s, dtype=u.dtype)) * 2.0 - 1.0).astype(u.dtype, copy=False)
    if gn == 3:
        return np.where(np.abs(d) <= s, 1.0, -1.0).astype(u.dtype)
    raise ValueError(f"unknown growth type gn={gn}")


def _wrapped_offsets(n: int) -> np.ndarray:
    """Signed periodic offsets of each index from 0, in ``[-n/2, n/2)``."""
    i = np.arange(n, dtype=np.float64)
    return np.where(i < n / 2, i, i - n)  # n even: 0..n/2-1, -n/2..-1


def make_kernel(shape: Sequence[int], R: float, r: float, b: Sequence[float],
                kn: int) -> np.ndarray:
    """Normalised real-space kernel on the periodic grid, origin at index 0.

    Returns float64 of ``shape`` (``(Z, Y, X)``), summing to 1.
    """
    oz, oy, ox = (_wrapped_offsets(n) for n in shape)
    d = np.sqrt(oz[:, None, None] ** 2 + oy[None, :, None] ** 2 + ox[None, None, :] ** 2)
    D = d / (R * r)
    bs = np.asarray([float(x) for x in b], dtype=np.float64)
    B = len(bs)
    Br = B * D
    fl = np.floor(Br)
    idx = np.minimum(fl, B - 1).astype(np.int64)
    f = Br - fl
    K = np.where(D < 1.0, bs[np.minimum(idx, B - 1)] * kernel_core(kn, f), 0.0)
    tot = K.sum()
    if tot <= 0:
        raise ValueError("kernel is identically zero")
    return K / tot


def normalize_rule(rule: dict) -> dict:
    """Fill defaults / coerce types of a rule dict (returns a new dict)."""
    ks = []
    for k in rule["kernels"]:
        ks.append({
            "src": int(k.get("src", 0)), "dst": int(k.get("dst", 0)),
            "r": float(k.get("r", 1.0)), "b": [float(x) for x in k.get("b", [1.0])],
            "m": float(k["m"]), "s": float(k["s"]), "h": float(k.get("h", 1.0)),
            "kn": int(k.get("kn", 1)), "gn": int(k.get("gn", 1)),
        })
    C = int(rule.get("channels", 1 + max(max(k["src"], k["dst"]) for k in ks)))
    return {"R": float(rule["R"]), "T": float(rule["T"]), "channels": C, "kernels": ks}


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

class Lenia:
    """Stateless stepper for one rule on one grid shape.

    ``A`` is a float32 array of shape ``(C, Z, Y, X)``.

    >>> sim = Lenia(rule, (64, 64, 64))
    >>> A = sim.step(A)            # returns a new array
    """

    def __init__(self, rule: dict, shape: Sequence[int]):
        self.rule = normalize_rule(rule)
        self.shape = tuple(int(n) for n in shape)
        self.C = self.rule["channels"]
        self.dt = np.float32(1.0 / self.rule["T"])
        self.kernels = self.rule["kernels"]
        # Kernel spectra (real-valued because kernels are even; keep complex64).
        self.kfft = []
        cache = {}
        for k in self.kernels:
            key = (k["r"], tuple(k["b"]), k["kn"])
            if key not in cache:
                K = make_kernel(self.shape, self.rule["R"], k["r"], k["b"], k["kn"])
                cache[key] = _rfftn(K.astype(np.float32)).astype(np.complex64)
            self.kfft.append(cache[key])
        self.sources = sorted({k["src"] for k in self.kernels})
        self.dsts = sorted({k["dst"] for k in self.kernels})

    def potentials(self, A: np.ndarray) -> list:
        """List of potentials ``U_k`` (float32) for every kernel."""
        F = {c: _rfftn(A[c]) for c in self.sources}
        return [_irfftn(F[k["src"]] * kf, self.shape).astype(np.float32, copy=False)
                for k, kf in zip(self.kernels, self.kfft)]

    def growth_field(self, A: np.ndarray) -> np.ndarray:
        """``D_c`` for all channels, shape ``(C, Z, Y, X)`` (zeros where no kernel)."""
        D = np.zeros((self.C,) + self.shape, dtype=np.float32)
        for k, U in zip(self.kernels, self.potentials(A)):
            G = growth(k["gn"], U, k["m"], k["s"])
            if k["h"] == 1.0:
                D[k["dst"]] += G
            else:
                D[k["dst"]] += np.float32(k["h"]) * G
        return D

    def step(self, A: np.ndarray) -> np.ndarray:
        """One synchronous Lenia step.  Returns a new float32 array."""
        A = np.asarray(A, dtype=np.float32)
        D = self.growth_field(A)
        out = A.copy()
        for c in self.dsts:
            out[c] = np.clip(A[c] + self.dt * D[c], 0.0, 1.0)
        return out

    def run(self, A: np.ndarray, steps: int) -> np.ndarray:
        for _ in range(steps):
            A = self.step(A)
        return A


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def _wrapped_dist(shape, center):
    """Distance from every cell index to ``center`` on the periodic grid."""
    acc = 0.0
    for ax, (n, c) in enumerate(zip(shape, center)):
        d = np.arange(n, dtype=np.float64) - c
        d = (d + n / 2) % n - n / 2
        sh = [1, 1, 1]
        sh[ax] = n
        acc = acc + (d ** 2).reshape(sh)
    return np.sqrt(acc)


def blob(shape, center, radius, rng, density=1.0, skew=None):
    """One noisy blob (float32 of ``shape``), smoothstep falloff to 0 at ``radius``.

    ``skew`` (optional 3-vector) multiplies the blob by ``clip(1 + skew . d/radius, 0, 1)``
    where ``d`` is the offset from the centre; used by the search to make
    lopsided seeds that tend to start moving.
    """
    d = _wrapped_dist(shape, center)
    t = np.clip(1.0 - d / radius, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    v = rng.random(shape) * w
    if density < 1.0:
        v = v * (rng.random(shape) < density)
    if skew is not None:
        offs = []
        for ax, (n, c) in enumerate(zip(shape, center)):
            o = (np.arange(n, dtype=np.float64) - c + n / 2) % n - n / 2
            sh = [1, 1, 1]
            sh[ax] = n
            offs.append(o.reshape(sh))
        proj = sum(float(s) * o for s, o in zip(skew, offs)) / radius
        v = v * np.clip(1.0 + proj, 0.0, 1.0)
    return v.astype(np.float32)


def seed_blobs(shape, channels, R, blobs=12, blob_radius=1.0, density=0.5,
               seed=None) -> np.ndarray:
    """Random-blob soup initialisation (see module docstring for the spec)."""
    rng = np.random.default_rng(seed)
    A = np.zeros((channels,) + tuple(shape), dtype=np.float32)
    rb = blob_radius * R
    for _ in range(int(blobs)):
        c = [rng.uniform(0, n) for n in shape]
        d = _wrapped_dist(shape, c)
        inside = d < rb
        t = 1.0 - d[inside] / rb
        w = t * t * (3.0 - 2.0 * t)
        for ch in range(channels):
            v = rng.random(w.shape) * w
            v = v * (rng.random(w.shape) < density)
            A[ch][inside] = np.maximum(A[ch][inside], v.astype(np.float32))
    return A


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------

def periodic_com(M: np.ndarray) -> np.ndarray:
    """Periodic centre of mass ``(z, y, x)`` of a non-negative 3D field."""
    tot = float(M.sum())
    out = []
    for ax in range(3):
        n = M.shape[ax]
        proj = M.sum(axis=tuple(a for a in range(3) if a != ax)).astype(np.float64)
        th = 2 * np.pi * np.arange(n) / n
        cz, sz = (proj * np.cos(th)).sum(), (proj * np.sin(th)).sum()
        if tot <= 0 or (cz == 0 and sz == 0):
            out.append(0.0)
        else:
            out.append((math.atan2(sz, cz) % (2 * np.pi)) * n / (2 * np.pi))
    return np.asarray(out)


def stats(A: np.ndarray) -> dict:
    """mass (total, per channel), periodic COM, radius of gyration, max value."""
    M = A.sum(axis=0) if A.ndim == 4 else A
    mass = float(M.sum())
    com = periodic_com(M)
    rg = 0.0
    if mass > 0:
        acc = 0.0
        for ax in range(3):
            n = M.shape[ax]
            o = (np.arange(n) - com[ax] + n / 2) % n - n / 2
            proj = M.sum(axis=tuple(a for a in range(3) if a != ax)).astype(np.float64)
            acc += (proj * o ** 2).sum()
        rg = math.sqrt(acc / mass)
    return {
        "mass": mass,
        "mass_ch": [float(a.sum()) for a in A] if A.ndim == 4 else [mass],
        "com": com, "rg": rg, "max": float(M.max()),
    }


def center_world(A: np.ndarray) -> np.ndarray:
    """Roll ``A`` (C,Z,Y,X) so its periodic centre of mass sits at the grid centre."""
    M = A.sum(axis=0)
    com = periodic_com(M)
    shifts = [int(round(n / 2 - c)) for n, c in zip(M.shape, com)]
    return np.roll(A, shifts, axis=(1, 2, 3))


def crop_creature(A: np.ndarray, margin: int = 2, threshold: float = 0.5 / 255) -> np.ndarray:
    """Centre the creature, then crop to its bounding box (+``margin`` cells)."""
    A = center_world(A)
    M = A.max(axis=0) >= threshold
    if not M.any():
        raise ValueError("empty world")
    sl = []
    for ax in range(3):
        idx = np.where(M.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        lo, hi = max(int(idx[0]) - margin, 0), min(int(idx[-1]) + 1 + margin, M.shape[ax])
        sl.append(slice(lo, hi))
    return A[(slice(None),) + tuple(sl)].copy()


def place_cells(world_shape, cells: np.ndarray, center=None) -> np.ndarray:
    """Return a (C, *world_shape) zero world with ``cells`` (C,z,y,x) pasted centred
    (periodic wrap if it does not fit)."""
    C = cells.shape[0]
    W = np.zeros((C,) + tuple(world_shape), dtype=np.float32)
    if center is None:
        center = [n // 2 for n in world_shape]
    idx = []
    for ax in range(3):
        n, m = world_shape[ax], cells.shape[ax + 1]
        idx.append((np.arange(m) - m // 2 + center[ax]) % n)
    W[np.ix_(range(C), idx[0], idx[1], idx[2])] = cells
    return W


# --------------------------------------------------------------------------
# Preset I/O
# --------------------------------------------------------------------------

def encode_cells(cells: np.ndarray) -> dict:
    """(C,Z,Y,X) float -> {"shape": [Z,Y,X], "data": [base64 uint8 per channel]}."""
    q = np.rint(np.clip(cells, 0, 1) * 255).astype(np.uint8)
    return {"shape": [int(x) for x in cells.shape[1:]],
            "data": [base64.b64encode(np.ascontiguousarray(c).tobytes()).decode("ascii") for c in q]}


def decode_cells(d: dict) -> np.ndarray:
    shape = tuple(d["shape"])
    chans = [np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(shape) for s in d["data"]]
    return (np.stack(chans).astype(np.float32) / np.float32(255.0))


def save_preset(path, name, description, rule, cells=None, world_size=64,
                kind="creature", init=None, extra=None):
    rule = normalize_rule(rule)
    out = {
        "name": name,
        "description": description,
        "R": rule["R"],
        "T": rule["T"],
        "channels": rule["channels"],
        "kernels": rule["kernels"],
        "cells": encode_cells(cells) if cells is not None else None,
        "world_size": int(world_size),
        "kind": kind,
    }
    if init is not None:
        out["init"] = init
    if extra:
        out.update(extra)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    return out


def load_preset(path) -> dict:
    """Load a preset JSON.  Adds ``rule`` (normalised) and ``cells_array``
    ((C,Z,Y,X) float32 or None)."""
    with open(path) as fh:
        p = json.load(fh)
    p["rule"] = normalize_rule({k: p[k] for k in ("R", "T", "channels", "kernels")})
    p["cells_array"] = decode_cells(p["cells"]) if p.get("cells") else None
    return p


def initial_world(preset: dict, size: int | None = None, seed=None) -> np.ndarray:
    """Build the starting world for a loaded preset (creature pasted in the
    centre, or random-blob soup)."""
    N = int(size or preset["world_size"])
    shape = (N, N, N)
    if preset.get("cells_array") is not None:
        return place_cells(shape, preset["cells_array"])
    ini = preset.get("init", {})
    return seed_blobs(shape, preset["channels"], preset["R"], ini.get("blobs", 12),
                      ini.get("blob_radius", 1.0), ini.get("density", 0.5), seed=seed)


# --------------------------------------------------------------------------
# Bert Chan RLE (only for sanity checks against animals3D.json; not shipped)
# --------------------------------------------------------------------------

def decode_chan_rle(st: str) -> np.ndarray:
    """Decode a 3D Lenia RLE string into a float array [z][y][x] in [0, 1]."""
    layers, rows, row = [], [], []
    count, prefix = "", ""
    for ch in st:
        if ch.isdigit():
            count += ch
            continue
        if ch in "pqrstuvwxy":
            prefix = ch
            continue
        n = int(count) if count else 1
        count = ""
        if ch == "$":
            rows.append(row)
            rows.extend([[]] * (n - 1))
            row = []
        elif ch == "%" or ch == "!":
            rows.append(row)
            row = []
            layers.append(rows)
            rows = []
            if ch == "%":
                layers.extend([[]] * (n - 1))
            else:
                break
        else:
            tok = prefix + ch
            prefix = ""
            if tok in (".", "b"):
                v = 0
            elif tok == "o":
                v = 255
            elif len(tok) == 1:
                v = ord(tok) - ord("A") + 1
            else:
                v = (ord(tok[0]) - ord("p")) * 24 + (ord(tok[1]) - ord("A") + 25)
            row.extend([v] * n)
    if row or rows:
        rows.append(row)
        layers.append(rows)
    Z = len(layers)
    Y = max((len(l) for l in layers), default=0)
    X = max((len(r) for l in layers for r in l), default=0)
    out = np.zeros((Z, Y, X), dtype=np.float32)
    for z, l in enumerate(layers):
        for y, r in enumerate(l):
            out[z, y, :len(r)] = r
    return out / 255.0


def parse_fraction_list(s: str) -> list:
    """'1,5/12' -> [1.0, 0.41666]."""
    from fractions import Fraction
    return [float(Fraction(x)) for x in str(s).split(",")]


if __name__ == "__main__":  # tiny smoke test
    import time
    rule = {"R": 10.0, "T": 10.0, "channels": 1,
            "kernels": [{"src": 0, "dst": 0, "r": 1.0, "b": [1.0], "m": 0.15,
                         "s": 0.016, "h": 1.0, "kn": 1, "gn": 1}]}
    sim = Lenia(rule, (64, 64, 64))
    A = seed_blobs((64, 64, 64), 1, 10.0, blobs=1, blob_radius=1.0, density=1.0, seed=1)
    t = time.time()
    for _ in range(50):
        A = sim.step(A)
    print("50 steps", round(time.time() - t, 3), "s", stats(A)["mass"])
