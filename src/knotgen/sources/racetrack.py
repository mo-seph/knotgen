"""Racetrack (stadium) layout for weaving knots — the braid-closure picture.

The weave of W(p, q) is drawn the way the papers draw it: p parallel lanes
run around a stadium-shaped loop, and the braid word (s1 s2^-1 ...)^q plays
out along ONE straight — all q(p-1) crossings concentrated there, each an
adjacent-lane swap with an over/under z bump. Around the rest of the track
the lanes are nested and never cross.

Construction is in sample space (centerline + lateral lane offset + z),
then FFT'd into the standard FourierKnot representation, so all downstream
machinery (styling, strip frames, preflight, export) applies unchanged.

Controls:
  aspect          straight length / stadium width — stretches the racetrack
  braid_fraction  how much of the braid straight the crossings occupy
                  (small = tight woven patch, 1.0 = spread the full straight)
"""

from __future__ import annotations

from math import gcd

import numpy as np

from knotgen.fourier import FourierKnot
from knotgen.link import FourierLink


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Quintic smoothstep, C2: 0 -> 1 over x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _bump(x: np.ndarray) -> np.ndarray:
    """C1 bump, 0 -> 1 -> 0 over x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * x))


def _stadium(u: np.ndarray, R: float, Ls: float) -> tuple[np.ndarray, np.ndarray]:
    """Stadium centerline: points and outward normals at arc positions u.

    Segments (counter-clockwise): bottom straight (the braid side) from
    (-Ls/2, -R) to (+Ls/2, -R), right cap, top straight, left cap.
    """
    cap = np.pi * R
    total = 2 * Ls + 2 * cap
    u = np.mod(u, total)
    pts = np.zeros((len(u), 2))
    nrm = np.zeros((len(u), 2))

    m = u < Ls  # bottom straight
    pts[m, 0] = -Ls / 2 + u[m]
    pts[m, 1] = -R
    nrm[m] = (0.0, -1.0)

    m = (u >= Ls) & (u < Ls + cap)  # right cap
    th = (u[m] - Ls) / R - np.pi / 2
    nrm[m, 0] = np.cos(th)
    nrm[m, 1] = np.sin(th)
    pts[m, 0] = Ls / 2 + R * np.cos(th)
    pts[m, 1] = R * np.sin(th)

    m = (u >= Ls + cap) & (u < 2 * Ls + cap)  # top straight
    pts[m, 0] = Ls / 2 - (u[m] - Ls - cap)
    pts[m, 1] = R
    nrm[m] = (0.0, 1.0)

    m = u >= 2 * Ls + cap  # left cap
    th = (u[m] - 2 * Ls - cap) / R + np.pi / 2
    nrm[m, 0] = np.cos(th)
    nrm[m, 1] = np.sin(th)
    pts[m, 0] = -Ls / 2 + R * np.cos(th)
    pts[m, 1] = R * np.sin(th)

    return pts, nrm


def racetrack_weave(
    p: int,
    q: int,
    aspect: float = 2.0,
    braid_fraction: float = 0.8,
    lane_gap: float | None = None,
    braid_split: float = 0.0,
    samples_per_lap: int = 2048,
    n_harmonics: int = 200,
) -> FourierKnot | FourierLink:
    """W(p, q) laid out as a closed braid on a stadium racetrack.

    lane_gap: spacing between adjacent lanes as a fraction of the stadium
    half-width (default auto). Smaller = gentler crossings (lower curvature
    in the woven patch) but less clearance between the nested lanes.

    braid_split: fraction of the crossings moved to the OTHER straight
    (0 = all on one side, 0.5 = half and half). Splitting the cyclic braid
    word around the closure is an isotopy, so the knot type is unchanged —
    the alternation check still verifies each result.
    """
    if p < 2 or q < 1:
        raise ValueError(f"W({p},{q}): need p >= 2 and q >= 1")
    if not 0.05 <= braid_fraction <= 1.0:
        raise ValueError("braid_fraction must be in [0.05, 1]")
    if aspect <= 0:
        raise ValueError("aspect must be positive")
    if not 0.0 <= braid_split <= 1.0:
        raise ValueError("braid_split must be in [0, 1]")

    R = 1.0
    Ls = aspect * 2.0 * R
    total = 2 * Ls + 2 * np.pi * R
    if lane_gap is not None:
        if not 0.02 <= lane_gap <= 0.9:
            raise ValueError("lane_gap must be in [0.02, 0.9]")
        gap = lane_gap * R
    else:
        gap = min(0.8 * R / max(p - 1, 1), 0.35 * R)
    # small: this is a flat 2.5D layout (depth styling rescales z); a large
    # amplitude would trip the auto-depth 3D detection at small aspects
    bump_amp = 0.3

    def lane_offset(track: float) -> float:
        return (track - (p - 1) / 2.0) * gap

    # the braid word of W(p,q): (s1 s2^-1 s3 s4^-1 ...)^q
    word = []
    for _ in range(q):
        for i in range(p - 1):
            word.append((i, +1 if i % 2 == 0 else -1))
    n_letters = len(word)

    # distribute the letters over the two straights: the first block on the
    # bottom straight, the rest on the top straight (in travel order, so the
    # cyclic word is preserved). Each side gets its own evenly-spaced zone.
    cap = np.pi * R
    n_top = int(round(braid_split * n_letters))
    n_bottom = n_letters - n_top
    positions: list[tuple[float, float]] = []  # (u_start, window) per letter
    if n_bottom:
        zone = braid_fraction * Ls
        w_b = zone / n_bottom
        start = (Ls - zone) / 2.0
        positions += [(start + k * w_b, w_b) for k in range(n_bottom)]
    if n_top:
        zone = braid_fraction * Ls
        w_t = zone / n_top
        start = Ls + cap + (Ls - zone) / 2.0
        positions += [(start + k * w_t, w_t) for k in range(n_top)]

    u = np.linspace(0.0, total, samples_per_lap, endpoint=False)

    # per-lap lane and z profiles for a strand entering the lap in `track`
    lap_lane: list[np.ndarray] = []
    lap_z: list[np.ndarray] = []
    end_track = [0] * p
    for t0 in range(p):
        lane = np.full(samples_per_lap, float(t0))
        z = np.zeros(samples_per_lap)
        ct = t0
        for (i, sign), (u0, w) in zip(word, positions):
            frac = (u - u0) / w
            inside = (frac >= 0.0) & (frac < 1.0)
            if ct == i or ct == i + 1:
                other = i + 1 if ct == i else i
                lane[inside] = ct + (other - ct) * _smoothstep(frac[inside])
                lane[u >= u0 + w] = other
                over = (sign > 0) == (ct == i)
                z[inside] += (bump_amp if over else -bump_amp) * _bump(frac[inside])
                ct = other
        lap_lane.append(lane)
        lap_z.append(z)
        end_track[t0] = ct

    # follow strand cycles -> components
    d = gcd(p, q)
    components = []
    seen = set()
    for start in range(p):
        if start in seen:
            continue
        cycle = [start]
        seen.add(start)
        t = end_track[start]
        while t != start:
            cycle.append(t)
            seen.add(t)
            t = end_track[t]

        lanes = np.concatenate([lap_lane[t] for t in cycle])
        zs = np.concatenate([lap_z[t] for t in cycle])
        u_all = np.tile(u, len(cycle))
        pts2, nrm2 = _stadium(u_all, R, Ls)
        offs = lane_offset(lanes)  # vectorized: plain arithmetic
        pts3 = np.column_stack(
            [
                pts2[:, 0] + offs * nrm2[:, 0],
                pts2[:, 1] + offs * nrm2[:, 1],
                zs,
            ]
        )
        components.append(
            FourierKnot.from_samples(
                pts3,
                n_harmonics=min(n_harmonics * len(cycle), len(pts3) // 2 - 1),
                name=f"W({p},{q})",
                meta={"source": "weaving-racetrack", "p": p, "q": q,
                      "aspect": aspect, "braid_fraction": braid_fraction,
                      "braid_split": braid_split},
            )
        )

    if len(components) != d:
        raise RuntimeError(
            f"W({p},{q}) racetrack: expected {d} components, got {len(components)}"
        )
    if d == 1:
        return components[0]
    return FourierLink(
        components=components,
        name=f"W({p},{q})",
        meta=dict(components[0].meta, components=d),
    )
