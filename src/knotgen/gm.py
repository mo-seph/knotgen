"""Gonzalez-Maddocks global radius of curvature.

The tangent-point radius r_tp(x, y) is the radius of the circle through y
that is TANGENT to the curve at x:

    r_tp = |y - x|^2 / (2 * d_perp)

where d_perp is the distance from y to the tangent line at x. Its infimum
over all pairs is the curve's THICKNESS: the largest tube radius that can
be swept without self-intersection (Gonzalez & Maddocks, PNAS 1999).

Why this is the right quantity for knotgen: it unifies the two failure
modes the preflight tracks separately. As y slides toward x along a smooth
arc, r_tp converges to the LOCAL curvature radius (the bend check); for a
strand doubling back past x, r_tp is half the passage clearance (the gap
check) — with no along-curve exclusion window, no arc/chord heuristics,
and no coupling to the worst kink elsewhere on the curve. A tight cusp in
one lobe cannot change what counts as a violation in another.

Cost note: the exact infimum is over all pairs, but r_tp >= |y - x| / 2
always (d_perp <= |y - x|), so only pairs closer than 2x the current best
can compete — the search prunes with a KD-tree radius query seeded by the
local curvature bound.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from knotgen.fourier import TAU, FourierKnot
from knotgen.link import FourierLink, as_link


def tangent_point_radii(
    P: np.ndarray, T: np.ndarray, i: np.ndarray, j: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """r_tp for pairs (i, j): circle through P[j] tangent at P[i].

    Returns (radii, d_perp unit vectors pointing from the tangent line at
    P[i] toward P[j] — the direction that INCREASES the radius).
    Degenerate pairs (y on the tangent line) get radius inf.
    """
    chord = P[j] - P[i]
    d2 = np.einsum("ij,ij->i", chord, chord)
    along = np.einsum("ij,ij->i", chord, T[i])
    perp = chord - along[:, None] * T[i]
    dp = np.linalg.norm(perp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dp > 1e-12, d2 / (2.0 * dp), np.inf)
        perp_hat = np.where(dp[:, None] > 1e-12, perp / np.maximum(dp, 1e-12)[:, None], 0.0)
    return r, perp_hat


def _sampled(link: FourierLink, samples: int):
    """Arc-length samples with unit tangents, all components stacked."""
    lengths = [c.total_length() for c in link.components]
    total = sum(lengths)
    pts, tans, comp_id, idx, counts = [], [], [], [], []
    for ci, comp in enumerate(link.components):
        n = max(256, int(round(samples * lengths[ci] / total)))
        t, p = comp.sample_arclength(n)
        d1 = comp.deriv(t, 1)
        tans.append(d1 / np.linalg.norm(d1, axis=1, keepdims=True))
        pts.append(p)
        comp_id.append(np.full(n, ci))
        idx.append(np.arange(n))
        counts.append(n)
    return (np.vstack(pts), np.vstack(tans), np.concatenate(comp_id),
            np.concatenate(idx), counts)


def thickness(
    knot: FourierKnot | FourierLink, samples: int = 1024
) -> dict:
    """The curve's GM thickness: the largest sweepable tube RADIUS.

    Returns a dict with 'thickness' (mm), 'min_local_radius' (the classical
    bend limit), 'min_tp_radius' (the global/tangent-point limit over
    non-neighbour pairs), and 'max_tube_diameter' (= 2 * thickness).
    """
    link = as_link(knot)
    P, T, comp_id, idx, counts = _sampled(link, samples)

    # local bound: classical curvature radius (r_tp limit as y -> x)
    tloc = np.linspace(0.0, TAU, 2048, endpoint=False)
    kmax = max(float(c.curvature(tloc).max()) for c in link.components)
    best = 1.0 / kmax

    # prune: only pairs with |y - x| < 2 * best can beat the current best
    tree = cKDTree(P)
    pairs = tree.query_pairs(r=2.0 * best, output_type="ndarray")
    min_tp = np.inf
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        # exclude immediately adjacent samples: their r_tp estimates the
        # local radius, which the analytic curvature above already covers
        # more accurately than sampled chords do
        same = comp_id[i] == comp_id[j]
        gap = np.abs(idx[i] - idx[j])
        n_arr = np.array(counts)[comp_id[i]]
        gap = np.minimum(gap, n_arr - gap)
        keep = ~(same & (gap <= 2))
        i, j = i[keep], j[keep]
        if len(i):
            # tangent-point radius is asymmetric: check both orientations
            r_ij, _ = tangent_point_radii(P, T, i, j)
            r_ji, _ = tangent_point_radii(P, T, j, i)
            min_tp = float(min(r_ij.min(), r_ji.min()))

    thick = min(best, min_tp)
    return {
        "thickness": thick,
        "min_local_radius": best,
        "min_tp_radius": min_tp if np.isfinite(min_tp) else None,
        "max_tube_diameter": 2.0 * thick,
    }


def tight_spots(
    knot: FourierKnot | FourierLink,
    slack: float = 1.35,
    samples: int = 1024,
    max_spots: int = 12,
) -> list[dict]:
    """The places that limit (or nearly limit) the tube: every location
    whose GM radius is within `slack` of the curve's thickness.

    Returns [{'xyz': [x,y,z], 'radius': r, 'kind': 'turn'|'gap'}, ...],
    tightest first, de-duplicated so markers don't pile up on one feature.
    'turn' = local bending (open the turn to improve); 'gap' = two
    passages too close (more depth/width to improve).
    """
    link = as_link(knot)
    P, T, comp_id, idx, counts = _sampled(link, samples)
    ds_comp = np.array([c.total_length() for c in link.components]) / np.array(counts)
    thick = thickness(link, samples)["thickness"]
    limit = slack * thick

    cand: list[tuple[float, np.ndarray, str]] = []

    # local curvature spots
    tloc = np.linspace(0.0, TAU, 2048, endpoint=False)
    for comp in link.components:
        kap = comp.curvature(tloc)
        r = 1.0 / np.maximum(kap, 1e-12)
        hot = r < limit
        if not hot.any():
            continue
        pts = comp.eval(tloc)
        # local minima of radius within the hot region
        is_min = (r <= np.roll(r, 1)) & (r <= np.roll(r, -1)) & hot
        for i in np.flatnonzero(is_min):
            cand.append((float(r[i]), pts[i], "turn"))

    # tangent-point spots (doubled-back passages and cross-component)
    tree = cKDTree(P)
    pairs = tree.query_pairs(r=2.0 * limit, output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        same = comp_id[i] == comp_id[j]
        gap = np.abs(idx[i] - idx[j])
        n_arr = np.array(counts)[comp_id[i]]
        gap = np.minimum(gap, n_arr - gap)
        keep = ~(same & (gap <= 2))
        i, j, same, gap = i[keep], j[keep], same[keep], gap[keep]
        if len(i):
            r_ij, _ = tangent_point_radii(P, T, i, j)
            r_ji, _ = tangent_point_radii(P, T, j, i)
            r = np.minimum(r_ij, r_ji)
            hot = r < limit
            arc = gap * ds_comp[comp_id[i]]
            for k in np.flatnonzero(hot):
                mid = 0.5 * (P[i[k]] + P[j[k]])
                near_turn = same[k] and arc[k] <= np.pi * r[k] * 1.5
                cand.append((float(r[k]),
                             mid, "turn" if near_turn else "gap"))

    cand.sort(key=lambda c: c[0])
    spots: list[dict] = []
    for r, xyz, kind in cand:
        if any(np.linalg.norm(xyz - np.array(s["xyz"])) < 4.0 * thick
               for s in spots):
            continue
        spots.append({"xyz": [round(float(v), 2) for v in xyz],
                      "radius": round(r, 2), "kind": kind})
        if len(spots) >= max_spots:
            break
    return spots
