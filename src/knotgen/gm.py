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
