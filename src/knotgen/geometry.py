"""Sampled-geometry analysis: crossings, clearance, bend radius.

All entry points accept a single FourierKnot or a multi-component
FourierLink. For links, strand clearance counts *cross-component* pairs
unconditionally — two components touching is just as fatal to a sweep as
one strand folding back on itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from knotgen.fourier import TAU, FourierKnot
from knotgen.link import FourierLink, as_link


@dataclass
class Crossing:
    """An xy-projection crossing between two strand passages."""

    xy: tuple[float, float]
    z_over: float
    z_under: float
    seg_over: int  # global dense-sample segment index (see crossings_xy)
    seg_under: int
    comp_over: int = 0
    comp_under: int = 0
    t_over: float = 0.0  # curve parameter of each passage, within its component
    t_under: float = 0.0

    @property
    def separation(self) -> float:
        return self.z_over - self.z_under


def max_curvature(
    knot: FourierKnot | FourierLink, samples: int = 4096
) -> tuple[float, float]:
    """(max curvature, t at max) over all components."""
    link = as_link(knot)
    t = np.linspace(0.0, TAU, samples, endpoint=False)
    best_k, best_t = -1.0, 0.0
    for comp in link.components:
        k = comp.curvature(t)
        i = int(np.argmax(k))
        if k[i] > best_k:
            best_k, best_t = float(k[i]), float(t[i])
    return best_k, best_t


def _sampled_components(
    link: FourierLink, total_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    """Arc-length samples of every component.

    Returns (points, comp_id per point, index-within-component per point,
    per-component sample counts, curve parameter t per point).
    """
    lengths = [k.total_length() for k in link.components]
    total_len = sum(lengths)
    counts = [max(256, int(round(total_samples * ln / total_len))) for ln in lengths]
    pts_list, comp_ids, idx_in_comp, t_list = [], [], [], []
    for ci, (comp, n) in enumerate(zip(link.components, counts)):
        t_vals, p = comp.sample_arclength(n)
        pts_list.append(p)
        comp_ids.append(np.full(n, ci))
        idx_in_comp.append(np.arange(n))
        t_list.append(t_vals)
    return (
        np.vstack(pts_list),
        np.concatenate(comp_ids),
        np.concatenate(idx_in_comp),
        counts,
        np.concatenate(t_list),
    )


def min_clearance(
    knot: FourierKnot | FourierLink, samples: int = 2048
) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimum distance between distinct strands.

    Same-component pairs within an along-curve exclusion window are ignored
    (that's local bending, covered by the bend-radius check); the window is
    pi / max_curvature capped at a quarter of the component length.
    Cross-component pairs always count.

    Returns (distance, point_a, point_b).
    """
    link = as_link(knot)
    pts, comp_id, idx, counts, _ = _sampled_components(link, samples)

    kappa_max, _ = max_curvature(link)
    w_idx = np.zeros(len(counts), dtype=int)
    for ci, (comp, n) in enumerate(zip(link.components, counts)):
        length = comp.total_length()
        window = min(np.pi / kappa_max, length / 4.0)
        w_idx[ci] = max(int(np.ceil(window / (length / n))), 1)

    tree = cKDTree(pts)
    k_query = min(int(2 * w_idx.max() + 8), len(pts))
    dists, idxs = tree.query(pts, k=k_query)

    best = np.inf
    best_pair = (0, 0)
    for i in range(len(pts)):
        for d, j in zip(dists[i, 1:], idxs[i, 1:]):
            if comp_id[i] == comp_id[j]:
                n = counts[comp_id[i]]
                gap = abs(int(idx[i]) - int(idx[j]))
                gap = min(gap, n - gap)
                if gap <= w_idx[comp_id[i]]:
                    continue
            if d < best:
                best = d
                best_pair = (i, j)
            break  # neighbours are sorted; first valid one is the min for i
    if not np.isfinite(best):
        # every neighbour of every point was in-window: brute force
        d2 = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        same = comp_id[:, None] == comp_id[None, :]
        gap = np.abs(idx[:, None] - idx[None, :])
        n_arr = np.array(counts)[comp_id]
        gap = np.minimum(gap, n_arr[:, None] - gap)
        excl = same & (gap <= w_idx[comp_id][:, None])
        np.fill_diagonal(excl, True)
        d2[excl] = np.inf
        i, j = np.unravel_index(np.argmin(d2), d2.shape)
        best, best_pair = float(d2[i, j]), (int(i), int(j))

    i, j = best_pair
    return float(best), pts[i], pts[j]


def crossings_xy(
    knot: FourierKnot | FourierLink, samples: int = 2048
) -> list[Crossing]:
    """Crossings of the xy projection, within and across components."""
    link = as_link(knot)
    pts, comp_id, idx, counts, t_all = _sampled_components(link, samples)
    n_total = len(pts)

    # segment i runs from point i to nxt[i] (wrapping within its component)
    offsets = np.cumsum([0] + counts[:-1])
    nxt = np.empty(n_total, dtype=int)
    for ci, n in enumerate(counts):
        o = offsets[ci]
        nxt[o : o + n] = o + (np.arange(n) + 1) % n

    p = pts[:, :2]
    a, b = p, p[nxt]
    d = b - a

    crossings: list[Crossing] = []
    for i in range(n_total):
        j = np.arange(i + 2, n_total)
        if len(j) == 0:
            continue
        # drop pairs adjacent within the same component (shared endpoint)
        adjacent = (nxt[j] == i) | (nxt[i] == j)
        j = j[~adjacent]
        if len(j) == 0:
            continue
        r = d[i]
        s = d[j]
        qp = a[j] - a[i]
        denom = r[0] * s[:, 1] - r[1] * s[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / denom
            uu = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / denom
        hit = (np.abs(denom) > 1e-12) & (tt >= 0) & (tt < 1) & (uu >= 0) & (uu < 1)
        for jj, ttt, uuu in zip(j[hit], tt[hit], uu[hit]):
            z_i = pts[i, 2] + ttt * (pts[nxt[i], 2] - pts[i, 2])
            z_j = pts[jj, 2] + uuu * (pts[nxt[jj], 2] - pts[jj, 2])
            t_i = float(t_all[i] + ttt * ((t_all[nxt[i]] - t_all[i]) % TAU))
            t_j = float(t_all[jj] + uuu * ((t_all[nxt[jj]] - t_all[jj]) % TAU))
            xy = tuple(a[i] + ttt * r)
            ci_i, ci_j = int(comp_id[i]), int(comp_id[jj])
            if z_i >= z_j:
                crossings.append(
                    Crossing(xy, float(z_i), float(z_j), i, int(jj), ci_i, ci_j, t_i, t_j)
                )
            else:
                crossings.append(
                    Crossing(xy, float(z_j), float(z_i), int(jj), i, ci_j, ci_i, t_j, t_i)
                )
    return crossings


def passage_sequence(
    knot: FourierKnot | FourierLink, samples: int = 2048
) -> dict[int, list[tuple[float, bool]]]:
    """Per component: crossing passages as (t, is_over), sorted along the strand."""
    passages: dict[int, list[tuple[float, bool]]] = {}
    for c in crossings_xy(knot, samples):
        passages.setdefault(c.comp_over, []).append((c.t_over, True))
        passages.setdefault(c.comp_under, []).append((c.t_under, False))
    for seq in passages.values():
        seq.sort()
    return passages


def is_alternating(knot: FourierKnot | FourierLink, samples: int = 2048) -> bool:
    """True if every strand alternates over/under along its length (a
    necessary condition for the diagram to be alternating)."""
    for seq in passage_sequence(knot, samples).values():
        overs = [o for _, o in seq]
        if len(overs) < 2:
            continue
        for a_, b_ in zip(overs, overs[1:] + overs[:1]):
            if a_ == b_:
                return False
    return True
