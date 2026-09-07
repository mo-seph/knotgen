"""Clearance relaxation: open up a knot so a fatter tube fits.

Opt-in post-styling step (--relax). Two forces, applied to arc-length
samples and re-fit to the Fourier representation each iteration:

  * repulsion between strand points closer than the target gap
    (tube diameter x safety), pushing tight passages apart;
  * bend relief where the curvature radius is below the tube radius,
    pushing the curve away from its centre of curvature.

Each iteration the design is uniformly rescaled back to its original
xy diameter, so the effect is "redistribute space at fixed size", and —
for knots with exact n-fold symmetry — the Fourier coefficients are
projected back onto the symmetry-allowed subspace, so a pentafoil stays
exactly 5-fold.

Topology safety: every step is capped at a quarter of the CURRENT minimum
strand gap, so strands can never cross through each other; the repulsion
only ever increases separation at the tight spots.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from knotgen.fourier import TAU, FourierKnot
from knotgen.link import FourierLink, as_link

CLEARANCE_SAFETY = 1.15  # target gap = tube * this
BEND_SAFETY = 1.15  # target bend radius = tube/2 * this
REPULSE_GAIN = 0.35
BEND_GAIN = 0.25


def _symmetry_projection(knot: FourierKnot, n: int) -> FourierKnot:
    """Project coefficients onto the exact n-fold-symmetric subspace."""
    j = np.arange(knot.a.shape[1])
    ax, ay, az = knot.a
    bx, by, bz = knot.b
    # w = x + iy in exponential form
    c_pos = 0.5 * ((bx + ay) + 1j * (by - ax))
    c_neg = 0.5 * ((bx - ay) + 1j * (by + ax))
    # dominant residue class of the w-frequencies
    power = np.zeros(n)
    for f, c in [(jj, c_pos[jj]) for jj in j[1:]] + [
        (-jj, c_neg[jj]) for jj in j[1:]
    ]:
        power[f % n] += abs(c) ** 2
    m = int(np.argmax(power))
    keep_pos = (j % n) == m
    keep_neg = ((-j) % n) == m
    c_pos = np.where(keep_pos, c_pos, 0.0)
    c_neg = np.where(keep_neg, c_neg, 0.0)
    c_pos[0] = 0.0  # no offset for a symmetric curve
    c_neg[0] = 0.0
    # back to sin/cos
    bx2 = (c_pos + c_neg).real
    ay2 = (c_pos - c_neg).real
    by2 = (c_pos + c_neg).imag
    ax2 = -(c_pos - c_neg).imag
    a = np.vstack([ax2, ay2, np.where(j % n == 0, az, 0.0)])
    b = np.vstack([bx2, by2, np.where(j % n == 0, bz, 0.0)])
    b[2, 0] = knot.b[2, 0]
    return replace(knot, a=a, b=b)


def _sample_all(link: FourierLink, per_comp: list[int]):
    ts, pts, comp_id, idx = [], [], [], []
    for ci, (comp, n) in enumerate(zip(link.components, per_comp)):
        t, p = comp.sample_arclength(n)
        ts.append(t)
        pts.append(p)
        comp_id.append(np.full(n, ci))
        idx.append(np.arange(n))
    return ts, pts, np.concatenate(comp_id), np.concatenate(idx)


def relax(
    design: FourierKnot | FourierLink,
    tube: float,
    iterations: int = 150,
    verbose: bool = False,
) -> tuple[FourierKnot | FourierLink, dict]:
    """Return (relaxed design, info). Sizes in mm; run AFTER apply_style."""
    from knotgen.geometry import max_curvature, min_clearance

    single = isinstance(design, FourierKnot)
    link = as_link(design)

    target_gap = tube * CLEARANCE_SAFETY
    target_bend = 0.5 * tube * BEND_SAFETY
    kappa_limit = 1.0 / target_bend

    width0 = link.extents()["xy_diameter"]
    gap0, _, _ = min_clearance(link)
    kap0, _ = max_curvature(link)

    # symmetry to preserve (single component only)
    sym = link.components[0].rotational_symmetry_order() if single else 1

    # sampling / refit resolution per component
    per_comp = []
    harmonics = []
    for comp in link.components:
        h = max(comp.n_harmonics, 24)
        per_comp.append(int(min(max(256, 2 * h + 64), 768)))
        harmonics.append(h)

    comps = list(link.components)
    done_at = iterations
    for it in range(iterations):
        work = FourierLink(components=comps, name=link.name, meta=link.meta)
        ts, pts_list, comp_id, idx = _sample_all(work, per_comp)
        P = np.vstack(pts_list)
        F = np.zeros_like(P)

        # exclusion window per comp: half-turn at the current tightest bend
        kmax, _ = max_curvature(work)
        w_idx = []
        for ci, comp in enumerate(comps):
            L = comp.total_length()
            window = min(np.pi / kmax, L / 4.0)
            w_idx.append(max(int(np.ceil(window / (L / per_comp[ci]))), 1))

        # repulsion between close strand points
        tree = cKDTree(P)
        pairs = tree.query_pairs(r=target_gap, output_type="ndarray")
        if len(pairs):
            i, jj = pairs[:, 0], pairs[:, 1]
            same = comp_id[i] == comp_id[jj]
            gap_idx = np.abs(idx[i] - idx[jj])
            n_arr = np.array([per_comp[c] for c in comp_id[i]])
            gap_idx = np.minimum(gap_idx, n_arr - gap_idx)
            w_arr = np.array([w_idx[c] for c in comp_id[i]])
            keep = ~(same & (gap_idx <= w_arr))
            i, jj = i[keep], jj[keep]
            if len(i):
                d = P[i] - P[jj]
                dist = np.linalg.norm(d, axis=1)
                dist = np.maximum(dist, 1e-9)
                push = (REPULSE_GAIN * (target_gap - dist) / dist)[:, None] * d
                np.add.at(F, i, push)
                np.add.at(F, jj, -push)

        # bend relief: push away from the centre of curvature where too tight
        off = 0
        for ci, comp in enumerate(comps):
            t = ts[ci]
            d1 = comp.deriv(t, 1)
            d2 = comp.deriv(t, 2)
            sp2 = np.sum(d1 * d1, axis=1, keepdims=True)
            T = d1 / np.sqrt(sp2)
            kv = (d2 - np.sum(d2 * T, axis=1, keepdims=True) * T) / sp2
            kappa = np.linalg.norm(kv, axis=1)
            excess = np.maximum(kappa - kappa_limit, 0.0)
            hot = excess > 0
            if hot.any():
                khat = kv[hot] / kappa[hot, None]
                F[off:off + per_comp[ci]][hot] -= (
                    BEND_GAIN * target_bend**2 * excess[hot, None] * khat
                )
            off += per_comp[ci]

        # topology-safe step cap: never move more than 1/4 of the current gap
        cur_gap, _, _ = min_clearance(work)
        max_move = np.linalg.norm(F, axis=1).max()
        if max_move < 1e-9:
            done_at = it
            break
        cap = max(0.25 * cur_gap, 0.05)
        if max_move > cap:
            F *= cap / max_move

        # move, refit, re-symmetrize, re-normalize size
        off = 0
        new_comps = []
        for ci, comp in enumerate(comps):
            moved = P[off:off + per_comp[ci]] + F[off:off + per_comp[ci]]
            off += per_comp[ci]
            k = FourierKnot.from_samples(
                moved, n_harmonics=min(harmonics[ci], per_comp[ci] // 2 - 1),
                name=comp.name, meta=comp.meta,
            )
            if sym > 1:
                k = _symmetry_projection(k, sym)
            new_comps.append(k)
        work = FourierLink(components=new_comps, name=link.name, meta=link.meta)
        s = width0 / work.extents()["xy_diameter"]
        comps = work.scaled(s, s, s).components

        if it % 10 == 0 or it == iterations - 1:
            check = FourierLink(components=comps, name=link.name, meta=link.meta)
            g, _, _ = min_clearance(check)
            km, _ = max_curvature(check)
            if verbose:
                print(f"    relax {it:3d}: gap {g:6.1f} mm, bend r {1/km:6.1f} mm")
            if g >= 0.995 * target_gap and 1.0 / km >= 0.995 * target_bend:
                done_at = it + 1
                break

    result = FourierLink(components=comps, name=link.name, meta=dict(link.meta))
    gap1, _, _ = min_clearance(result)
    kap1, _ = max_curvature(result)
    result.meta["relaxed"] = {
        "tube": tube, "iterations": done_at,
        "gap_before": round(float(gap0), 2), "gap_after": round(float(gap1), 2),
        "bend_before": round(1.0 / kap0, 2), "bend_after": round(1.0 / kap1, 2),
    }
    info = result.meta["relaxed"]
    if single:
        out = result.components[0]
        out.name = design.name
        out.meta = result.meta
        return out, info
    return result, info
