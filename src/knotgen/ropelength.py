"""Fixed-rope relaxation: inflate the TUBE, never the rope.

The earlier relaxers bought clearance by manufacturing arc length —
pushing strands apart at an ambitious working target lengthens the curve,
and surplus rope confined to a slab has exactly one place to go: wiggles
and coils. The max-tube metric rewarded it, so numbers improved while
shapes got worse.

This one takes the ideal-knot algorithms' discipline (SONO, ridgerunner):
the rope is inextensible. Each iteration

  1. finds every pair whose Gonzalez-Maddocks tangent-point radius is
     below the current working radius and pushes them apart (moving rope),
  2. applies the depth budget and a little fairing,
  3. PROJECTS the curve back onto its length budget with curve-shortening
     steps — which shrink high-curvature features first, i.e. wiggles are
     the slack that gets spent — then refits, re-symmetrises and restores
     the footprint,

and inflates the working tube diameter by 3% every time the design is
clean at the current one. When it stalls, it grants a little more rope
(up to the budget: the knot's natural length plus a small slack, for
rounding crushed kinks) and tries again; when rope and patience are both
spent, it stops. The result is the fattest tube this knot's rope can carry
in this slab — with no coils, by construction, because there is no rope
to make them from.

The requested tube only enters scoring and reporting (the best state
returned is the one that best satisfies the request, ties broken by the
absolute achievable tube), so the dynamics are request-independent.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from knotgen.fourier import FourierKnot
from knotgen.link import FourierLink, as_link
from knotgen.relax import (
    BEND_SAFETY,
    CLEARANCE_SAFETY,
    DEPTH_GAIN,
    _sample_all,
    _smooth_circular,
    _symmetry_projection,
    spectral_polish,
)

GM_GAIN = 1.0  # push per mm of tangent-point-radius deficit (SONO resolves overlaps fully)
FAIR_GAIN = 0.12  # gentle always-on curve-shortening (wobble hygiene)
INFLATE = 1.03  # working tube growth per clean iteration
ROPE_GRANT = 1.02  # rope allowance growth when stuck (up to the budget)
SHRINK_STEP = 0.3  # Laplacian step used by the length projection
KEEP_GAIN = 1.5  # keep_diagram spring per unit weight: 1 = firm (~halves xy drift, costs tube)
STIFF_GAIN = 0.05  # elastica flow per unit of (stiffness - 1), as a fraction of the strand gap
PRESSURE = 1.25  # overlaps are resolved against a radius this much above the
#                  working one: the rope cap makes high pressure safe (it can
#                  only move rope, never make it), and low pressure stalls


def _polyline_length(pts: np.ndarray) -> float:
    d = np.roll(pts, -1, axis=0) - pts
    return float(np.linalg.norm(d, axis=1).sum())


def _laplacian(pts: np.ndarray) -> np.ndarray:
    return 0.5 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0)) - pts


def _shorten_to(blocks: list[np.ndarray], allowed: float, max_steps: int = 120) -> tuple[list[np.ndarray], int]:
    """Curve-shortening steps on the sampled components until their total
    polyline length is within `allowed`. High-curvature features shrink
    fastest, so this spends wiggles before it spends lobes."""
    steps = 0
    total = sum(_polyline_length(b) for b in blocks)
    while total > allowed and steps < max_steps:
        blocks = [b + SHRINK_STEP * _laplacian(b) for b in blocks]
        total = sum(_polyline_length(b) for b in blocks)
        steps += 1
    return blocks, steps


def relax_fixed_rope(
    design: FourierKnot | FourierLink,
    tube: float,
    iterations: int = 150,
    max_depth: float | None = None,
    push: bool = False,
    verbose: bool = False,
    rope_budget: float | None = None,
    rope_slack: float = 0.10,
    polish_floor: float = 0.995,
    snapshot: Callable[[int, FourierLink, dict], None] | None = None,
    snapshot_every: int = 5,
    hops: int = 0,
    seed: int = 0,
    stiffness=None,
    inflate=None,
    keep_diagram: float = 0.0,
) -> tuple[FourierKnot | FourierLink, dict]:
    """Return (relaxed design, info). See the module docstring.

    Aesthetics (all per component, lists in component order):
      rope_slack may be a list — a component on a tight rope budget stays a
        clean ring while the others do the deforming. Per-component budgets
        are SOFT (a few percent): at a fixed footprint a strand can only
        shed length by changing shape, and the width re-normalisation
        partly undoes each shrink — on a fully symmetric link they cannot
        differentiate at all; stiffness and inflate are the strong
        per-component levers. The total budget is what's firm;
      stiffness: resistance to bending (1 = default): a stiff component
        follows an elastica flow — curvature flow at fixed length — so it
        rounds toward the circle of its own length and responds less to
        pushes; the others move around it;
      inflate: tube scale per component (1 = the base tube): a fatter rope
        demands more clearance from its neighbours and pushes them aside;
      keep_diagram: a spring (0 = off, 1 = firm) holding each strand's xy
        near where it started, so the drawn presentation survives while z
        and roundness adapt.

    rope_budget: the most rope the design may use (mm); default = its
    current length. rope_slack: extra fraction on top of the budget, for
    rounding crushed kinks (default 0.10). snapshot(it, link, metrics) is
    called every `snapshot_every` iterations and once at the end.

    hops: basin hopping. The pushes are all perpendicular to the strand, so
    a passage can never SLIDE along to relocate a crossing — the optimiser
    is local. Each hop restarts from the best state with a random
    low-harmonic (whole-arc scale, never wiggle scale) perturbation,
    capped below half the strand gap so strands cannot pass through each
    other, and the run continues; the best state is kept only if the hop
    beats it. Costs iterations: pair with --relax-max.
    """
    from knotgen.geometry import max_curvature, min_clearance
    from knotgen.gm import tangent_point_radii

    single = isinstance(design, FourierKnot)
    link = as_link(design)
    target_gap = tube * CLEARANCE_SAFETY
    target_bend = 0.5 * tube * BEND_SAFETY
    width0 = link.extents()["xy_diameter"]
    sym = link.components[0].rotational_symmetry_order() if single else 1

    per_comp, harmonics = [], []
    for comp in link.components:
        h = min(max(comp.n_harmonics, 24), 128)
        per_comp.append(int(min(max(256, 2 * h + 64), 768)))
        harmonics.append(h)

    comps = list(link.components)
    if push:
        iterations = max(iterations, 600)
    patience = 40 if push else 20

    def total_length(cs) -> float:
        return sum(c.total_length() for c in cs)

    n_c = len(comps)
    slack = list(rope_slack) if isinstance(rope_slack, (list, tuple)) else [float(rope_slack)] * n_c
    stiff = [float(v) for v in (stiffness or [1.0] * n_c)]
    infl = [float(v) for v in (inflate or [1.0] * n_c)]
    lengths0 = [c.total_length() for c in comps]
    length0 = sum(lengths0)
    # rope budgets are per component: the input length plus slack (rounding
    # crushed kinks needs a little); rope_budget, when given, only ever CAPS
    # the total — a crushed ideal knot's natural length is far more rope
    # than its flat self should be allowed, and surplus rope is wiggle
    rope_max_c = [L * (1.0 + sl) for L, sl in zip(lengths0, slack)]
    if rope_budget is not None:
        cap_total = max(rope_budget, length0)
        if sum(rope_max_c) > cap_total:
            f = cap_total / sum(rope_max_c)
            rope_max_c = [max(L, r * f) for L, r in zip(lengths0, rope_max_c)]
    rope_max = sum(rope_max_c)
    rope_allow_c = list(lengths0)  # grow toward rope_max_c only when stuck
    rope_allow = sum(rope_allow_c)
    orig_comps = list(comps)  # for keep_diagram

    def measure(cs):
        cand = FourierLink(components=list(cs), name=link.name, meta=link.meta)
        g, _, _ = min_clearance(cand)
        km, _ = max_curvature(cand)
        est = min(g / CLEARANCE_SAFETY, 2.0 * (1.0 / km) / BEND_SAFETY)
        score = min(g / target_gap, (1.0 / km) / target_bend)
        return score, g, 1.0 / km, est

    score, gap, bend, est = measure(comps)
    gap0, bend0 = gap, bend
    best = {"key": (round(min(score, 1.0), 4), round(est, 2)),
            "comps": list(comps), "score": score, "gap": gap,
            "bend": bend, "est": est}
    d_work = est  # working tube diameter: clean here -> inflate
    stuck = 0
    grants = 0
    done_at = iterations

    def emit(it: int, cs, extra: dict) -> None:
        if snapshot is None:
            return
        snapshot(it, FourierLink(components=list(cs), name=link.name, meta=link.meta),
                 {"iteration": it, "gap": round(gap, 2), "bend": round(bend, 2),
                  "fits": round(est, 2), "d_work": round(d_work, 2),
                  "length": round(total_length(cs), 1),
                  "rope_allow": round(rope_allow, 1), **extra})

    emit(0, comps, {"phase": "start"})
    rng = np.random.default_rng(seed)
    hops_done = 0

    from knotgen.geometry import linking_numbers

    linking0 = linking_numbers(link) if len(link.components) > 1 else None

    def perturb(cs, max_disp: float):
        """Random whole-arc kick whose largest POINT displacement is capped
        at max_disp (coefficient noise summed over harmonics and axes moves
        a point far more than any single coefficient — capping the
        coefficients was the bug that let one strand pull through
        another). Retries with a softer kick if the linking numbers change
        anyway, and gives up on the hop rather than break topology."""
        t = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
        for _attempt in range(6):
            out = []
            for c in cs:
                a, b = c.a.copy(), c.b.copy()
                for j in range(1, min(4, a.shape[1])):
                    a[:, j] += rng.normal(0.0, 1.0 / j, 3)
                    b[:, j] += rng.normal(0.0, 1.0 / j, 3)
                k = FourierKnot(a=a, b=b, name=c.name, meta=c.meta)
                if sym > 1:
                    k = _symmetry_projection(k, sym)
                disp = np.linalg.norm(k.eval(t) - c.eval(t), axis=1).max()
                if disp > 1e-12:
                    f = max_disp / disp
                    k = FourierKnot(a=c.a + (k.a - c.a) * f, b=c.b + (k.b - c.b) * f,
                                    name=c.name, meta=c.meta)
                out.append(k)
            w = FourierLink(components=out, name=link.name, meta=link.meta)
            sc = width0 / w.extents()["xy_diameter"]
            cand = w.scaled(sc, sc, sc).components
            if linking0 is None or linking_numbers(
                    FourierLink(components=cand, name=link.name, meta=link.meta)) == linking0:
                return cand
            max_disp *= 0.5
        return list(cs)

    for it in range(iterations):
        r_work = 0.5 * d_work * BEND_SAFETY * PRESSURE  # GM radius to push for
        work = FourierLink(components=comps, name=link.name, meta=link.meta)
        ts, pts_list, comp_id, idx = _sample_all(work, per_comp)
        P = np.vstack(pts_list)
        T = np.vstack([
            (lambda d1: d1 / np.linalg.norm(d1, axis=1, keepdims=True))(
                comp.deriv(ts[ci], 1))
            for ci, comp in enumerate(comps)
        ])

        # 1. overlap removal: tangent-point radius below the working radius
        F_gm = np.zeros_like(P)
        tree = cKDTree(P)
        pairs = tree.query_pairs(r=2.0 * r_work, output_type="ndarray")
        n_viol = 0
        if len(pairs):
            i, jj = pairs[:, 0], pairs[:, 1]
            same = comp_id[i] == comp_id[jj]
            gap_idx = np.abs(idx[i] - idx[jj])
            n_arr = np.array([per_comp[c] for c in comp_id[i]])
            gap_idx = np.minimum(gap_idx, n_arr - gap_idx)
            keep = ~(same & (gap_idx <= 2))
            i, jj = i[keep], jj[keep]
            infl_pt = np.array(infl)[comp_id]
            stiff_pt = np.array(stiff)[comp_id]
            for x, y in ((i, jj), (jj, i)):
                r_tp, u = tangent_point_radii(P, T, x, y)
                # a fatter component demands more room: mean of the pair's scales
                r_req = r_work * 0.5 * (infl_pt[x] + infl_pt[y])
                viol = r_tp < r_req
                n_viol += int(viol.sum())
                if viol.any():
                    f = (GM_GAIN * (r_req[viol] - r_tp[viol]))[:, None] * u[viol]
                    # stiff components respond less to pushes (the partner
                    # still gets its full push, so the pair still separates)
                    np.add.at(F_gm, y[viol], f / np.minimum(stiff_pt[y[viol]], 3.0)[:, None])
                    np.add.at(F_gm, x[viol], -f / np.minimum(stiff_pt[x[viol]], 3.0)[:, None])

        # 2. depth budget (soft) and gentle fairing (unsmoothed, so it can
        #    act at wiggle wavelengths)
        F_other = np.zeros_like(P)
        if max_depth is not None:
            zc = 0.5 * (P[:, 2].min() + P[:, 2].max())
            dz = P[:, 2] - zc
            over_z = np.abs(dz) - 0.49 * max_depth  # use the whole slab
            F_other[:, 2] -= DEPTH_GAIN * np.sign(dz) * np.maximum(over_z, 0.0)
        off = 0
        F = np.zeros_like(P)
        for ci in range(len(comps)):
            n_i = per_comp[ci]
            ds = comps[ci].total_length() / n_i
            block = slice(off, off + n_i)
            lap = _laplacian(P[block])
            F[block] = (_smooth_circular(F_gm[block], max((0.75 * r_work) / ds, 1.5))
                        + F_other[block]
                        + FAIR_GAIN * lap)
            if stiff[ci] > 1.0:
                # resistance to bending as an elastica flow: curvature flow
                # strong enough to matter (normalised to the strand gap),
                # with its shrink undone below so the strand rounds toward
                # the circle of its own length instead of collapsing
                lmax = float(np.linalg.norm(lap, axis=1).max())
                if lmax > 1e-12:
                    F[block] += (stiff[ci] - 1.0) * STIFF_GAIN * gap * lap / lmax
            if keep_diagram > 0.0:
                # spring toward where this fraction of the strand started (xy
                # only): the drawn diagram survives, z and roundness adapt
                p0 = orig_comps[ci].eval(ts[ci])
                F[block, :2] += KEEP_GAIN * keep_diagram * (p0[:, :2] - P[block, :2])
            off += n_i

        # topology-safe step cap
        max_move = np.linalg.norm(F, axis=1).max()
        cap = max(0.25 * gap, 0.05)
        if max_move > cap:
            F *= cap / max_move
        moved = P + F

        # 3. length projection: the rope is inextensible
        blocks = []
        off = 0
        for ci in range(len(comps)):
            blocks.append(moved[off:off + per_comp[ci]])
            off += per_comp[ci]
        # elastica length restoration for stiff components: rounding must
        # not shrink them (scale about their own centroid, gently)
        for ci in range(len(comps)):
            if stiff[ci] > 1.0:
                L_pre = _polyline_length(pts_list[ci])
                L_now = _polyline_length(blocks[ci])
                if L_now < L_pre * 0.998:
                    f = min(L_pre / L_now, 1.03)
                    cen = blocks[ci].mean(axis=0)
                    blocks[ci] = cen + (blocks[ci] - cen) * f
        pre_shrink = [b.copy() for b in blocks]
        shrink_steps = 0
        for ci in range(len(comps)):
            allowed_i = _polyline_length(pts_list[ci]) * (
                rope_allow_c[ci] / max(comps[ci].total_length(), 1e-9))
            [blocks[ci]], st = _shorten_to([blocks[ci]], allowed_i)
            shrink_steps = max(shrink_steps, st)
        # the shrink is a topology-unaware flow: cap its displacement like
        # the forces (a quarter of the current strand gap) so it can never
        # drag a strand through another; any leftover length is spent on
        # later iterations
        if shrink_steps:
            disp = max(np.linalg.norm(b - b0, axis=1).max()
                       for b, b0 in zip(blocks, pre_shrink))
            if disp > cap:
                blocks = [b0 + (b - b0) * (cap / disp)
                          for b, b0 in zip(blocks, pre_shrink)]

        new_comps = []
        for ci, comp in enumerate(comps):
            k = FourierKnot.from_samples(
                blocks[ci], n_harmonics=min(harmonics[ci], per_comp[ci] // 2 - 1),
                name=comp.name, meta=comp.meta,
            )
            if sym > 1:
                k = _symmetry_projection(k, sym)
            new_comps.append(k)
        work = FourierLink(components=new_comps, name=link.name, meta=link.meta)
        s = width0 / work.extents()["xy_diameter"]
        comps = work.scaled(s, s, s).components
        if max_depth is not None:
            check = FourierLink(components=comps, name=link.name, meta=link.meta)
            z = check.extents()["z_extent"]
            if z > max_depth:
                comps = check.scaled(1.0, 1.0, max_depth / z).components

        # the footprint rescale above can re-inflate length after the
        # projection (a shrinking xy diameter scales everything back up),
        # which crept a zero-slack component up 9% over 60 iterations — so
        # re-project any component now over its allowance, once, in the
        # final frame of this iteration
        over = [ci for ci in range(len(comps))
                if comps[ci].total_length() > rope_allow_c[ci] * 1.003]
        if over:
            fixed = list(comps)
            for ci in over:
                _, pts_i = comps[ci].sample_arclength(per_comp[ci])
                allowed_i = _polyline_length(pts_i) * (
                    rope_allow_c[ci] / comps[ci].total_length())
                [blk], _ = _shorten_to([pts_i], allowed_i)
                k = FourierKnot.from_samples(
                    blk, n_harmonics=min(harmonics[ci], per_comp[ci] // 2 - 1),
                    name=comps[ci].name, meta=comps[ci].meta)
                if sym > 1:
                    k = _symmetry_projection(k, sym)
                fixed[ci] = k
            comps = fixed
            w = FourierLink(components=comps, name=link.name, meta=link.meta)
            s2 = width0 / w.extents()["xy_diameter"]
            comps = w.scaled(s2, s2, s2).components

        # 4. measure, inflate or grant rope, track the best
        score, gap, bend, est = measure(comps)
        key = (round(min(score, 1.0), 4), round(est, 2))
        if key > best["key"]:
            best = {"key": key, "comps": list(comps), "score": score,
                    "gap": gap, "bend": bend, "est": est}
        if est >= 0.97 * d_work:
            d_work = max(d_work, est) * INFLATE
            stuck = 0
        else:
            stuck += 1
        # rope grants are cheap and gradual: decide them on a shorter clock
        if (stuck >= max(8, patience // 2) and hops_done >= hops
                and rope_allow < rope_max * 0.999):
            rope_allow_c = [min(r * ROPE_GRANT, m) for r, m in zip(rope_allow_c, rope_max_c)]
            rope_allow = sum(rope_allow_c)
            grants += 1
            stuck = 0
            phase = "grant-rope"
        phase = "inflate"
        if stuck >= patience:
            if hops_done < hops:
                # basin hop first when asked for: a whole-arc kick from the
                # best state (under half the gap, so no strand can cross
                # another in the kick) is a bigger move than +2% rope
                hops_done += 1
                comps = perturb(best["comps"], 0.4 * best["gap"])
                score, gap, bend, est = measure(comps)
                d_work = est
                stuck = 0
                phase = f"hop-{hops_done}"
                if verbose:
                    print(f"    hop {hops_done}/{hops}: kicked the best state "
                          f"(fits \u2300{best['est']:.1f}) to \u2300{est:.1f}")
            elif rope_allow < rope_max * 0.999:
                rope_allow_c = [min(r * ROPE_GRANT, m) for r, m in zip(rope_allow_c, rope_max_c)]
                rope_allow = sum(rope_allow_c)
                grants += 1
                stuck = 0
                phase = "grant-rope"
            else:
                done_at = it + 1
                emit(it + 1, comps, {"phase": "stop"})
                break

        if verbose and (it % 5 == 4 or it == iterations - 1):
            print(f"    relax {it:3d}: gap {gap:6.1f} mm, bend r {bend:6.1f} mm "
                  f"(fits ⌀{est:.1f}, working ⌀{d_work:.1f}, "
                  f"rope {total_length(comps):.0f}/{rope_allow:.0f} mm, "
                  f"{n_viol} overlaps, {shrink_steps} shrink steps)")
        if (it + 1) % snapshot_every == 0:
            emit(it + 1, comps, {"phase": phase})

    result_comps = list(best["comps"])
    polished, polish_info = spectral_polish(
        FourierLink(components=result_comps, name=link.name, meta=link.meta),
        max_depth=max_depth, floor=polish_floor,
    )
    if polish_info["passes"] > 0:
        result_comps = list(polished.components)
        _, gap1, bend1, _ = measure(result_comps)
    else:
        gap1, bend1 = best["gap"], best["bend"]
    if verbose and polish_info["passes"] > 0:
        print(f"    polish: {polish_info['passes']} passes, wobble "
              f"-{100 * polish_info['wobble_reduction']:.0f}%, "
              f"fits ⌀{polish_info['est_before']} -> "
              f"⌀{polish_info['est_after']}")

    topology_ok = True
    if linking0 is not None:
        topology_ok = linking_numbers(
            FourierLink(components=result_comps, name=link.name, meta=link.meta)) == linking0
        if not topology_ok and verbose:
            print("    ! linking numbers changed during relax — a strand passed "
                  "through another; discard this result")
    result = FourierLink(components=result_comps, name=link.name, meta=dict(link.meta))
    result.meta["relaxed"] = {
        "tube": tube, "iterations": done_at, "method": "sono",
        "topology_ok": topology_ok,
        "gap_before": round(float(gap0), 2), "gap_after": round(float(gap1), 2),
        "bend_before": round(float(bend0), 2), "bend_after": round(float(bend1), 2),
        "length_before": round(length0, 1),
        "length_after": round(total_length(result_comps), 1),
        "rope_budget": round(rope_max, 1), "rope_grants": grants,
        "hops": hops_done,
        "stiffness": stiff, "inflate": infl, "keep_diagram": keep_diagram,
        "converged": bool(gap1 >= 0.99 * target_gap and bend1 >= 0.99 * target_bend),
        "max_tube_est": round(min(2.0 * bend1 / 1.1, gap1 / 1.05), 1),
        "polish": polish_info,
    }
    info = result.meta["relaxed"]
    emit(done_at, result_comps, {"phase": "final"})
    if single:
        out = result.components[0]
        out.name = design.name
        out.meta = result.meta
        return out, info
    return result, info
