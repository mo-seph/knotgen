"""Aesthetic/styling transforms applied to a knot or link.

All transforms here stay in Fourier space, so they preserve exact symmetry
and keep derivatives/curvature analytic. Multi-component links are styled
jointly: one shared scale so the *design* hits the requested size, with
tightness reweighting applied per component.
"""

from __future__ import annotations

import numpy as np

from knotgen.fourier import FourierKnot
from knotgen.link import FourierLink, as_link

TIGHTNESS_ALPHA = 0.25  # strength of the harmonic reweighting per unit tightness

FLAT_RATIO = 0.30  # natural z/xy above this = a genuinely 3D conformation
DEFAULT_FLAT_DEPTH = 25.0  # mm, the 2.5D "crossing bump" depth


def floor_z(design: FourierKnot | FourierLink) -> FourierKnot | FourierLink:
    """Translate the design so its lowest point sits at z = 0 (e.g. so a
    wall-layout knot's flat parts lie on the mounting plane)."""
    import numpy as np

    from knotgen.fourier import TAU

    link = as_link(design)
    t = np.linspace(0.0, TAU, 4096, endpoint=False)
    z_min = min(float(c.eval(t)[:, 2].min()) for c in link.components)
    comps = []
    for c in link.components:
        b = c.b.copy()
        b[2, 0] -= z_min
        from dataclasses import replace

        comps.append(replace(c, a=c.a.copy(), b=b))
    shifted = FourierLink(components=comps, name=link.name, meta=dict(link.meta))
    if isinstance(design, FourierKnot):
        out = shifted.components[0]
        out.name = design.name
        out.meta = shifted.meta
        return out
    return shifted


def natural_aspect(knot: FourierKnot | FourierLink) -> float:
    """Natural z-extent / xy-diameter of the unstyled curve."""
    e = as_link(knot).extents()
    return e["z_extent"] / e["xy_diameter"] if e["xy_diameter"] > 0 else 0.0


def apply_style(
    knot: FourierKnot | FourierLink,
    *,
    width: float,
    breadth: float | None = None,
    depth: float | None = None,
    tightness: float = 0.0,
) -> FourierKnot | FourierLink:
    """Return a styled copy (same type as the input).

    width      target xy diameter of the whole design in mm (uniform in-plane
               scale — keeps n-fold symmetry exact)
    breadth    optional separate y extent in mm (deliberate symmetry break)
    depth      target z extent in mm (crossing separation). None = auto:
               25 mm for flat/2.5D embeddings; for genuinely 3D conformations
               (ideal knots — natural z/xy above 0.3) the natural proportion
               is preserved, because squashing them creates jagged near-cusps.
    tightness  in [-1, 1]: reweights harmonic j by exp(alpha*(j-1)).
               Positive sharpens lobes, negative smooths.
    """
    if not -1.0 <= tightness <= 1.0:
        raise ValueError(f"tightness must be in [-1, 1], got {tightness}")

    single = isinstance(knot, FourierKnot)
    aspect = natural_aspect(knot)
    depth_auto = depth is None
    if depth_auto:
        depth = width * aspect if aspect > FLAT_RATIO else DEFAULT_FLAT_DEPTH

    link = as_link(knot).centered()

    if tightness != 0.0:
        comps = []
        for k in link.components:
            j = np.arange(k.a.shape[1], dtype=float)
            expo = TIGHTNESS_ALPHA * tightness * (j - 1.0)
            # positive tightness must not blow up dense spectra (the ideal
            # embeddings carry ~256 harmonics whose tail is numerical noise):
            # cap the boost at 3x, and never amplify negligible harmonics
            expo = np.minimum(expo, np.log(3.0))
            w = np.exp(expo)
            amp = np.sqrt((k.a**2 + k.b**2).sum(axis=0))
            noise = amp < 1e-4 * max(float(amp.max()), 1e-30)
            w[noise & (w > 1.0)] = 1.0
            w[0] = 1.0
            comps.append(k.harmonic_filtered(w))
        link = FourierLink(components=comps, name=link.name, meta=link.meta)

    e = link.extents()
    if e["xy_diameter"] <= 0 or e["z_extent"] <= 0:
        raise ValueError(f"degenerate curve: extents {e}")

    s_xy = width / e["xy_diameter"]
    s_z = depth / e["z_extent"]
    link = link.scaled(s_xy, s_xy, s_z)

    if breadth is not None:
        e2 = link.extents()
        link = link.scaled(1.0, breadth / e2["y_extent"], 1.0)

    link.meta = dict(as_link(knot).meta)
    link.meta["style"] = {
        "width": width,
        "breadth": breadth,
        "depth": depth,
        "depth_auto": depth_auto,
        "natural_aspect": round(aspect, 4),
        "tightness": tightness,
    }
    link.name = knot.name

    if single:
        out = link.components[0]
        out.name = knot.name
        out.meta = link.meta
        return out
    return link
