"""Multi-component designs: a link is one or more closed Fourier curves.

`FourierKnot` stays the single-curve workhorse; `FourierLink` is the thin
container the pipeline passes around. Single knots are 1-component links.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from knotgen.fourier import TAU, FourierKnot


def as_link(obj: "FourierLink | FourierKnot") -> "FourierLink":
    if isinstance(obj, FourierLink):
        return obj
    return FourierLink(components=[obj], name=obj.name, meta=dict(obj.meta))


@dataclass
class FourierLink:
    components: list[FourierKnot]
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a link needs at least one component")

    @property
    def n_components(self) -> int:
        return len(self.components)

    # ------------------------------------------------------------ transforms

    def centered(self) -> FourierLink:
        """Translate so the mean of the component centres is at the origin.

        Deliberately unweighted: an unweighted mean commutes with anisotropic
        scaling (weights based on arc length would not), so centering is
        idempotent through apply_style.
        """
        means = np.array([k.b[:, 0] for k in self.components])
        centroid = means.mean(axis=0)
        comps = []
        for k in self.components:
            b = k.b.copy()
            b[:, 0] -= centroid
            comps.append(replace(k, a=k.a.copy(), b=b))
        return replace(self, components=comps)

    def scaled(self, sx: float, sy: float, sz: float) -> FourierLink:
        return replace(self, components=[k.scaled(sx, sy, sz) for k in self.components])

    # -------------------------------------------------------------- measures

    def extents(self, samples: int = 4096) -> dict[str, float]:
        """Joint extents of the centred union (same keys as FourierKnot)."""
        t = np.linspace(0.0, TAU, samples, endpoint=False)
        pts = np.vstack([k.eval(t) for k in self.centered().components])
        r_xy = np.hypot(pts[:, 0], pts[:, 1])
        span = pts.max(axis=0) - pts.min(axis=0)
        return {
            "xy_diameter": 2.0 * float(r_xy.max()),
            "x_extent": float(span[0]),
            "y_extent": float(span[1]),
            "z_extent": float(span[2]),
        }

    def total_length(self) -> float:
        return float(sum(k.total_length() for k in self.components))

    def rotational_symmetry_order(self, max_order: int = 24) -> int:
        """Symmetry of the design as a whole (the union of components).

        Single component: exact frequency-space test. Multi-component:
        numeric set test — rotate all sampled points by 2*pi/n and check the
        union maps onto itself (components may permute).
        """
        if self.n_components == 1:
            return self.components[0].rotational_symmetry_order(max_order=max_order)

        from scipy.spatial import cKDTree

        t = np.linspace(0.0, TAU, 512, endpoint=False)
        pts = np.vstack([k.eval(t) for k in self.centered().components])
        scale = np.linalg.norm(pts, axis=1).max()
        # tolerance: a fraction of typical sample spacing
        spacing = self.total_length() / len(pts)
        tol = max(0.75 * spacing, 1e-9 * scale)
        tree = cKDTree(pts)
        for n in range(max_order, 1, -1):
            c, s = np.cos(TAU / n), np.sin(TAU / n)
            rot = pts @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            d, _ = tree.query(rot)
            if d.max() < tol:
                return n
        return 1
