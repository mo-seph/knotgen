"""Periodic cubic B-spline fit of a knot, in forms Fusion 360 can ingest.

We interpolate arc-length-spaced samples with a *uniform periodic* cubic
B-spline (C2 everywhere, including the seam), then also derive an exactly
equivalent *clamped* representation via knot insertion. The Fusion import
script tries: periodic NURBS -> clamped NURBS -> fitted spline points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from knotgen.fourier import FourierKnot

DEGREE = 3


@dataclass
class PeriodicNurbs:
    """Uniform periodic cubic B-spline through n arc-length samples.

    control_points: (n, 3), NOT wrapped (P_0 .. P_{n-1})
    The unclamped uniform knot vector is arange(-3, n+4); the curve domain
    is [0, n], and evaluation wraps control points periodically.
    """

    control_points: np.ndarray

    @property
    def n(self) -> int:
        return len(self.control_points)

    def wrapped(self) -> tuple[np.ndarray, np.ndarray]:
        """(knots, control_points) of the equivalent unclamped open B-spline."""
        cp = np.vstack([self.control_points, self.control_points[:DEGREE]])
        knots = np.arange(-DEGREE, self.n + DEGREE + 1, dtype=float)
        return knots, cp

    def bspline(self) -> BSpline:
        knots, cp = self.wrapped()
        return BSpline(knots, cp, DEGREE)

    def clamped(self) -> tuple[np.ndarray, np.ndarray]:
        """Exactly equivalent standard clamped cubic NURBS.

        Returns (knots, control_points): knots start/end with multiplicity
        degree+1 at 0 and n; control point count = len(knots) - degree - 1.
        """
        spl = self.bspline()
        for _ in range(DEGREE):
            spl = spl.insert_knot(0.0)
            spl = spl.insert_knot(float(self.n))
        t, c = spl.t, spl.c
        # keep exactly degree+1 copies of the boundary knots
        lo = np.searchsorted(t, 0.0, side="left")
        hi = np.searchsorted(t, float(self.n), side="right")
        knots = np.concatenate([t[lo:hi]])
        ctrl = c[lo : hi - DEGREE - 1]
        return knots, ctrl


def fit_periodic_points(points: np.ndarray) -> PeriodicNurbs:
    """Interpolating uniform periodic cubic B-spline through a closed loop of
    points (last point must NOT repeat the first).

    At integer parameters, a uniform cubic B-spline evaluates to
    (P_{i-1} + 4 P_i + P_{i+1}) / 6 — solve that cyclic system for P.
    """
    q = np.asarray(points, dtype=float)
    n = len(q)
    A = np.zeros((n, n))
    idx = np.arange(n)
    A[idx, idx] = 4.0 / 6.0
    A[idx, (idx - 1) % n] = 1.0 / 6.0
    A[idx, (idx + 1) % n] = 1.0 / 6.0
    control = np.linalg.solve(A, q)
    return PeriodicNurbs(control_points=control)


def fit_periodic(knot: FourierKnot, n_samples: int = 400) -> PeriodicNurbs:
    """Periodic cubic B-spline through arc-length samples of the knot."""
    _, q = knot.sample_arclength(n_samples)
    return fit_periodic_points(q)


def max_deviation(nurbs: PeriodicNurbs, knot: FourierKnot, dense: int = 65536) -> float:
    """Max distance from the fitted spline to the analytic curve (approx).

    Accuracy is limited by the density of the reference cloud: the reported
    value overestimates the true deviation by at most ~half the reference
    point spacing (a few hundredths of a mm at default settings).
    """
    u = np.linspace(0.0, nurbs.n, 4096, endpoint=False)
    spline_pts = nurbs.bspline()(u)
    t = np.linspace(0.0, 2 * np.pi, dense, endpoint=False)
    analytic = knot.eval(t)
    tree = cKDTree(analytic)
    d, _ = tree.query(spline_pts)
    return float(d.max())
