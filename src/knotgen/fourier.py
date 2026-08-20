"""Fourier-series representation of closed space curves.

A knot is stored as one truncated Fourier series per coordinate:

    r_c(t) = sum_j  b[c, j] * cos(j t) + a[c, j] * sin(j t),   t in [0, 2*pi)

with c in {x, y, z} and harmonic index j = 0..N (a[:, 0] is unused since
sin(0) == 0; b[:, 0] is the constant offset).

This representation is what makes the tool work:
  * derivatives are analytic (termwise), so curvature is exact,
  * n-fold symmetry is preserved exactly by uniform xy scaling and by
    per-harmonic reweighting,
  * "depth" is just a z-amplitude scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

TAU = 2.0 * np.pi


@dataclass
class FourierKnot:
    """A closed 3D curve defined by Fourier coefficients.

    a: sin coefficients, shape (3, N+1) — a[:, 0] must be zero.
    b: cos coefficients, shape (3, N+1) — b[:, 0] is the constant term.
    """

    a: np.ndarray
    b: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.a = np.atleast_2d(np.asarray(self.a, dtype=float))
        self.b = np.atleast_2d(np.asarray(self.b, dtype=float))
        if self.a.shape != self.b.shape or self.a.shape[0] != 3:
            raise ValueError(
                f"coefficient arrays must both be (3, N+1); got a{self.a.shape} b{self.b.shape}"
            )

    @classmethod
    def from_samples(
        cls,
        points: np.ndarray,
        n_harmonics: int = 128,
        name: str = "",
        meta: dict | None = None,
    ) -> FourierKnot:
        """Build a FourierKnot from uniformly-parametrized samples of a
        closed curve (last point must NOT repeat the first) via FFT."""
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        c = np.fft.fft(pts, axis=0) / n
        n_harmonics = min(n_harmonics, n // 2 - 1)
        a = np.zeros((3, n_harmonics + 1))
        b = np.zeros((3, n_harmonics + 1))
        b[:, 0] = c[0].real
        for j in range(1, n_harmonics + 1):
            b[:, j] = 2.0 * c[j].real  # cos coefficients
            a[:, j] = -2.0 * c[j].imag  # sin coefficients
        return cls(a=a, b=b, name=name, meta=meta or {})

    # ------------------------------------------------------------------ eval

    @property
    def n_harmonics(self) -> int:
        return self.a.shape[1] - 1

    def eval(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the curve at parameter values t. Returns (len(t), 3)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        j = np.arange(self.a.shape[1])
        jt = np.outer(t, j)  # (T, N+1)
        return np.sin(jt) @ self.a.T + np.cos(jt) @ self.b.T

    def deriv_coeffs(self, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Coefficients of the order-th derivative (termwise differentiation)."""
        a, b = self.a, self.b
        j = np.arange(a.shape[1])
        for _ in range(order):
            # d/dt [a sin(jt) + b cos(jt)] = (j a) cos(jt) + (-j b) sin(jt)
            a, b = -j * b, j * a
        return a, b

    def deriv(self, t: np.ndarray, order: int = 1) -> np.ndarray:
        """Evaluate the order-th derivative at t. Returns (len(t), 3)."""
        a, b = self.deriv_coeffs(order)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        j = np.arange(a.shape[1])
        jt = np.outer(t, j)
        return np.sin(jt) @ a.T + np.cos(jt) @ b.T

    def curvature(self, t: np.ndarray) -> np.ndarray:
        """Curvature kappa(t) = |r' x r''| / |r'|^3 (exact)."""
        d1 = self.deriv(t, 1)
        d2 = self.deriv(t, 2)
        cross = np.cross(d1, d2)
        speed = np.linalg.norm(d1, axis=1)
        return np.linalg.norm(cross, axis=1) / speed**3

    # ------------------------------------------------------------ transforms

    def scaled(self, sx: float, sy: float, sz: float) -> FourierKnot:
        """Anisotropic scale about the origin (constant term included)."""
        s = np.array([sx, sy, sz], dtype=float)[:, None]
        return replace(self, a=self.a * s, b=self.b * s)

    def centered(self) -> FourierKnot:
        """Drop the constant term so the curve is centred on the origin."""
        b = self.b.copy()
        b[:, 0] = 0.0
        return replace(self, a=self.a.copy(), b=b)

    def rotated_z(self, angle: float) -> FourierKnot:
        """Rotate the curve about the z axis (mixes x/y coefficient rows)."""
        c, s = np.cos(angle), np.sin(angle)
        a, b = self.a.copy(), self.b.copy()
        a[0], a[1] = c * self.a[0] - s * self.a[1], s * self.a[0] + c * self.a[1]
        b[0], b[1] = c * self.b[0] - s * self.b[1], s * self.b[0] + c * self.b[1]
        return replace(self, a=a, b=b)

    def harmonic_filtered(self, weights: np.ndarray) -> FourierKnot:
        """Multiply each harmonic j by weights[j] (weights[0] applies to the
        constant term; pass 1.0 there to leave centring alone)."""
        w = np.asarray(weights, dtype=float)
        if w.shape != (self.a.shape[1],):
            raise ValueError(f"weights must have shape ({self.a.shape[1]},)")
        return replace(self, a=self.a * w, b=self.b * w)

    # -------------------------------------------------------------- symmetry

    def rotational_symmetry_order(self, tol: float = 1e-6, max_order: int = 24) -> int:
        """Largest n such that the curve has exact n-fold rotational symmetry
        about the z axis (as a set, allowing a parameter shift).

        Works in frequency space: writing w = x + iy as a sum of e^{ift}
        terms, the curve is n-fold symmetric iff all its w-frequencies are
        congruent to a single residue m mod n with gcd(m, n) == 1, and all
        z-frequencies are multiples of n.
        """
        from math import gcd

        j = np.arange(self.a.shape[1])
        bx, by = self.b[0], self.b[1]
        ax, ay = self.a[0], self.a[1]
        # coefficient of e^{+ijt} and e^{-ijt} in w = x + iy
        c_pos = 0.5 * ((bx + ay) + 1j * (by - ax))
        c_neg = 0.5 * ((bx - ay) + 1j * (by + ax))
        scale = max(np.abs(c_pos).max(), np.abs(c_neg).max())
        if scale == 0:
            return 1
        w_freqs = [int(f) for f in j[1:][np.abs(c_pos[1:]) > tol * scale]]
        w_freqs += [-int(f) for f in j[1:][np.abs(c_neg[1:]) > tol * scale]]

        z_amp = np.hypot(self.a[2], self.b[2])
        z_scale = max(z_amp.max(), scale)
        z_freqs = [int(f) for f in j[1:][z_amp[1:] > tol * z_scale]]

        if not w_freqs:
            return 1
        for n in range(max_order, 1, -1):
            if any(f % n for f in z_freqs):
                continue
            residues = {f % n for f in w_freqs}
            if len(residues) == 1 and gcd(residues.pop(), n) == 1:
                return n
        return 1

    # -------------------------------------------------------------- measures

    def extents(self, samples: int = 4096) -> dict[str, float]:
        """Characteristic sizes of the (centred) curve.

        xy_diameter: 2 * max radial distance in the xy plane
        x_extent, y_extent, z_extent: bounding-box sizes per axis
        """
        t = np.linspace(0.0, TAU, samples, endpoint=False)
        p = self.centered().eval(t)
        r_xy = np.hypot(p[:, 0], p[:, 1])
        span = p.max(axis=0) - p.min(axis=0)
        return {
            "xy_diameter": 2.0 * float(r_xy.max()),
            "x_extent": float(span[0]),
            "y_extent": float(span[1]),
            "z_extent": float(span[2]),
        }

    def arclength_table(self, samples: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Dense (t, cumulative_arclength) table over one period.

        Returns (t of length samples+1 spanning [0, 2pi], s of same length,
        s[0] == 0, s[-1] == total length).
        """
        t = np.linspace(0.0, TAU, samples + 1)
        speed = np.linalg.norm(self.deriv(t, 1), axis=1)
        # trapezoidal cumulative integral
        ds = 0.5 * (speed[1:] + speed[:-1]) * np.diff(t)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        return t, s

    def total_length(self, samples: int = 4096) -> float:
        _, s = self.arclength_table(samples)
        return float(s[-1])

    def sample_arclength(
        self, n: int, samples: int = 4096
    ) -> tuple[np.ndarray, np.ndarray]:
        """n points equally spaced in arc length (closed: last point != first).

        Returns (t_values shape (n,), points shape (n, 3)).
        """
        t_dense, s = self.arclength_table(samples)
        s_targets = np.linspace(0.0, s[-1], n, endpoint=False)
        t_values = np.interp(s_targets, s, t_dense)
        return t_values, self.eval(t_values)
