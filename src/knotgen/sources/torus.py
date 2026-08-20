"""Exact torus knots T(p, q) — and torus links when gcd(p, q) > 1.

Single component (gcd 1):

    x(t) = (1 + rho*cos(q t)) * cos(p t)
    y(t) = (1 + rho*cos(q t)) * sin(p t)
    z(t) =      rho*sin(q t)

expanded with product-to-sum identities into exact Fourier coefficients.
The curve has exact q-fold rotational symmetry about z for the usual lamp
look (T(2,3) = trefoil with 3 lobes, T(2,5) = pentafoil).

gcd(p, q) = d > 1 gives the d-component torus link: each component is the
(p/d, q/d) curve on the same torus, and component k is component 0 rotated
by -2*pi*k/q about z (which realises the tube-phase offset 2*pi*k/p).

rho = r/R is the native "tightness" control: small rho -> shallow lobes.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from knotgen.fourier import FourierKnot
from knotgen.link import FourierLink


def _rosette_xy(P: int, Q: int, rho: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of x=(1+rho*cos(Q s))cos(P s), y=... in (3, n+1) arrays."""
    a = np.zeros((3, n + 1))
    b = np.zeros((3, n + 1))
    # x = cos(Ps) + rho/2 * [cos((Q-P)s) + cos((Q+P)s)]
    b[0, P] += 1.0
    b[0, abs(Q - P)] += rho / 2.0
    b[0, Q + P] += rho / 2.0
    # y = sin(Ps) + rho/2 * [sin((P+Q)s) + sin((P-Q)s)]
    a[1, P] += 1.0
    a[1, P + Q] += rho / 2.0
    if P != Q:
        a[1, abs(P - Q)] += np.sign(P - Q) * rho / 2.0
    return a, b


def torus_knot(
    p: int, q: int, rho: float = 0.4, name: str = ""
) -> FourierKnot | FourierLink:
    """T(p, q): a FourierKnot when gcd(p,q)=1, else a FourierLink."""
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must be in (0, 1), got {rho}")
    d = gcd(p, q)
    P, Q = p // d, q // d

    n = P + Q + Q  # room for all harmonics (z uses Q)
    a, b = _rosette_xy(P, Q, rho, n)
    a[2, Q] += rho  # z = rho*sin(Q s)
    comp0 = FourierKnot(
        a=a,
        b=b,
        name=name or f"T({p},{q})",
        meta={"source": "torus", "p": p, "q": q, "rho": rho, "symmetry": q},
    )
    if d == 1:
        return comp0

    components = [comp0.rotated_z(-2.0 * np.pi * k / q) for k in range(d)]
    return FourierLink(
        components=components,
        name=name or f"T({p},{q})",
        meta={"source": "torus", "p": p, "q": q, "rho": rho, "components": d},
    )
