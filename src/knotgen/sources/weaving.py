"""Weaving knots and links W(p, q) — a.k.a. Turk's head / rosette knots.

W(p, q) is the ALTERNATING knot/link with the same projection as the torus
knot T(p, q): the closure of the p-strand braid (s1 s2^-1 s3 s4^-1 ...)^q.
It has q(p-1) crossings and gcd(p, q) components. W(3,2) = figure-eight,
W(3,3) = Borromean rings, W(3,4) = 8_18, W(3,5) = 10_123, W(2,q) = T(2,q).
References: Champanerkar-Kofman-Purcell arXiv:1506.04139; Turk's head
survey arXiv:2409.20106.

Construction: same xy rosette as the torus source; only z differs —
frequency q(p-1) (each of the q(p-1) crossings is passed twice, once over,
once under) with a phase chosen numerically so that z extrema land on the
crossings and every strand alternates over/under:

    x(s) = (1 + rho*cos(Q s)) cos(P s)      P = p/gcd, Q = q/gcd
    y(s) = (1 + rho*cos(Q s)) sin(P s)
    z(s) = sin(Qz s + psi)                  Qz = (q/gcd)*(p-1)

Multi-component (gcd d > 1): component k = component 0 rotated by
-2*pi*k/q about z, exactly as for torus links. z(s) is shared.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from knotgen.fourier import FourierKnot
from knotgen.geometry import crossings_xy, is_alternating
from knotgen.link import FourierLink, as_link
from knotgen.sources.torus import _rosette_xy


def _build(p: int, q: int, rho: float, psi: float) -> FourierKnot | FourierLink:
    d = gcd(p, q)
    P, Q = p // d, q // d
    Qz = Q * (p - 1)

    n = max(P + Q + P, Qz)
    a, b = _rosette_xy(P, Q, rho, n)
    # small z amplitude: this is a flat 2.5D layout (the final depth is set
    # by styling); a large amplitude here would trip the auto-depth logic
    # into treating it as a fully-3D conformation
    amp = 0.1
    a[2, Qz] += amp * np.cos(psi)  # z = amp * sin(Qz s + psi)
    b[2, Qz] += amp * np.sin(psi)
    comp0 = FourierKnot(
        a=a,
        b=b,
        name=f"W({p},{q})",
        meta={"source": "weaving", "p": p, "q": q, "rho": rho, "psi": psi,
              "symmetry": q},
    )
    if d == 1:
        return comp0
    return FourierLink(
        components=[comp0.rotated_z(-2.0 * np.pi * k / q) for k in range(d)],
        name=f"W({p},{q})",
        meta=dict(comp0.meta, components=d),
    )


def _tune_phase(p: int, q: int, rho: float, samples: int = 4096) -> float:
    """Choose psi so the diagram is alternating with maximal z margin.

    The xy projection is independent of psi, so we collect the crossing
    passages once (parameter values t of both strands at each crossing) and
    score candidate phases analytically.
    """
    d = gcd(p, q)
    Qz = (q // d) * (p - 1)
    trial = _build(p, q, rho, 0.0)
    crossings = crossings_xy(trial, samples=samples)
    if not crossings:
        raise ValueError(f"W({p},{q}) projection has no crossings?")

    passages = [
        (c.comp_over, c.t_over, c.comp_under, c.t_under) for c in crossings
    ]

    def score(psi: float) -> float:
        # margin: smallest |z difference| across all crossings
        dz = [
            abs(np.sin(Qz * ta + psi) - np.sin(Qz * tb + psi))
            for _, ta, _, tb in passages
        ]
        return min(dz)

    def alternating(psi: float) -> bool:
        seqs: dict[int, list[tuple[float, bool]]] = {}
        for ca, ta, cb, tb in passages:
            za = np.sin(Qz * ta + psi)
            zb = np.sin(Qz * tb + psi)
            seqs.setdefault(ca, []).append((ta, za > zb))
            seqs.setdefault(cb, []).append((tb, zb > za))
        for seq in seqs.values():
            seq.sort()
            overs = [o for _, o in seq]
            if any(x == y for x, y in zip(overs, overs[1:] + overs[:1])):
                return False
        return True

    candidates = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    best_psi, best_margin = None, -1.0
    for psi in candidates:
        m = score(psi)
        if m > best_margin and alternating(psi):
            best_psi, best_margin = float(psi), m
    if best_psi is None:
        raise ValueError(
            f"could not find an alternating phase for W({p},{q}) — "
            "try a different rho"
        )
    return best_psi


_PSI_CACHE: dict[tuple[int, int, float], float] = {}


def weaving(p: int, q: int, rho: float = 0.4) -> FourierKnot | FourierLink:
    """The weaving knot/link W(p, q) (gcd(p,q) components)."""
    if p < 2 or q < 1:
        raise ValueError(f"W({p},{q}): need p >= 2 and q >= 1")
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must be in (0, 1), got {rho}")
    key = (p, q, round(rho, 6))
    if key not in _PSI_CACHE:
        _PSI_CACHE[key] = _tune_phase(p, q, rho)
    return _build(p, q, rho, _PSI_CACHE[key])
