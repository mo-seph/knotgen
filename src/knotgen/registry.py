"""Resolve knot/link names to curves from the available sources."""

from __future__ import annotations

import re

from knotgen.fourier import FourierKnot
from knotgen.link import FourierLink
from knotgen.sources import fremlin, ideal, torus, weaving

_PARAM_RE = re.compile(r"^([TtWw])\((\d+),\s*(\d+)\)$")

# Rolfsen-named knots that are torus knots, with their (p, q)
TORUS_KNOTS: dict[str, tuple[int, int]] = {
    "3_1": (2, 3),
    "5_1": (2, 5),
    "7_1": (2, 7),
    "8_19": (3, 4),
    "9_1": (2, 9),
    "10_124": (3, 5),
    "11_367": (2, 11),  # 11a367 = T(2,11)
}


def resolve(
    name: str,
    source: str = "auto",
    variant: str | None = None,
    rho: float = 0.4,
    layout: str = "rosette",
    aspect: float = 2.0,
    braid_fraction: float = 0.8,
    lane_gap: float | None = None,
    braid_split: float = 0.0,
) -> FourierKnot | FourierLink:
    """Get a curve by name.

    Accepted: Rolfsen names ("5_1"), 11-crossing names ("11a42", "11n34",
    also "K11n34"/"11n_34"), link names ("L6a4"), torus "T(p,q)" and
    weaving "W(p,q)" families.
    """
    name = name.strip()

    m = _PARAM_RE.match(name)
    if m:
        p, q = int(m.group(2)), int(m.group(3))
        if m.group(1).upper() == "T":
            return torus.torus_knot(p, q, rho=rho)
        if layout == "racetrack":
            from knotgen.sources.racetrack import racetrack_weave

            return racetrack_weave(
                p, q, aspect=aspect, braid_fraction=braid_fraction,
                lane_gap=lane_gap, braid_split=braid_split,
            )
        return weaving.weaving(p, q, rho=rho)

    if source == "torus":
        if name not in TORUS_KNOTS:
            raise KeyError(
                f"{name} is not a torus knot; torus source covers {sorted(TORUS_KNOTS)}"
            )
        p, q = TORUS_KNOTS[name]
        return torus.torus_knot(p, q, rho=rho, name=name)

    if source == "fremlin":
        return fremlin.load(name, variant=variant)

    if source == "ideal":
        return ideal.load(name)

    if source == "auto":
        # preference order: Fremlin (symmetrised) > torus (exact symmetric)
        # > ideal (tight-rope organic, but covers everything to 11 crossings)
        try:
            return fremlin.load(name, variant=variant)
        except KeyError:
            pass
        if name in TORUS_KNOTS:
            p, q = TORUS_KNOTS[name]
            return torus.torus_knot(p, q, rho=rho, name=name)
        try:
            return ideal.load(name)
        except KeyError:
            raise KeyError(
                f"no source for {name!r} — try `knotgen list` to see what's available"
            ) from None

    raise ValueError(
        f"unknown source {source!r} (expected auto, fremlin, torus or ideal)"
    )


def available() -> list[dict]:
    """Catalogue for `knotgen list`: one row per knot name."""
    rows: dict[str, dict] = {}
    for name, variants in fremlin.catalogue().items():
        rows[name] = {"name": name, "sources": ["fremlin"], "variants": variants}
    for name, (p, q) in TORUS_KNOTS.items():
        row = rows.setdefault(name, {"name": name, "sources": [], "variants": []})
        row["sources"].append(f"torus T({p},{q})")

    def sort_key(n: str) -> tuple[int, int]:
        try:
            crossings, index = n.split("_")
            return int(crossings), int(index)
        except ValueError:
            return (99, 0)

    return [rows[n] for n in sorted(rows, key=sort_key)]
