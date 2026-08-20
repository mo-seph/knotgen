"""Load David Fremlin's symmetric knot embeddings (.fseries files).

Data source: https://david.fremlin.de/knots/index.htm ("Knots and their
symmetries"). Files are vendored in knotgen/data/fremlin/ — see
ATTRIBUTION.md there.

File format (from the files' own headers):

    % lines  a_x(j) b_x(j)  a_y(j)  b_y(j)  a_z(j)  b_z(j)
    % corresponding to  x(s) = sum_j a_x(j)cos(js) + b_x(j)sin(js)  etc.

so column a = cosine coefficient, column b = sine coefficient, one row per
harmonic j starting at j=1. Some files carry a leading 3-value row: the
j=0 constant term (a_x, a_y, a_z) — detected by column count.
"""

from __future__ import annotations

import re
from importlib import resources

import numpy as np

from knotgen.fourier import FourierKnot

_NAME_RE = re.compile(r"^(\d+_\d+)([a-z]*)$")


def _data_dir():
    return resources.files("knotgen.data") / "fremlin"


def parse_fseries(text: str, name: str = "") -> FourierKnot:
    rows: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        values = [float(v) for v in line.split()]
        if len(values) not in (3, 6):
            raise ValueError(f"{name}: unexpected row with {len(values)} values: {line!r}")
        rows.append(values)
    if not rows:
        raise ValueError(f"{name}: no data rows found")

    constant = np.zeros(3)
    if len(rows[0]) == 3:
        constant = np.array(rows[0])
        rows = rows[1:]
    if any(len(r) != 6 for r in rows):
        raise ValueError(f"{name}: mixed row widths after constant row")

    data = np.array(rows)  # (N, 6): a_x b_x a_y b_y a_z b_z, j = 1..N
    n = len(data)
    a = np.zeros((3, n + 1))  # our sin coefficients
    b = np.zeros((3, n + 1))  # our cos coefficients
    b[:, 0] = constant
    # file's a_* = cos -> our b; file's b_* = sin -> our a
    b[0, 1:] = data[:, 0]
    a[0, 1:] = data[:, 1]
    b[1, 1:] = data[:, 2]
    a[1, 1:] = data[:, 3]
    b[2, 1:] = data[:, 4]
    a[2, 1:] = data[:, 5]

    return FourierKnot(a=a, b=b, name=name, meta={"source": "fremlin"})


def catalogue() -> dict[str, list[str]]:
    """{knot_name: [variant, ...]} for all vendored files ('' = default)."""
    result: dict[str, list[str]] = {}
    data = _data_dir()
    if not data.is_dir():
        return result
    for entry in data.iterdir():
        if not entry.name.endswith(".fseries"):
            continue
        m = _NAME_RE.match(entry.name.removesuffix(".fseries"))
        if not m:
            continue
        result.setdefault(m.group(1), []).append(m.group(2))
    for variants in result.values():
        variants.sort()
    return result


def load(name: str, variant: str | None = None) -> FourierKnot:
    """Load knot `name` (e.g. "5_1"); variant '', 'r', 'p', 'd' etc."""
    cat = catalogue()
    if name not in cat:
        raise KeyError(f"no Fremlin data for knot {name!r}")
    variants = cat[name]
    if variant is None:
        chosen = "" if "" in variants else variants[0]
    else:
        if variant not in variants:
            raise KeyError(
                f"knot {name} has no variant {variant!r}; available: {variants}"
            )
        chosen = variant

    fname = f"{name}{chosen}.fseries"
    text = (_data_dir() / fname).read_text()
    knot = parse_fseries(text, name=name)
    knot.meta["variant"] = chosen
    knot.meta["file"] = fname
    knot.meta["symmetry"] = knot.rotational_symmetry_order()
    return knot
