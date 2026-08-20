"""Knot Atlas "Ideal knots" databases: SONO-tightened knots and links.

Data: Brian Gilbert, https://katlas.org/wiki/Ideal_knots — vendored
verbatim (gzipped) in knotgen/data/ideal/, see ATTRIBUTION.md there.

Coverage: every prime knot 3_1..10_166 (incl. both Perko twins), K11a1..367,
K11n1..185, and the full Thistlethwaite link table L2a1..L11n459.

File convention: X(t) = A[0]/2 + sum_i A[i] cos(i t) + B[i] sin(i t),
one <Coeff I=".." A="ax,ay,az" B="bx,by,bz"/> per harmonic (sparse — zero
rows are omitted, so index by I); links/11-crossing entries wrap each
component in <STRING>. Everything is normalised to unit tube diameter.

These embeddings are tight-rope "organic" conformations — not symmetrised
and not 2.5D-flat like the Fremlin set. Negative --tightness doubles as a
smoother for their high harmonic content.
"""

from __future__ import annotations

import gzip
import re
from importlib import resources

import numpy as np

from knotgen.fourier import FourierKnot
from knotgen.link import FourierLink

_ENTRY_RE = re.compile(r'<(AB|HT|TL)\s+Id="([^"]+)"([^>]*)>')
_STRING_RE = re.compile(r"<STRING\b")
_COEFF_RE = re.compile(
    r'<Coeff\s+I="\s*(\d+)"\s+A="([^"]+)"\s+B="([^"]+)"'
)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

KNOT_FILES = ["Ideal.txt.gz", "Ideal_11a.txt.gz", "Ideal_11n.txt.gz"]
LINK_FILES = [
    "IdealLinks.txt.gz",
    "IdealLinks_10a.txt.gz",
    "IdealLinks_10n.txt.gz",
    "IdealLinks_11a1.txt.gz",
    "IdealLinks_11a2.txt.gz",
    "IdealLinks_11n1.txt.gz",
    "IdealLinks_11n2.txt.gz",
]

_cache: dict[str, dict[str, FourierLink]] = {}


def canonical_name(name: str) -> str:
    """Normalise the various naming schemes to knotgen's canonical form.

    'K11n34' -> '11n34';  '11n_34' -> '11n34';  '10:1:124' -> '10_124';
    'L6a4{0,1}' -> 'L6a4';  '3_1' stays '3_1'.
    """
    name = name.strip()
    m = re.match(r"^K?(\d+)([an])_?(\d+)$", name)
    if m and int(m.group(1)) >= 11:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = re.match(r"^(\d+):(\d+):(\d+)$", name)
    if m:
        return f"{m.group(1)}_{m.group(3)}"
    m = re.match(r"^(L\d+[an]\d+)(\{.*\})?$", name)
    if m:
        return m.group(1)
    return name


def _data_dir():
    return resources.files("knotgen.data") / "ideal"


def _parse_file(fname: str) -> dict[str, FourierLink]:
    if fname in _cache:
        return _cache[fname]
    path = _data_dir() / fname
    if not path.is_file():
        raise FileNotFoundError(
            f"{fname} not vendored — run `uv run python scripts/fetch_ideal.py`"
        )
    entries: dict[str, FourierLink] = {}
    name = None
    attrs: dict[str, str] = {}
    strings: list[list[tuple[int, list[float], list[float]]]] = []

    def finish() -> None:
        if name is None or not strings:
            return
        comps = []
        for rows in strings:
            n = max(i for i, _, _ in rows)
            a = np.zeros((3, n + 1))
            b = np.zeros((3, n + 1))
            for i, avals, bvals in rows:
                if i == 0:
                    b[:, 0] = np.array(avals) / 2.0  # A[0]/2 convention
                else:
                    b[:, i] = avals  # file A = cos -> our b
                    a[:, i] = bvals  # file B = sin -> our a
            comps.append(FourierKnot(a=a, b=b, name=name))
        entries[name] = FourierLink(
            components=comps,
            name=name,
            meta={
                "source": "ideal",
                "file": fname,
                "conway": attrs.get("Conway", ""),
                "ropelength": float(attrs["L"]) if "L" in attrs else None,
            },
        )

    with gzip.open(str(path), "rt", errors="replace") as fh:
        for line in fh:
            m = _ENTRY_RE.search(line)
            if m:
                finish()
                name = canonical_name(m.group(2))
                attrs = dict(_ATTR_RE.findall(m.group(0)))
                strings = [[]]  # implicit first string (no <STRING> in <AB> files)
                continue
            if _STRING_RE.search(line):
                if strings and strings[-1]:
                    strings.append([])
                continue
            m = _COEFF_RE.search(line)
            if m and name is not None:
                strings[-1].append(
                    (
                        int(m.group(1)),
                        [float(v) for v in m.group(2).split(",")],
                        [float(v) for v in m.group(3).split(",")],
                    )
                )
    finish()
    _cache[fname] = entries
    return entries


def _file_for(name: str) -> list[str]:
    """Candidate vendored files for a canonical name."""
    m = re.match(r"^(\d+)_(\d+)$", name)
    if m:
        return ["Ideal.txt.gz"] if int(m.group(1)) <= 10 else []
    m = re.match(r"^11([an])(\d+)$", name)
    if m:
        return [f"Ideal_11{m.group(1)}.txt.gz"]
    m = re.match(r"^L(\d+)([an])(\d+)$", name)
    if m:
        c, an, i = int(m.group(1)), m.group(2), int(m.group(3))
        if c <= 9:
            return ["IdealLinks.txt.gz"]
        if c == 10:
            return [f"IdealLinks_10{an}.txt.gz"]
        if c == 11 and an == "a":
            return ["IdealLinks_11a1.txt.gz", "IdealLinks_11a2.txt.gz"]
        if c == 11 and an == "n":
            return ["IdealLinks_11n1.txt.gz", "IdealLinks_11n2.txt.gz"]
    return []


def load(name: str) -> FourierLink:
    canon = canonical_name(name)
    for fname in _file_for(canon):
        entries = _parse_file(fname)
        if canon in entries:
            return entries[canon]
    raise KeyError(f"no ideal-knot data for {name!r}")


def catalogue_summary() -> dict[str, int]:
    """Entry counts per family without forcing a full parse of every file."""
    counts = {}
    for fname in KNOT_FILES + LINK_FILES:
        try:
            counts[fname] = len(_parse_file(fname))
        except FileNotFoundError:
            counts[fname] = 0
    return counts


def names(crossings: int | None = None, links: bool | None = None) -> list[str]:
    """All available canonical names, optionally filtered."""
    out: list[str] = []
    for fname in KNOT_FILES + LINK_FILES:
        try:
            out.extend(_parse_file(fname).keys())
        except FileNotFoundError:
            continue
    if links is not None:
        out = [n for n in out if n.startswith("L") == links]
    if crossings is not None:
        pat = re.compile(rf"^L?{crossings}(?:_|[an])\d+$")
        out = [n for n in out if pat.match(n)]
    return out
