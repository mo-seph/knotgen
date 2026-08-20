"""Fetch David Fremlin's .fseries knot embeddings for vendoring.

Dev-only; run once (or to refresh):  uv run python scripts/fetch_fremlin.py

Scrapes each per-knot page ({name}.htm) for knot.{name}{variant}.fseries
links and saves them into src/knotgen/data/fremlin/{name}{variant}.fseries.
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import requests

BASE = "https://david.fremlin.de/knots/"
DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "knotgen" / "data" / "fremlin"

KNOTS = (
    ["3_1", "4_1", "5_1", "5_2"]
    + [f"6_{i}" for i in range(1, 4)]
    + [f"7_{i}" for i in range(1, 8)]
    + [f"8_{i}" for i in range(1, 22)]
)


def _has_data(fseries_text: str) -> bool:
    return any(
        line.strip() and not line.strip().startswith("%")
        for line in fseries_text.splitlines()
    )


def short_to_fseries(short_text: str, name: str, n_harmonics: int = 40) -> str:
    """Convert a knot.*.short coordinate list into an .fseries file.

    The .short points are unevenly spaced polyline vertices, so we resample
    uniformly by arc length before taking the FFT, then truncate.
    Output uses the same column convention as Fremlin's files:
    a_* = cos coefficients, b_* = sin coefficients, one row per harmonic.
    """
    pts = np.array(
        [
            [float(v) for v in line.split()]
            for line in short_text.splitlines()
            if line.strip() and not line.strip().startswith("%")
        ]
    )
    # resample the closed polyline uniformly by arc length
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = 512
    s_targets = np.linspace(0.0, s[-1], n, endpoint=False)
    uniform = np.column_stack(
        [np.interp(s_targets, s, closed[:, i]) for i in range(3)]
    )

    c = np.fft.fft(uniform, axis=0) / n
    n_harmonics = min(n_harmonics, n // 2 - 1)
    rows = []
    for j in range(1, n_harmonics + 1):
        cos_c = 2.0 * c[j].real  # a_* columns
        sin_c = -2.0 * c[j].imag  # b_* columns
        rows.append(
            f"{cos_c[0]:12.6f}{sin_c[0]:12.6f}{cos_c[1]:12.6f}"
            f"{sin_c[1]:12.6f}{cos_c[2]:12.6f}{sin_c[2]:12.6f}"
        )
    const = c[0].real
    header = (
        f"% Knot {name} — computed by knotgen scripts/fetch_fremlin.py from "
        f"knot.{name}.short\n"
        f"% (site's .fseries had no data rows; arc-length resampled + FFT, "
        f"{n_harmonics} harmonics)\n"
        f"% lines  a_x(j) b_x(j)  a_y(j)  b_y(j)  a_z(j)  b_z(j)\n"
        f"% corresponding to  x(s) = sum_j a_x(j)cos(js)+b_x(j)sin(js)  etc\n"
        f"{const[0]:12.6f}{const[1]:12.6f}{const[2]:12.6f}\n"
    )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "knotgen-fetch (personal knot-lamp tool)"
    fetched: list[str] = []

    for name in KNOTS:
        page = session.get(f"{BASE}{name}.htm", timeout=30)
        if page.status_code != 200:
            print(f"!! no page for {name} ({page.status_code})")
            continue
        pattern = rf"knot\.({re.escape(name)}[a-z]*)\.fseries"
        variants = sorted(set(re.findall(pattern, page.text)))
        if not variants:
            print(f"!! no fseries links on {name}.htm")
            continue
        for full in variants:
            url = f"{BASE}knot.{full}.fseries"
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"!! {url} -> {resp.status_code}")
                continue
            text = resp.text
            if not _has_data(text):
                short = session.get(f"{BASE}knot.{full}.short", timeout=30)
                if short.status_code == 200 and _has_data(short.text):
                    text = short_to_fseries(short.text, full)
                    print(f"   {full}.fseries was empty -> computed from .short")
                else:
                    print(f"!! {full}.fseries empty and no usable .short — skipped")
                    continue
            out = DATA_DIR / f"{full}.fseries"
            out.write_text(text)
            fetched.append(full)
            print(f"   {full}.fseries  ({len(text)} bytes)")
            time.sleep(0.3)  # be polite

    (DATA_DIR / "ATTRIBUTION.md").write_text(
        f"""# Attribution

The `.fseries` files in this directory are from David Fremlin's
"Knots and their symmetries": https://david.fremlin.de/knots/index.htm

Each file gives Fourier coefficients for a deliberately symmetrised
embedding of a prime knot. Fetched {date.today().isoformat()} by
`scripts/fetch_fremlin.py` ({len(fetched)} files). Filenames are
`{{knot}}{{variant}}.fseries` (variant suffixes as on the site, e.g.
`u`, `p`, `d`; no suffix = the site's default embedding).

No explicit license is stated on the site; the data is used here in a
personal fabrication tool with attribution.
"""
    )
    print(f"\nfetched {len(fetched)} files -> {DATA_DIR}")


if __name__ == "__main__":
    main()
