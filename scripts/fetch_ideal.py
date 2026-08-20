"""Fetch the Knot Atlas 'Ideal knots' Fourier databases for vendoring.

Dev-only; run once:  uv run python scripts/fetch_ideal.py

These files (by Brian Gilbert, SONO-tightened conformations) are Fourier
series in exactly the representation knotgen uses. They are vendored
verbatim (gzipped) and parsed lazily by knotgen.sources.ideal.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import requests

BASE = "https://katlas.org/images"
FILES = {
    # knots
    "Ideal.txt.gz": f"{BASE}/d/d2/Ideal.txt.gz",  # 3_1 .. 10_166
    "Ideal_11a.txt.gz": f"{BASE}/4/42/Ideal_11a.txt.gz",  # K11a1 .. K11a367
    "Ideal_11n.txt.gz": f"{BASE}/8/85/Ideal_11n.txt.gz",  # K11n1 .. K11n185
    # links (full Thistlethwaite table)
    "IdealLinks.txt.gz": f"{BASE}/5/5a/IdealLinks.txt.gz",  # L2a1 .. L9n28
    "IdealLinks_10a.txt.gz": f"{BASE}/e/ec/IdealLinks_10a.txt.gz",
    "IdealLinks_10n.txt.gz": f"{BASE}/d/de/IdealLinks_10n.txt.gz",
    "IdealLinks_11a1.txt.gz": f"{BASE}/f/f3/IdealLinks_11a1.txt.gz",
    "IdealLinks_11a2.txt.gz": f"{BASE}/9/99/IdealLinks_11a2.txt.gz",
    "IdealLinks_11n1.txt.gz": f"{BASE}/2/26/IdealLinks_11n1.txt.gz",
    "IdealLinks_11n2.txt.gz": f"{BASE}/b/bb/IdealLinks_11n2.txt.gz",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "knotgen" / "data" / "ideal"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "knotgen-fetch (personal knot-lamp tool)"
    total = 0
    for fname, url in FILES.items():
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        out = DATA_DIR / fname
        out.write_bytes(resp.content)
        total += len(resp.content)
        print(f"   {fname}  ({len(resp.content) // 1024} KB)")

    (DATA_DIR / "ATTRIBUTION.md").write_text(
        f"""# Attribution

The gzipped files in this directory are the "Ideal knots" databases by
Brian Gilbert, published on the Knot Atlas:
https://katlas.org/wiki/Ideal_knots

They contain Fourier-series representations of SONO-tightened ("ideal")
conformations of all prime knots up to 11 crossings and all prime links
of the Thistlethwaite table up to 11 crossings, normalised to unit tube
diameter. Convention: X(t) = A[0]/2 + sum_i A[i] cos(i t) + B[i] sin(i t).

Fetched {date.today().isoformat()} by scripts/fetch_ideal.py
({len(FILES)} files, {total // 1024} KB). Used with attribution in a
personal fabrication tool.
"""
    )
    print(f"\nfetched {len(FILES)} files ({total // 1024} KB) -> {DATA_DIR}")


if __name__ == "__main__":
    main()
