"""Fixed-rope relaxation (method='sono') and relax snapshots."""

import json

import pytest

from knotgen.checks import preflight
from knotgen.link import as_link
from knotgen.registry import resolve
from knotgen.relax import relax
from knotgen.transforms import apply_style


def _length(k):
    return sum(c.total_length() for c in as_link(k).components)


def test_sono_never_adds_rope_beyond_budget():
    k = apply_style(resolve("L10n60"), width=180, depth=40)
    natural = _length(apply_style(resolve("L10n60"), width=180))
    out, info = relax(k, tube=15, max_depth=40, method="sono",
                      rope_budget=natural, rope_slack=0.10, iterations=60)
    assert _length(out) <= natural * 1.10 * 1.01
    assert info["method"] == "sono"
    assert "length_after" in info and "rope_budget" in info


def test_sono_improves_and_holds_depth():
    k = apply_style(resolve("11a2", source="ideal"), width=300, depth=50,
                    tightness=-0.15)
    before = preflight(k).max_tube_diameter_mm
    out, _ = relax(k, tube=40, max_depth=50, method="sono", iterations=60)
    assert preflight(out).max_tube_diameter_mm >= before
    assert as_link(out).extents()["z_extent"] <= 50.0 * 1.001


def test_sono_preserves_symmetry_and_width():
    k = apply_style(resolve("5_1"), width=300, depth=25)
    out, _ = relax(k, tube=16, method="sono", iterations=40)
    assert out.rotational_symmetry_order() == 5
    assert out.extents()["xy_diameter"] == pytest.approx(300.0, rel=1e-3)


def test_snapshots_are_delivered():
    seen = []
    k = apply_style(resolve("3_1"), width=150, depth=15)
    relax(k, tube=8, method="sono", iterations=12, snapshot_every=4,
          snapshot=lambda it, link, m: seen.append((it, m)))
    its = [it for it, _ in seen]
    assert 0 in its and 4 in its and 8 in its  # start + every 4
    assert all("fits" in m and "length" in m for _, m in seen)
    # the old methods report snapshots too
    seen.clear()
    relax(k, tube=8, method="gm", iterations=8, snapshot_every=4,
          snapshot=lambda it, link, m: seen.append((it, m)))
    assert len(seen) >= 2


def test_cli_snapshots_write_pngs_and_trace(tmp_path, monkeypatch):
    from knotgen.cli import main

    monkeypatch.chdir(tmp_path)
    snapdir = tmp_path / "snaps"
    main(["3_1", "--width", "150", "--depth", "15", "--tube", "8", "--relax",
          "--relax-iterations", "10", "--snapshot-every", "5",
          "--relax-snapshots", str(snapdir)])
    pngs = sorted(snapdir.glob("*.png"))
    assert len(pngs) >= 2
    trace = json.loads((snapdir / "trace.json").read_text())
    assert len(trace) == len(pngs)
    assert all("fits" in t and "png" in t for t in trace)


def test_gui_async_job_reports_progress():
    import time

    from knotgen.gui.server import api_generate_start, api_progress

    job = api_generate_start({"argv": ["3_1", "--width", "150", "--tube", "8",
                                       "--relax", "--relax-iterations", "10"]})["job"]
    snaps = 0
    for _ in range(300):
        p = api_progress(job)
        if p["snapshot"]:
            snaps = max(snaps, p["snapshot"]["iteration"])
            assert p["snapshot"]["components"]
        if p["status"] != "running":
            break
        time.sleep(0.1)
    assert p["status"] == "done", p.get("error")
    assert p["result"]["name"] == "3_1"
    assert snaps >= 5
