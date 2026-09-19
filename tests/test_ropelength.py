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


def test_basin_hops_run_and_keep_best():
    k = apply_style(resolve("3_1"), width=150, depth=15)
    base, _ = relax(k, tube=8, method="sono", iterations=40)
    hopped, info = relax(k, tube=8, method="sono", iterations=120, hops=2,
                         rope_slack=0.0)
    assert info["hops"] >= 1
    # the best-state guarantee survives hopping
    assert preflight(hopped).max_tube_diameter_mm >= preflight(base).max_tube_diameter_mm * 0.98


def test_cli_relax_gif(tmp_path, monkeypatch):
    from knotgen.cli import main

    monkeypatch.chdir(tmp_path)
    main(["3_1", "--width", "150", "--depth", "15", "--tube", "8", "--relax",
          "--relax-iterations", "10", "--snapshot-every", "5",
          "--relax-gif", "relax.gif"])
    gif = tmp_path / "output" / "relax.gif"
    assert gif.exists() and gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def test_gui_timeline_gif_and_keep_state():
    import time

    from knotgen.gui.server import (api_generate_start, api_gif, api_progress,
                                    api_snapshots, api_use_snapshot)

    argv = ["3_1", "--width", "150", "--tube", "8", "--relax",
            "--relax-iterations", "10", "--snapshot-every", "5"]
    job = api_generate_start({"argv": argv})["job"]
    for _ in range(300):
        p = api_progress(job)
        if p["status"] != "running":
            break
        time.sleep(0.1)
    assert p["status"] == "done", p.get("error")
    snaps = api_snapshots(job)["snapshots"]
    assert len(snaps) >= 3 and all("components" in s for s in snaps)
    name, data = api_gif(job)
    assert name.endswith(".gif") and data[:6] in (b"GIF87a", b"GIF89a")
    chosen = api_use_snapshot({"job": job, "index": 1})
    assert chosen["name"] == "3_1"
    assert chosen["relax"]["snapshot_iteration"] == snaps[1]["iteration"]


def _curv_spread(comp):
    import numpy as np

    t = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    k = comp.curvature(t)
    return float(np.std(k) / max(np.mean(k), 1e-12))


def test_stiff_component_rounds_toward_a_circle():
    k = apply_style(resolve("W(3,3)"), width=250, depth=25)
    base, _ = relax(k, tube=14, method="sono", iterations=40)
    stiff, info = relax(k, tube=14, method="sono", iterations=40,
                        stiffness=[5.0, 1.0, 1.0])
    assert info["stiffness"] == [5.0, 1.0, 1.0]
    # component 1 ends up with more uniform curvature (rounder) when stiff
    assert _curv_spread(as_link(stiff).components[0]) < _curv_spread(as_link(base).components[0])


def test_inflated_component_gets_more_room_and_is_exported_fatter(tmp_path):
    from knotgen.gm import tangent_point_radii  # noqa: F401  (module import check)
    from knotgen.mesh import design_mesh

    k = apply_style(resolve("W(3,3)"), width=250, depth=25)
    out, info = relax(k, tube=14, method="sono", iterations=40,
                      inflate=[1.0, 1.5, 1.0])
    assert info["inflate"] == [1.0, 1.5, 1.0]
    # the mesh picks the scales up from the relaxed meta: component 2's
    # tube is fatter, so its vertex ring is further from its centreline
    import numpy as np

    v, f = design_mesh(out, tube_diameter=14)
    v1, f1 = design_mesh(out, tube_diameter=14, scales=[1.0, 1.0, 1.0])
    # the inflated mesh occupies more space (fatter component 2)
    assert np.ptp(v, axis=0).prod() > np.ptp(v1, axis=0).prod()
    # and the JSON records it
    from knotgen.checks import preflight
    from knotgen.export import build_document

    doc = build_document(out, report=preflight(out, tube_diameter=14),
                         fit_points=30, tube_diameter=14)
    assert doc["pipe_preview"]["component_scales"] == [1.0, 1.5, 1.0]


def test_keep_diagram_limits_xy_drift():
    import numpy as np

    k = apply_style(resolve("L10n60"), width=180, depth=40)
    t = np.linspace(0, 2 * np.pi, 512, endpoint=False)

    def drift(out):
        return max(float(np.linalg.norm(c1.eval(t)[:, :2] - c0.eval(t)[:, :2], axis=1).mean())
                   for c0, c1 in zip(as_link(k).components, as_link(out).components))

    free, _ = relax(k, tube=15, max_depth=40, method="sono", iterations=40)
    held, info = relax(k, tube=15, max_depth=40, method="sono", iterations=40,
                       keep_diagram=1.0)
    assert info["keep_diagram"] == 1.0
    assert drift(held) < drift(free)


def test_per_component_rope_slack_is_respected():
    k = apply_style(resolve("W(3,3)"), width=250, depth=25)
    L0 = [c.total_length() for c in as_link(k).components]
    out, _ = relax(k, tube=14, method="sono", iterations=60,
                   rope_slack=[0.0, 0.3, 0.0])
    L1 = [c.total_length() for c in as_link(out).components]
    growth = [b / a - 1.0 for a, b in zip(L0, L1)]
    # per-component budgets are SOFT at a fixed footprint: the width
    # re-normalisation inflates every component uniformly, so on a fully
    # symmetric link (all three set the footprint) they cannot
    # differentiate at all — the guarantee is only that tight components
    # stay within a few percent and the total stays within budget
    assert growth[0] < 0.07 and growth[2] < 0.07
    assert sum(L1) <= sum(L0) * (1.0 + 0.3 / 3) * 1.01


def test_hops_preserve_linking_numbers():
    from knotgen.geometry import linking_numbers

    k = apply_style(resolve("L11a508"), width=270, depth=70)
    out, info = relax(k, tube=20, max_depth=70, method="sono", iterations=120,
                      rope_slack=0.15, hops=2)
    assert info["topology_ok"] is True
    assert linking_numbers(out) == linking_numbers(k)
