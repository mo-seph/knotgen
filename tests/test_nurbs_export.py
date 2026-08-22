import json

import numpy as np
import pytest
from scipy.interpolate import BSpline

from knotgen.checks import preflight
from knotgen.export import build_document, export_json, load_knot
from knotgen.nurbs import DEGREE, fit_periodic, max_deviation
from knotgen.sources import fremlin
from knotgen.transforms import apply_style


@pytest.fixture(scope="module")
def styled():
    return apply_style(fremlin.load("5_1"), width=300.0, depth=25.0)


@pytest.fixture(scope="module")
def nurbs(styled):
    return fit_periodic(styled)


def test_fit_deviation_under_tolerance(styled, nurbs):
    assert max_deviation(nurbs, styled) < 0.05  # mm at 300 mm width


def test_periodic_seam_is_c2(nurbs):
    """Evaluate derivatives on both sides of the seam (u=0 == u=n)."""
    spl = nurbs.bspline()
    eps = 1e-9
    for order in range(3):
        d = spl.derivative(order) if order else spl
        np.testing.assert_allclose(
            d(0.0 + eps), d(float(nurbs.n) - eps), atol=1e-4
        )


def test_clamped_equivalent_matches_periodic(nurbs):
    knots_c, ctrl_c = nurbs.clamped()
    # standard clamped invariants
    assert np.all(knots_c[: DEGREE + 1] == knots_c[0])
    assert np.all(knots_c[-(DEGREE + 1) :] == knots_c[-1])
    assert len(ctrl_c) == len(knots_c) - DEGREE - 1
    clamped = BSpline(knots_c, ctrl_c, DEGREE)
    u = np.linspace(0.0, float(nurbs.n), 1000)
    np.testing.assert_allclose(clamped(u), nurbs.bspline()(u), atol=1e-9)
    # clamped curve starts and ends at the same point (closed)
    np.testing.assert_allclose(ctrl_c[0], ctrl_c[-1], atol=1e-9)


def test_export_roundtrip(tmp_path, styled, nurbs):
    report = preflight(styled, tube_diameter=16.0)
    doc = build_document(styled, report=report, tube_diameter=16.0)
    path = export_json(tmp_path / "k.json", doc)

    data = json.loads(path.read_text())
    assert data["schema_version"] == 2
    assert data["units"] == "mm"
    assert data["n_components"] == 1
    seg = data["paths"][0]["segments"][0]
    assert seg["closed"] is True
    assert len(seg["fit_points"]) == 60
    # v1 compatibility mirror for single-component designs
    assert data["path"]["segments"][0]["fit_points"] == seg["fit_points"]
    assert data["pipe_preview"]["diameter_mm"] == 16.0
    nc = seg["nurbs_clamped"]
    assert len(nc["control_points"]) == len(nc["knots"]) - nc["degree"] - 1

    back = load_knot(path)
    t = np.linspace(0, 2 * np.pi, 50)
    np.testing.assert_allclose(back.eval(t), styled.eval(t), atol=1e-12)


def test_export_multi_component_link(tmp_path):
    from knotgen.link import FourierLink
    from knotgen.sources.weaving import weaving

    borromean = apply_style(weaving(3, 3), width=300.0, depth=30.0)
    report = preflight(borromean)
    doc = build_document(borromean, report=report)
    path = export_json(tmp_path / "borromean.json", doc)

    data = json.loads(path.read_text())
    assert data["n_components"] == 3
    assert len(data["paths"]) == 3
    assert "path" not in data  # no single-path alias for true links

    back = load_knot(path)
    assert isinstance(back, FourierLink)
    assert back.n_components == 3


def test_connector_spacing_is_a_maximum(styled):
    from knotgen.export import build_connectors_section
    from knotgen.frames import compute_frames

    frames = [compute_frames(styled)]
    length = frames[0].length

    spacing = length / 2.4  # forces ceil -> 3 connectors
    sec = build_connectors_section(frames, spacing=spacing)
    pc = sec["per_component"][0]
    assert pc["count"] == 3
    assert pc["spacing_mm"] <= spacing
    assert pc["spacing_mm"] == pytest.approx(length / 3, rel=1e-3)

    # frames are orthonormal right-handed systems
    for fr in sec["frames"]:
        x, y, z = (np.array(fr[k]) for k in ("x_axis", "y_axis", "z_axis"))
        assert abs(np.dot(x, y)) < 1e-6
        np.testing.assert_allclose(np.cross(x, y), z, atol=1e-6)
        np.testing.assert_allclose(
            [np.linalg.norm(v) for v in (x, y, z)], 1.0, atol=1e-6
        )


def test_connector_offset_shifts_positions(styled):
    from knotgen.export import build_connectors_section
    from knotgen.frames import compute_frames

    frames = [compute_frames(styled)]
    base = build_connectors_section(frames, count=4)
    shifted = build_connectors_section(frames, count=4, offset=50.0)
    d = np.linalg.norm(
        np.array(base["frames"][0]["origin"])
        - np.array(shifted["frames"][0]["origin"])
    )
    # 50 mm along the curve moves the point by a comparable chord distance
    assert 20.0 < d <= 51.0
    # offset preserves count and spacing
    assert shifted["per_component"] == base["per_component"]


def test_connectors_validation(styled):
    from knotgen.export import build_connectors_section
    from knotgen.frames import compute_frames

    frames = [compute_frames(styled)]
    with pytest.raises(ValueError, match="exactly one"):
        build_connectors_section(frames)
    with pytest.raises(ValueError, match="exactly one"):
        build_connectors_section(frames, count=3, spacing=100.0)
    with pytest.raises(ValueError, match="positive"):
        build_connectors_section(frames, spacing=-5.0)


def test_mounts_section(styled):
    from knotgen.export import build_mounts_section
    from knotgen.frames import compute_frames

    frames = [compute_frames(styled)]
    sec = build_mounts_section(frames, [(0, 250.0), (0, 900.0)])
    assert sec["count"] == 2
    assert sec["frames"][0]["s_mm"] == pytest.approx(250.0, abs=0.1)
    for fr in sec["frames"]:
        x, y, z = (np.array(fr[k]) for k in ("x_axis", "y_axis", "z_axis"))
        assert abs(np.dot(x, y)) < 1e-6
        np.testing.assert_allclose(np.cross(x, y), z, atol=1e-6)
    with pytest.raises(ValueError, match="no such component"):
        build_mounts_section(frames, [(3, 100.0)])


def test_preflight_gates_oversized_tube(styled):
    report = preflight(styled, tube_diameter=200.0)
    assert report.ok_for_tube is False
    assert report.warnings
