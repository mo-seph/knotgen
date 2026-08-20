import numpy as np
import pytest

from knotgen.geometry import crossings_xy, min_clearance
from knotgen.sources import fremlin
from knotgen.sources.torus import torus_knot
from knotgen.transforms import apply_style


@pytest.fixture
def pentafoil():
    return fremlin.load("5_1")


def test_width_and_depth_hit_targets(pentafoil):
    k = apply_style(pentafoil, width=300.0, depth=25.0)
    e = k.extents()
    assert e["xy_diameter"] == pytest.approx(300.0, abs=1e-9)
    assert e["z_extent"] == pytest.approx(25.0, abs=1e-9)


def test_breadth_override(pentafoil):
    k = apply_style(pentafoil, width=300.0, breadth=200.0, depth=25.0)
    assert k.extents()["y_extent"] == pytest.approx(200.0, abs=1e-6)


def test_symmetry_preserved_by_style(pentafoil):
    k = apply_style(pentafoil, width=280.0, depth=30.0, tightness=0.4)
    assert k.rotational_symmetry_order() == 5


def test_tightness_changes_curvature_not_width(pentafoil):
    from knotgen.geometry import max_curvature

    base = apply_style(pentafoil, width=300.0, depth=25.0)
    smooth = apply_style(pentafoil, width=300.0, depth=25.0, tightness=-0.6)
    sharp = apply_style(pentafoil, width=300.0, depth=25.0, tightness=0.6)
    assert smooth.extents()["xy_diameter"] == pytest.approx(300.0, abs=1e-9)
    assert sharp.extents()["xy_diameter"] == pytest.approx(300.0, abs=1e-9)
    # negative tightness genuinely rounds the tightest corner...
    k_base, _ = max_curvature(base)
    k_smooth, _ = max_curvature(smooth)
    assert k_smooth < 0.5 * k_base
    # ...positive tightness reshapes (corners become small loops), so max
    # curvature is NOT monotonic — just require a substantially different curve
    t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    rms = np.sqrt(np.mean(np.sum((sharp.eval(t) - base.eval(t)) ** 2, axis=1)))
    assert rms > 5.0  # mm, at 300 mm width


def test_tightness_range_validated(pentafoil):
    with pytest.raises(ValueError, match="tightness"):
        apply_style(pentafoil, width=300, depth=25, tightness=1.5)


def test_clearance_flags_flattened_knot(pentafoil):
    tall = apply_style(pentafoil, width=300.0, depth=30.0)
    flat = apply_style(pentafoil, width=300.0, depth=3.0)
    c_tall, _, _ = min_clearance(tall)
    c_flat, _, _ = min_clearance(flat)
    assert c_flat < c_tall
    assert c_flat < 5.0  # a 3mm-deep knot can't have more than ~3mm of gap + slack


def test_crossing_count_trefoil():
    k = apply_style(torus_knot(2, 3, 0.5), width=250.0, depth=20.0)
    crossings = crossings_xy(k)
    assert len(crossings) == 3
    for c in crossings:
        assert c.separation == pytest.approx(20.0, rel=0.2)


def test_crossing_count_pentafoil(pentafoil):
    k = apply_style(pentafoil, width=300.0, depth=25.0)
    assert len(crossings_xy(k)) == 5
