"""Gonzalez-Maddocks thickness and the GM relax method."""

import numpy as np
import pytest

from knotgen.checks import preflight
from knotgen.fourier import FourierKnot
from knotgen.gm import thickness
from knotgen.link import FourierLink, as_link
from knotgen.registry import resolve
from knotgen.relax import relax
from knotgen.transforms import apply_style


def _circle(r=50.0, dz=0.0):
    a = np.zeros((3, 2))
    b = np.zeros((3, 2))
    a[0, 1] = r
    b[1, 1] = r
    b[2, 0] = dz  # b[:, 0] is the constant (DC) term
    return FourierKnot(a=a, b=b, name="circle", meta={})


def test_thickness_circle_is_its_radius():
    r = thickness(_circle(50.0))
    assert r["thickness"] == pytest.approx(50.0, rel=1e-3)
    assert r["max_tube_diameter"] == pytest.approx(100.0, rel=1e-3)


def test_thickness_stacked_circles_is_half_separation():
    link = FourierLink(components=[_circle(50.0), _circle(50.0, dz=12.0)],
                       name="stack", meta={})
    r = thickness(link)
    assert r["thickness"] == pytest.approx(6.0, rel=1e-2)
    assert r["min_local_radius"] == pytest.approx(50.0, rel=1e-3)


def test_thickness_ellipse_is_tip_curvature():
    # 100 x 10 ellipse: tip radius b^2/a = 1
    a = np.zeros((3, 2))
    b = np.zeros((3, 2))
    a[0, 1] = 100.0
    b[1, 1] = 10.0
    r = thickness(FourierKnot(a=a, b=b, name="e", meta={}))
    assert r["thickness"] == pytest.approx(1.0, rel=1e-2)


def test_thickness_agrees_with_preflight_on_real_knot():
    styled = apply_style(resolve("5_1"), width=300, depth=25)
    gm = thickness(styled)["max_tube_diameter"]
    pf = preflight(styled).max_tube_diameter_mm
    # GM is the true criterion; the separate checks are conservative,
    # so GM >= preflight, and they should be in the same ballpark
    assert gm >= pf * 0.95
    assert gm <= pf * 1.6


def test_relax_gm_method_opens_clearance():
    k = apply_style(resolve("5_1"), width=300.0, depth=25.0)
    rk, info = relax(k, tube=16.0, method="gm")
    assert preflight(rk).max_tube_diameter_mm >= 16.0
    assert rk.rotational_symmetry_order() == 5  # symmetry survives


def test_relax_gm_respects_depth_budget():
    k = apply_style(resolve("11a2", source="ideal"), width=300.0, depth=50.0,
                    tightness=-0.15)
    before = preflight(k).max_tube_diameter_mm
    rk, _ = relax(k, tube=40.0, iterations=60, max_depth=50.0, method="gm")
    assert as_link(rk).extents()["z_extent"] <= 50.0 * 1.001
    assert preflight(rk).max_tube_diameter_mm > before


def test_relax_gm_never_worse_than_input():
    k = apply_style(resolve("W(3,3)"), width=300.0, depth=15.0)
    s0 = preflight(k).max_tube_diameter_mm
    rk, _ = relax(k, tube=16.0, iterations=20, method="gm")
    assert preflight(rk).max_tube_diameter_mm >= s0 * 0.999
