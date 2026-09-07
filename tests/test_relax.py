import numpy as np
import pytest

from knotgen.checks import preflight
from knotgen.geometry import crossings_xy, is_alternating
from knotgen.link import FourierLink, as_link
from knotgen.registry import resolve
from knotgen.relax import relax
from knotgen.transforms import apply_style


def test_relax_opens_clearance_and_preserves_topology():
    k = apply_style(resolve("5_1"), width=300.0, depth=25.0)
    assert preflight(k).max_tube_diameter_mm < 16.0  # genuinely too tight
    rk, info = relax(k, tube=16.0)
    assert preflight(rk).max_tube_diameter_mm >= 16.0
    assert info["gap_after"] > info["gap_before"]
    # diagram unchanged: same crossings, still alternating
    assert len(crossings_xy(rk)) == 5
    assert is_alternating(rk)


def test_relax_preserves_exact_symmetry_and_width():
    k = apply_style(resolve("5_1"), width=300.0, depth=25.0)
    rk, _ = relax(k, tube=16.0)
    assert rk.rotational_symmetry_order() == 5  # exact, via projection
    assert rk.extents()["xy_diameter"] == pytest.approx(300.0, rel=1e-3)


def test_relax_closure():
    from knotgen.fourier import TAU

    k = apply_style(resolve("3_1"), width=200.0, depth=10.0)
    rk, _ = relax(k, tube=14.0, iterations=40)
    np.testing.assert_allclose(
        rk.eval(np.array([0.0])), rk.eval(np.array([TAU])), atol=1e-9
    )


def test_relax_multi_component():
    b = apply_style(resolve("W(3,3)"), width=300.0, depth=15.0)
    rb, info = relax(b, tube=16.0)
    assert isinstance(rb, FourierLink)
    assert as_link(rb).n_components == 3
    assert preflight(rb).max_tube_diameter_mm >= 16.0


def test_relax_untouched_without_optin():
    # sanity that relax is a pure function: input unchanged
    k = apply_style(resolve("5_1"), width=300.0, depth=25.0)
    a0, b0 = k.a.copy(), k.b.copy()
    relax(k, tube=16.0, iterations=5)
    np.testing.assert_array_equal(k.a, a0)
    np.testing.assert_array_equal(k.b, b0)
