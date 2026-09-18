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


def test_relax_respects_depth_budget():
    # depth-limited regime: budget held (soft force + one final squash),
    # and clearance still improves over the input
    k = apply_style(resolve("11a2", source="ideal"), width=300.0, depth=50.0,
                    tightness=-0.15)
    before = preflight(k).max_tube_diameter_mm
    rk, info = relax(k, tube=40.0, iterations=60, max_depth=50.0)
    assert as_link(rk).extents()["z_extent"] <= 50.0 * 1.001
    assert preflight(rk).max_tube_diameter_mm > before


def test_relax_controls_length_growth():
    # repulsion used to pump length into the curve (wrinkles); the always-on
    # shortening flow holds it near the original
    k = apply_style(resolve("11a2", source="ideal"), width=300.0, depth=50.0,
                    tightness=-0.15)
    len0 = as_link(k).components[0].total_length()
    rk, _ = relax(k, tube=40.0, iterations=80)
    len1 = as_link(rk).components[0].total_length()
    assert len1 <= 1.35 * len0


def test_spectral_polish_reduces_wobble_not_tube():
    from knotgen.relax import spectral_polish

    k = apply_style(resolve("11a2", source="ideal"), width=300, depth=50,
                    tightness=-0.15)
    polished, info = spectral_polish(k)
    assert info["passes"] >= 1
    assert info["wobble_reduction"] > 0.0
    # tube budget respected: never loses more than the floor allows
    assert info["est_after"] >= 0.99 * info["est_before"]


def test_spectral_polish_preserves_symmetry_and_width():
    from knotgen.relax import spectral_polish

    k = apply_style(resolve("5_1"), width=300, depth=25)
    polished, _ = spectral_polish(k)
    assert polished.rotational_symmetry_order() == 5
    assert polished.extents()["xy_diameter"] == pytest.approx(300.0, rel=1e-3)


def test_relax_reports_polish():
    k = apply_style(resolve("5_1"), width=300, depth=25)
    _, info = relax(k, tube=16.0, iterations=30)
    assert "polish" in info
