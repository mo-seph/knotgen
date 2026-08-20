import numpy as np
import pytest

from knotgen.fourier import TAU, FourierKnot
from knotgen.sources.torus import torus_knot


@pytest.fixture
def trefoil() -> FourierKnot:
    return torus_knot(2, 3, rho=0.5)


def test_closure(trefoil):
    p0 = trefoil.eval(np.array([0.0]))
    p1 = trefoil.eval(np.array([TAU]))
    np.testing.assert_allclose(p0, p1, atol=1e-12)


def test_deriv_matches_finite_difference(trefoil):
    t = np.linspace(0.1, TAU, 50)
    h = 1e-6
    fd = (trefoil.eval(t + h) - trefoil.eval(t - h)) / (2 * h)
    np.testing.assert_allclose(trefoil.deriv(t, 1), fd, rtol=1e-6, atol=1e-6)

    fd2 = (trefoil.eval(t + h) - 2 * trefoil.eval(t) + trefoil.eval(t - h)) / h**2
    np.testing.assert_allclose(trefoil.deriv(t, 2), fd2, rtol=1e-3, atol=1e-3)


def test_torus_knot_matches_direct_formula():
    p, q, rho = 2, 5, 0.35
    k = torus_knot(p, q, rho)
    t = np.linspace(0, TAU, 200, endpoint=False)
    expected = np.stack(
        [
            (1 + rho * np.cos(q * t)) * np.cos(p * t),
            (1 + rho * np.cos(q * t)) * np.sin(p * t),
            rho * np.sin(q * t),
        ],
        axis=1,
    )
    np.testing.assert_allclose(k.eval(t), expected, atol=1e-12)


def test_trefoil_threefold_symmetry(trefoil):
    """Shifting t by 2pi/3 must equal rotating the curve by 2pi*p/3 about z."""
    t = np.linspace(0, TAU, 97, endpoint=False)
    shifted = trefoil.eval(t + TAU / 3)
    theta = 2 * TAU / 3  # 2*pi*p/q with p=2, q=3
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotated = trefoil.eval(t) @ rot.T
    np.testing.assert_allclose(shifted, rotated, atol=1e-12)


def test_curvature_of_circle():
    # unit circle in xy: curvature exactly 1 everywhere
    a = np.zeros((3, 2))
    b = np.zeros((3, 2))
    b[0, 1] = 1.0  # x = cos t
    a[1, 1] = 1.0  # y = sin t
    circle = FourierKnot(a=a, b=b)
    t = np.linspace(0, TAU, 40)
    np.testing.assert_allclose(circle.curvature(t), np.ones_like(t), atol=1e-12)


def test_scaled_extents(trefoil):
    k = trefoil.scaled(2.0, 2.0, 0.5)
    e0 = trefoil.extents()
    e1 = k.extents()
    assert e1["xy_diameter"] == pytest.approx(2.0 * e0["xy_diameter"])
    assert e1["z_extent"] == pytest.approx(0.5 * e0["z_extent"])


def test_sample_arclength_even_spacing(trefoil):
    _, pts = trefoil.sample_arclength(400)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # closed curve: include wrap-around segment
    seg = np.append(seg, np.linalg.norm(pts[0] - pts[-1]))
    assert seg.std() / seg.mean() < 0.01
    # last point must not duplicate the first
    assert np.linalg.norm(pts[0] - pts[-1]) > 1e-6


def test_gcd_gives_torus_link():
    from knotgen.link import FourierLink

    link = torus_knot(2, 4)
    assert isinstance(link, FourierLink)
    assert link.n_components == 2
    # component 1 is a rotated copy of component 0 (as a set)
    assert link.rotational_symmetry_order() >= 2
