import numpy as np
import pytest

from knotgen.fourier import FourierKnot
from knotgen.geometry import crossings_xy, min_clearance
from knotgen.link import FourierLink, as_link
from knotgen.transforms import apply_style


def circle(radius: float, center, plane: str = "xy") -> FourierKnot:
    a = np.zeros((3, 2))
    b = np.zeros((3, 2))
    b[:, 0] = center
    if plane == "xy":
        b[0, 1] = radius  # x = cx + r cos t
        a[1, 1] = radius  # y = cy + r sin t
    elif plane == "xz":
        b[0, 1] = radius
        a[2, 1] = radius
    return FourierKnot(a=a, b=b)


@pytest.fixture
def hopf() -> FourierLink:
    """Two unit circles in perpendicular planes, linked like L2a1."""
    return FourierLink(
        components=[
            circle(1.0, (0.0, 0.0, 0.0), "xy"),
            circle(1.0, (1.0, 0.0, 0.0), "xz"),
        ],
        name="hopf",
    )


def test_as_link_wraps_knot():
    k = circle(1.0, (0, 0, 0))
    link = as_link(k)
    assert link.n_components == 1
    assert as_link(link) is link


def test_joint_extents(hopf):
    e = hopf.extents()
    # centred union spans x in [-1.5, 1.5], y in [-1, 1], z in [-1, 1]
    assert e["x_extent"] == pytest.approx(3.0, abs=1e-3)
    assert e["z_extent"] == pytest.approx(2.0, abs=1e-3)


def test_joint_styling_hits_targets(hopf):
    styled = apply_style(hopf, width=300.0, depth=40.0)
    assert isinstance(styled, FourierLink)
    e = styled.extents()
    assert e["xy_diameter"] == pytest.approx(300.0, abs=1e-6)
    assert e["z_extent"] == pytest.approx(40.0, abs=1e-6)
    assert styled.n_components == 2


def test_apply_style_preserves_input_type(hopf):
    from knotgen.sources.torus import torus_knot

    styled_knot = apply_style(torus_knot(2, 3, 0.5), width=100.0, depth=10.0)
    assert isinstance(styled_knot, FourierKnot)
    styled_link = apply_style(hopf, width=100.0, depth=10.0)
    assert isinstance(styled_link, FourierLink)


def test_flat_circle_degenerate_raises():
    with pytest.raises(ValueError, match="degenerate"):
        apply_style(circle(1.0, (0, 0, 0)), width=100.0, depth=10.0)


def test_cross_component_clearance():
    # two parallel circles 5 apart in z: clearance must be ~5, and it is a
    # CROSS-component distance (each circle alone has no close approach)
    link = FourierLink(
        components=[
            circle(50.0, (0, 0, 0.0), "xy"),
            circle(50.0, (0, 0, 5.0), "xy"),
        ]
    )
    d, pa, pb = min_clearance(link)
    assert d == pytest.approx(5.0, rel=1e-3)


def test_cross_component_crossings(hopf):
    crossings = crossings_xy(hopf, samples=1024)
    cross_pairs = [c for c in crossings if c.comp_over != c.comp_under]
    assert len(cross_pairs) >= 2  # Hopf projection: the circles cross twice


def test_multi_component_symmetry():
    # three unit circles centred on a radius-2 ring, rotated copies of each
    # other -> the design has 3-fold symmetry as a set
    comps = []
    for k in range(3):
        ang = 2 * np.pi * k / 3
        comps.append(circle(1.0, (2 * np.cos(ang), 2 * np.sin(ang), 0.0), "xy"))
    link = FourierLink(components=comps)
    assert link.rotational_symmetry_order() % 3 == 0
