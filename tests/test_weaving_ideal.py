import numpy as np
import pytest

from knotgen.fourier import TAU, FourierKnot
from knotgen.geometry import crossings_xy, is_alternating, min_clearance
from knotgen.link import FourierLink, as_link
from knotgen.sources import ideal
from knotgen.sources.weaving import weaving
from knotgen.transforms import apply_style


# ------------------------------------------------------------------ weaving


@pytest.mark.parametrize(
    "p,q,components",
    [(3, 2, 1), (3, 4, 1), (3, 5, 1), (2, 3, 1), (3, 3, 3), (2, 4, 2), (4, 3, 1)],
)
def test_weaving_crossings_and_alternation(p, q, components):
    w = as_link(weaving(p, q))
    assert w.n_components == components
    assert len(crossings_xy(w)) == q * (p - 1)
    assert is_alternating(w)


def test_weaving_symmetry():
    assert weaving(3, 5).rotational_symmetry_order() == 5
    assert weaving(3, 4).rotational_symmetry_order() == 4


def test_borromean_components_disjoint():
    b = apply_style(weaving(3, 3), width=300.0, depth=30.0)
    d, _, _ = min_clearance(b)
    assert d > 5.0  # components keep real distance at lamp scale


def test_weaving_validation():
    with pytest.raises(ValueError):
        weaving(1, 5)
    with pytest.raises(ValueError):
        weaving(3, 4, rho=1.5)


# -------------------------------------------------------------------- ideal


def test_canonical_names():
    cn = ideal.canonical_name
    assert cn("K11n34") == "11n34"
    assert cn("11n_34") == "11n34"
    assert cn("11a367") == "11a367"
    assert cn("3:1:1") == "3_1"
    assert cn("L6a4{0,1}") == "L6a4"
    assert cn("5_2") == "5_2"


def test_ideal_knot_loads_and_closes():
    for name in ["3_1", "9_35", "10_124", "11a42", "11n34"]:
        link = ideal.load(name)
        assert link.n_components == 1
        comp = link.components[0]
        np.testing.assert_allclose(
            comp.eval(np.array([0.0])), comp.eval(np.array([TAU])), atol=1e-9
        )


def test_ideal_links_component_counts():
    assert ideal.load("L2a1").n_components == 2
    assert ideal.load("L6a4").n_components == 3  # Borromean rings
    assert ideal.load("L10n113").n_components == 5


def test_l2a1_is_two_perpendicular_circles():
    # the research sample showed L2a1 as two unit circles
    hopf = ideal.load("L2a1")
    d, _, _ = min_clearance(hopf)
    assert d > 0.5  # genuinely separated strands (unit tube diameter data)


def test_registry_resolves_all_families():
    from knotgen.registry import resolve

    assert isinstance(resolve("5_1"), FourierKnot)  # fremlin
    assert isinstance(resolve("9_1"), FourierKnot)  # torus (preferred over ideal)
    assert resolve("9_1").meta["source"] == "torus"
    assert resolve("10_45").meta["source"] == "ideal"
    assert isinstance(resolve("W(3,3)"), FourierLink)
    assert isinstance(resolve("T(2,4)"), FourierLink)
    assert as_link(resolve("L4a1")).n_components == 2
    assert resolve("K11n34").name == "11n34"


def test_styled_link_preflight_end_to_end():
    from knotgen.checks import preflight

    l6a4 = apply_style(ideal.load("L6a4"), width=300.0, depth=40.0)
    report = preflight(l6a4, tube_diameter=16.0)
    assert report.min_clearance_mm > 0
    assert report.max_tube_diameter_mm > 0
