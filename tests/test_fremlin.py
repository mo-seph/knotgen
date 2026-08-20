import numpy as np
import pytest

from knotgen.fourier import TAU
from knotgen.sources import fremlin


def test_catalogue_covers_wikipedia_knots():
    cat = fremlin.catalogue()
    expected = (
        ["3_1", "4_1", "5_1", "5_2"]
        + [f"6_{i}" for i in range(1, 4)]
        + [f"7_{i}" for i in range(1, 8)]
        + [f"8_{i}" for i in range(1, 22)]
    )
    missing = [k for k in expected if k not in cat]
    assert not missing, f"missing knots: {missing}"


def test_parse_3_1_known_coefficients():
    k = fremlin.load("3_1")
    # x(t) = 0.224483 sin(t) + 0.995730 cos(2t) + 0.221238 sin(5t)
    assert k.a[0, 1] == pytest.approx(0.224483)  # sin t
    assert k.b[0, 2] == pytest.approx(0.995730)  # cos 2t
    assert k.a[0, 5] == pytest.approx(0.221238)  # sin 5t
    # z(t) = 0.445727 cos(3t)
    assert k.b[2, 3] == pytest.approx(0.445727)
    assert np.all(k.a[2] == 0.0)


@pytest.mark.parametrize(
    "name,order",
    [("3_1", 3), ("5_1", 5), ("7_1", 7)],
)
def test_symmetry_orders(name, order):
    k = fremlin.load(name)
    assert k.rotational_symmetry_order() == order


def test_5_1_fivefold_symmetry_pointwise():
    """R_z(theta) r(t) == r(t + 2pi/5) for the appropriate theta."""
    k = fremlin.load("5_1")
    t = np.linspace(0, TAU, 173, endpoint=False)
    shifted = k.eval(t + TAU / 5)
    # frequencies are congruent to 2 mod 5 -> shift by 2pi/5 rotates by 2*2pi/5
    theta = 2 * TAU / 5
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotated = k.eval(t) @ rot.T
    np.testing.assert_allclose(shifted, rotated, atol=1e-9)


def test_every_vendored_file_parses_and_closes():
    cat = fremlin.catalogue()
    for name, variants in cat.items():
        for v in variants:
            k = fremlin.load(name, variant=v)
            p0 = k.eval(np.array([0.0]))
            p1 = k.eval(np.array([TAU]))
            np.testing.assert_allclose(p0, p1, atol=1e-9, err_msg=f"{name}{v}")
            # a knot must be a genuinely 3D closed curve
            e = k.extents()
            assert e["z_extent"] > 0, f"{name}{v} is flat"


def test_variant_selection():
    k = fremlin.load("3_1", variant="p")
    assert k.meta["variant"] == "p"
    with pytest.raises(KeyError, match="variant"):
        fremlin.load("3_1", variant="zzz")
    with pytest.raises(KeyError, match="no Fremlin data"):
        fremlin.load("99_9")
