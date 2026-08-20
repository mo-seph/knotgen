import numpy as np
import pytest

from knotgen.fourier import TAU
from knotgen.geometry import crossings_xy, is_alternating
from knotgen.link import as_link
from knotgen.sources.racetrack import racetrack_weave
from knotgen.transforms import apply_style, natural_aspect


@pytest.mark.parametrize(
    "p,q,components",
    [(3, 2, 1), (3, 5, 1), (2, 3, 1), (3, 3, 3), (4, 3, 1)],
)
def test_racetrack_crossings_and_alternation(p, q, components):
    w = as_link(racetrack_weave(p, q))
    assert w.n_components == components
    assert len(crossings_xy(w)) == q * (p - 1)
    assert is_alternating(w)


def test_racetrack_is_flat():
    # the layout is 2.5D by construction, so auto depth gives 25 mm
    w = racetrack_weave(3, 5)
    assert natural_aspect(w) < 0.3
    styled = apply_style(w, width=400.0)
    assert styled.meta["style"]["depth"] == pytest.approx(25.0)


def test_racetrack_components_close():
    link = as_link(racetrack_weave(3, 3))
    for comp in link.components:
        np.testing.assert_allclose(
            comp.eval(np.array([0.0])), comp.eval(np.array([TAU])), atol=1e-6
        )


def test_braid_fraction_concentrates_crossings():
    tight = as_link(racetrack_weave(3, 5, braid_fraction=0.3))
    spread = as_link(racetrack_weave(3, 5, braid_fraction=1.0))
    for link, max_span in ((tight, 0.35), (spread, 1.05)):
        xs = [c.xy[0] for c in crossings_xy(link)]
        ys = [c.xy[1] for c in crossings_xy(link)]
        e = link.extents()
        span = (max(xs) - min(xs)) / e["x_extent"]
        assert span < max_span
        # all crossings on the bottom straight
        assert max(ys) < 0


@pytest.mark.parametrize("split", [0.5, 1.0])
def test_braid_split_preserves_diagram(split):
    link = as_link(racetrack_weave(3, 5, braid_split=split))
    crossings = crossings_xy(link)
    assert len(crossings) == 10
    assert is_alternating(link)
    ys = [c.xy[1] for c in crossings]
    if split == 0.5:
        assert sum(1 for y in ys if y > 0) == 5
        assert sum(1 for y in ys if y < 0) == 5
    else:
        assert all(y > 0 for y in ys)  # all moved to the top straight


def test_braid_split_component_count_unchanged():
    assert as_link(racetrack_weave(3, 3, braid_split=0.5)).n_components == 3


def test_wall_mode():
    from knotgen.transforms import floor_z

    w = racetrack_weave(3, 5, wall=True)
    link = as_link(w)
    assert len(crossings_xy(link)) == 10
    assert is_alternating(link)
    # raw construction: nothing meaningfully below the lane plane (the
    # truncated Fourier series rings by a fraction of a percent)
    t = np.linspace(0, TAU, 4096, endpoint=False)
    z = link.components[0].eval(t)[:, 2]
    assert z.min() > -0.01
    # most of the curve lies flat on the plane (lanes + under-passages)
    assert np.mean(np.abs(z) < 1e-2) > 0.5

    # styled + floored: flat plane at z = 0, everything above
    styled = floor_z(apply_style(w, width=500.0, depth=20.0))
    zs = styled.eval(t)[:, 2]
    assert zs.min() == pytest.approx(0.0, abs=1e-6)
    assert zs.max() == pytest.approx(20.0, abs=0.5)


def test_racetrack_validation():
    with pytest.raises(ValueError):
        racetrack_weave(1, 3)
    with pytest.raises(ValueError):
        racetrack_weave(3, 4, braid_fraction=0.01)
    with pytest.raises(ValueError):
        racetrack_weave(3, 4, aspect=-1.0)


def test_registry_layout_passthrough():
    from knotgen.registry import resolve

    rt = resolve("W(3,5)", layout="racetrack", aspect=3.0)
    assert rt.meta["source"] == "weaving-racetrack"
    assert rt.meta["aspect"] == 3.0
    ro = resolve("W(3,5)")
    assert ro.meta["source"] == "weaving"
