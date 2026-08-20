import numpy as np
import pytest

from knotgen.frames import compute_frames, resolve_light_dir
from knotgen.sources import fremlin
from knotgen.sources.torus import torus_knot
from knotgen.transforms import apply_style


@pytest.fixture(scope="module")
def trefoil():
    return apply_style(torus_knot(2, 3, 0.5), width=250.0, depth=25.0)


def _closure_gap(frames) -> float:
    """Angle (rad) between the last width dir transported to the first sample."""
    from knotgen.frames import _transport

    w_wrap = _transport(frames.width_dirs[-1], frames.tangents[-1], frames.tangents[0])
    dot = np.clip(np.dot(w_wrap, frames.width_dirs[0]), -1, 1)
    return float(np.arccos(dot))


def test_frames_orthonormal(trefoil):
    f = compute_frames(trefoil, follow=1.0)
    for a, b in [(f.tangents, f.width_dirs), (f.tangents, f.normals), (f.width_dirs, f.normals)]:
        assert np.abs(np.sum(a * b, axis=1)).max() < 1e-9
    for v in (f.tangents, f.width_dirs, f.normals):
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-9)


def test_frames_close_around_loop(trefoil):
    for follow in (0.0, 0.5, 1.0):
        f = compute_frames(trefoil, follow=follow)
        # gap should be about one sample step of twist, not a seam jump
        step = np.abs(f.twist_rate).max() * f.length / f.params["n"]
        assert _closure_gap(f) < max(3 * step, 0.05), f"follow={follow}"


def test_follow1_kills_edgewise_curvature(trefoil):
    f = compute_frames(trefoil, follow=1.0, twist_smooth=0.0)
    kappa = np.hypot(f.edgewise_curvature, f.inplane_curvature)
    # edgewise component should be a tiny fraction of total curvature
    assert np.abs(f.edgewise_curvature).max() < 0.05 * kappa.max()


def test_follow0_faces_light(trefoil):
    f = compute_frames(trefoil, follow=0.0, light_dir="up", twist_smooth=0.0)
    # face normals should point mostly up wherever the tangent allows it
    conf = np.linalg.norm(np.cross(f.tangents, [0.0, 0.0, 1.0]), axis=1)
    good = conf > 0.5
    dots = f.normals[good] @ np.array([0.0, 0.0, 1.0])
    assert dots.min() > 0.7


def test_smoothing_reduces_twist(trefoil):
    rough = compute_frames(trefoil, follow=0.7, twist_smooth=0.0)
    smooth = compute_frames(trefoil, follow=0.7, twist_smooth=20.0)
    assert np.abs(smooth.twist_rate).max() <= np.abs(rough.twist_rate).max() + 1e-12
    assert np.std(smooth.twist_rate) < np.std(rough.twist_rate)


def test_edges_are_width_apart(trefoil):
    f = compute_frames(trefoil, follow=1.0)
    a, b = f.edges(8.0)
    np.testing.assert_allclose(np.linalg.norm(b - a, axis=1), 8.0, atol=1e-9)


def test_light_dir_parsing():
    np.testing.assert_allclose(resolve_light_dir("down"), [0, 0, -1])
    np.testing.assert_allclose(resolve_light_dir("0,0,2"), [0, 0, 1])
    assert resolve_light_dir("out") == "radial-out"
    with pytest.raises(ValueError):
        resolve_light_dir("sideways")


def test_radial_light_dir(trefoil):
    f = compute_frames(trefoil, follow=0.0, light_dir="out", twist_smooth=0.0)
    radial = f.points * [1.0, 1.0, 0.0]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    conf = np.linalg.norm(np.cross(f.tangents, radial), axis=1)
    good = conf > 0.5
    dots = np.sum(f.normals[good] * radial[good], axis=1)
    assert np.mean(dots > 0.5) > 0.8


def test_symmetric_knot_symmetric_frames():
    """5-fold knot with follow=1: twist pattern should repeat 5x around."""
    k = apply_style(fremlin.load("5_1"), width=300.0, depth=25.0)
    f = compute_frames(k, n=400, follow=1.0, twist_smooth=5.0)
    kg = f.edgewise_curvature.reshape(5, 80)
    for i in range(1, 5):
        np.testing.assert_allclose(kg[i], kg[0], atol=1e-4)
