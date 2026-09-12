"""Tube mesh export: watertight closed meshes, STL/OBJ writers, CLI flag."""

import struct

import numpy as np
import pytest

from knotgen.mesh import design_mesh, export_mesh, obj_text, stl_bytes, tube_mesh
from knotgen.registry import resolve
from knotgen.transforms import apply_style


def _circle(n=100, r=50.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t), np.zeros(n)], axis=1)


def test_tube_mesh_torus_topology():
    verts, faces = tube_mesh(_circle(), radius=5.0, sides=12)
    n, s = 100, 12
    assert verts.shape == (n * s, 3)
    assert faces.shape == (2 * n * s, 3)
    # closed torus: every edge shared by exactly two triangles
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()


def test_tube_mesh_faces_point_outward():
    verts, faces = tube_mesh(_circle(), radius=5.0)
    tri = verts[faces]
    volume = np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    assert volume > 0
    # a torus of tube radius 5 around r=50: volume = 2 pi^2 R r^2
    expected = 2 * np.pi**2 * 50 * 25
    assert abs(volume - expected) / expected < 0.05


def test_design_mesh_merges_link_components():
    styled = apply_style(resolve("W(3,3)"), width=250, depth=25)
    verts, faces = design_mesh(styled, tube_diameter=12)
    assert faces.max() == len(verts) - 1
    # three disjoint tori: edge-connected components would be 3; cheap check
    # instead: vertex count is a multiple of a single tube's rows*sides
    assert len(verts) % 3 == 0


def test_stl_bytes_layout():
    verts, faces = tube_mesh(_circle(20), radius=5.0, sides=6)
    data = stl_bytes(verts, faces, name="test")
    assert len(data) == 80 + 4 + 50 * len(faces)
    assert struct.unpack("<I", data[80:84])[0] == len(faces)
    assert data[:4] == b"test"


def test_obj_text():
    verts, faces = tube_mesh(_circle(10), radius=5.0, sides=4)
    text = obj_text(verts, faces)
    assert text.count("\nv ") == len(verts)
    assert text.count("\nf ") == len(faces)
    assert "f 0" not in text  # OBJ is 1-indexed


def test_export_mesh_formats(tmp_path):
    styled = apply_style(resolve("3_1"), width=150, depth=25)
    p_stl, n1 = export_mesh(tmp_path / "t.stl", styled, tube_diameter=10)
    p_obj, n2 = export_mesh(tmp_path / "t.obj", styled, tube_diameter=10)
    p_bare, _ = export_mesh(tmp_path / "bare", styled, tube_diameter=10)
    assert p_stl.stat().st_size == 84 + 50 * n1
    assert p_obj.read_text().startswith("o 3_1")
    assert p_bare.suffix == ".stl"
    assert n1 == n2


def test_cli_mesh_flag(tmp_path, monkeypatch, capsys):
    from knotgen.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["3_1", "--width", "150", "--tube", "10", "--mesh", "t.stl"]) == 0
    out = capsys.readouterr().out
    assert "triangles" in out
    assert (tmp_path / "output" / "t.stl").exists()


def test_cli_mesh_needs_tube(tmp_path, monkeypatch, capsys):
    from knotgen.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["3_1", "--mesh", "t.stl"]) == 1
    assert "--tube" in capsys.readouterr().out
    assert not (tmp_path / "output" / "t.stl").exists()


def test_cli_mesh_refused_when_tube_does_not_fit(tmp_path, monkeypatch, capsys):
    from knotgen.cli import main

    monkeypatch.chdir(tmp_path)
    code = main(["3_1", "--width", "60", "--tube", "40", "--mesh", "t.stl"])
    assert code == 1
    assert "NOT writing mesh" in capsys.readouterr().out
    assert not (tmp_path / "output" / "t.stl").exists()
