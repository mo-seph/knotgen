"""The GUI backend: same pipeline as the CLI, JSON in/out."""

import json

import pytest

from knotgen.gui.server import (
    api_catalogue,
    api_download,
    api_export,
    api_generate,
    api_mesh,
)


def test_catalogue_lists_everything():
    cat = api_catalogue()
    curated_names = [r["name"] for r in cat["curated"]]
    assert "3_1" in curated_names and "8_21" in curated_names
    assert "11n34" in cat["ideal"]
    assert "L6a4" in cat["ideal"]
    # natural sort: 9-crossing knots before 10-crossing, knots before links
    assert cat["ideal"].index("9_1") < cat["ideal"].index("10_1")
    assert cat["ideal"].index("10_1") < cat["ideal"].index("L2a1")
    assert json.dumps(cat)  # serializable


def test_generate_simple_knot():
    out = api_generate({"argv": ["5_1", "--width", "200"]})
    assert out["name"].startswith("5_1")
    assert out["n_components"] == 1
    assert out["symmetry"] == 5
    assert abs(out["extents"]["xy_diameter"] - 200) < 1.0
    assert len(out["components"]) == 1
    assert len(out["components"][0]) == 400
    assert "clearance" in out["report"]["summary"].lower() or out["report"]["summary"]
    assert out["strips"] is None
    assert json.dumps(out)  # everything must be plain JSON types


def test_generate_link_with_strip():
    out = api_generate({"argv": ["W(3,3)", "--width", "250", "--strip", "10"]})
    assert out["n_components"] == 3
    assert len(out["components"]) == 3
    assert len(out["strips"]) == 3
    left = out["strips"][0]["left"]
    right = out["strips"][0]["right"]
    assert len(left) == len(right) > 0
    assert out["strip_metrics"]["max_twist_deg_per_cm"] > 0
    assert json.dumps(out)


def test_generate_tube_check_reported():
    out = api_generate({"argv": ["3_1", "--width", "150", "--tube", "10"]})
    assert out["report"]["ok_for_tube"] in (True, False)
    assert out["tube_mm"] == 10


def test_generate_unknown_name_is_value_error():
    with pytest.raises(ValueError):
        api_generate({"argv": ["99_99"]})


def test_generate_bad_flag_is_value_error():
    with pytest.raises(ValueError):
        api_generate({"argv": ["5_1", "--nonsense"]})


def test_generate_relax_requires_tube():
    with pytest.raises(ValueError, match="tube"):
        api_generate({"argv": ["5_1", "--relax"]})


def test_export_runs_the_real_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = api_export(
        {"argv": ["3_1", "--width", "150", "--out", "gui_test.json"]}
    )
    assert out["returncode"] == 0
    assert "wrote" in out["log"]
    doc = json.loads((tmp_path / "output" / "gui_test.json").read_text())
    assert doc["command"].startswith("knotgen ")
    assert "--width 150" in doc["command"]


def test_export_requires_out():
    with pytest.raises(ValueError, match="--out"):
        api_export({"argv": ["3_1"]})


def test_download_returns_clean_document():
    out = api_download(
        {"argv": ["3_1", "--width", "150"], "filename": "my knot.json"}
    )
    assert out["returncode"] == 0
    assert out["filename"] == "my_knot.json"  # sanitized
    doc = json.loads(out["content"])
    # no temp-dir paths leak into the reproducibility record
    assert doc["command"] == "knotgen gen 3_1 --width 150 --out my_knot.json"
    assert "/" not in doc["cli_options"]["out"]


def test_download_rejects_explicit_out():
    with pytest.raises(ValueError, match="--out"):
        api_download({"argv": ["3_1", "--out", "x.json"]})


def test_mesh_returns_binary_stl():
    filename, data = api_mesh(
        {"argv": ["3_1", "--width", "150", "--tube", "10"]}
    )
    assert filename == "3_1.stl"
    n_tris = int.from_bytes(data[80:84], "little")
    assert len(data) == 84 + 50 * n_tris
    assert n_tris > 1000


def test_mesh_requires_tube():
    with pytest.raises(ValueError, match="tube"):
        api_mesh({"argv": ["3_1"]})


def test_mesh_refuses_nonfitting_tube():
    with pytest.raises(ValueError, match="not fit"):
        api_mesh({"argv": ["3_1", "--width", "60", "--tube", "40"]})
