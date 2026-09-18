"""CLI behaviors: ignored-flag warnings and the relax depth default."""

from knotgen.cli import main


def test_layout_flags_warn_for_non_weaving_names(capsys):
    main(["L6a4", "--layout", "racetrack", "--braid-split", "0.5"])
    out = capsys.readouterr().out
    assert "only apply to weaving knots" in out
    assert "--layout" in out and "--braid-split" in out
    assert 'W(3,3)' in out  # the Borromean hint


def test_layout_flags_no_warning_for_weaving(capsys):
    main(["W(3,3)", "--layout", "racetrack"])
    out = capsys.readouterr().out
    assert "only apply to weaving knots" not in out


def test_source_warns_for_parametric_names(capsys):
    main(["W(3,3)", "--source", "ideal"])
    out = capsys.readouterr().out
    assert "--source is ignored" in out


def test_relax_holds_explicit_depth_by_default(capsys):
    main(["5_1", "--width", "250", "--depth", "30", "--tube", "18",
          "--relax", "--relax-iterations", "25"])
    out = capsys.readouterr().out
    assert "holding depth <= 30" in out


def test_relax_max_depth_zero_lifts_the_hold(capsys):
    main(["5_1", "--width", "250", "--depth", "30", "--tube", "18",
          "--relax", "--relax-iterations", "5", "--relax-max-depth", "0"])
    out = capsys.readouterr().out
    assert "holding depth" not in out


def test_force_writes_failing_export(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # a 40 mm tube on a tiny trefoil cannot fit
    code = main(["3_1", "--width", "60", "--tube", "40",
                 "--out", "f.json", "--mesh", "f.stl", "--force"])
    out = capsys.readouterr().out
    assert "--force writes anyway" in out or "forced" in out.lower()
    assert (tmp_path / "output" / "f.json").exists()
    assert (tmp_path / "output" / "f.stl").exists()
    assert code == 0  # forced writes are an accepted outcome


def test_without_force_failing_export_blocked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["3_1", "--width", "60", "--tube", "40", "--out", "f.json"])
    assert code == 1
    assert not (tmp_path / "output" / "f.json").exists()


def test_relax_parks_result_when_not_exported(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["5_1", "--width", "250", "--depth", "30", "--tube", "40",
          "--relax", "--relax-iterations", "10"])
    out = capsys.readouterr().out
    assert "parked in" in out
    parked = tmp_path / "output" / "relaxed_5_1.json"
    assert parked.exists()
    # and the parked curve is reusable: check + mesh without re-relaxing
    code = main([
        "check", str(parked), "--tube", "40", "--mesh", "r.stl", "--force"
    ])
    assert code == 0
    assert (tmp_path / "output" / "r.stl").exists()


def test_check_mesh_from_saved_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["3_1", "--width", "200", "--tube", "12", "--out", "k.json"])
    code = main(["check", str(tmp_path / "output" / "k.json"),
                 "--tube", "12", "--mesh", "k.stl"])
    assert code == 0
    assert "triangles" in capsys.readouterr().out
    assert (tmp_path / "output" / "k.stl").stat().st_size > 1000


def test_squeeze_to_flag(capsys):
    main(["5_1", "--width", "250", "--depth", "40", "--squeeze-to", "25"])
    out = capsys.readouterr().out
    assert "squeezed z" in out


def test_relax_anneal_needs_relax_and_depth(capsys):
    main(["5_1", "--width", "250", "--relax-anneal"])
    assert "--relax-anneal needs" in capsys.readouterr().out


def test_limiter_reported(capsys):
    main(["5_1", "--width", "250", "--depth", "25"])
    assert "limited by" in capsys.readouterr().out
