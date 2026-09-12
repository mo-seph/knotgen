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
