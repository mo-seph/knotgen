"""JSON export for the Fusion 360 import script (and other consumers).

Schema notes:
  * everything geometric is in mm; the Fusion script converts to cm.
  * `path.segments` is a list so that future open paths (e.g. a broken knot
    with two straight stand stubs) fit the same schema: v1 always emits a
    single closed spline segment.
  * the Fourier coefficients are included so a JSON file can be re-loaded
    for `knotgen check/preview` without regenerating.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from knotgen import __version__
from knotgen.checks import PreflightReport
from knotgen.fourier import FourierKnot
from knotgen.nurbs import DEGREE, PeriodicNurbs, max_deviation

SCHEMA_VERSION = 2


def _round(arr: np.ndarray, places: int = 6) -> list:
    return np.round(np.asarray(arr, dtype=float), places).tolist()


def _nurbs_dict(nurbs: PeriodicNurbs) -> dict[str, Any]:
    knots_p, cp_wrapped = nurbs.wrapped()
    knots_c, cp_clamped = nurbs.clamped()
    return {
        "nurbs_periodic": {
            "degree": DEGREE,
            "control_points": _round(nurbs.control_points),
            "control_points_wrapped": _round(cp_wrapped),
            "knots": _round(knots_p),
        },
        "nurbs_clamped": {
            "degree": DEGREE,
            "control_points": _round(cp_clamped),
            "knots": _round(knots_c),
        },
    }


def build_strip_section(
    strip,  # StripFrames
    *,
    width: float,
    light_dir_spec: str,
    frame_count: int = 25,
    component: int = 0,
) -> dict[str, Any]:
    from knotgen.nurbs import fit_periodic_points

    edge_a, edge_b = strip.edges(width)
    step = max(len(strip.points) // frame_count, 1)
    sel = list(range(0, len(strip.points), step))[:frame_count]
    frame_lines = [
        {
            "s_fraction": round(float(strip.s_fraction[i]), 6),
            "center": _round(strip.points[i]),
            "a": _round(edge_a[i]),
            "b": _round(edge_b[i]),
            "normal": _round(strip.normals[i]),
        }
        for i in sel
    ]
    return {
        "component": component,
        "width_mm": width,
        "params": {**strip.params, "light_dir": light_dir_spec},
        "metrics": {k: round(v, 3) for k, v in strip.metrics().items()},
        "samples": {
            "s_fraction": _round(strip.s_fraction),
            "center": _round(strip.points),
            "width_dir": _round(strip.width_dirs),
            "normal": _round(strip.normals),
        },
        "edges": {
            "a": {"fit_points": _round(edge_a[:: max(len(edge_a) // 100, 1)]),
                  **_nurbs_dict(fit_periodic_points(edge_a))},
            "b": {"fit_points": _round(edge_b[:: max(len(edge_b) // 100, 1)]),
                  **_nurbs_dict(fit_periodic_points(edge_b))},
        },
        "frame_lines": frame_lines,
    }


def _frame_at(s, u: float) -> dict[str, Any]:
    """Interpolated placement frame at continuous sample position u in [0, n).

    Linear interpolation between adjacent samples, then re-orthonormalised
    (x = tangent wins, y is squared up against it, z = x cross y).
    """
    n = len(s.points)
    i0 = int(np.floor(u)) % n
    i1 = (i0 + 1) % n
    f = u - np.floor(u)

    origin = (1 - f) * s.points[i0] + f * s.points[i1]
    x = (1 - f) * s.tangents[i0] + f * s.tangents[i1]
    x /= np.linalg.norm(x)
    y = (1 - f) * s.width_dirs[i0] + f * s.width_dirs[i1]
    y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    # axes at high precision: CAD kernels reject rotation matrices that are
    # even slightly non-orthonormal
    return {
        "s_fraction": round(float(u / n), 6),
        "origin": _round(origin),
        "x_axis": _round(x, 12),
        "y_axis": _round(y, 12),
        "z_axis": _round(z, 12),
    }


def build_connectors_section(
    strips,
    count: int | None = None,
    spacing: float | None = None,
    offset: float = 0.0,
) -> dict[str, Any]:
    """Connector placement frames, evenly spaced in arc length per component.

    count    total frames (split across components of a link), OR
    spacing  MAXIMUM arc distance in mm between frames: each component gets
             ceil(length / spacing) connectors at equal (smaller) spacing —
             250 mm around a 600 mm loop gives 3 at 200 mm, never 250+250+100.
    offset   arc-length shift in mm of the first frame along each component
             (choose where the joins land).

    Each frame is a full right-handed coordinate system matching the strip:
    x = tangent (along path), y = width direction, z = LED face normal.
    The Fusion script places component occurrences with these transforms —
    unlike Pattern Along Path, the rotation matches the strip by construction.
    """
    if (count is None) == (spacing is None):
        raise ValueError("give exactly one of count / spacing")

    frames = []
    per_component = []
    for ci, s in enumerate(strips):
        length = s.length
        n = len(s.points)
        if spacing is not None:
            if spacing <= 0:
                raise ValueError(f"spacing must be positive, got {spacing}")
            n_conn = max(1, int(np.ceil(length / spacing)))
        else:
            n_conn = max(1, round(count / len(strips)))
        step_mm = length / n_conn
        for j in range(n_conn):
            s_mm = (offset + j * step_mm) % length
            frame = _frame_at(s, s_mm / length * n)
            frame["component"] = ci
            frames.append(frame)
        per_component.append(
            {"component": ci, "count": n_conn, "spacing_mm": round(step_mm, 2)}
        )
    return {
        "count": len(frames),
        "requested": {"count": count, "spacing_mm": spacing, "offset_mm": offset},
        "per_component": per_component,
        "frames": frames,
    }


def build_mounts_section(strips, specs: list[tuple[int, float]]) -> dict[str, Any]:
    """Mount frames at explicit arc positions: (component, mm along path).

    Same frame convention as connectors (x = tangent, y = width, z = LED
    normal); the Fusion script turns each into a Joint Origin so a base can
    be attached with a single rigid joint — no manual plane construction.
    """
    frames = []
    for ci, s_mm in specs:
        if not 0 <= ci < len(strips):
            raise ValueError(f"mount c{ci + 1}: no such component")
        s = strips[ci]
        n = len(s.points)
        u = (s_mm % s.length) / s.length * n
        fr = _frame_at(s, u)
        fr["component"] = ci
        fr["s_mm"] = round(s_mm % s.length, 1)
        frames.append(fr)
    return {"count": len(frames), "frames": frames}


def build_document(
    design,  # FourierKnot | FourierLink
    *,
    report: PreflightReport,
    fit_points: int = 60,
    dense_points: int = 400,
    tube_diameter: float | None = None,
    strips: list[dict[str, Any]] | None = None,
    command: str | None = None,
    cli_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from knotgen.link import as_link
    from knotgen.nurbs import fit_periodic

    link = as_link(design)

    paths = []
    worst_dev = 0.0
    for ci, comp in enumerate(link.components):
        _, fit = comp.sample_arclength(fit_points)
        nurbs = fit_periodic(comp)
        worst_dev = max(worst_dev, max_deviation(nurbs, comp))
        paths.append(
            {
                "component": ci,
                "closed": True,
                "length_mm": round(comp.total_length(), 1),
                "segments": [
                    {
                        "type": "spline",
                        "closed": True,
                        "fit_points": _round(fit),
                        **_nurbs_dict(nurbs),
                    }
                ],
            }
        )

    dense = [
        _round(comp.sample_arclength(dense_points)[1]) for comp in link.components
    ]

    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": f"knotgen {__version__}",
        "command": command,
        "cli_options": cli_options,
        "name": link.name,
        "source": link.meta.get("source", "unknown"),
        "meta": {k: v for k, v in link.meta.items() if k != "style"},
        "params": link.meta.get("style", {}),
        "units": "mm",
        "n_components": link.n_components,
        "total_length_mm": round(link.total_length(), 1),
        "paths": paths,
        "dense_points": dense if link.n_components > 1 else dense[0],
        "fourier": {
            "components": [
                {"a": c.a.tolist(), "b": c.b.tolist()} for c in link.components
            ]
        },
        "checks": {
            "min_bend_radius_mm": round(report.min_bend_radius_mm, 3),
            "min_clearance_mm": round(report.min_clearance_mm, 3),
            "max_tube_diameter_mm": round(report.max_tube_diameter_mm, 3),
            "fit_max_deviation_mm": round(worst_dev, 4),
        },
    }
    if link.n_components == 1:
        # compatibility with schema v1 consumers (installed Fusion scripts)
        doc["path"] = paths[0]
        doc["fourier"]["a"] = link.components[0].a.tolist()
        doc["fourier"]["b"] = link.components[0].b.tolist()
    if tube_diameter is not None:
        doc["pipe_preview"] = {"diameter_mm": tube_diameter}
    if strips:
        doc["strips"] = strips
        doc["strip"] = strips[0]  # v1 compatibility for single-component files
    return doc


def export_json(path: str | Path, doc: dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(doc, indent=1))
    return path


def load_knot(path: str | Path):
    """Rebuild the design from an exported JSON file.

    Returns a FourierKnot for single-component files, FourierLink otherwise.
    """
    from knotgen.link import FourierLink

    doc = json.loads(Path(path).read_text())
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {doc.get('schema_version')}")
    meta = doc.get("meta", {})
    meta["style"] = doc.get("params", {})
    comps = [
        FourierKnot(
            a=np.array(c["a"]), b=np.array(c["b"]), name=doc.get("name", ""), meta=meta
        )
        for c in doc["fourier"]["components"]
    ]
    if len(comps) == 1:
        return comps[0]
    return FourierLink(components=comps, name=doc.get("name", ""), meta=meta)
