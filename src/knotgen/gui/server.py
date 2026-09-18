"""HTTP backend for `knotgen gui`.

Stdlib-only local server (binds 127.0.0.1). The GUI is a command builder:
the page assembles a `knotgen gen` argv, /api/generate parses it with the
real CLI parser and runs the same resolve -> style -> relax -> preflight
pipeline to return geometry for the viewer; downloads and exports are
served from a design cache via the CLI's own document assembly, so a GUI
file is byte-identical to the CLI export of the same command.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from knotgen.fourier import TAU

STATIC_DIR = Path(__file__).parent / "static"
DOCS_DIR = Path(__file__).parent.parent / "docs"

# curated example grids for the parametric families
TORUS_EXAMPLES = ["T(2,3)", "T(2,5)", "T(2,7)", "T(2,9)", "T(2,11)",
                  "T(3,4)", "T(3,5)", "T(3,7)", "T(4,5)",
                  "T(2,4)", "T(2,6)", "T(3,6)"]
WEAVING_EXAMPLES = ["W(3,2)", "W(3,3)", "W(3,4)", "W(3,5)", "W(3,7)",
                    "W(4,3)", "W(4,5)", "W(5,4)", "W(5,6)",
                    "W(2,4)", "W(3,6)", "W(4,6)"]

_NAME_SORT_RE = re.compile(r"^(L?)(\d+)([an]?)_?(\d+)$")


def _name_sort_key(name: str) -> tuple:
    m = _NAME_SORT_RE.match(name)
    if not m:
        return (2, 99, "", 0, name)
    link, crossings, letter, index = m.groups()
    return (1 if link else 0, int(crossings), letter, int(index), name)


def _parse_gen_argv(argv: list[str]):
    """Parse a GUI-built argv with the real CLI parser (defaults included)."""
    from knotgen.cli import KNOT_NAME_RE, build_parser

    argv = [str(a) for a in argv]
    if argv and KNOT_NAME_RE.match(argv[0]):
        argv.insert(0, "gen")
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            args = build_parser().parse_args(argv)
    except SystemExit:
        raise ValueError(err.getvalue().strip().splitlines()[-1]
                         if err.getvalue().strip() else "invalid arguments")
    if args.command != "gen":
        raise ValueError("only `gen` commands can be previewed")
    args.argv = argv[1:] if argv[0] == "gen" else argv
    return args


def api_catalogue() -> dict:
    from knotgen.registry import available
    from knotgen.sources import ideal

    curated = available()
    return {
        "curated": curated,
        "ideal": sorted(ideal.names(), key=_name_sort_key),
        "families": ["T(2,7)", "T(3,5)", "W(3,4)", "W(3,5)", "W(4,3)"],
    }


def _styled_from_args(args):
    """The gen pipeline up to a styled (and relaxed) curve — shared by
    /api/generate and /api/mesh."""
    from knotgen.registry import resolve
    from knotgen.transforms import apply_style

    try:
        knot = resolve(
            args.knot, source=args.source, variant=args.variant, rho=args.rho,
            layout=args.layout, aspect=args.aspect,
            braid_fraction=args.braid_fraction, lane_gap=args.lane_gap,
            braid_split=args.braid_split, wall=args.wall,
        )
    except KeyError as exc:
        raise ValueError(str(exc.args[0]) if exc.args else str(exc)) from None
    styled = apply_style(
        knot, width=args.width, breadth=args.breadth, depth=args.depth,
        tightness=args.tightness,
    )

    relax_info = None
    if args.relax:
        if args.tube is None:
            raise ValueError("--relax needs --tube (the diameter to open clearance for)")
        from knotgen.relax import relax

        # same depth-budget default as the CLI: an explicit depth is held
        # (relax-max-depth 0 lifts it) — the GUI must not quietly diverge
        max_depth = args.relax_max_depth
        if max_depth == 0:
            max_depth = None
        elif max_depth is None and args.depth is not None:
            max_depth = args.depth
        styled, relax_info = relax(
            styled, tube=args.tube, iterations=args.relax_iterations,
            max_depth=max_depth, push=args.relax_max, verbose=False,
        )
        relax_info = {k: (float(v) if hasattr(v, "item") or isinstance(v, float) else v)
                      for k, v in relax_info.items()}

    if args.wall:
        from knotgen.transforms import floor_z

        styled = floor_z(styled)
    return knot, styled, relax_info


_DESIGN_CACHE: dict[tuple, tuple] = {}


def _design_for(argv: list) -> tuple:
    """(args, knot, styled, relax_info) for an argv — cached, so a Download
    after a Generate serves EXACTLY the design on screen instead of
    re-running the pipeline (and any --relax) a second time."""
    key = tuple(str(a) for a in argv)
    if key in _DESIGN_CACHE:
        return _DESIGN_CACHE[key]
    args = _parse_gen_argv(list(key))
    result = (args, *_styled_from_args(args))
    if len(_DESIGN_CACHE) >= 8:
        _DESIGN_CACHE.pop(next(iter(_DESIGN_CACHE)))
    _DESIGN_CACHE[key] = result
    return result


def api_groups() -> dict:
    """The catalogue organised for the graphical browser: groups of names,
    each pointing at the doc page that explains its naming system."""
    from knotgen.registry import available
    from knotgen.sources import ideal

    groups = [
        {"id": "classic", "title": "Classic knots 3₁–8₂₁",
         "doc": "classic", "names": [r["name"] for r in available()]},
        {"id": "torus", "title": "Torus knots T(p,q)", "doc": "torus",
         "names": TORUS_EXAMPLES, "parametric": True},
        {"id": "weaving", "title": "Weaving / Turk's head W(p,q)",
         "doc": "weaving", "names": WEAVING_EXAMPLES, "parametric": True},
    ]
    for c in (9, 10):
        groups.append({"id": f"k{c}", "title": f"{c}-crossing knots",
                       "doc": "ht",
                       "names": sorted(ideal.names(crossings=c, links=False),
                                       key=_name_sort_key)})
    names11 = sorted(ideal.names(crossings=11, links=False), key=_name_sort_key)
    groups.append({"id": "k11a", "title": "11-crossing knots, alternating",
                   "doc": "ht", "names": [n for n in names11 if "a" in n]})
    groups.append({"id": "k11n", "title": "11-crossing knots, non-alternating",
                   "doc": "ht", "names": [n for n in names11 if "n" in n]})
    for c in range(2, 12):
        sub = sorted(ideal.names(crossings=c, links=True), key=_name_sort_key)
        if sub:
            groups.append({"id": f"l{c}", "title": f"Links · {c} crossings",
                           "doc": "links", "names": sub})
    return {"groups": groups}


_THUMB_CACHE: dict[str, list] = {}


def api_thumbs(payload: dict) -> dict:
    """Small preview polylines (per-component, xyz) for a batch of names.

    Raw curve shapes at default parameters — the client normalises and
    draws them; anything unresolvable is reported, not fatal.
    """
    from knotgen.link import as_link
    from knotgen.registry import resolve

    names = [str(n) for n in (payload.get("names") or [])][:80]
    out: dict[str, list | None] = {}
    for name in names:
        if name in _THUMB_CACHE:
            out[name] = _THUMB_CACHE[name]
            continue
        try:
            link = as_link(resolve(name))
            comps = []
            for comp in link.components:
                t = np.linspace(0.0, TAU, 96, endpoint=False)
                pts = comp.eval(t)
                comps.append([[round(float(c), 3) for c in p] for p in pts])
            _THUMB_CACHE[name] = comps
            out[name] = comps
        except Exception:
            out[name] = None
    return {"thumbs": out}


def api_doc(doc_id: str) -> dict:
    path = DOCS_DIR / f"{doc_id}.md"
    if not re.fullmatch(r"[a-z]+", doc_id) or not path.exists():
        raise ValueError(f"no doc page {doc_id!r}")
    return {"id": doc_id, "markdown": path.read_text()}


def api_generate(payload: dict) -> dict:
    from knotgen.checks import preflight
    from knotgen.link import as_link

    args, knot, styled, relax_info = _design_for(payload.get("argv") or [])
    link = as_link(styled)
    report = preflight(styled, tube_diameter=args.tube)
    e = styled.extents()
    style = styled.meta.get("style", {})

    warnings: list[str] = []
    aspect_nat = style.get("natural_aspect", 0.0)
    if style.get("depth_auto") and aspect_nat > 0.3:
        warnings.append(
            f"depth auto {style['depth']:.0f} mm — this embedding is genuinely 3D "
            f"(natural z/xy {aspect_nat:.2f}), keeping its proportions"
        )
    elif not style.get("depth_auto") and aspect_nat > 0.3:
        natural = aspect_nat * args.width
        if args.depth is not None and args.depth < 0.5 * natural:
            warnings.append(
                f"depth {args.depth:g} mm crushes a fully-3D conformation "
                f"(natural ≈ {natural:.0f} mm) — expect jagged tight bends; "
                f"leave depth blank to keep proportions"
            )
    if relax_info is not None and not relax_info.get("converged", True):
        warnings.append(
            f"relax plateaued short of the target: this layout tops out around "
            f"a {relax_info['max_tube_est']:g} mm tube"
        )

    samples = int(payload.get("samples", 400))
    samples = max(64, min(samples, 1200))
    components = []
    for comp in link.components:
        _, pts = comp.sample_arclength(samples)
        components.append([[round(float(c), 3) for c in p] for p in pts])

    strips = None
    strip_metrics = None
    if args.strip:
        from knotgen.frames import compute_frames

        frames = [
            compute_frames(comp, follow=args.follow, light_dir=args.light_dir,
                           twist_smooth=args.twist_smooth)
            for comp in link.components
        ]
        strips = []
        for f in frames:
            left, right = f.edges(args.strip)
            strips.append({
                "left": [[round(float(c), 3) for c in p] for p in left],
                "right": [[round(float(c), 3) for c in p] for p in right],
            })
        per = [f.metrics() for f in frames]
        strip_metrics = {
            "max_twist_deg_per_cm": max(m["max_twist_deg_per_cm"] for m in per),
            "min_edgewise_bend_radius_mm": min(m["min_edgewise_bend_radius_mm"] for m in per),
            "min_inplane_bend_radius_mm": min(m["min_inplane_bend_radius_mm"] for m in per),
        }
        strip_metrics = {k: (v if v != float("inf") else None)
                         for k, v in strip_metrics.items()}

    sym = styled.rotational_symmetry_order()
    return {
        "name": styled.name,
        "source": knot.meta.get("source"),
        "variant": knot.meta.get("variant"),
        "n_components": link.n_components,
        "symmetry": int(sym),
        "extents": {k: float(v) for k, v in e.items()},
        "length_mm": float(styled.total_length()),
        "component_lengths_mm": [float(c.total_length()) for c in link.components],
        "components": components,
        "strips": strips,
        "strip_metrics": strip_metrics,
        "tube_mm": args.tube,
        "report": {
            "summary": report.summary(),
            "ok_for_tube": (None if args.tube is None else bool(report.ok_for_tube)),
        },
        "relax": relax_info,
        "warnings": warnings,
    }


def _safe_filename(name: str, suffix: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name.strip()) or "knot"
    if not name.lower().endswith(suffix):
        name += suffix
    return Path(name).name


def _document_for(argv: list, out_name: str, force: bool) -> dict:
    """The full export document for an argv, from the cached design — the
    same assembly the CLI uses, with the command recorded as the user
    would type it. Raises _TubeFailure when the check fails without force."""
    from knotgen.checks import preflight
    from knotgen.cli import assemble_document, compute_strip_frames
    from knotgen.link import as_link

    args, _, styled, _ = _design_for(argv)
    if args.connectors and args.connector_spacing:
        raise ValueError("give either connectors count OR max spacing, not both")
    report = preflight(styled, tube_diameter=args.tube)
    if args.tube is not None and not report.ok_for_tube and not force:
        raise _TubeFailure(report.summary())
    frames = compute_strip_frames(as_link(styled), args)
    args.argv = [*getattr(args, "argv", list(argv)), "--out", out_name]
    try:
        return assemble_document(styled, report, args, frames)
    finally:
        args.argv = args.argv[:-2]  # the cached args must stay pristine


class _TubeFailure(Exception):
    """Tube check failed and the caller did not force."""


def api_export(payload: dict) -> dict:
    """Write the export JSON into output/ (like the CLI's --out), from the
    cached design — no second pipeline run."""
    argv = [str(a) for a in (payload.get("argv") or []) if a != "--preview"]
    if "--out" in argv:
        i = argv.index("--out")
        out_name = argv[i + 1] if i + 1 < len(argv) else None
        argv = argv[:i] + argv[i + 2:]
    else:
        out_name = payload.get("filename")
    if not out_name:
        raise ValueError("export needs an output filename (--out)")
    from knotgen.cli import _resolve_out
    from knotgen.export import export_json

    try:
        doc = _document_for(argv, out_name, bool(payload.get("force")))
    except _TubeFailure as exc:
        return {"returncode": 1,
                "log": f"{exc}\n  NOT exporting — tube does not fit"}
    out = export_json(_resolve_out(out_name), doc)
    return {"returncode": 0,
            "log": f"wrote {out}  (fit deviation "
                   f"{doc['checks']['fit_max_deviation_mm']} mm)"}


def api_download(payload: dict) -> dict:
    """Same document as api_export, returned for a browser download."""
    argv = [str(a) for a in (payload.get("argv") or []) if a != "--preview"]
    if "--out" in argv:
        raise ValueError("download builds its own --out; leave it off the argv")
    filename = _safe_filename(payload.get("filename") or "knot.json", ".json")
    try:
        doc = _document_for(argv, filename, bool(payload.get("force")))
    except _TubeFailure as exc:
        return {"returncode": 1,
                "log": f"{exc}\n  NOT exporting — tube does not fit"}
    content = json.dumps(doc, indent=1)
    return {"returncode": 0, "filename": filename, "content": content,
            "log": f"built {filename}  (fit deviation "
                   f"{doc['checks']['fit_max_deviation_mm']} mm)"}


def api_mesh(payload: dict) -> tuple[str, bytes]:
    """Binary STL of the swept tube, for a browser download — built from
    the cached design, i.e. exactly the geometry in the viewer."""
    from knotgen.checks import preflight
    from knotgen.mesh import design_mesh, stl_bytes

    args, _, styled, _ = _design_for(payload.get("argv") or [])
    if args.tube is None:
        raise ValueError("mesh export needs a tube diameter (set tube ⌀ mm)")
    report = preflight(styled, tube_diameter=args.tube)
    if not report.ok_for_tube and not payload.get("force"):
        raise ValueError("tube does not fit — the mesh would self-intersect "
                         "(see the report; try --relax or a smaller tube)")
    detail = float(payload.get("detail") or args.mesh_detail)
    verts, faces = design_mesh(styled, tube_diameter=args.tube, detail=detail)
    filename = _safe_filename(payload.get("filename") or f"{styled.name}.stl", ".stl")
    return filename, stl_bytes(verts, faces, name=styled.name)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # keep the terminal quiet
        pass

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path in ("/", "/index.html"):
            body = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/catalogue":
            self._json(api_catalogue())
        elif self.path == "/api/groups":
            self._json(api_groups())
        elif self.path.startswith("/api/doc/"):
            try:
                self._json(api_doc(self.path.rsplit("/", 1)[1]))
            except ValueError as exc:
                self._json({"error": str(exc)}, status=404)
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad JSON"}, status=400)
            return
        try:
            if self.path == "/api/generate":
                self._json(api_generate(payload))
            elif self.path == "/api/thumbs":
                self._json(api_thumbs(payload))
            elif self.path == "/api/export":
                self._json(api_export(payload))
            elif self.path == "/api/download":
                self._json(api_download(payload))
            elif self.path == "/api/mesh":
                filename, body = api_mesh(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, status=404)
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:  # surfaced in the GUI report panel
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)


def run(port: int = 8642, open_browser: bool = True) -> int:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)  # port busy
    host, actual_port = server.server_address[:2]
    url = f"http://{host}:{actual_port}/"
    print(f"knotgen gui at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0
