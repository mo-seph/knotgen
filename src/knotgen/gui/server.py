"""HTTP backend for `knotgen gui`.

Stdlib-only local server (binds 127.0.0.1). The GUI is a command builder:
the page assembles a `knotgen gen` argv, /api/generate parses it with the
real CLI parser and runs the same resolve -> style -> relax -> preflight
pipeline to return geometry for the viewer, and /api/export hands the argv
verbatim to `knotgen.cli.main` — an export from the GUI *is* a CLI run.
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

STATIC_DIR = Path(__file__).parent / "static"

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

        styled, relax_info = relax(
            styled, tube=args.tube, iterations=args.relax_iterations,
            max_depth=args.relax_max_depth, push=args.relax_max, verbose=False,
        )
        relax_info = {k: (float(v) if hasattr(v, "item") or isinstance(v, float) else v)
                      for k, v in relax_info.items()}

    if args.wall:
        from knotgen.transforms import floor_z

        styled = floor_z(styled)
    return knot, styled, relax_info


def api_generate(payload: dict) -> dict:
    args = _parse_gen_argv(payload.get("argv") or [])

    from knotgen.checks import preflight
    from knotgen.link import as_link

    knot, styled, relax_info = _styled_from_args(args)
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


def api_export(payload: dict) -> dict:
    """Run the argv through the real CLI entry point, capturing its output."""
    from knotgen.cli import main as cli_main

    argv = [str(a) for a in (payload.get("argv") or [])]
    # the GUI is the preview; these would pop windows on the server side
    argv = [a for a in argv if a not in ("--preview",)]
    if "--out" not in argv:
        raise ValueError("export needs an output filename (--out)")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(list(argv))
    return {"returncode": int(code), "log": buf.getvalue()}


def _safe_filename(name: str, suffix: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name.strip()) or "knot"
    if not name.lower().endswith(suffix):
        name += suffix
    return Path(name).name


def api_download(payload: dict) -> dict:
    """Same run as api_export, but the JSON comes back for a browser
    download instead of landing in output/."""
    import tempfile

    from knotgen.cli import main as cli_main

    argv = [str(a) for a in (payload.get("argv") or []) if a != "--preview"]
    if "--out" in argv:
        raise ValueError("download builds its own --out; leave it off the argv")
    filename = _safe_filename(payload.get("filename") or "knot.json", ".json")
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / filename
        with contextlib.redirect_stdout(buf):
            code = cli_main([*argv, "--out", str(tmp)])
        if code != 0 or not tmp.exists():
            return {"returncode": int(code), "log": buf.getvalue()}
        # the recorded command should read as the user would type it
        content = tmp.read_text().replace(str(tmp), filename)
    log = buf.getvalue().replace(str(tmp), filename)
    return {"returncode": 0, "log": log, "filename": filename, "content": content}


def api_mesh(payload: dict) -> tuple[str, bytes]:
    """Binary STL of the swept tube, for a browser download."""
    args = _parse_gen_argv(payload.get("argv") or [])
    if args.tube is None:
        raise ValueError("mesh export needs a tube diameter (set tube ⌀ mm)")

    from knotgen.checks import preflight
    from knotgen.mesh import design_mesh, stl_bytes

    _, styled, _ = _styled_from_args(args)
    report = preflight(styled, tube_diameter=args.tube)
    if not report.ok_for_tube:
        raise ValueError("tube does not fit — the mesh would self-intersect "
                         "(see the report; try --relax or a smaller tube)")
    verts, faces = design_mesh(styled, tube_diameter=args.tube)
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
