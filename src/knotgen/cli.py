"""knotgen command-line interface.

    knotgen 5_1 --width 300 --depth 25 --preview
    knotgen gen 5_1 --width 300 --depth 25 --tube 16 --out pentafoil.json
    knotgen list
    knotgen check pentafoil.json --tube 18
    knotgen preview pentafoil.json
    knotgen identify 5_1
"""

from __future__ import annotations

import argparse
import re
import sys

from knotgen import __version__

KNOT_NAME_RE = re.compile(
    r"^(\d+_\d+|K?\d+[an]_?\d+|L\d+[an]\d+|[tTwW]\(\d+,\s*\d+\))$"
)


def _knot_catalogue_epilog() -> str:
    try:
        from knotgen.registry import available

        names = [row["name"] for row in available()]
    except Exception:
        return ""
    lines, line = [], "  "
    for name in names:
        if len(line) + len(name) + 2 > 78:
            lines.append(line.rstrip(", "))
            line = "  "
        line += name + ", "
    lines.append(line.rstrip(", "))
    return (
        "curated symmetric knots (see `knotgen list -v` for variants and symmetry):\n"
        + "\n".join(lines)
        + '\n...plus torus knots "T(p,q)", weaving/Turk\'s head knots "W(p,q)",\n'
        "and the ideal-conformation catalogue: every knot to 11 crossings\n"
        '("9_35", "11a42", "11n34") and every link of the Thistlethwaite table\n'
        '("L2a1".."L11n459") — enumerate with `knotgen list --crossings N [--links]`'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knotgen",
        description="Generate mathematical knots as closed 3D curves for Fusion 360.",
        epilog=_knot_catalogue_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"knotgen {__version__}")
    sub = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="{gen,list,check,preview,identify}",
        help="a knot name on its own implies `gen`: `knotgen 5_1 ...`",
    )

    gen = sub.add_parser(
        "gen",
        help="generate a knot curve (also implied by `knotgen 5_1 ...`)",
        description="Generate a knot curve: run pre-flight checks, optionally "
                    "preview it and export a JSON for the Fusion KnotImport script.",
        epilog=_knot_catalogue_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gen.add_argument("knot",
                     help='knot or link name: "5_1", "11n34", "L6a4", torus "T(2,7)", '
                          'weaving "W(3,5)"')
    gen.add_argument("--width", type=float, default=300.0, metavar="MM",
                     help="knot size: diameter of the circumscribed circle in the xy plane, "
                          "applied as a uniform in-plane scale so n-fold symmetry stays exact "
                          "(default 300)")
    gen.add_argument("--breadth", type=float, default=None, metavar="MM",
                     help="separate y extent (deliberately breaks n-fold symmetry for oval "
                          "layouts; default: keep uniform)")
    gen.add_argument("--depth", type=float, default=None, metavar="MM",
                     help="z extent = strand separation at crossings. Default auto: "
                          "25 for flat/2.5D embeddings (fremlin, torus, weaving); "
                          "fully-3D ideal conformations keep their natural "
                          "proportions instead — squashing those creates jagged "
                          "near-cusps")
    gen.add_argument("--tightness", type=float, default=0.0, metavar="T",
                     help="corner aesthetics, -1..1: negative rounds corners off (relaxed "
                          "rope), positive pinches lobes into small loops (petal look); "
                          "sane range is about -0.6..0.6 (default 0)")
    gen.add_argument("--source", choices=["auto", "fremlin", "torus", "ideal"],
                     default="auto",
                     help="curve source: 'fremlin' = symmetrised embeddings (knots to 8 "
                          "crossings), 'torus' = exact torus-knot formula, 'ideal' = "
                          "tightened conformations (all knots to 11 crossings + all links, "
                          "organic rope look), 'auto' = fremlin, then torus, then ideal")
    gen.add_argument("--variant", default=None, metavar="V",
                     help="Fremlin embedding variant (e.g. p, d, u) — alternative "
                          "symmetrisations of the same knot; see `knotgen list`")
    gen.add_argument("--rho", type=float, default=0.4, metavar="R",
                     help="torus/weaving rosette: tube ratio r/R in (0,1); bigger = "
                          "deeper lobes (default 0.4)")
    gen.add_argument("--layout", choices=["rosette", "racetrack"], default="rosette",
                     help="weaving knots W(p,q) only: 'rosette' = round, q-fold "
                          "symmetric (default); 'racetrack' = stadium shape with all "
                          "crossings woven along one straight, as drawn in the papers")
    gen.add_argument("--aspect", type=float, default=2.0, metavar="A",
                     help="racetrack layout: straight length / stadium width "
                          "(default 2.0 — bigger stretches the oval)")
    gen.add_argument("--braid-fraction", type=float, default=0.8, metavar="F",
                     help="racetrack layout: how much of the braid straight the "
                          "crossings occupy, 0.05-1: small = tight woven patch in the "
                          "middle, 1 = spread along the whole straight (default 0.8)")
    gen.add_argument("--lane-gap", type=float, default=None, metavar="F",
                     help="racetrack layout: lane spacing as a fraction of the "
                          "stadium half-width (default auto). Smaller = gentler "
                          "crossings but tighter clearance between the nested lanes")
    gen.add_argument("--braid-split", type=float, default=0.0, metavar="F",
                     help="racetrack layout: fraction of the crossings moved to the "
                          "other straight, 0-1 (0.5 = half on each side; same knot "
                          "either way — splitting the cyclic braid word is an isotopy)")
    gen.add_argument("--wall", action="store_true",
                     help="racetrack layout: wall-mount mode — under-strands stay "
                          "flat at lane level and only over-strands arch up, and the "
                          "whole knot is shifted so the flat plane sits at z = 0 "
                          "(lies against a wall)")
    gen.add_argument("--tube", type=float, default=None, metavar="MM",
                     help="intended tube/profile diameter: refuses to export if it would "
                          "self-intersect, and enables the Pipe preview offer in Fusion")
    gen.add_argument("--fit-points", type=int, default=60, metavar="N",
                     help="points for the editable fitted-spline representation in the "
                          "export (default 60)")
    gen.add_argument("--strip", type=float, default=None, metavar="MM",
                     help="LED strip width in mm: computes orientation frames and "
                          "exports a sweepable strip surface")
    gen.add_argument("--follow", type=float, default=1.0, metavar="F",
                     help="strip orientation, 0..1: 1 = follow curvature (no edgewise "
                          "bend, twists freely), 0 = face the light direction (default 1)")
    gen.add_argument("--light-dir", default="up", metavar="DIR",
                     help="light direction for --follow < 1: up, down, out, in "
                          "(radial), or 'x,y,z' (default up)")
    gen.add_argument("--twist-smooth", type=float, default=5.0, metavar="MM",
                     help="twist stiffness: arc-length scale over which strip angle is "
                          "smoothed; 0 = off (default 5)")
    gen.add_argument("--frame-count", type=int, default=25, metavar="N",
                     help="control lines exported for Fusion's editable surface mode "
                          "(default 25)")
    gen.add_argument("--connectors", type=int, default=None, metavar="N",
                     help="export N evenly-spaced connector frames (exact position + "
                          "orientation matching the strip) so the Fusion script can "
                          "place your connector component at each; needs --strip")
    gen.add_argument("--connector-spacing", type=float, default=None, metavar="MM",
                     help="alternative to --connectors: MAXIMUM arc distance between "
                          "connector frames — the count is rounded up and the actual "
                          "spacing shrunk to fit evenly (250 around a 600 mm loop "
                          "gives 3 at 200, never 250+250+100)")
    gen.add_argument("--connector-offset", type=float, default=0.0, metavar="MM",
                     help="shift the first connector this far along each component "
                          "(choose where the joins land; default 0)")
    gen.add_argument("--out", default=None, metavar="FILE",
                     help="write the JSON export (for the Fusion KnotImport script); "
                          "a bare filename goes into output/, move keepers to designs/")
    gen.add_argument("--preview", action="store_true", help="open a 3D preview window")
    gen.add_argument("--save-png", default=None, metavar="FILE",
                     help="save the preview as a PNG instead of opening a window "
                          "(bare filenames go into output/)")

    lst = sub.add_parser(
        "list",
        help="list available knots and links",
        description="List available knots and links. By default shows the curated "
                    "table (Fremlin symmetric embeddings + torus knots) plus a "
                    "summary of the ideal-conformation catalogue; use --crossings "
                    "and/or --links to enumerate that catalogue.",
    )
    lst.add_argument("--verbose", "-v", action="store_true",
                     help="also show each knot's rotational symmetry order (slower)")
    lst.add_argument("--crossings", type=int, default=None, metavar="N",
                     help="list all ideal-catalogue entries with N crossings")
    lst.add_argument("--links", action="store_true",
                     help="restrict --crossings listing to links (multi-component)")

    chk = sub.add_parser(
        "check",
        help="re-run feasibility checks on an exported JSON",
        description="Re-run the pre-flight checks (bend radius, strand clearance) "
                    "on a previously exported JSON — e.g. to test a different tube "
                    "or profile-diagonal size without regenerating.",
    )
    chk.add_argument("file", help="a JSON file written by `knotgen gen --out`")
    chk.add_argument("--tube", type=float, default=None, metavar="MM",
                     help="tube/profile diameter to test against")

    prv = sub.add_parser(
        "preview",
        help="preview an exported JSON",
        description="Re-open the 3D viewer on a previously exported JSON "
                    "(including its strip surface, if it has one).",
    )
    prv.add_argument("file", help="a JSON file written by `knotgen gen --out`")
    prv.add_argument("--tube", type=float, default=None, metavar="MM",
                     help="overlay a translucent tube of this diameter")
    prv.add_argument("--save-png", default=None, metavar="FILE",
                     help="save the preview as a PNG instead of opening a window")

    idf = sub.add_parser(
        "identify",
        help="verify knot type with pyknotid (optional dependency)",
        description="Topologically identify a generated curve with pyknotid — "
                    "useful as a sanity check after extreme styling. Requires "
                    'pyknotid: uv pip install "git+https://github.com/SPOCKnots/pyknotid"',
    )
    idf.add_argument("knot", help="knot name or an exported JSON file")
    idf.add_argument("--tightness", type=float, default=0.0, metavar="T",
                     help="apply this tightness before identifying (to confirm an "
                          "extreme value hasn't changed the knot type)")

    return parser


def _resolve_out(path_str: str) -> "Path":
    """Bare filenames land in output/ (created on demand); explicit paths
    are respected."""
    from pathlib import Path

    p = Path(path_str)
    if str(p.parent) == ".":
        outdir = Path("output")
        outdir.mkdir(exist_ok=True)
        return outdir / p
    return p


def cmd_gen(args: argparse.Namespace) -> int:
    from knotgen.checks import preflight
    from knotgen.registry import resolve
    from knotgen.transforms import apply_style

    knot = resolve(
        args.knot,
        source=args.source,
        variant=args.variant,
        rho=args.rho,
        layout=args.layout,
        aspect=args.aspect,
        braid_fraction=args.braid_fraction,
        lane_gap=args.lane_gap,
        braid_split=args.braid_split,
        wall=args.wall,
    )
    styled = apply_style(
        knot,
        width=args.width,
        breadth=args.breadth,
        depth=args.depth,
        tightness=args.tightness,
    )
    if args.wall:
        from knotgen.transforms import floor_z

        styled = floor_z(styled)

    from knotgen.link import as_link

    styled_link = as_link(styled)
    sym = styled.rotational_symmetry_order()
    n_comp = styled_link.n_components
    print(f"{styled.name}  (source: {knot.meta.get('source')}"
          + (f", variant {knot.meta['variant']!r}" if knot.meta.get("variant") else "")
          + (f", {n_comp} components" if n_comp > 1 else "")
          + (f", {sym}-fold symmetric" if sym > 1 else "") + ")")
    e = styled.extents()
    style = styled.meta.get("style", {})
    print(f"  size: ⌀{e['xy_diameter']:.0f} circumcircle, "
          f"bbox {e['x_extent']:.0f} x {e['y_extent']:.0f} x {e['z_extent']:.1f} mm, "
          f"path length {styled.total_length():.0f} mm")
    if n_comp > 1:
        lengths = ", ".join(
            f"c{ci + 1}: {c.total_length():.0f}"
            for ci, c in enumerate(styled_link.components)
        )
        print(f"  component lengths: {lengths} mm")
    aspect = style.get("natural_aspect", 0.0)
    if style.get("depth_auto") and aspect > 0.3:
        print(f"  depth: auto {style['depth']:.0f} mm — this embedding is genuinely "
              f"3D (natural z/xy {aspect:.2f}), keeping its proportions")
    elif not style.get("depth_auto") and aspect > 0.3:
        natural = aspect * args.width
        if args.depth < 0.5 * natural:
            print(f"  ! depth {args.depth:g} mm crushes a fully-3D conformation "
                  f"(natural ≈ {natural:.0f} mm) — expect jagged tight bends; "
                  f"omit --depth to keep proportions")

    report = preflight(styled, tube_diameter=args.tube)
    print(report.summary())

    strip_frames = None
    if args.strip:
        from knotgen.frames import compute_frames

        strip_frames = [
            compute_frames(
                comp,
                follow=args.follow,
                light_dir=args.light_dir,
                twist_smooth=args.twist_smooth,
            )
            for comp in styled_link.components
        ]
        agg = {
            "max_twist_deg_per_cm": max(
                f.metrics()["max_twist_deg_per_cm"] for f in strip_frames
            ),
            "min_edgewise_bend_radius_mm": min(
                f.metrics()["min_edgewise_bend_radius_mm"] for f in strip_frames
            ),
            "min_inplane_bend_radius_mm": min(
                f.metrics()["min_inplane_bend_radius_mm"] for f in strip_frames
            ),
        }
        print(f"  strip ({args.strip:g} mm wide, follow {args.follow:g}, "
              f"light {args.light_dir}"
              + (f", worst of {n_comp} components" if n_comp > 1 else "") + "):")
        print(f"    max twist:            {agg['max_twist_deg_per_cm']:8.1f} deg/cm")
        print(f"    min edgewise bend r:  {agg['min_edgewise_bend_radius_mm']:8.1f} mm"
              "   (strip hates this — bigger is better)")
        print(f"    min in-plane bend r:  {agg['min_inplane_bend_radius_mm']:8.1f} mm"
              "   (check your strip's rated bend radius)")

    failed = args.tube is not None and not report.ok_for_tube

    if args.out and not failed:
        from knotgen.export import build_document, build_strip_section, export_json

        strip_sections = None
        if strip_frames is not None:
            strip_sections = [
                build_strip_section(
                    f,
                    width=args.strip,
                    light_dir_spec=args.light_dir,
                    frame_count=args.frame_count,
                    component=ci,
                )
                for ci, f in enumerate(strip_frames)
            ]
        import shlex

        cli_options = {
            k: v for k, v in vars(args).items()
            if k not in ("command", "argv") and not k.startswith("_")
        }
        doc = build_document(
            styled,
            report=report,
            fit_points=args.fit_points,
            tube_diameter=args.tube,
            strips=strip_sections,
            command="knotgen " + shlex.join(getattr(args, "argv", [])),
            cli_options=cli_options,
        )
        want_connectors = args.connectors or args.connector_spacing
        if want_connectors and strip_frames is not None:
            from knotgen.export import build_connectors_section

            if args.connectors and args.connector_spacing:
                print("  give either --connectors or --connector-spacing, not both")
                return 1
            doc["connectors"] = build_connectors_section(
                strip_frames,
                count=args.connectors,
                spacing=args.connector_spacing,
                offset=args.connector_offset,
            )
            for pc in doc["connectors"]["per_component"]:
                comp_label = (
                    f"component {pc['component'] + 1}: " if n_comp > 1 else ""
                )
                print(f"  connectors: {comp_label}{pc['count']} at "
                      f"{pc['spacing_mm']:g} mm spacing"
                      + (f" (max {args.connector_spacing:g})"
                         if args.connector_spacing else ""))
        elif want_connectors:
            print("  (--connectors/--connector-spacing need --strip for "
                  "orientation frames; skipped)")
        out = export_json(_resolve_out(args.out), doc)
        print(f"  wrote {out}  (fit deviation {doc['checks']['fit_max_deviation_mm']} mm)")
    elif args.out and failed:
        print("  NOT exporting — tube does not fit (see above)")

    if args.preview or args.save_png:
        from knotgen.viz import preview

        preview(
            styled,
            tube_diameter=args.tube,
            strip=strip_frames,
            strip_width=args.strip,
            save=str(_resolve_out(args.save_png)) if args.save_png else None,
            show=args.preview,
            title=f"{styled.name}  {e['x_extent']:.0f}x{e['y_extent']:.0f}x{e['z_extent']:.0f}mm",
        )

    return 1 if failed else 0


def cmd_list(args: argparse.Namespace) -> int:
    from knotgen.registry import available, resolve
    from knotgen.sources import ideal

    if args.crossings is not None:
        names = ideal.names(
            crossings=args.crossings, links=True if args.links else None
        )
        if not names:
            print(f"nothing with {args.crossings} crossings"
                  + (" (links)" if args.links else ""))
            return 1
        for name in names:
            print(name)
        print(f"\n{len(names)} entries (source: ideal — tightened conformations)")
        return 0

    rows = available()
    if not rows:
        print("no knots available — is the Fremlin data vendored?")
        return 1
    print(f"{'knot':<8} {'sources':<24} {'variants':<16}" + ("symmetry" if args.verbose else ""))
    for row in rows:
        variants = ",".join(v if v else "(default)" for v in row["variants"]) or "-"
        line = f"{row['name']:<8} {'/'.join(row['sources']):<24} {variants:<16}"
        if args.verbose:
            try:
                sym = resolve(row["name"]).rotational_symmetry_order()
                line += f"{sym}-fold" if sym > 1 else "none"
            except Exception:
                line += "?"
        print(line)

    counts = ideal.catalogue_summary()
    n_knots = sum(v for k, v in counts.items() if not k.startswith("IdealLinks"))
    n_links = sum(v for k, v in counts.items() if k.startswith("IdealLinks"))
    print(f"\nplus the ideal-conformation catalogue: {n_knots} knots to 11 crossings "
          f"(9_1..9_49, 10_1..10_166, 11a1..11a367, 11n1..11n185)\n"
          f"and {n_links} links (L2a1 .. L11n459) — `knotgen list --crossings N [--links]`\n"
          f'and parametric families: torus "T(p,q)", weaving/Turk\'s head "W(p,q)"')
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from knotgen.checks import preflight
    from knotgen.export import load_knot

    knot = load_knot(args.file)
    print(f"{knot.name}  (from {args.file})")
    report = preflight(knot, tube_diameter=args.tube)
    print(report.summary())
    return 1 if (args.tube is not None and not report.ok_for_tube) else 0


def cmd_preview(args: argparse.Namespace) -> int:
    import json

    from knotgen.export import load_knot
    from knotgen.viz import preview

    knot = load_knot(args.file)
    strip_frames = None
    strip_width = None
    doc = json.loads(open(args.file).read())
    strip_docs = doc.get("strips") or ([doc["strip"]] if doc.get("strip") else [])
    if strip_docs:
        from knotgen.frames import compute_frames
        from knotgen.link import as_link

        p = strip_docs[0]["params"]
        strip_width = strip_docs[0]["width_mm"]
        strip_frames = [
            compute_frames(
                comp,
                follow=p.get("follow", 1.0),
                light_dir=p.get("light_dir", "up"),
                twist_smooth=p.get("twist_smooth", 5.0),
            )
            for comp in as_link(knot).components
        ]
    preview(
        knot,
        tube_diameter=args.tube,
        strip=strip_frames,
        strip_width=strip_width,
        save=args.save_png,
        show=not args.save_png,
    )
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    try:
        from pyknotid.spacecurves import Knot  # type: ignore
    except ImportError:
        print(
            "pyknotid is not installed. Install it with:\n"
            '  uv pip install "git+https://github.com/SPOCKnots/pyknotid"'
        )
        return 1

    if args.knot.endswith(".json"):
        from knotgen.export import load_knot

        knot = load_knot(args.knot)
    else:
        from knotgen.registry import resolve
        from knotgen.transforms import apply_style

        knot = apply_style(
            resolve(args.knot), width=300, depth=25, tightness=args.tightness
        )
    _, pts = knot.sample_arclength(400)
    k = Knot(pts, verbose=False)
    print(f"{knot.name}: pyknotid identifies {k.identify()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `knotgen 5_1 ...` is shorthand for `knotgen gen 5_1 ...`
    if argv and KNOT_NAME_RE.match(argv[0]):
        argv.insert(0, "gen")

    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = argv  # recorded into exports for reproducibility
    if args.command is None:
        parser.print_help()
        return 0
    handler = {
        "gen": cmd_gen,
        "list": cmd_list,
        "check": cmd_check,
        "preview": cmd_preview,
        "identify": cmd_identify,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
