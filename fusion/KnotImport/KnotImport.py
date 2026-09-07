"""KnotImport — import a knotgen JSON as continuous 3D path(s) in Fusion 360.

Flow: pick the JSON file, then ONE dialog with everything:
  * curve type (exact NURBS / editable fitted spline)
  * strip surface mode (quick rung loft / editable angle-lines / none)
  * pipe preview on/off
  * profile region(s) to sweep — a real Fusion selection input, so composite
    profiles work — plus a reference sketch POINT
  * connector placement on/off (+ 180-degree flip toggle)

Alignment convention (one reference drives everything): author your profile
sketch inside the connector component; pick a reference point in that sketch
(e.g. its origin point). That point lands ON the path; the sketch's x-axis
goes ACROSS the strip (width) and its y-axis points OUT of the LED face.
The swept body and every placed connector share this mapping, and the
connector component is taken from the selected point's parent — no name
typing.

Install: copy or symlink this folder into
  ~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/Scripts/
"""

import json
import traceback

import adsk.core
import adsk.fusion

MM = 0.1  # JSON is in mm; the Fusion API works in cm
CMD_ID = "knotgenKnotImport"

_app = None
_ui = None
_data = None
_handlers = []
_scale = 1.0  # dialog scale factor: applied to every imported coordinate
# (path, strip, connector/mount positions) but NOT to user geometry —
# profiles, connector components and label text stay physical size


def _point3d(xyz):
    return adsk.core.Point3D.create(
        xyz[0] * MM * _scale, xyz[1] * MM * _scale, xyz[2] * MM * _scale
    )


def _vector3d(xyz):
    return adsk.core.Vector3D.create(xyz[0], xyz[1], xyz[2])


# --------------------------------------------------------------- frame maths

def _frame_vectors(fr):
    """(origin Point3D, tangent, width, led_normal Vector3D), orthonormalised.

    led_normal is recomputed as width x tangent — the frames.py definition —
    so no sign convention in the JSON can flip it.
    """

    def norm(v):
        m = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        return [v[0] / m, v[1] / m, v[2] / m]

    def cross(u, v):
        return [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]

    t = norm(fr["x_axis"])
    w0 = fr["y_axis"]
    d = t[0] * w0[0] + t[1] * w0[1] + t[2] * w0[2]
    w = norm([w0[i] - d * t[i] for i in range(3)])
    n = cross(w, t)  # LED face normal
    return _point3d(fr["origin"]), _vector3d(t), _vector3d(w), _vector3d(n)


# ------------------------------------------------------------ curve creation

def _add_fixed_nurbs(sketch, seg, log):
    nc = seg.get("nurbs_clamped")
    if not nc:
        return None
    try:
        points = [_point3d(p) for p in nc["control_points"]]
        nurbs = adsk.core.NurbsCurve3D.createNonRational(
            points, nc["degree"], list(nc["knots"]), False
        )
        curve = sketch.sketchCurves.sketchFixedSplines.addByNurbsCurve(nurbs)
        if curve:
            log.append("exact NURBS")
            return curve
    except Exception as exc:
        log.append("NURBS failed ({}), using fitted".format(exc))
    return None


def _add_fitted(sketch, seg, log):
    coll = adsk.core.ObjectCollection.create()
    for p in seg["fit_points"]:
        coll.add(_point3d(p))
    curve = sketch.sketchCurves.sketchFittedSplines.add(coll)
    if seg.get("closed", True):
        curve.isClosed = True
    log.append("fitted spline ({} pts)".format(len(seg["fit_points"])))
    return curve


def _add_section(comp, loft_input, curve):
    try:
        loft_input.loftSections.add(curve)
    except Exception:
        loft_input.loftSections.add(comp.features.createPath(curve, False))


def _new_surface_loft_input(comp):
    loft_input = comp.features.loftFeatures.createInput(
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )
    loft_input.isSolid = False
    return loft_input


# ------------------------------------------------------------ strip surfaces

def _strip_quick(comp, strip, center_curve, want_exact, log,
                 sketches_out=None):
    """Rung-line loft (edge-to-edge lofts are rejected for twisted closed
    bands). Edge splines are still created for use as sweep guide rails.
    Returns the two edge curves."""
    sketch = comp.sketches.add(comp.xYConstructionPlane)
    sketch.name = "strip edges c{}".format(strip.get("component", 0) + 1)
    sketch.isComputeDeferred = True
    edges = []
    for key in ("a", "b"):
        seg = dict(strip["edges"][key])
        seg["closed"] = True
        curve = _add_fixed_nurbs(sketch, seg, log) if want_exact else None
        if curve is None:
            curve = _add_fitted(sketch, seg, log)
        edges.append(curve)
    sketch.isComputeDeferred = False

    rung_sketch = comp.sketches.add(comp.xYConstructionPlane)
    rung_sketch.name = "strip rungs c{}".format(strip.get("component", 0) + 1)
    rung_sketch.isComputeDeferred = True
    lines = []
    for fl in strip["frame_lines"]:
        lines.append(
            rung_sketch.sketchCurves.sketchLines.addByTwoPoints(
                _point3d(fl["a"]), _point3d(fl["b"])
            )
        )
    rung_sketch.isComputeDeferred = False
    loft_input = _new_surface_loft_input(comp)
    for line in lines:
        _add_section(comp, loft_input, line)
    try:
        loft_input.isClosed = True
    except Exception as exc:
        log.append("loft isClosed not accepted: {}".format(exc))
    try:
        loft_input.centerLineOrRails.addCenterLine(center_curve)
    except Exception:
        try:
            loft_input.centerLineOrRails.addRail(center_curve)
        except Exception:
            pass
    loft = comp.features.loftFeatures.add(loft_input)
    ci = strip.get("component", 0) + 1
    try:
        for i in range(loft.bodies.count):
            loft.bodies.item(i).name = "strip c{}".format(ci)
    except Exception as exc:
        log.append("strip body rename failed: {}".format(exc))
    log.append("strip: rung loft ({} sections)".format(len(lines)))
    rung_sketch.isVisible = False
    if sketches_out is not None:
        sketches_out += [sketch, rung_sketch]
    return edges


def _strip_editable(comp, strip, center_curve, log):
    path = comp.features.createPath(center_curve, False)
    lines = []
    ci = strip.get("component", 0)
    for i, fl in enumerate(strip["frame_lines"]):
        plane_input = comp.constructionPlanes.createInput()
        plane_input.setByDistanceOnPath(
            path, adsk.core.ValueInput.createByReal(fl["s_fraction"])
        )
        plane = comp.constructionPlanes.add(plane_input)
        plane.name = "strip c{} {:02d}".format(ci + 1, i)
        sk = comp.sketches.add(plane)
        sk.name = "strip line c{} {:02d}".format(ci + 1, i)
        a = sk.modelToSketchSpace(_point3d(fl["a"]))
        b = sk.modelToSketchSpace(_point3d(fl["b"]))
        a.z = 0.0
        b.z = 0.0
        lines.append(sk.sketchCurves.sketchLines.addByTwoPoints(a, b))

    loft_input = _new_surface_loft_input(comp)
    for line in lines:
        _add_section(comp, loft_input, line)
    try:
        loft_input.isClosed = True
    except Exception as exc:
        log.append("loft isClosed not accepted: {}".format(exc))
    try:
        loft_input.centerLineOrRails.addCenterLine(center_curve)
    except Exception:
        try:
            loft_input.centerLineOrRails.addRail(center_curve)
        except Exception as exc:
            log.append("loft rail not accepted: {}".format(exc))
    loft = comp.features.loftFeatures.add(loft_input)
    try:
        for i in range(loft.bodies.count):
            loft.bodies.item(i).name = "strip c{}".format(ci + 1)
    except Exception as exc:
        log.append("strip body rename failed: {}".format(exc))
    log.append("strip: editable ({} angle lines)".format(len(lines)))


# ------------------------------------------------------------------- sweep

def _ref_in_sketch_space(ref_point, sketch):
    """Selected reference point in `sketch`'s sketch space."""
    try:
        if hasattr(ref_point, "parentSketch") and ref_point.parentSketch == sketch:
            return ref_point.geometry
    except Exception:
        pass
    world = ref_point.worldGeometry if hasattr(ref_point, "worldGeometry") else None
    if world is None and hasattr(ref_point, "geometry"):
        world = ref_point.geometry
    return sketch.modelToSketchSpace(world)


def _sweep_profiles(comp, center_curve, rail_curves, frame0, suffix, log,
                    src_profiles, ref_sketch_pt, allow_plain, plain_used,
                    profile_scale=1.0):
    """Copy the selected profile set onto a plane perpendicular to the path
    start and sweep it. ref_sketch_pt: reference point in SOURCE sketch
    coords — it lands on the path."""
    src_sketch = src_profiles[0].parentSketch

    path = comp.features.createPath(center_curve, False)
    plane_input = comp.constructionPlanes.createInput()
    plane_input.setByDistanceOnPath(path, adsk.core.ValueInput.createByReal(0.0))
    plane = comp.constructionPlanes.add(plane_input)
    plane.name = "profile plane {}".format(suffix)
    sk = comp.sketches.add(plane)
    sk.name = "profile {}".format(suffix)

    if frame0 is not None:
        start_model = _point3d(frame0["center"])
        w_model = _vector3d([frame0["b"][i] - frame0["a"][i] for i in range(3)])
        t_model = None
        n_model = _vector3d(frame0["normal"])
    else:
        eva = center_curve.worldGeometry.evaluator
        _, p_min, _ = eva.getParameterExtents()
        _, start_model = eva.getPointAtParameter(p_min)
        w_model = n_model = None

    to_sketch = sk.transform.copy()
    if not to_sketch.invert():
        log.append("sweep: could not invert sketch transform")
        return False
    target_center = start_model.copy()
    target_center.transformBy(to_sketch)

    src_objs = adsk.core.ObjectCollection.create()
    for c in src_sketch.sketchCurves:
        src_objs.add(c)

    move = adsk.core.Matrix3D.create()
    move.translation = adsk.core.Vector3D.create(
        -ref_sketch_pt.x, -ref_sketch_pt.y, -ref_sketch_pt.z
    )
    xform = move
    if w_model is not None:
        wt = w_model.copy()
        wt.transformBy(to_sketch)
        wt = adsk.core.Vector3D.create(wt.x, wt.y, 0.0)
        nt = None
        if n_model is not None:
            nt0 = n_model.copy()
            nt0.transformBy(to_sketch)
            nt = adsk.core.Vector3D.create(nt0.x, nt0.y, 0.0)
        if wt.length > 1e-9 and nt is not None and nt.length > 1e-9:
            wt.normalize()
            d = nt.x * wt.x + nt.y * wt.y
            nt = adsk.core.Vector3D.create(nt.x - d * wt.x, nt.y - d * wt.y, 0.0)
            if nt.length > 1e-9:
                nt.normalize()
                # keep the map a proper rotation: LED normal is sacred,
                # the width sign is arbitrary
                det = wt.x * nt.y - wt.y * nt.x
                if det < 0:
                    wt = adsk.core.Vector3D.create(-wt.x, -wt.y, 0.0)
                    log.append("sweep: width sign flipped to keep handedness")
                rot = adsk.core.Matrix3D.create()
                rot.setWithCoordinateSystem(
                    adsk.core.Point3D.create(0, 0, 0),
                    wt, nt, adsk.core.Vector3D.create(0, 0, 1),
                )
                xform.transformBy(rot)
    shift = adsk.core.Matrix3D.create()
    shift.translation = adsk.core.Vector3D.create(
        target_center.x, target_center.y, 0.0
    )
    xform.transformBy(shift)

    try:
        src_sketch.copy(src_objs, xform, sk)
    except Exception as exc:
        log.append(
            "sweep: profile copy failed ({}) — draw the profile on sketch "
            "'{}' and sweep manually".format(exc, sk.name)
        )
        return False

    # parametric profile scale: bound to the 'knotProfileScale' user
    # parameter about the path point — edit the parameter later and the
    # sweep rebuilds with the resized profile
    try:
        design = comp.parentDesign
        _ensure_param(design, "knotProfileScale", profile_scale,
                      "knotgen: swept profile scale")
        pivot = sk.sketchPoints.add(
            adsk.core.Point3D.create(target_center.x, target_center.y, 0.0)
        )
        coll = adsk.core.ObjectCollection.create()
        coll.add(sk)
        _add_scale_feature(
            comp, coll, pivot, "knotProfileScale", profile_scale,
            log, "profile scale",
        )
    except Exception as exc:
        log.append("profile scale failed: {}".format(exc))

    # match copied profiles to the selected set by area
    src_areas = []
    for p in src_profiles:
        try:
            # the copied sketch is scaled by knotProfileScale before matching
            src_areas.append(p.areaProperties().area * profile_scale ** 2)
        except Exception:
            src_areas.append(None)
    chosen = adsk.core.ObjectCollection.create()
    used = set()
    for area in src_areas:
        best_i, best_err = None, None
        for i in range(sk.profiles.count):
            if i in used:
                continue
            try:
                a = sk.profiles.item(i).areaProperties().area
            except Exception:
                continue
            err = abs(a - area) if area is not None else 0.0
            if best_err is None or err < best_err:
                best_i, best_err = i, err
        if best_i is not None:
            used.add(best_i)
            chosen.add(sk.profiles.item(best_i))
    if chosen.count == 0:
        log.append("sweep: no matching profiles after copy")
        return False
    prof = chosen if chosen.count > 1 else chosen.item(0)

    def attempt(rail, distance):
        sweep_input = comp.features.sweepFeatures.createInput(
            prof, path, adsk.fusion.FeatureOperations.NewBodyFeatureOperation
        )
        if rail is not None:
            sweep_input.guideRail = comp.features.createPath(rail, False)
            sweep_input.profileScaling = (
                adsk.fusion.SweepProfileScalingOptions.SweepProfileNoScalingOption
            )
        if distance is not None:
            sweep_input.distanceOne = adsk.core.ValueInput.createByReal(distance)
        return comp.features.sweepFeatures.add(sweep_input)

    # fallback chain: each strip edge as guide rail, then (only if allowed)
    # a plain rail-less sweep. A plain sweep uses Fusion's own minimal-twist
    # frame — it IGNORES the strip frames entirely, so the body's twist
    # will not match the strip surface or the connectors.
    rail_options = [(r, "strip edge {}".format("ab"[k])) for k, r in
                    enumerate(rail_curves or [])]
    if allow_plain or not rail_options:
        rail_options.append((None, None))
    last_error = None
    for rail, rail_label in rail_options:
        for distance, dist_label in ((None, ""), (0.99999, " 0.99999 workaround")):
            try:
                feature = attempt(rail, distance)
                if rail_label:
                    log.append("sweep: created {} (guided by {}{})".format(
                        suffix, rail_label, dist_label))
                elif rail_curves:
                    plain_used.append(suffix)
                    log.append("sweep: created {} (PLAIN{})".format(
                        suffix, dist_label))
                else:
                    log.append("sweep: created {}{}".format(suffix, dist_label))
                return feature.bodies.item(0) if feature.bodies.count else None
            except Exception as exc:
                last_error = exc
    log.append(
        "sweep failed ({}){} — profile sketch '{}' is in place for a manual "
        "attempt".format(
            last_error,
            "" if allow_plain or not rail_curves
            else " [plain fallback disabled — enable it in the dialog to "
                 "force a body, at the cost of twist mismatch]",
            sk.name,
        )
    )
    return None


# --------------------------------------------------------------- connectors

def _place_connectors(conn_comp, connectors, target_comp, ref_sketch,
                      ref_sketch_pt, flip, log):
    """Place copies of target_comp at every frame, inside conn_comp.

    Mapping (same as the sweep): ref point -> path point, ref sketch x ->
    strip width, sketch y -> LED normal. `flip` rotates every placement 180
    degrees about the LED normal (if the connector faces backwards).

    Returns a list of (occurrence, frame, transform) for the cut/label steps.
    """
    sketch_to_comp = ref_sketch.transform.copy()
    if not sketch_to_comp.invert():
        log.append("connectors: could not invert sketch transform")
        return []

    t_ref = adsk.core.Matrix3D.create()
    t_ref.translation = adsk.core.Vector3D.create(
        -ref_sketch_pt.x, -ref_sketch_pt.y, -ref_sketch_pt.z
    )

    flip_m = None
    if flip:
        flip_m = adsk.core.Matrix3D.create()
        flip_m.setWithCoordinateSystem(
            adsk.core.Point3D.create(0, 0, 0),
            adsk.core.Vector3D.create(-1, 0, 0),
            adsk.core.Vector3D.create(0, 1, 0),
            adsk.core.Vector3D.create(0, 0, -1),
        )

    result, failed = [], []
    for fr in connectors["frames"]:
        try:
            origin, tangent, width, led = _frame_vectors(fr)
            z_f = width.crossProduct(led)  # proper right-handed third axis
            frame_m = adsk.core.Matrix3D.create()
            frame_m.setWithCoordinateSystem(origin, width, led, z_f)
            m = sketch_to_comp.copy()
            m.transformBy(t_ref)
            if flip_m is not None:
                m.transformBy(flip_m)
            m.transformBy(frame_m)
            o = conn_comp.occurrences.addExistingComponent(target_comp, m)
            result.append((o, fr, m))
        except Exception as exc:
            failed.append(str(exc))
    log.append("connectors: {} copies of {} placed{}".format(
        len(result), target_comp.name, " (flipped)" if flip else ""))
    if failed:
        log.append("connectors: {} failed ({})".format(len(failed), failed[0]))
    return result


def _cut_connector_bodies(design, knot_occ, conn_occ, placed, cut_bodies,
                          swept_bodies, curves, log):
    """Combine-Cut the selected connector bodies out of the swept pieces.

    Tool bodies stay (isKeepToolBodies) — the placed connectors remain
    visible parts. One combine per knot component, all tools together.
    When the cut severs the sweep, the resulting pieces are renamed
    "Segment 1..N" in path order (probing the path midpoint between
    consecutive connectors for containment).
    """
    root = design.rootComponent
    natives = []
    for b in cut_bodies:
        natives.append(b.nativeObject if b.nativeObject else b)

    cut_count = 0
    for ci, target_body in swept_bodies.items():
        if target_body is None:
            continue
        tools = adsk.core.ObjectCollection.create()
        frames_ci = []
        for o, fr, _ in placed:
            if fr.get("component", 0) != ci:
                continue
            frames_ci.append(fr)
            try:
                o_proxy = o.createForAssemblyContext(conn_occ)
                for nb in natives:
                    tools.add(nb.createForAssemblyContext(o_proxy))
            except Exception as exc:
                log.append("cut: proxy failed ({})".format(exc))
        if tools.count == 0:
            continue
        try:
            target_proxy = target_body.createForAssemblyContext(knot_occ)
            cin = root.features.combineFeatures.createInput(target_proxy, tools)
            cin.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
            cin.isKeepToolBodies = True
            combine = root.features.combineFeatures.add(cin)
            cut_count += tools.count
        except Exception as exc:
            log.append("cut failed on c{}: {}".format(ci + 1, exc))
            continue

        # rename the pieces Segment 1..N along the path
        try:
            frames_ci.sort(key=lambda f: f.get("s_fraction", 0.0))
            curve = curves.get(ci)
            eva = curve.worldGeometry.evaluator
            _, pmin, pmax = eva.getParameterExtents()
            seg_bodies = [
                combine.bodies.item(i) for i in range(combine.bodies.count)
            ]
            n_seg = len(frames_ci)
            named = 0
            for k in range(n_seg):
                s0 = frames_ci[k].get("s_fraction", 0.0)
                s1 = frames_ci[(k + 1) % n_seg].get("s_fraction", 0.0)
                if k == n_seg - 1:
                    s1 += 1.0
                smid = ((s0 + s1) / 2.0) % 1.0
                ok_, pt = eva.getPointAtParameter(pmin + smid * (pmax - pmin))
                for b in seg_bodies:
                    try:
                        cont = b.pointContainment(pt)
                        if cont in (
                            adsk.fusion.PointContainment.PointInsidePointContainment,
                            adsk.fusion.PointContainment.PointOnPointContainment,
                        ):
                            b.name = "Segment {}".format(k + 1)
                            named += 1
                            break
                    except Exception:
                        continue
            log.append("segments: {} of {} named".format(named, n_seg))
        except Exception as exc:
            log.append("segment naming failed: {}".format(exc))
    if cut_count:
        log.append("cut: {} connector bodies subtracted".format(cut_count))


def _add_labels(comp, placed, label_point, prefix, height_cm, depth_cm,
                swept_bodies, flip_text, log):
    """One engraving per joint, cut symmetrically about the segmentation
    plane through the still-unsegmented body.

    The text lies IN the joint's cross-section plane (the plane the part
    ends meet on), positioned at the authored label anchor projected onto
    that plane. A symmetric two-way extrude stamps BOTH mating ends with
    one feature: after segmentation, the number reads normally on one part
    end and mirror-image on the other — which also tells you at a glance
    which side of the joint an end belongs to. Neither direction cuts air,
    because the body is still continuous through the joint.

    depth_cm is the engrave depth per side, measured from the joint plane —
    set it DEEPER than the connector socket half-thickness, or the socket
    cut will remove the whole imprint.

    flip_text rotates the text 180 degrees in its plane. Numbering: prefix
    + joint number per component, 'c2-' inserted for multi-component links.
    """
    import math

    label_sketch = label_point.parentSketch
    p = label_point.geometry
    s_to_comp = label_sketch.transform

    anchors = comp.sketches.add(comp.xYConstructionPlane)
    anchors.name = "label anchors"
    anchors.isComputeDeferred = True

    def to_world(m, sx, sy, sz):
        pt = adsk.core.Point3D.create(sx, sy, sz)
        pt.transformBy(s_to_comp)
        pt.transformBy(m)
        return pt

    made = 0
    counters = {}
    multi = len(swept_bodies) > 1
    jobs = []
    for o, fr, m in placed:
        ci = fr.get("component", 0)
        counters[ci] = counters.get(ci, 0) + 1
        text = "{}{}{}".format(
            prefix, "c{}-".format(ci + 1) if multi else "", counters[ci]
        )
        origin, tangent, width, led = _frame_vectors(fr)
        # anchor position, projected onto the joint's cross-section plane
        a = to_world(m, p.x, p.y, p.z)
        d = ((a.x - origin.x) * tangent.x + (a.y - origin.y) * tangent.y
             + (a.z - origin.z) * tangent.z)
        q0 = adsk.core.Point3D.create(
            a.x - d * tangent.x, a.y - d * tangent.y, a.z - d * tangent.z
        )
        qx = adsk.core.Point3D.create(
            q0.x + width.x, q0.y + width.y, q0.z + width.z
        )
        qy = adsk.core.Point3D.create(
            q0.x + led.x, q0.y + led.y, q0.z + led.z
        )
        sp0 = anchors.sketchPoints.add(q0)
        spx = anchors.sketchPoints.add(qx)
        spy = anchors.sketchPoints.add(qy)
        jobs.append((text, ci, q0, qx, sp0, spx, spy))
    anchors.isComputeDeferred = False

    for text, ci, q0, qx, sp0, spx, spy in jobs:
        try:
            target = swept_bodies.get(ci)
            if target is None:
                log.append("label '{}' skipped: no swept body".format(text))
                continue
            plane_input = comp.constructionPlanes.createInput()
            plane_input.setByThreePoints(sp0, spx, spy)
            plane = comp.constructionPlanes.add(plane_input)
            plane.name = "label {}".format(text)
            sk = comp.sketches.add(plane)
            sk.name = "label {} text".format(text)

            corner = sk.modelToSketchSpace(q0.copy())
            xdir = sk.modelToSketchSpace(qx.copy())
            angle = math.atan2(xdir.y - corner.y, xdir.x - corner.x)
            box_w = max(len(text), 1) * height_cm * 1.2
            box_h = height_cm * 1.6
            if flip_text:
                corner = adsk.core.Point3D.create(
                    corner.x + box_w * math.cos(angle) - box_h * math.sin(angle),
                    corner.y + box_w * math.sin(angle) + box_h * math.cos(angle),
                    0.0,
                )
                angle += math.pi

            tin = sk.sketchTexts.createInput2(text, height_cm)
            try:
                tin.setAsMultiLine(
                    adsk.core.Point3D.create(corner.x, corner.y, 0.0),
                    adsk.core.Point3D.create(
                        corner.x + box_w, corner.y + box_h, 0.0
                    ),
                    adsk.core.HorizontalAlignments.LeftHorizontalAlignment,
                    adsk.core.VerticalAlignments.BottomVerticalAlignment,
                    0.0,
                )
            except Exception:
                pass
            try:
                tin.angle = angle
            except Exception:
                pass
            text_obj = sk.sketchTexts.add(tin)

            # symmetric two-way cut through the joint: depth_cm into EACH
            # mating end (total extent 2*depth_cm)
            done = False
            try:
                ein = comp.features.extrudeFeatures.createInput(
                    text_obj, adsk.fusion.FeatureOperations.CutFeatureOperation
                )
                ein.setSymmetricExtent(
                    adsk.core.ValueInput.createByReal(2.0 * depth_cm), True
                )
                ein.participantBodies = [target]
                comp.features.extrudeFeatures.add(ein)
                made += 1
                done = True
            except Exception:
                pass
            if not done:
                try:
                    ein = comp.features.extrudeFeatures.createInput(
                        text_obj,
                        adsk.fusion.FeatureOperations.CutFeatureOperation,
                    )
                    ein.setSymmetricExtent(
                        adsk.core.ValueInput.createByReal(depth_cm), False
                    )
                    ein.participantBodies = [target]
                    comp.features.extrudeFeatures.add(ein)
                    made += 1
                except Exception as exc:
                    log.append("label '{}' cut failed: {}".format(text, exc))
        except Exception as exc:
            log.append("label '{}' failed: {}".format(text, exc))
    anchors.isVisible = False
    if made:
        log.append("labels: {} joints stamped (both ends each)".format(made))
    else:
        log.append("labels: none engraved — text sketches may need manual cut")

# ------------------------------------------------------- user parameters

def _ensure_param(design, name, value, comment):
    """Create (or update) a unitless user parameter.

    Note: UserParameters.add rejects ValueInput.createByReal with
    'invalid expression' — the value must be a STRING expression.
    """
    p = design.userParameters.itemByName(name)
    if p:
        try:
            p.expression = "{:g}".format(value)
        except Exception:
            pass
        return p
    return design.userParameters.add(
        name,
        adsk.core.ValueInput.createByString("{:g}".format(value)),
        "", comment,
    )


def _add_scale_feature(comp, entities, pivot, param_name, value, log, label):
    """Scale feature driven by a user parameter, with fallback.

    Some Fusion builds reject a parameter-name expression in
    ScaleFeatures.createInput ('invalid expression'); the reliable route is
    to create the feature numerically and then bind its model parameter."""
    scales = comp.features.scaleFeatures
    try:
        sin = scales.createInput(
            entities, pivot, adsk.core.ValueInput.createByString(param_name)
        )
        feat = scales.add(sin)
        log.append("{} bound to '{}'".format(label, param_name))
        return feat
    except Exception:
        pass
    sin = scales.createInput(
        entities, pivot, adsk.core.ValueInput.createByReal(value)
    )
    feat = scales.add(sin)
    try:
        feat.scaleFactor.expression = param_name
        log.append("{} bound to '{}'".format(label, param_name))
    except Exception as exc:
        log.append(
            "{}: Scale feature created but NOT bound to '{}' ({}) — "
            "link it by editing the feature's scale value".format(
                label, param_name, exc
            )
        )
    return feat


def _apply_knot_scale(design, comp, sketches, log):
    """Scale feature over the knot's geometry sketches, driven by the
    'knotScale' user parameter — edit the parameter later and the path,
    strip and everything downstream (lofts, sweeps) rebuilds. Connector /
    label / mount placements are static transforms and will NOT follow."""
    try:
        _ensure_param(design, "knotScale", 1.0,
                      "knotgen: post-import knot scale (sketches rebuild; "
                      "re-run connectors after changing)")
        coll = adsk.core.ObjectCollection.create()
        for s in sketches:
            coll.add(s)
        if coll.count == 0:
            return
        _add_scale_feature(
            comp, coll, comp.originConstructionPoint, "knotScale", 1.0,
            log, "knot scale ({} sketches)".format(coll.count),
        )
    except Exception as exc:
        log.append("knotScale feature failed: {}".format(exc))


# ------------------------------------------------------------------- mounts

def _add_mounts(comp, mounts, log):
    """A Joint Origin at each exported mount frame: origin on the path,
    x = tangent, z = LED face normal. Attach a base with one rigid joint
    against it — all three rotations come from the frame, and the joint
    origin's offset/angle parameters stay editable for fine adjustment."""
    made = 0
    for i, fr in enumerate(mounts["frames"]):
        try:
            origin, tangent, width, led = _frame_vectors(fr)
            sk = comp.sketches.add(comp.xYConstructionPlane)
            sk.name = "mount {:02d} frame".format(i + 1)
            scale = 1.5  # cm; axis carrier lines
            p0 = sk.sketchPoints.add(origin)
            ln_x = sk.sketchCurves.sketchLines.addByTwoPoints(
                adsk.core.Point3D.create(origin.x, origin.y, origin.z),
                adsk.core.Point3D.create(
                    origin.x + tangent.x * scale,
                    origin.y + tangent.y * scale,
                    origin.z + tangent.z * scale,
                ),
            )
            ln_z = sk.sketchCurves.sketchLines.addByTwoPoints(
                adsk.core.Point3D.create(origin.x, origin.y, origin.z),
                adsk.core.Point3D.create(
                    origin.x + led.x * scale,
                    origin.y + led.y * scale,
                    origin.z + led.z * scale,
                ),
            )
            try:
                geo = adsk.fusion.JointGeometry.createByPoint(p0)
                jo_input = comp.jointOrigins.createInput(geo)
                jo_input.xAxisEntity = ln_x
                jo_input.zAxisEntity = ln_z
                jo = comp.jointOrigins.add(jo_input)
                jo.name = "mount {:02d}".format(i + 1)
                made += 1
            except Exception as exc:
                log.append(
                    "mount {:02d}: joint origin failed ({}) — sketch '{}' "
                    "carries the frame (point + tangent/normal lines) for a "
                    "manual joint origin".format(i + 1, exc, sk.name)
                )
        except Exception as exc:
            log.append("mount {:02d} failed: {}".format(i + 1, exc))
    if made:
        log.append("mounts: {} joint origin(s) created".format(made))


# ------------------------------------------------------------------ command

class _Created(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            inputs = cmd.commandInputs

            dd = inputs.addDropDownCommandInput(
                "curveType", "Curve type",
                adsk.core.DropDownStyles.TextListDropDownStyle,
            )
            dd.listItems.add("Exact NURBS", True)
            dd.listItems.add("Editable fitted spline", False)

            inputs.addValueInput(
                "scaleFactor", "Scale factor", "",
                adsk.core.ValueInput.createByReal(1.0),
            )

            if _data.get("strips") or _data.get("strip"):
                sd = inputs.addDropDownCommandInput(
                    "stripMode", "LED strip surface",
                    adsk.core.DropDownStyles.TextListDropDownStyle,
                )
                sd.listItems.add("Quick (rung loft)", True)
                sd.listItems.add("Editable (angle lines)", False)
                sd.listItems.add("None", False)

            if _data.get("pipe_preview"):
                inputs.addBoolValueInput(
                    "pipe",
                    "Pipe preview ({} mm)".format(
                        _data["pipe_preview"].get("diameter_mm", "?")
                    ),
                    True, "", False,
                )

            prof_sel = inputs.addSelectionInput(
                "profiles", "Sweep profile(s)",
                "Profile region(s) to sweep along the path (leave empty to skip)",
            )
            prof_sel.addSelectionFilter("Profiles")
            prof_sel.setSelectionLimits(0, 0)
            inputs.addValueInput(
                "profileScale", "Profile scale", "",
                adsk.core.ValueInput.createByReal(1.0),
            )

            if _data.get("strips") or _data.get("strip"):
                inputs.addBoolValueInput(
                    "allowPlainSweep",
                    "Allow plain-sweep fallback (twist will NOT match "
                    "strip/connectors)",
                    True, "", False,
                )

            pt_sel = inputs.addSelectionInput(
                "refPoint", "Reference point",
                "Sketch point that lands ON the path (e.g. the profile "
                "sketch's origin point, inside your connector component)",
            )
            pt_sel.addSelectionFilter("SketchPoints")
            pt_sel.addSelectionFilter("ConstructionPoints")
            pt_sel.setSelectionLimits(0, 1)

            if _data.get("connectors"):
                inputs.addBoolValueInput(
                    "placeConnectors",
                    "Place connectors ({} frames)".format(
                        _data["connectors"]["count"]
                    ),
                    True, "", True,
                )
                inputs.addBoolValueInput(
                    "flipConnectors",
                    "Flip connectors 180° (if they face backwards)",
                    True, "", False,
                )
                cut_sel = inputs.addSelectionInput(
                    "cutBodies", "Cut bodies",
                    "Body/bodies in the connector component to SUBTRACT from "
                    "the swept piece at every connector (leave empty for no cut)",
                )
                cut_sel.addSelectionFilter("Bodies")
                cut_sel.setSelectionLimits(0, 0)
                lbl_sel = inputs.addSelectionInput(
                    "labelPoint", "Label anchor point",
                    "Sketch point in the connector component: its position "
                    "(projected onto the joint plane) is where each joint "
                    "number sits within the cross-section. One symmetric cut "
                    "stamps both mating ends — normal on one, mirrored on "
                    "the other (empty = no labels)",
                )
                lbl_sel.addSelectionFilter("SketchPoints")
                lbl_sel.setSelectionLimits(0, 1)
                inputs.addStringValueInput("labelPrefix", "Label prefix", "")
                inputs.addBoolValueInput(
                    "flipLabels", "Rotate label text 180°", True, "", False,
                )
                inputs.addValueInput(
                    "labelHeight", "Label text height", "mm",
                    adsk.core.ValueInput.createByReal(0.6),
                )
                inputs.addValueInput(
                    "labelDepth", "Label depth per side (exceed socket!)", "mm",
                    adsk.core.ValueInput.createByReal(0.3),
                )

            on_execute = _Execute()
            cmd.execute.add(on_execute)
            _handlers.append(on_execute)
            on_destroy = _Destroy()
            cmd.destroy.add(on_destroy)
            _handlers.append(on_destroy)
        except Exception:
            _ui.messageBox("KnotImport dialog failed:\n" + traceback.format_exc())


class _Destroy(adsk.core.CommandEventHandler):
    def notify(self, args):
        adsk.terminate()


class _Execute(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            _do_import(args.command.commandInputs)
        except Exception:
            _ui.messageBox("KnotImport failed:\n" + traceback.format_exc())


def _do_import(inputs):
    global _scale
    data = _data
    design = adsk.fusion.Design.cast(_app.activeProduct)
    paths = data.get("paths") or [data["path"]]

    _scale = 1.0
    if inputs.itemById("scaleFactor"):
        v = inputs.itemById("scaleFactor").value
        if v > 0:
            _scale = v

    want_exact = (
        inputs.itemById("curveType").selectedItem.name == "Exact NURBS"
    )
    strip_mode = "none"
    if inputs.itemById("stripMode"):
        strip_mode = {
            "Quick (rung loft)": "quick",
            "Editable (angle lines)": "editable",
            "None": "none",
        }[inputs.itemById("stripMode").selectedItem.name]
    want_pipe = bool(
        inputs.itemById("pipe") and inputs.itemById("pipe").value
    )

    profile_scale = 1.0
    if inputs.itemById("profileScale"):
        v = inputs.itemById("profileScale").value
        if v > 0:
            profile_scale = v

    prof_input = inputs.itemById("profiles")
    src_profiles = [
        adsk.fusion.Profile.cast(prof_input.selection(i).entity)
        for i in range(prof_input.selectionCount)
    ]
    src_profiles = [p for p in src_profiles if p]

    ref_entity = None
    pt_input = inputs.itemById("refPoint")
    if pt_input and pt_input.selectionCount > 0:
        ref_entity = pt_input.selection(0).entity

    place_conn = bool(
        inputs.itemById("placeConnectors")
        and inputs.itemById("placeConnectors").value
    )
    flip_conn = bool(
        inputs.itemById("flipConnectors")
        and inputs.itemById("flipConnectors").value
    )

    cut_bodies = []
    cut_input = inputs.itemById("cutBodies")
    if cut_input:
        for i in range(cut_input.selectionCount):
            b = adsk.fusion.BRepBody.cast(cut_input.selection(i).entity)
            if b:
                cut_bodies.append(b)

    label_entity = None
    lbl_input = inputs.itemById("labelPoint")
    if lbl_input and lbl_input.selectionCount > 0:
        label_entity = lbl_input.selection(0).entity
    label_prefix = (
        inputs.itemById("labelPrefix").value
        if inputs.itemById("labelPrefix") else ""
    )
    label_height = (
        inputs.itemById("labelHeight").value
        if inputs.itemById("labelHeight") else 0.6
    )
    label_depth = (
        inputs.itemById("labelDepth").value
        if inputs.itemById("labelDepth") else 0.3
    )
    flip_labels = bool(
        inputs.itemById("flipLabels") and inputs.itemById("flipLabels").value
    )

    root = design.rootComponent
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    comp = occ.component
    comp.name = "Knot {}".format(data.get("name", "?"))

    log = []
    curves = []
    geom_sketches = []
    for pi, pathdoc in enumerate(paths):
        sketch = comp.sketches.add(comp.xYConstructionPlane)
        geom_sketches.append(sketch)
        sketch.name = "{} path c{}".format(data.get("name", "knot"), pi + 1)
        sketch.isComputeDeferred = True
        first = None
        for seg in pathdoc["segments"]:
            curve = None
            if seg["type"] == "spline":
                if want_exact:
                    curve = _add_fixed_nurbs(sketch, seg, log)
                if curve is None:
                    curve = _add_fitted(sketch, seg, log)
            elif seg["type"] == "line":
                curve = sketch.sketchCurves.sketchLines.addByTwoPoints(
                    _point3d(seg["start"]), _point3d(seg["end"])
                )
            if curve is not None and first is None:
                first = curve
        sketch.isComputeDeferred = False
        if first is None:
            _ui.messageBox("Component {}: no curves created.".format(pi + 1))
            return
        curves.append(first)

    # a simple origin circle to joint against later
    try:
        mount_sk = comp.sketches.add(comp.xYConstructionPlane)
        mount_sk.name = "central mount point"
        mount_sk.sketchCurves.sketchCircles.addByCenterRadius(
            adsk.core.Point3D.create(0.0, 0.0, 0.0), 1.0
        )
    except Exception as exc:
        log.append("central mount point failed: {}".format(exc))

    rails = {}
    strips = data.get("strips") or ([data["strip"]] if data.get("strip") else [])
    if strips and strip_mode != "none":
        try:
            for strip in strips:
                ci = strip.get("component", 0)
                if strip_mode == "quick":
                    edges = _strip_quick(comp, strip, curves[ci], want_exact,
                                         log, sketches_out=geom_sketches)
                    rails[ci] = edges
                else:
                    _strip_editable(comp, strip, curves[ci], log)
        except Exception as exc:
            log.append("strip failed: {}".format(exc))

    _apply_knot_scale(design, comp, geom_sketches, log)

    if want_pipe and data.get("pipe_preview"):
        try:
            for pi, curve in enumerate(curves):
                path = comp.features.createPath(curve, False)
                pipe_input = comp.features.pipeFeatures.createInput(
                    path, adsk.fusion.FeatureOperations.NewBodyFeatureOperation
                )
                pipe_input.sectionSize = adsk.core.ValueInput.createByReal(
                    data["pipe_preview"]["diameter_mm"] * MM
                )
                pipe = comp.features.pipeFeatures.add(pipe_input)
                try:
                    for i in range(pipe.bodies.count):
                        pipe.bodies.item(i).name = "pipe c{}".format(pi + 1)
                except Exception:
                    pass
            log.append("pipe preview(s) created")
        except Exception as exc:
            log.append("pipe failed: {}".format(exc))

    swept_bodies = {}
    plain_used = []
    allow_plain = bool(
        inputs.itemById("allowPlainSweep")
        and inputs.itemById("allowPlainSweep").value
    )
    if src_profiles:
        src_sketch = src_profiles[0].parentSketch
        if any(p.parentSketch != src_sketch for p in src_profiles):
            log.append("sweep skipped: selected profiles are in different sketches")
        else:
            if ref_entity is not None:
                ref_pt = _ref_in_sketch_space(ref_entity, src_sketch)
            else:
                try:
                    ref_pt = src_profiles[0].areaProperties().centroid
                    log.append("sweep: no reference point selected — using centroid")
                except Exception:
                    ref_pt = adsk.core.Point3D.create(0, 0, 0)
            for pi, curve in enumerate(curves):
                frame0 = None
                for strip in strips:
                    if strip.get("component", 0) == pi:
                        frame0 = strip["frame_lines"][0]
                        break
                swept_bodies[pi] = _sweep_profiles(
                    comp, curve, rails.get(pi) or [], frame0,
                    "c{}".format(pi + 1), log, src_profiles, ref_pt,
                    allow_plain, plain_used, profile_scale,
                )

    if place_conn and data.get("connectors"):
        if ref_entity is None:
            log.append("connectors skipped: select a reference point (its "
                       "parent component is what gets placed)")
        else:
            try:
                ref_sketch = ref_entity.parentSketch
                target_comp = ref_sketch.parentComponent
                if target_comp == comp or target_comp == root:
                    log.append(
                        "connectors skipped: the reference point must live in "
                        "the connector component (its parent is {})".format(
                            target_comp.name
                        )
                    )
                else:
                    timeline = design.timeline
                    group_start = timeline.markerPosition

                    conn_occ = root.occurrences.addNewComponent(
                        adsk.core.Matrix3D.create()
                    )
                    conn_comp = conn_occ.component
                    conn_comp.name = "{} connectors".format(
                        data.get("name", "knot")
                    )
                    ref_pt = _ref_in_sketch_space(ref_entity, ref_sketch)
                    placed = _place_connectors(
                        conn_comp, data["connectors"], target_comp,
                        ref_sketch, ref_pt, flip_conn, log,
                    )

                    # labels BEFORE the segmentation cut: the target is the
                    # single swept body, so the engraves need no containment
                    # probing and use a symmetric two-way extrude — and the
                    # segments inherit their numbers when the cut splits them
                    if placed and label_entity is not None:
                        if label_entity.parentSketch.parentComponent != target_comp:
                            log.append(
                                "labels skipped: anchor point must be in the "
                                "connector component"
                            )
                        else:
                            _add_labels(
                                comp, placed, label_entity, label_prefix,
                                label_height, label_depth, swept_bodies,
                                flip_labels, log,
                            )

                    if placed and cut_bodies:
                        _cut_connector_bodies(
                            design, occ, conn_occ, placed, cut_bodies,
                            swept_bodies, dict(enumerate(curves)), log,
                        )
                    elif cut_bodies and not placed:
                        log.append("cut skipped: no connectors placed")

                    # collapse everything connector-related into one group
                    try:
                        group_end = timeline.markerPosition - 1
                        if group_end > group_start:
                            grp = timeline.timelineGroups.add(
                                group_start, group_end
                            )
                            grp.name = "connectors"
                            grp.isCollapsed = True
                    except Exception as exc:
                        log.append("timeline group failed: {}".format(exc))
            except Exception as exc:
                log.append("connectors failed: {}".format(exc))

    if data.get("mounts"):
        try:
            _add_mounts(comp, data["mounts"], log)
        except Exception as exc:
            log.append("mounts failed: {}".format(exc))

    checks = data.get("checks", {})
    length_note = ""
    if data.get("total_length_mm"):
        length_note = "\npath length: {} mm".format(data["total_length_mm"])
        if len(paths) > 1:
            length_note += " ({})".format(
                ", ".join(
                    "c{}: {}".format(pi + 1, pd.get("length_mm", "?"))
                    for pi, pd in enumerate(paths)
                )
            )
    warning = ""
    if plain_used:
        warning = (
            "*** WARNING: PLAIN SWEEP used on {} — the body's twist does "
            "NOT follow the strip frames, so it will not match the strip "
            "surface, connectors, or the LED plan. ***\n\n".format(
                ", ".join(plain_used)
            )
        )
    _ui.messageBox(
        "{}Imported {} ({}, {} component{}).\n\n{}\n"
        "max tube diameter: {} mm{}".format(
            warning,
            data.get("name", "knot"),
            data.get("source", "?"),
            len(paths),
            "s" if len(paths) > 1 else "",
            "; ".join(log),
            checks.get("max_tube_diameter_mm", "?"),
            length_note,
        )
    )


def run(context):
    global _app, _ui, _data
    try:
        _app = adsk.core.Application.get()
        _ui = _app.userInterface
        design = adsk.fusion.Design.cast(_app.activeProduct)
        if not design:
            _ui.messageBox("Open a design first.")
            return

        dlg = _ui.createFileDialog()
        dlg.title = "Select a knotgen JSON file"
        dlg.filter = "knotgen JSON (*.json)"
        if dlg.showOpen() != adsk.core.DialogResults.DialogOK:
            return
        with open(dlg.filename, "r") as f:
            _data = json.load(f)
        if _data.get("schema_version") not in (1, 2):
            _ui.messageBox(
                "Unsupported schema_version {}.".format(
                    _data.get("schema_version")
                )
            )
            return

        cmd_def = _ui.commandDefinitions.itemById(CMD_ID)
        if cmd_def:
            cmd_def.deleteMe()
        cmd_def = _ui.commandDefinitions.addButtonDefinition(
            CMD_ID,
            "KnotImport: {}".format(_data.get("name", "knot")),
            "Import a knotgen path with optional strip, sweep and connectors",
        )
        on_created = _Created()
        cmd_def.commandCreated.add(on_created)
        _handlers.append(on_created)
        cmd_def.execute()
        adsk.autoTerminate(False)
    except Exception:
        if _ui:
            _ui.messageBox("KnotImport failed:\n" + traceback.format_exc())
