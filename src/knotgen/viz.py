"""3D preview of knots with matplotlib."""

from __future__ import annotations

import numpy as np

from knotgen.fourier import FourierKnot
from knotgen.geometry import crossings_xy


def _parallel_transport_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation-minimizing normal/binormal along a closed polyline."""
    n = len(points)
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # initial normal: any vector orthogonal to t0
    t0 = tangents[0]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t0, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    normal = np.cross(t0, ref)
    normal /= np.linalg.norm(normal)

    normals = np.empty_like(points)
    normals[0] = normal
    for i in range(1, n):
        # rotate previous normal by the rotation taking t[i-1] to t[i]
        v = np.cross(tangents[i - 1], tangents[i])
        c = np.dot(tangents[i - 1], tangents[i])
        if np.linalg.norm(v) < 1e-12:
            normals[i] = normals[i - 1]
            continue
        # Rodrigues rotation
        k = v / np.linalg.norm(v)
        angle = np.arctan2(np.linalg.norm(v), c)
        nrm = normals[i - 1]
        normals[i] = (
            nrm * np.cos(angle)
            + np.cross(k, nrm) * np.sin(angle)
            + k * np.dot(k, nrm) * (1 - np.cos(angle))
        )
    # close the loop: transport the last normal across the wrap and spread
    # the residual holonomy angle along the curve, so the tube's final ring
    # meets the first without a twist pinch at the seam
    v = np.cross(tangents[-1], tangents[0])
    s = np.linalg.norm(v)
    n_wrap = normals[-1]
    if s > 1e-12:
        k = v / s
        angle = np.arctan2(s, np.dot(tangents[-1], tangents[0]))
        n_wrap = (
            n_wrap * np.cos(angle)
            + np.cross(k, n_wrap) * np.sin(angle)
            + k * np.dot(k, n_wrap) * (1 - np.cos(angle))
        )
    b0 = np.cross(tangents[0], normals[0])
    holonomy = np.arctan2(np.dot(n_wrap, b0), np.dot(n_wrap, normals[0]))
    corr = -holonomy * np.arange(n) / n
    for i in range(1, n):
        c, sn = np.cos(corr[i]), np.sin(corr[i])
        t_i = tangents[i]
        nrm = normals[i]
        normals[i] = (
            nrm * c + np.cross(t_i, nrm) * sn + t_i * np.dot(t_i, nrm) * (1 - c)
        )
    binormals = np.cross(tangents, normals)
    return normals, binormals


def _tube_mesh(
    points: np.ndarray, radius: float, sides: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals, binormals = _parallel_transport_frames(points)
    theta = np.linspace(0, 2 * np.pi, sides)
    # closed in both directions: wrap the curve samples
    pts = np.vstack([points, points[:1]])
    nrm = np.vstack([normals, normals[:1]])
    bnm = np.vstack([binormals, binormals[:1]])
    circ = (
        nrm[:, None, :] * np.cos(theta)[None, :, None]
        + bnm[:, None, :] * np.sin(theta)[None, :, None]
    )
    surf = pts[:, None, :] + radius * circ
    return surf[..., 0], surf[..., 1], surf[..., 2]


def _autocrop(path: str, pad: int = 24) -> None:
    """Trim white margins from a saved PNG (clean mode leaves the invisible
    3D axes' full extent as border)."""
    import matplotlib.pyplot as plt

    img = plt.imread(path)
    content = (img[..., :3] < 0.99).any(axis=2)
    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return
    r0 = max(rows[0] - pad, 0)
    r1 = min(rows[-1] + pad, img.shape[0] - 1)
    c0 = max(cols[0] - pad, 0)
    c1 = min(cols[-1] + pad, img.shape[1] - 1)
    plt.imsave(path, img[r0 : r1 + 1, c0 : c1 + 1])


def preview(
    knot,  # FourierKnot | FourierLink
    *,
    tube_diameter: float | None = None,
    strip=None,  # StripFrames | list[StripFrames] (one per component)
    strip_width: float | None = None,
    color_by: str = "curvature",
    save: str | None = None,
    show: bool = True,
    samples: int = 800,
    title: str | None = None,
    clean: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    from knotgen.link import as_link

    link = as_link(knot)
    multi = link.n_components > 1
    strips = strip if isinstance(strip, (list, tuple)) else ([strip] if strip else [])

    sampled = [comp.sample_arclength(samples) for comp in link.components]
    all_closed = [np.vstack([pts, pts[:1]]) for _, pts in sampled]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection="3d")

    # dark theme for the working viewer; --clean stays white (beauty shots,
    # and the autocrop keys on white margins)
    dark = not clean
    fg = "#d8d8d8" if dark else "black"
    arrow_colour = "#e8e8e8" if dark else "black"
    if dark:
        bg = "#15171c"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.set_pane_color((0.11, 0.12, 0.15, 1.0))
                axis.label.set_color(fg)
                axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.12)
            except Exception:
                pass
        ax.tick_params(colors="#8a8a8a")

    if multi:
        # solid colour per component beats curvature shading for links
        palette = plt.get_cmap("tab10").colors
        for ci, closed_pts in enumerate(all_closed):
            ax.plot(
                *closed_pts.T, lw=2.5, color=palette[ci % len(palette)],
                label=f"component {ci + 1}",
            )
        if not clean:
            ax.legend(loc="upper left", fontsize=8)
    elif color_by == "curvature":
        t, _ = sampled[0]
        closed_pts = all_closed[0]
        kappa = link.components[0].curvature(t)
        segs = np.stack([closed_pts[:-1], closed_pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap="viridis", linewidths=2.5)
        lc.set_array(kappa)
        ax.add_collection3d(lc)
        if not clean:
            cbar = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label("curvature (1/mm)", color=fg)
            cbar.ax.tick_params(colors=fg)
            cbar.outline.set_edgecolor(fg)
    else:
        ax.plot(*all_closed[0].T, lw=2.5)
    closed = np.vstack(all_closed)

    # crossing markers: line from under-z to over-z at each xy crossing.
    # Only meaningful for flat-ish (2.5D) designs — on a genuinely 3D
    # conformation the xy projection is arbitrary and the markers are noise.
    e = link.extents(samples=1024)
    is_flat = e["xy_diameter"] > 0 and e["z_extent"] / e["xy_diameter"] <= 0.35
    show_crossings = is_flat and not clean
    for c in crossings_xy(knot, samples=min(2048, samples * 2)) if show_crossings else []:
        ax.plot(
            [c.xy[0], c.xy[0]],
            [c.xy[1], c.xy[1]],
            [c.z_under, c.z_over],
            color="red",
            lw=1.0,
            alpha=0.6,
        )

    tube_artists = []
    if tube_diameter:
        for _, pts in sampled:
            # decimate evenly (keeps the wrap segment the same size as
            # the rest); opaque shaded tubes get heavy at full res
            n_keep = min(len(pts), 600)
            sel = np.linspace(0, len(pts), n_keep, endpoint=False).astype(int)
            X, Y, Z = _tube_mesh(pts[sel], tube_diameter / 2.0, sides=24)
            surf = ax.plot_surface(
                X, Y, Z, color="#e8934a", linewidth=0, antialiased=False,
                shade=True, rcount=X.shape[0], ccount=X.shape[1],
            )
            surf.set_visible(not strips)  # hidden by default under a strip
            tube_artists.append(surf)

    ribbon_modes: dict[str, list] = {
        "edgewise": [], "curvature": [], "sides": []
    }
    arrow_artists = []
    if strips and strip_width:
        cmap_e = plt.get_cmap("plasma")
        cmap_c = plt.get_cmap("viridis")
        vmax_e = max(
            max(np.abs(s.edgewise_curvature).max() for s in strips), 1e-6
        )
        vmax_c = max(
            max(np.hypot(s.edgewise_curvature, s.inplane_curvature).max()
                for s in strips),
            1e-6,
        )
        for s in strips:
            a, b = s.edges(strip_width)
            a = np.vstack([a, a[:1]])
            b = np.vstack([b, b[:1]])
            quads = [
                [a[i], b[i], b[i + 1], a[i + 1]] for i in range(len(a) - 1)
            ]
            # mode 1: across-strip (edgewise) curvature — the strain map
            strain = np.abs(s.edgewise_curvature)
            colors = cmap_e(strain / vmax_e)
            colors[:, 3] = 0.85
            poly = Poly3DCollection(quads, facecolors=colors, edgecolors="none")
            ax.add_collection3d(poly)
            ribbon_modes["edgewise"].append(poly)
            # mode 2: total curvature along the path
            kappa = np.hypot(s.edgewise_curvature, s.inplane_curvature)
            colors = cmap_c(kappa / vmax_c)
            colors[:, 3] = 0.85
            poly = Poly3DCollection(quads, facecolors=colors, edgecolors="none")
            poly.set_visible(False)
            ax.add_collection3d(poly)
            ribbon_modes["curvature"].append(poly)
            # mode 3: sides — LED face blue, back face yellow. Both shells
            # must live in ONE collection: mplot3d depth-sorts polygons only
            # within a collection, so separate collections paint over each
            # other wholesale and flip with the view.
            off = 0.06 * strip_width * s.normals
            off = np.vstack([off, off[:1]])
            from matplotlib.colors import to_rgba

            side_quads = []
            side_colors = []
            for sign, colour in ((+1.0, "#3070d0"), (-1.0, "#e8c02a")):
                side_quads += [
                    [a[i] + sign * off[i], b[i] + sign * off[i],
                     b[i + 1] + sign * off[i + 1], a[i + 1] + sign * off[i + 1]]
                    for i in range(len(a) - 1)
                ]
                side_colors += [to_rgba(colour)] * (len(a) - 1)
            poly = Poly3DCollection(
                side_quads, facecolors=np.array(side_colors), edgecolors="none"
            )
            poly.set_visible(False)
            ax.add_collection3d(poly)
            ribbon_modes["sides"].append(poly)
            # LED face direction arrows, a few per lobe (always on)
            step = max(len(s.points) // 36, 1)
            arrow = 0.9 * strip_width
            for i in range(0, len(s.points), step):
                p, nrm = s.points[i], s.normals[i]
                (al,) = ax.plot(
                    [p[0], p[0] + arrow * nrm[0]],
                    [p[1], p[1] + arrow * nrm[1]],
                    [p[2], p[2] + arrow * nrm[2]],
                    color=arrow_colour, lw=0.8, alpha=0.7,
                )
                arrow_artists.append(al)
        if not clean:
            sm = plt.cm.ScalarMappable(
                cmap=cmap_e, norm=plt.Normalize(vmin=0.0, vmax=vmax_e)
            )
            cb2 = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, location="left")
            cb2.set_label("strip edgewise curvature (1/mm) — lower is kinder",
                          color=fg)
            cb2.ax.tick_params(colors=fg)
            cb2.outline.set_edgecolor(fg)

    # ---- orientation aids (toggleable) -----------------------------------
    lims = np.array([closed.min(axis=0), closed.max(axis=0)])
    center = lims.mean(axis=0)
    half = (lims[1] - lims[0]).max() / 2.0

    groups: dict[str, list] = {"triad": [], "start": []}
    # axis triad at the origin, Fusion colours: X red, Y green, Z blue (up)
    if clean:
        decorate = False
    else:
        decorate = True
    if decorate:
        tri_len = 0.45 * half
        for axis_dir, colour, label_txt in (
            ((1, 0, 0), "red", "X"),
            ((0, 1, 0), "green", "Y"),
            ((0, 0, 1), "blue", "Z up"),
        ):
            d = np.array(axis_dir) * tri_len
            (ln,) = ax.plot([0, d[0]], [0, d[1]], [0, d[2]], color=colour, lw=2)
            txt = ax.text(d[0] * 1.1, d[1] * 1.1, d[2] * 1.1, label_txt,
                          color=colour, fontsize=10, weight="bold")
            ln.set_visible(False)
            txt.set_visible(False)
            groups["triad"] += [ln, txt]
        # path start + direction per component (this is where the sweep seam and
        # connector 1 sit — compare against Fusion before committing)
        for ci, (_, pts) in enumerate(sampled):
            p0 = pts[0]
            d0 = pts[1] - pts[0]
            d0 = d0 / np.linalg.norm(d0) * 0.18 * half
            dot = ax.scatter(*p0, color="limegreen", s=60, depthshade=False)
            (arrow,) = ax.plot(
                [p0[0], p0[0] + d0[0]], [p0[1], p0[1] + d0[1]],
                [p0[2], p0[2] + d0[2]], color="limegreen", lw=3,
            )
            txt = ax.text(*(p0 + d0 * 1.3), f"start c{ci + 1}",
                          color="limegreen", fontsize=9)
            dot.set_visible(False)
            arrow.set_visible(False)
            txt.set_visible(False)
            groups["start"] += [dot, arrow, txt]

    # equal aspect + minimal whitespace
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    if clean:
        ax.set_axis_off()
    else:
        ax.set_title(title or knot.name, y=0.99, color=fg)
    # reserve a sliver at the bottom for the colour-mode buttons when a strip
    # is shown — if the 3D axes overlaps them, interactive rotation repaints
    # the axes over the buttons and they vanish after the first drag
    fig.subplots_adjust(
        left=0.0, right=1.0,
        bottom=0.055 if ((strips or show) and not clean) else 0.0, top=1.0,
    )

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        if clean:
            _autocrop(save)
    if show:
        # free our shortcut keys from matplotlib's default bindings
        # (e.g. 's' is Save, 'g' is grid — they'd fire alongside ours)
        our_keys = {"t", "m", "g", "p", "n", "1", "2", "3"}
        for param in list(plt.rcParams):
            if param.startswith("keymap."):
                for k in our_keys & set(plt.rcParams[param]):
                    plt.rcParams[param].remove(k)

        # trackpad/scroll zoom about the current view centre
        def on_scroll(event):
            base = 0.85 if event.button == "up" else 1.0 / 0.85
            for get_lim, set_lim in (
                (ax.get_xlim3d, ax.set_xlim3d),
                (ax.get_ylim3d, ax.set_ylim3d),
                (ax.get_zlim3d, ax.set_zlim3d),
            ):
                lo, hi = get_lim()
                mid, span = (lo + hi) / 2.0, (hi - lo) / 2.0 * base
                set_lim(mid - span, mid + span)
            fig.canvas.draw_idle()

        def set_mode(mode):
            for name, artists in ribbon_modes.items():
                for artist in artists:
                    artist.set_visible(name == mode)
            fig.canvas.draw_idle()

        def toggle(artists):
            for artist in artists:
                artist.set_visible(not artist.get_visible())
            fig.canvas.draw_idle()

        axes_on = [True]

        def toggle_axes():
            axes_on[0] = not axes_on[0]
            if axes_on[0]:
                ax.set_axis_on()
            else:
                ax.set_axis_off()
            fig.canvas.draw_idle()

        mode_keys = {"1": "curvature", "2": "edgewise", "3": "sides"}
        toggle_keys = {
            "t": groups["triad"],
            "m": groups["start"],
            "p": tube_artists,
            "n": arrow_artists,
        }

        def on_key(event):
            if event.key in toggle_keys and toggle_keys[event.key]:
                toggle(toggle_keys[event.key])
            elif event.key in mode_keys and strips:
                set_mode(mode_keys[event.key])
            elif event.key == "g":
                toggle_axes()
            elif event.key in ("cmd+q", "ctrl+q", "q"):
                plt.close(fig)

        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("key_press_event", on_key)

        # button row along the bottom: colour modes (strip only) + toggles
        if not clean:
            from matplotlib.widgets import Button

            entries = []
            if strips:
                entries += [
                    ("curvature", lambda: set_mode("curvature")),
                    ("edgewise", lambda: set_mode("edgewise")),
                    ("sides", lambda: set_mode("sides")),
                ]
            entries += [
                ("origin [t]", lambda: toggle(groups["triad"])),
                ("start [m]", lambda: toggle(groups["start"])),
            ]
            if tube_artists:
                entries.append(("tube [p]", lambda: toggle(tube_artists)))
            if arrow_artists:
                entries.append(("normals [n]", lambda: toggle(arrow_artists)))
            entries.append(("grid [g]", toggle_axes))

            buttons = []
            x = 0.01
            for label, cb in entries:
                w = 0.013 + 0.0105 * len(label)
                bax = fig.add_axes([x, 0.005, w, 0.042])
                if dark:
                    btn = Button(bax, label, color="#2a2d34",
                                 hovercolor="#3f434e")
                    btn.label.set_color(fg)
                else:
                    btn = Button(bax, label)
                btn.label.set_fontsize(8)
                btn.on_clicked(lambda _, f=cb: f())
                buttons.append(btn)
                x += w + 0.008
            fig._knotgen_buttons = buttons  # keep references alive

            fig.text(
                0.01, 0.985,
                "drag rotate · scroll zoom · q quit",
                color=fg, fontsize=8, alpha=0.75, va="top",
            )

        help_line = ("viewer: scroll = zoom, drag = rotate | buttons or keys: "
                     "[t] origin, [m] start, [p] tube, [n]ormals, [g] grid, [q]uit")
        if strips:
            help_line += " | strip colours: [1] curvature, [2] edgewise, [3] sides (blue=LED face, yellow=back)"
        print(help_line)
        plt.show()
    else:
        plt.close(fig)
