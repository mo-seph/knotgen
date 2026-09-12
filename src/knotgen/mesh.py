"""Triangle-mesh export: swept tube meshes as binary STL or OBJ.

For 3D printing or taking the tube into a mesh modeller, as an alternative
to the JSON -> Fusion sweep route. Frames use the same holonomy-corrected
parallel transport as the viewer, so the tube has no seam twist.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from knotgen.link import as_link
from knotgen.viz import _parallel_transport_frames


def tube_mesh(
    points: np.ndarray, radius: float, sides: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """Closed tube around a closed polyline -> (vertices (n*sides, 3),
    triangle faces (2*n*sides, 3), outward-facing)."""
    n = len(points)
    normals, binormals = _parallel_transport_frames(points)
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    circ = (
        normals[:, None, :] * np.cos(theta)[None, :, None]
        + binormals[:, None, :] * np.sin(theta)[None, :, None]
    )
    verts = (points[:, None, :] + radius * circ).reshape(n * sides, 3)

    i = np.arange(n)[:, None]
    k = np.arange(sides)[None, :]
    a = (i * sides + k).ravel()
    b = (((i + 1) % n) * sides + k).ravel()
    c = (((i + 1) % n) * sides + (k + 1) % sides).ravel()
    d = (i * sides + (k + 1) % sides).ravel()
    faces = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    ).astype(np.int64)

    # orient outward (positive enclosed volume)
    tri = verts[faces]
    volume = np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    if volume < 0:
        faces = faces[:, ::-1]
    return verts, faces


def design_mesh(
    knot,  # FourierKnot | FourierLink
    tube_diameter: float,
    sides: int = 24,
    max_rows: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Tube mesh for a whole design (all link components merged)."""
    link = as_link(knot)
    radius = tube_diameter / 2.0
    all_verts, all_faces = [], []
    offset = 0
    for comp in link.components:
        rows = int(np.clip(comp.total_length() / (0.5 * radius), 64, max_rows))
        _, pts = comp.sample_arclength(rows)
        verts, faces = tube_mesh(pts, radius, sides=sides)
        all_verts.append(verts)
        all_faces.append(faces + offset)
        offset += len(verts)
    return np.vstack(all_verts), np.vstack(all_faces)


def stl_bytes(verts: np.ndarray, faces: np.ndarray, name: str = "knotgen") -> bytes:
    """Binary STL."""
    tri = verts[faces].astype(np.float32)  # (m, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lengths > 1e-12, normals / np.maximum(lengths, 1e-12), 0.0)
    m = len(faces)
    rec = np.zeros(
        m, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]
    )
    rec["n"] = normals
    rec["v"] = tri
    header = name.encode()[:80].ljust(80, b"\0")
    return header + struct.pack("<I", m) + rec.tobytes()


def obj_text(verts: np.ndarray, faces: np.ndarray, name: str = "knotgen") -> str:
    lines = [f"o {name}"]
    lines += [f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in verts]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces]  # 1-indexed
    return "\n".join(lines) + "\n"


def export_mesh(
    path: str | Path,
    knot,
    tube_diameter: float,
    sides: int = 24,
) -> tuple[Path, int]:
    """Write the design's tube mesh; format from the suffix (.stl binary,
    .obj text; anything else gets .stl appended). Returns (path, n_triangles)."""
    path = Path(path)
    if path.suffix.lower() not in (".stl", ".obj"):
        path = path.with_suffix(path.suffix + ".stl")
    verts, faces = design_mesh(knot, tube_diameter, sides=sides)
    name = getattr(knot, "name", "knotgen")
    if path.suffix.lower() == ".obj":
        path.write_text(obj_text(verts, faces, name=name))
    else:
        path.write_bytes(stl_bytes(verts, faces, name=name))
    return path, len(faces)
