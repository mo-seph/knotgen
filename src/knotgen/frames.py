"""Orientation frames for sweeping an LED strip along a knot.

The physical strip bends readily in its own plane (around its width axis)
and twists around its long axis, but cannot bend edgewise. Zero edgewise
bending means the strip's width direction lies along the curve's binormal
(the curvature vector stays in the strip's bend plane). The opposite ideal
is a *fixed* face direction so all the light points one way — which costs
edgewise bending wherever the curve turns within the strip plane.

We express both as target angles measured against the rotation-minimizing
frame (RMF, the zero-twist reference), blend them with pointwise confidence
weights, then smooth the angle along the curve by penalizing twist rate.
Everything stays a closed loop: the RMF's holonomy is distributed evenly
and the smoothing is cyclic, so the frame comes back to itself exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from knotgen.fourier import FourierKnot

LIGHT_DIRS = {
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "out": "radial-out",  # handled specially: radially outward per sample
    "in": "radial-in",
}


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * np.dot(axis, v) * (1.0 - np.cos(angle))
    )


def _transport(normal: np.ndarray, t_from: np.ndarray, t_to: np.ndarray) -> np.ndarray:
    """Parallel-transport `normal` across the rotation taking t_from to t_to."""
    v = np.cross(t_from, t_to)
    s = np.linalg.norm(v)
    if s < 1e-14:
        return normal
    angle = np.arctan2(s, np.dot(t_from, t_to))
    return _rotate_about(normal, v / s, angle)


@dataclass
class StripFrames:
    """Computed strip orientation along the (closed) curve."""

    s_fraction: np.ndarray  # (n,) arc-length fractions in [0, 1)
    points: np.ndarray  # (n, 3)
    tangents: np.ndarray  # (n, 3) unit
    width_dirs: np.ndarray  # (n, 3) unit, along the strip's width
    normals: np.ndarray  # (n, 3) unit, LED face direction (width x tangent)
    theta: np.ndarray  # (n,) angle vs the RMF (radians)
    twist_rate: np.ndarray  # (n,) rad/mm about the tangent
    edgewise_curvature: np.ndarray  # (n,) 1/mm, curvature along width (the bad kind)
    inplane_curvature: np.ndarray  # (n,) 1/mm, curvature along the face normal (the ok kind)
    length: float
    params: dict = field(default_factory=dict)

    def metrics(self) -> dict[str, float]:
        kg = np.abs(self.edgewise_curvature)
        kn = np.abs(self.inplane_curvature)
        return {
            "max_twist_deg_per_cm": float(np.abs(self.twist_rate).max() * 180 / np.pi * 10),
            "min_edgewise_bend_radius_mm": float(1.0 / kg.max()) if kg.max() > 1e-9 else float("inf"),
            "min_inplane_bend_radius_mm": float(1.0 / kn.max()) if kn.max() > 1e-9 else float("inf"),
        }

    def edges(self, width: float) -> tuple[np.ndarray, np.ndarray]:
        """The two strip edge curves for a strip of the given width (mm)."""
        offset = 0.5 * width * self.width_dirs
        return self.points - offset, self.points + offset


def resolve_light_dir(spec: str) -> str | np.ndarray:
    """'up'/'down'/'out'/'in' or 'x,y,z' -> unit vector or radial marker."""
    if spec in LIGHT_DIRS:
        v = LIGHT_DIRS[spec]
        return v if isinstance(v, str) else np.array(v)
    parts = [float(p) for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"light dir must be up/down/out/in or 'x,y,z', got {spec!r}")
    return _normalize(np.array(parts))


def compute_frames(
    knot: FourierKnot,
    *,
    n: int = 400,
    follow: float = 1.0,
    light_dir: str | np.ndarray = "up",
    twist_smooth: float = 5.0,
) -> StripFrames:
    """Compute strip orientation frames along the closed curve.

    follow        1.0 = pure curvature-following (no edgewise bend, twists as
                  needed); 0.0 = face the light direction everywhere possible;
                  in between blends, weighted by pointwise confidence.
    light_dir     'up'/'down'/'out'/'in', an 'x,y,z' string, or a unit vector.
    twist_smooth  stiffness (mm): roughly the arc-length scale over which the
                  angle is averaged. 0 disables smoothing.
    """
    if not 0.0 <= follow <= 1.0:
        raise ValueError(f"follow must be in [0, 1], got {follow}")
    if isinstance(light_dir, str):
        light_dir = resolve_light_dir(light_dir)

    t_vals, pts = knot.sample_arclength(n)
    d1 = knot.deriv(t_vals, 1)
    d2 = knot.deriv(t_vals, 2)
    speed2 = np.sum(d1 * d1, axis=1, keepdims=True)
    T = _normalize(d1)
    # spatial curvature vector kappa * n_hat = component of r'' normal to T, / |r'|^2
    kappa_vec = (d2 - np.sum(d2 * T, axis=1, keepdims=True) * T) / speed2
    kappa = np.linalg.norm(kappa_vec, axis=1)

    _, s_table = knot.arclength_table()
    length = float(s_table[-1])
    h = length / n  # mm between samples
    s_fraction = np.arange(n) / n

    # --- rotation-minimizing reference frame, closed up ------------------
    N = np.empty_like(pts)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(T[0], ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    N[0] = _normalize(np.cross(T[0], ref))
    for i in range(1, n):
        N[i] = _transport(N[i - 1], T[i - 1], T[i])
        N[i] = _normalize(N[i] - np.dot(N[i], T[i]) * T[i])
    # holonomy: transport the last normal back to sample 0 and measure the gap
    N_wrap = _transport(N[-1], T[-1], T[0])
    B0 = np.cross(T[0], N[0])
    holonomy = np.arctan2(np.dot(N_wrap, B0), np.dot(N_wrap, N[0]))
    # distribute the correction so the frame closes exactly
    corrections = -holonomy * np.arange(n) / n
    for i in range(1, n):
        N[i] = _rotate_about(N[i], T[i], corrections[i])
    B = np.cross(T, N)

    # --- target angles in the (N, B) basis --------------------------------
    # fixed-face target: width = T x d, so the face normal leans toward d
    if isinstance(light_dir, str):  # radial marker
        sign = 1.0 if light_dir == "radial-out" else -1.0
        radial = pts * np.array([1.0, 1.0, 0.0])
        d = sign * _normalize(radial)
    else:
        d = np.broadcast_to(light_dir, pts.shape)
    w_fx = np.cross(T, d)
    conf_fx = np.linalg.norm(w_fx, axis=1)  # 0 when tangent parallel to light
    w_fx = w_fx / np.maximum(conf_fx[:, None], 1e-12)
    theta_fx = np.arctan2(np.sum(w_fx * B, axis=1), np.sum(w_fx * N, axis=1))

    # curvature-following target: width along +-binormal (sign made continuous)
    n_hat = kappa_vec / np.maximum(kappa[:, None], 1e-12)
    w_fr = np.cross(T, n_hat)  # binormal
    for i in range(1, n):
        if np.dot(w_fr[i], w_fr[i - 1]) < 0:
            w_fr[i] = -w_fr[i]
    if np.dot(w_fr[0], w_fr[-1]) < 0:
        # an odd number of sign flips around the loop cannot be fixed by a
        # global flip; the blend/smoothing has to absorb the one seam
        pass
    # global sign: prefer the choice whose face normal agrees with the light
    face_fr = np.cross(w_fr, T)
    if np.mean(np.sum(face_fr * d, axis=1)) < 0:
        w_fr = -w_fr
    # quadratic confidence: where curvature is weak the principal-normal
    # direction is noise-driven and can spin rapidly — a linear weight let
    # such stretches drag the strip through a quick twist-and-return blip
    conf_fr = (kappa / max(kappa.max(), 1e-12)) ** 2
    theta_fr = np.arctan2(np.sum(w_fr * B, axis=1), np.sum(w_fr * N, axis=1))

    # --- blend on the circle ----------------------------------------------
    wa = follow * conf_fr
    wb = (1.0 - follow) * conf_fx
    target_vec = wa * np.exp(1j * theta_fr) + wb * np.exp(1j * theta_fx)
    weight = np.abs(target_vec)
    theta_tgt = np.angle(target_vec)
    # where both targets are hopeless, fall back to "no twist" (theta = 0)
    weak = weight < 1e-6
    theta_tgt[weak] = 0.0
    weight = np.maximum(weight, 1e-6)

    # --- cyclic smoothing: min sum w (theta - tgt)^2 + lam int (dtheta/ds)^2
    diffs = np.angle(np.exp(1j * np.diff(theta_tgt, append=theta_tgt[0])))
    winding = np.round(np.sum(diffs) / (2 * np.pi))
    theta_unwrapped = theta_tgt[0] + np.concatenate([[0.0], np.cumsum(diffs[:-1])])
    trend = 2 * np.pi * winding * np.arange(n) / n
    phi_tgt = theta_unwrapped - trend  # cyclic (phi[0] ~ phi[n])

    if twist_smooth > 0:
        lam = twist_smooth**2 / h  # so stiffness scales like an arc length
        D = np.zeros((n, n))
        idx = np.arange(n)
        D[idx, idx] = -1.0
        D[idx, (idx + 1) % n] = 1.0
        A = np.diag(weight) + (lam / h) * (D.T @ D)
        phi = np.linalg.solve(A, weight * phi_tgt)
    else:
        phi = phi_tgt

    theta = phi + trend

    # --- assemble frames ----------------------------------------------------
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    W = c * N + s * B
    face = np.cross(W, T)

    dtheta = np.angle(np.exp(1j * np.diff(theta, append=theta[0] + 2 * np.pi * winding)))
    twist_rate = dtheta / h
    kg = np.sum(kappa_vec * W, axis=1)
    kn = np.sum(kappa_vec * face, axis=1)

    return StripFrames(
        s_fraction=s_fraction,
        points=pts,
        tangents=T,
        width_dirs=W,
        normals=face,
        theta=theta,
        twist_rate=twist_rate,
        edgewise_curvature=kg,
        inplane_curvature=kn,
        length=length,
        params={
            "follow": follow,
            "twist_smooth": twist_smooth,
            "n": n,
        },
    )
