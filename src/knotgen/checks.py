"""Pre-flight feasibility checks: will a tube of diameter d sweep cleanly?

Fusion refuses sweeps whose swept body self-intersects. Two independent
causes, both computable before we go anywhere near Fusion:

  * locally: tube radius exceeds the path's radius of curvature,
  * globally: two strands pass closer than the tube diameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knotgen.fourier import FourierKnot
from knotgen.geometry import max_curvature, min_clearance

BEND_SAFETY = 1.1  # required: bend radius >= BEND_SAFETY * tube radius
CLEARANCE_SAFETY = 1.05  # required: strand clearance >= CLEARANCE_SAFETY * tube diameter


@dataclass
class PreflightReport:
    min_bend_radius_mm: float
    min_clearance_mm: float
    max_tube_diameter_mm: float
    tube_diameter_mm: float | None = None
    ok_for_tube: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  min bend radius:    {self.min_bend_radius_mm:8.1f} mm",
            f"  min strand gap:     {self.min_clearance_mm:8.1f} mm",
            f"  max tube diameter:  {self.max_tube_diameter_mm:8.1f} mm",
        ]
        if self.tube_diameter_mm is not None:
            verdict = "OK" if self.ok_for_tube else "WON'T FIT"
            lines.append(f"  requested tube:     {self.tube_diameter_mm:8.1f} mm -> {verdict}")
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def preflight(knot: FourierKnot, tube_diameter: float | None = None) -> PreflightReport:
    kappa, _ = max_curvature(knot)
    bend_radius = 1.0 / kappa
    clearance, _, _ = min_clearance(knot)

    max_tube = min(2.0 * bend_radius / BEND_SAFETY, clearance / CLEARANCE_SAFETY)
    report = PreflightReport(
        min_bend_radius_mm=bend_radius,
        min_clearance_mm=clearance,
        max_tube_diameter_mm=max_tube,
    )

    if tube_diameter is not None:
        report.tube_diameter_mm = tube_diameter
        report.ok_for_tube = tube_diameter <= max_tube
        if not report.ok_for_tube:
            if tube_diameter > 2.0 * bend_radius / BEND_SAFETY:
                report.warnings.append(
                    f"tube {tube_diameter:g} mm exceeds bend limit "
                    f"{2.0 * bend_radius / BEND_SAFETY:.1f} mm — reduce --tightness or increase --width"
                )
            if tube_diameter > clearance / CLEARANCE_SAFETY:
                report.warnings.append(
                    f"tube {tube_diameter:g} mm exceeds strand gap limit "
                    f"{clearance / CLEARANCE_SAFETY:.1f} mm — increase --depth or --width"
                )
        if report.ok_for_tube and tube_diameter > 0.8 * max_tube:
            report.warnings.append(
                "note: a non-circular profile needs clearance for its diagonal — "
                "re-check with `knotgen check <file> --tube <diagonal>` if in doubt"
            )

    return report
