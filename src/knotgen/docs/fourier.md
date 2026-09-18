# How knotgen represents a knot: Fourier series

Every knot in knotgen is stored as **six short lists of numbers** — and
that choice quietly powers most of the tool's behaviour.

## The idea

A closed 3D curve is three coordinate functions of a parameter t that runs
once around the loop: x(t), y(t), z(t), each returning to its start after a
full turn. Any function that repeats like that can be written as a sum of
sine and cosine waves of increasing frequency — its **Fourier series**:

```
x(t) = b0 + a1*sin(t) + b1*cos(t) + a2*sin(2t) + b2*cos(2t) + ...
```

and likewise for y and z. So a knot is just the coefficient arrays
`a[3][N]` and `b[3][N]`: three coordinates, N "harmonics" each. A trefoil
needs a handful; the tightened 11-crossing conformations use ~256.

Low harmonics are the broad sweep of the shape; high harmonics are fine
detail. Truncating the series can only make a curve *smoother* — there is
no way to store a corner, which is exactly what you want for a path a
tube must sweep along.

## Why this representation earns its keep

- **Closure and smoothness are free.** Sines and cosines repeat perfectly,
  so the curve always closes with no seam, and it's infinitely
  differentiable everywhere — no spline segments, no knot vector, no
  endpoints to misalign. (When you see a "kink", it's genuinely in the
  shape, not a joint between segments.)
- **Derivatives are exact.** Tangents and curvature fall out by
  differentiating each term analytically — the preflight bend-radius check
  never numerically estimates a derivative.
- **Symmetry is frequency structure.** A pentafoil's 5-fold symmetry means
  only certain harmonics are allowed to be non-zero. knotgen detects
  symmetry by *looking at which coefficients are populated*, and `--relax`
  preserves it exactly by zeroing any disallowed coefficients each step
  (spectral projection) — the symmetry cannot drift, only stay perfect.
- **`--tightness` is a one-liner.** Reweighting harmonics (damping highs =
  rounder, boosting = pincher) changes the aesthetic without touching the
  topology.
- **Any curve can come in.** Sampling points around a loop and running an
  FFT converts them to coefficients — that's how the racetrack layouts and
  every `--relax` iteration get back into this form, and how external data
  (the Fremlin and Knot Atlas files are *published* as Fourier
  coefficients) drops straight in.

## Where you meet it

The exported JSON's `fourier.components` block holds exactly these arrays
(`a` = sin, `b` = cos; entry 0 of `b` is the constant offset). Everything
else in the file — fit points, NURBS control points, strip frames — is
derived from them, and `knotgen check/preview/mesh` rebuild the exact
curve from the coefficients alone.

```
knotgen 5_1 --preview        # ~10 harmonics doing all the work
knotgen 11n34 --preview      # ~256 harmonics of tightened rope
```
