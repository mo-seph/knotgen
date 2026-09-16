# Classic knots: 3₁ – 8₂₁ (Rolfsen numbering)

Names like `3_1`, `5_2`, `8_19` come from the **Alexander–Briggs / Rolfsen
tables**: the first number is the **crossing number** — the fewest crossings
any flat drawing of the knot can have — and the subscript just counts through
the distinct knots with that many crossings. The order within each crossing
number is historical convention (it roughly follows the 1927 Alexander–Briggs
enumeration, frozen by Rolfsen's 1976 book *Knots and Links*), so `6_2` isn't
"bigger" than `6_1` in any geometric sense — it's simply the next entry.

There are surprisingly few small knots: one with 3 crossings, one with 4,
two with 5, three with 6, seven with 7, and twenty-one with 8. Every knot
here is **prime** (not a sum of simpler knots) — composite knots like the
granny knot never get table names.

Worth knowing:

- `3_1` — the **trefoil**, the simplest possible knot. Chiral: its mirror
  image is a genuinely different curve.
- `4_1` — the **figure-eight**, the only 4-crossing knot, and amphichiral
  (equal to its own mirror image).
- `5_1` — the **cinquefoil / pentafoil** (Solomon's seal), with perfect
  5-fold symmetry; also the torus knot T(2,5).
- `7_1` and `9_1` continue that family: `(2,q)` torus knots with q-fold
  symmetry — the best lamp shapes in the catalogue.
- `8_19`, `8_20`, `8_21` — the first **non-alternating** knots: no drawing
  of them alternates over-under-over-under along the strand. Everything
  before them in the table alternates.

knotgen serves these from Fremlin's symmetrised Fourier embeddings, which
maximise rotational symmetry — many knots also offer alternative
symmetrisations via `--variant` (see `knotgen list`).

```
knotgen 5_1 --width 300 --depth 50 --tube 16 --preview
knotgen 8_19 --tightness -0.3 --preview
```
