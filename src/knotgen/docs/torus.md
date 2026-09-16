# Torus knots: T(p,q)

Wind a string around a doughnut **p** times through the hole while going
**q** times around the ring, then close the loop: that's the torus knot
`T(p,q)`. The two numbers are the whole story — everything about the knot
follows from them:

- **It's a knot only when p and q share no factor.** If gcd(p,q) = d > 1
  you get a *link* of d parallel components instead: `T(2,4)` is two loops,
  `T(3,6)` is three.
- **Crossing number**: q(p−1) for p < q — so `T(2,3)` has 3 crossings,
  `T(2,7)` has 7, `T(3,5)` has 10.
- **Symmetry**: perfect q-fold rotational symmetry (and swapping p and q
  gives the same knot: `T(3,4)` = `T(4,3)`).
- Every `T(2,q)` is alternating; from `T(3,q)` up they never are.

Several table knots are secretly torus knots, and knotgen knows the
aliases: `3_1` = T(2,3), `5_1` = T(2,5), `7_1` = T(2,7), `9_1` = T(2,9),
`8_19` = T(3,4), `10_124` = T(3,5), `11a367` = T(2,11).

For lamps, the `T(2,q)` family is the workhorse: clean q-lobed rosettes
with generous clearance. Higher p packs strands closer — prettier braiding,
tighter tube limits. The `--rho` flag sets the tube ratio of the underlying
doughnut (bigger = deeper lobes).

```
knotgen "T(2,7)" --width 350 --depth 50 --tube 20 --preview
knotgen "T(3,5)" --rho 0.55 --preview
knotgen "T(2,4)" --preview             # a 2-component torus LINK
```
