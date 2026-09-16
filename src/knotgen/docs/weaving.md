# Weaving knots: W(p,q) (Turk's head knots)

Take the torus knot T(p,q)'s drawing — p strands braided q times around a
ring — but make every crossing **alternate**: over, under, over, under.
That's the weaving knot `W(p,q)`, known to sailors and leatherworkers as
the **Turk's head** (p "leads", q "bights"). Same footprint as the torus
knot, completely different knot — and the woven look is the whole point.

The two numbers work like the torus family's:

- **p** = number of strands running side by side (the width of the weave);
- **q** = times the braid wraps around (the number of bights/lobes);
- gcd(p,q) = d > 1 makes it a **link** of d loops;
- crossing number q(p−1); exact q-fold symmetry in the rosette layout.

Small weaves are old friends in disguise: `W(2,q)` *is* the torus knot
T(2,q) (a 2-strand braid can't help alternating), `W(3,2)` is the
figure-eight knot 4₁, `W(3,3)` is the **Borromean rings**, `W(3,4)` = 8₁₈,
`W(3,5)` = 10₁₂₃. From there on they leave the tables and just get more
woven. (In knot theory these are celebrated for a different reason: among
all knots of a given crossing number they're conjectured to be the
"biggest" hyperbolic ones — maximal volume per crossing.)

knotgen draws them two ways:

- **rosette** (default) — a round mat, q-fold symmetric;
- **racetrack** (`--layout racetrack`) — a stadium shape with the crossings
  woven along the straights, the way braid diagrams are drawn in papers.
  `--braid-fraction` bunches or spreads the woven patch, `--braid-split`
  moves half the crossings to the other straight, `--wall` keeps the
  under-strands flat for wall mounting.

```
knotgen "W(3,5)" --width 350 --depth 50 --tube 16 --preview
knotgen "W(3,3)" --preview             # Borromean rings, flat and symmetric
knotgen "W(3,7)" --layout racetrack --braid-split 0.5 --preview
knotgen "W(4,3)" --rho 0.5 --preview
```
