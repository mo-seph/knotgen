# Links: L2a1 – L11n459 (Thistlethwaite numbering)

A **link** is several closed loops tangled together — for lamps, that means
separate pieces that interlock but never join. Names like `L6a4` come from
the **Thistlethwaite link table**:

- `L` — it's a link (more than one component);
- `6` — the total crossing number of the whole tangle;
- `a` / `n` — alternating or non-alternating, as for knots;
- `4` — the index in the enumeration.

One thing the name does *not* tell you is **how many loops** it has —
`L6a4` happens to have three, `L6a1` has two. knotgen prints the component
count (and per-component lengths) whenever it generates one, and the GUI
draws each component in its own colour.

The celebrities:

- `L2a1` — the **Hopf link**: two rings through each other once, the
  simplest possible link.
- `L4a1` — **Solomon's link**, two rings crossing four times.
- `L5a1` — the **Whitehead link**: two loops with zero linking number that
  still can't be pulled apart.
- `L6a4` — the **Borromean rings**: three rings, no two of which are
  actually linked — cut any one and the whole thing falls apart.

These come from ideal (tightened) conformations, so they're organic and
fully 3D. For a *flat, symmetric* take on some of the same links, check the
weaving family: `W(3,3)` is the Borromean rings as a flat 3-fold rosette
(or a racetrack), and `W(2,4)` is Solomon's link.

```
knotgen list --crossings 6 --links
knotgen L2a1 --tube 20 --preview
knotgen L6a4 --preview                 # organic Borromean rings
knotgen "W(3,3)" --preview             # the same rings, flat and symmetric
```
