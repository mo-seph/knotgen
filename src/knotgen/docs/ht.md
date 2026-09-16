# 9–11 crossing knots (Hoste–Thistlethwaite numbering)

Beyond 8 crossings the tables grow fast: **49** knots with 9 crossings,
**165** with 10, and **552** with 11. Two numbering systems meet here:

- `9_35`, `10_139` — 9- and 10-crossing knots keep **Rolfsen-style** names:
  crossing number, then an index in traditional table order.
- `11a42`, `11n34` — at 11 crossings the names switch to the
  **Hoste–Thistlethwaite** convention from the 1998 computer enumeration
  (Hoste, Thistlethwaite & Weeks tabulated everything to 16 crossings —
  1.7 million knots). The letter says **`a` = alternating** (367 of them)
  or **`n` = non-alternating** (185); the index is their enumeration order.
  Alternating means some drawing passes over, under, over, under… all the
  way around — non-alternating knots refuse to.

A famous quirk lives in this range: the **Perko pair**. Rolfsen's table
listed 10₁₆₁ and 10₁₆₂ as different knots for 75 years until Kenneth Perko
showed in 1973 they're the same knot — which is why "165 or 166?" depends
on which book you read.

Unlike the curated classics, these come from **ideal (tightened)
conformations** — the shape a frictionless rope settles into when pulled
snug (Brian Gilbert's SONO computations from the Knot Atlas). They're
organic and genuinely three-dimensional, not symmetric or flat: expect the
"pulled rope" look, and let `--depth` stay automatic unless you want the
squashed-flat aesthetic (the tool warns when a squash would create kinks —
`--relax` can open a squashed design back up within your depth).

```
knotgen list --crossings 11            # enumerate all of them
knotgen 11a367 --preview               # = the (2,11) torus knot, 11-fold
knotgen 10_124 --preview               # = the (3,5) torus knot
knotgen 11n34 --tube 12 --relax --preview
```
