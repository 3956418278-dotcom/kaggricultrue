# Fixed-road zone template parameters

This file records the current hand-authored road catalogue and the few values
that need replay calibration. The maintained source is
`src/kaggriculture_agent/zonal_templates.py`. The library is deliberately
independent of the production intraday executor until its integration is a
separate accepted change.

## Template contract

- A zone owns one literal simple road: an entry tile plus a movement string.
- The entry is a nearest tile of that zone to one of the four shed-access
  cells. A worker starts at that shed access and enters there.
- A spatial layout fixes zone ownership and every zone's normal road. It has a
  bounded set of fixed operating versions; each version says exactly which
  roads return. Multiple roads may return, but their routes are never fused.
- An explicit DROP must finish by turn 22. Every route, including preload,
  movement, tile actions, and DROP, must fit within 24 turns.
- There is no shape constraint such as rectangle, fixed area, distance band,
  INNER corridor, or connected-partition generator. The literal road is the
  authority.
- There is no pair repair, boundary borrowing, or chained zone transfer.

## Daily selection

`return_requirements.py` is the outside capacity/timing stage that marks task
tiles which must return today. D1-D10 output within shed distance 3 is forced;
otherwise it preserves the farthest indivisible output bundles for EOD until
the available shed capacity is full.
For each fixed operating version at the requested land mask and worker count:

1. Remove road tiles with no task today.
2. Preserve the surviving tiles' authored order.
3. Join successive task chains with a deterministic shortest Manhattan path
   that does not revisit any traversed cell. The only permitted repeat is the
   first/last shed-access cell of a returning loop.
4. Reject the version unless every `must_return` tile belongs to one of its
   predeclared returning roads. Other roads remain EOD.
5. Compile each road with the return behavior fixed by that version.
6. Reject per-tile workload mismatch, finish after turn 23, or DROP after turn
   22.
7. Among feasible templates, compare maximum finish turn, then total movement,
   then template id. If none work, try the next authored worker count.

The route accounting currently is:

```text
finish operations
= distinct required preload item types
 + traversed Manhattan edges
 + programme-provided tile actions
 + one DROP when that route returns
```

`output_units` is used only to require enough same-day returned product. Shed
capacity and sale quantity remain outside this library.

## Authored catalogue

| Land | Worker counts | Layouts | Road families |
|---|---:|---:|---|
| 2 quadrants | 3–12 | 41 | edge bands, central cross, columns, peak blocks, cross-quadrant spines, and local-road orientations |
| 3 quadrants | 5–12 | 55 | corner-edge-skeleton, top cross-spines, directional hotspot roads, central-return roads, balanced, skew-load and far-harvest roads |
| 4 quadrants | 6–12 | 42 | whole-farm bands/grids plus balanced, skew-load and far-harvest quadrant roads |

There are 138 spatial layouts and 428 fixed operating versions. A normal layout
has exactly three versions: `EOD`, `NARROW_RETURN`, and `WIDE_RETURN`.
Selected high-load layouts add one authored overflow/harvest-wave version.
There is no runtime subset enumeration. A multi-return version still means one
independent return per worker, never multiple returns by one worker.

The current coverage target is the 41 two-land and 55 three-land layouts.
Four-land entries remain provisional catalogue data and are not part of the
current integration or coverage acceptance.

The three-land six-worker ordinary layout is:

```text
AAABBBBCCC
AAABBBBCCC
AAABBBBCCC
DDDIIIICCC
DDDIIIICCC
DDDII.....
DDDII.....
EEEEE.....
EEEEE.....
EEEEE.....
```

`A` is the complete 3×3 corner road. `B` and `C` continue along the top edge;
`D` and `E` continue down the left edge; `I` is the single thick shed-facing
road. The road definitions, rather than the displayed shapes, determine the
actual traversal.

## Replay-tunable workload values

| Workload mode | Maximum task actions accepted at one active tile |
|---|---:|
| `STAGGERED` | 2 |
| `ORDINARY` | 3 |
| `DENSE` | 4 |
| `PEAK` | 4 |
| `HIGH_VARIANCE` | 5 |

These limits select between authored work modes; they do not replace the exact
24-turn calculation. Returning zones carry a checked certificate of four
active tiles with one action and six output units per tile. The tests prove
every declared return route's certificate reaches the shed by turn 22.
