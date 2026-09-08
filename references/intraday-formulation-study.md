# Intraday formulation study

Status: development decision record. This document compares representations; it
does not establish planner quality or promote a submission path.

The original Candidate A search decomposition below has been superseded by
[`route-solver-decomposition.md`](route-solver-decomposition.md).  The retained
representation findings still apply, but resource sources and logistics are now
a conditional compact flow/scheduling subproblem rather than peer LNS mutation
variables.  Workforce is solved in independent outer basins, and structural
candidate admission uses exact compilation rather than an abstract population.

## Fixed problem boundary

The runtime direction is:

`complete owned state -> fixed Daily Plan -> intraday realization -> actions`

The Plan states the economic effects that should be achieved and the constraints
implied by the actual farm. It may leave economically equivalent positions open.
The intraday planner jointly owns staffing, assignment, local service order,
material sources and allocation, batching, carrying, transfers, placement within
the Plan domains, routing, and timing. It reports unmet Plan goals; it may not
remove or replace them. Selling remains in the market owner.

The existing dense temporal CP-SAT formulation is not the selected runtime design.
It encoded every worker position and resource quantity on every turn. It produced
valid trajectories, but development searches did not improve their constructive
incumbents and materially exceeded the practical action budget. More parameter
tuning does not address that representation failure.

## Observed problem shape

The audited 2,880-player-day pilot was inspected without using demonstrated
actions as planner inputs. Direct replay/Plan statistics are:

| Quantity per player-day | Median | 90th percentile | Maximum |
| --- | ---: | ---: | ---: |
| semantic Plan goals | 112 | 139 | 162 |
| affected entities | 63 | 78 | 88 |
| demonstrated workers | 12 | 13 | 14 |
| demonstrated movement actions | 123 | 160 | 207 |
| demonstrated pickup/drop actions | 6 | 9 | 20 |
| entities with open placement | 7 | 19 | 31 |

Grouping consecutive non-movement actions by worker and position reduces a median
126 demonstrated service/logistics actions to 70.5 visit blocks, or about 5.8
visits per worker. With demonstrated assignment and visit order held fixed, the
sum of Manhattan distances between visits explains about 94% of demonstrated
movement on average; median extra movement is four actions. This matters because
workers do not collide and locked farm tiles may be traversed. Intermediate grid
coordinates usually need to be compiled, not optimized.

A fixed sample of 50 player-days (days 0, 8, 16, 24, and 29 from ten shards) was
also compiled into semantic entity work. All 2,606 entities had a complete local
service realization. An entity has at most four goals in the pilot, a median of
two, and a median of one legal local action order. The 90th percentile is six
orders and the maximum is 24. Local causal choice is real but small; global route
and resource organization is the larger combinatorial problem.

Entity work cannot nevertheless be made indivisible. In the pilot, 2,609 of
2,880 player-days assign one entity's effects to multiple workers, and 1,421 have
same-turn effects by multiple workers on one entity. Ordered unit semantics make
such cooperation useful: an earlier worker can build before a later worker places
an animal, or water before a later worker harvests. Likewise, 2,312 player-days
clear and reuse a tile during the day, including 13,092 harvest-then-plant and
2,933 dig-then-plant lifecycles. The compressed representation therefore needs
atomic service events, synchronization, and temporal tile leases; whole-entity
jobs and whole-day placement exclusion are not expressive enough.

These measurements are descriptive evidence about the pilot, not evidence that a
new planner is strong or that the demonstrations are optimal.

## Candidate formulations

### A. Route skeleton plus deterministic trajectory compiler

Represent a realization by worker routes over atomic service and logistics events.
Each entity selects one Plan-permitted position and one complete legal local
service order, which becomes a precedence chain rather than an indivisible job.
Its events may be routed to different workers and may form a synchronized
same-turn bundle when worker-index order makes the chain legal. Explicit resource
links choose starting stock, a purchase, same-worker production, or a
shed-mediated transfer as the source of each material demand. Temporal tile leases
connect creation and clearing events, allowing a later asset to reuse a position.
The number of routes is the staffing decision.

A deterministic compiler expands route arcs to canonical Manhattan movement,
batches unbounded carried inputs when they become available, schedules hires and
acquisitions, respects deadlines and ordered-unit dependencies, and applies the
real transition owner. It either returns a legal trajectory and exact reached
state or a structured infeasibility witness.

Canonical movement has one bounded exception. On a hire turn, two equal-length
movement prefixes can occupy different shed-access cells and thereby change the
new hand's spawn. That tie is retained as an explicit skeleton choice. Ordered
same-turn shed deposits/pickups and capacity-sensitive item deposit order are also
explicit when they affect feasibility or overflow; they are not hidden compiler
repairs.

Search uses adaptive large neighborhoods over the complete structure: event and
causal-chain relocation/exchange, order changes, route split/merge, local
service-order changes, placement/lease reassignment, resource-source changes,
synchronized-bundle creation/removal, and coupled destruction around a deadline,
material, or tile-lifecycle conflict. Every candidate is compiled and
transition-validated. Thus batching and materials are not frozen before routing;
they are parts of the same mutable solution even though ordinary primitive motion
is deterministic.

Strengths: removes almost all turn-by-turn position variables; naturally exposes
human-scale alternatives; supports incremental evaluation and retained plans;
can always report the exact reason a Plan goal was missed. Risks: large-neighborhood
search can miss a useful global rearrangement, and an overactive compiler can
silently make planning choices. Synchronization, resource links, temporal leases,
and capacity-sensitive logistics therefore remain explicit skeleton decisions.

### B. Macro scheduling CP-SAT with validation cuts

Use one optional interval per atomic event or synchronized bundle,
route-predecessor variables, Manhattan setup times, temporal tile-lease variables,
material precedence, and worker activation.
Do not represent every worker coordinate or item quantity per turn. Decode a
macro schedule through the same deterministic compiler; add a no-good or conflict
cut when exact execution rejects it.

Strengths: global propagation for deadlines, worker capacity, and compatible
temporal leases; retains solver bounds when the abstraction is exact. Risks:
sequence-dependent routing still creates many arcs, shared-material cuts may be
weak, and repeated solve/compile/cut cycles can again consume the action budget.
This is genuinely smaller than the rejected temporal lattice, but it remains a
solver-led formulation and should be attempted only if route search exposes a
specific global constraint it cannot handle.

### C. Event-driven forward beam or best-first search

Search directly over reachable decision epochs. A branch dispatches one or more
workers to visits, selects placements and resource actions, then advances them by
shortest-path execution until the next service, dependency, or availability
event. State dominance merges equal remaining goals, positions, inventories,
farm state, and worker availability.

Strengths: candidate states are reachable by construction and simultaneous
effects can be represented exactly. Risks: branching over roughly 60 entities and
12 workers is severe; beam pruning can discard a globally necessary material or
placement choice before its value appears. This is a useful fallback when compiler
feasibility—not route quality—proves to be the limiting issue.

### D. Route-column generation and a covering master

Generate feasible single-worker routes with resource-constrained shortest-path
labeling. A master problem selects routes to cover Plan goals, activates hired
workers, and enforces placement and shared-resource compatibility. Pricing adds
routes that improve the current dual solution; selected columns are compiled and
validated exactly.

Strengths: separates exponential route enumeration from global worker
coordination and can provide meaningful lower bounds. Risks: open placements and
cross-worker resource transfers couple routes strongly, making both the pricing
state and the master substantially harder. It becomes attractive if route-skeleton
search finds good feasible solutions but cannot establish or approach a useful
global bound.

## Selection for the next implementation

Candidate A, corrected to an event-route skeleton, is selected as the first
engineering hypothesis for the next complete model. This is not evidence that it
will beat candidate B. It follows from three game-specific facts: ordinary
shortest paths have no obstacle or collision choices, local entity service orders
are small, and the reference workload compresses to only about six position visits
per worker. The search will operate on real events, assignments, orders, batches,
temporal leases, synchronization, and material links, not on a small catalog of
route templates.

The implementation is not ready for a quality comparison until it supports, in
one coherent path:

- existing and new crop/animal/structure work, fertilizer, harvest, and digging;
- all Plan-permitted placement domains and local legal service orders;
- event-level multi-worker service and same-turn ordered bundles;
- temporal clear-then-reuse tile leases;
- variable hiring and route count;
- starting, purchased, carried, and newly produced material sources;
- same-worker chains and shed-mediated transfers;
- seed atomicity, ordered unit effects, storage overflow, ordinary auto-drop, and
  the terminal-day boundary;
- hire-turn spawn-sensitive movement ties and ordered shed transfers;
- exact market-entry protection for required hires/acquisitions while reactive
  selling remains outside the Plan;
- deterministic retention and explicit same-Plan repair after divergence.

No deliberately incomplete version is a model candidate. Focused compiler and
transition tests may run during construction; they are correctness checks, not
competitive comparisons.

## Decision tests

Development evidence proceeds in this order:

1. **Representation/compile check.** On development demonstrations only, strip
   movement paths while retaining their structural decisions. The compiler must
   reproduce the demonstrated Plan effects or return a specific missing semantic
   feature. This diagnoses expressiveness; it is not planner quality evidence.
2. **Search check.** Give the planner only `S_start + fixed Plan`. Compare reached
   Plan effects, resulting economic state, and then capacity use against the
   controlled strong reference. Demonstrated actions never seed the search.
3. **Coverage diagnosis.** When worse, identify whether the chosen structure could
   express a better realization, whether compilation rejected it, or whether
   search failed to find it. Add general neighborhoods or semantics, never a
   player-day-specific route rule.
4. **Runtime and integration.** Record build/search/compile time separately. Only
   after development performance is stable may the model receive full-episode
   legality and runtime checks. Held-out episodes remain untouched until the
   model, budget, and scoring protocol are frozen.

The old greedy scheduler remains a weak sanity comparison. The dense temporal
model remains a documented rejected representation and correctness oracle for
small cases; neither is the new candidate.
