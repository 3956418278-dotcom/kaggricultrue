# Route solver decomposition

Status: implemented design awaiting focused validation.  This record defines
the solver boundary; it is not evidence of solution quality.

The decomposition follows established patterns rather than a new population
heuristic: adaptive LNS for pickup/delivery routing
([Ropke and Pisinger, 2006](https://doi.org/10.1287/trsc.1050.0135)), LNS with
constraint-based synchronized scheduling
([Hojabri et al., 2018](https://doi.org/10.1016/j.cor.2017.11.011)), compact
[minimum-cost flow](https://developers.google.com/optimization/flow/mincostflow)
for conservation, and compiler conflicts in the role of logic-based
decomposition feedback
([Hooker, 2007](https://doi.org/10.1287/opre.1060.0371)).

## Optimization boundary

For workforce size `h`, structural search owns only

`X_h = (worker assignment, per-worker event order, open placement)`.

Every residual semantic Plan goal occurs exactly once in `X_h`.  Existing
placements are fixed and open placements stay inside the Plan domain.  Legal
local service paths are complete multi-modes: selecting a mode can change local
precedence and material deltas, but can never remove a goal.

Each workforce value is an independent LNS basin.  A smaller basin is searched
before a larger one.  A larger basin is opened because the prior basin returned
exact infeasibility, or because it is an economic challenger to a complete
solution.  Basins do not share a population and are never compared through a
surrogate.  Their final order is lexicographic:

1. exact Plan fulfillment;
2. `V(S_end)` from the deterministic compiled state.

Hire cost is already present in `S_end`; workforce has no extra penalty.

## Conditional support problem

For fixed `X_h`, let `d[g,i,m]` and `p[g,i,m]` be the demand and production of
item `i` at event `g` under the selected complete local mode `m`.  Flow variables
connect four source classes to mandatory event demands:

- opening carry of the assigned worker;
- opening shed or global seed inventory;
- market purchase;
- output of another service event.

For every goal/item pair the demand row is an equality:

`sum_s f[s,g,i] = sum_m d[g,i,m] y[m]`.

Opening and event-source conservation rows bound aggregate outgoing flow.
Purchase is a source, not permission to omit demand.  Without activation and
mode rows this is a compact node-arc flow model, not a worker-by-turn-by-item
lattice.

Binary variables are limited to genuine non-network couplings:

- one complete service mode per entity;
- activation of a batched worker/item shed pickup;
- activation of each selected cross-worker producer dependency/transfer.

Integer event ranks impose fixed route order, activated local-mode precedence,
and activated producer-before-consumer precedence.  Resource-source costs are
economic acquisition or opportunity costs.  Stable arc order only breaks ties;
movement, worker turns and logistics are not secondary objectives.

The selected flow is converted into explicit pickup and shed-mediated transfer
events.  Purchase batches, hire requirements, land entries, financing deposits,
shared shed capacity, exact cash timing, movement, service, synchronization,
waiting and hire timing are then evaluated by the deterministic compiler under
the real state transition.  Thus route capacity is the capacity of this full
realization; there is no fixed support margin.

The compiler either completes every residual Plan and returns `S_end`, or the
support solver returns a structured conflict containing missing goals, workers,
items and compiler blockers.  A partial compiler trace may be used by the upper
LNS only as an explicitly configured fallback/search signal.  It is never a
support-layer goal-selection decision.

## Fixed-workforce LNS

The LNS state is `X_h`, not a partially specified resource skeleton.  The main
neighborhoods are standard related removal with regret-2 reinsertion,
multi-route segment removal with regret-2 reinsertion, and block exchange.
They can alter several workers at once.  Open placements of destroyed entities
are reopened in the same neighborhood.  Every insertion retains at least one
complete legal local service mode, preserving precedence by construction.

Distance and service-count quantities are used only to construct a proposal and
to apply necessary bounds.  They never decide whether a candidate survives.
Every distinct proposal immediately runs the conditional support solve and exact
compiler, and exact `(fulfillment, V(S_end))` owns both LNS acceptance and the
basin incumbent.

There is no global population, `_abstract_schedule()`, exact shortlist, exact
repair, reconstruction descent, staffing compression, or final feedback stage.

## Validation gate

Focused validation is intentionally deferred until explicit confirmation.  It
must first establish mandatory-goal conservation, source conservation, economic
mode/source choice, pickup batching, cross-worker transfer timing, full compiled
capacity, deterministic search, exact score semantics and adaptive independent
workforce basins.  Only after those checks pass may the frozen development
benchmark be run; held-out evidence remains out of scope.
