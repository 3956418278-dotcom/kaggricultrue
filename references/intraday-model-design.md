# Joint temporal realization model

This research candidate uses a different principle from the legacy
template/neighborhood dispatcher. The legacy scheduler remains a weak benchmark
and the submitted policy is unchanged. This dense constraint encoding has **not
met the reference-quality or runtime requirements**. It is not a selected runtime
successor merely because it represents broad choices.

## Contract and objective

Input is the actual current farm and a fixed Daily Plan. Economic intent is never
rewritten. Existing positions and state-derived placement domains are constraints;
equivalent new locations remain decisions. Selling belongs to the market owner,
not Plan. A failed or incomplete realization must retain explicit unmet goals.

Solve for the joint remaining-day trajectory. Compare fulfillment of the fixed
semantic requirements first. Compare realizable resulting state next, and labor
consumption among equivalent outcomes. Movement, pickup, drop and placement are
not waste by definition: every necessary operation consumes real worker capacity.

## Design principle

Use a finite temporal constraint problem, not a family of candidate routes:

- Semantic work is compiled from commitment work/effects. Local service legality
  and dependencies are checked with the maintained rule owner. Reference worker
  identities, action order and routes are not planner inputs.
- Service event times, worker identities and permitted asset positions are joint
  variables. Individual workers have positions throughout the remaining day;
  only orthogonal one-tile moves are available. Service and logistics occupy a
  turn at the required position.
- Worker inventories and shared storage form dated resource flows. Pickup item,
  quantity and time remain decisions. Include pickup/use chains, harvested inputs,
  shared seed availability before simultaneous planting and ordered unit effects.
- Hiring is activation of additional worker trajectories with real Fibonacci cost,
  market entry capacity, spawn semantics and next-turn availability. It is not a
  worker count chosen by an outer heuristic before routing.
- Include seed/animal/product acquisition, construction dependencies, authorized
  land unlocks, crop harvest timing, fertilizer and feed/care interactions, storage
  and terminal deadlines. Do not require return travel merely for normal daily
  inventory preservation; the final day has no automatic drop.
- Solve these constraints together with symmetry reduction, reachability bounds,
  proven resource-domain reductions and bounded deterministic effort. Replay resulting trajectories through
  actual transitions; model abstractions must not cause invalid execution to be
  counted as fulfillment.

The solver may discover a partial realization under a time/resource limit. That is
a reported shortfall, not permission to change Plan. Solver bounds and whether
optimality is proved must remain separate from observed trajectory quality.

## Implementation

- `intent.py` compiles native and reference commitments to semantic requirements
  and local legal service orders using the shared rule owner.
- `temporal_model.py` owns the joint constraints and transition-checked decoding.
  Unsold products use projected realizable sale proceeds, including own supply
  and the price floor. Surviving assets use a simple capital/held-output option
  value; this is not an exact forecast of long-horizon terminal cash.
- `temporal_start.py` constructs and validates an initial feasible witness. It
  can reorder local service, visit input producers, batch pickups and reuse
  cleared sites. It is initialization, not the optimizer's feasible-space
  definition or a separately claimed model candidate. Only equivalent asset
  labels may be normalized without changing physical actions.
- Joint constraint neighborhoods free worker trajectories, services, placement,
  hiring and materials together; unrestricted search retains the modeled space.
  `search_improved_start` distinguishes optimization from merely retaining the
  starting witness. A neighborhood bound is not a global optimality certificate.
- `temporal_session.py` retains execution and acknowledges progress separately
  from Plan. Live sales remain in the market owner; prices/cash trigger repair
  only when they materially change the physical realization.

The search is still limited by its semantic compiler and resource/state encoding;
it is not a proof about every legal game trajectory. Solver `UNKNOWN` does not
establish infeasibility. `OPTIMAL` with a witness fixed certifies only that
constrained witness, not unrestricted planning quality.

## Development discipline

Complete the capabilities justified by this formulation before a quality benchmark.
Correctness tests during implementation are necessary and are not candidate-quality
comparisons. Do not create deliberately weakened models to manufacture a sequence
of improvements. A later distinct candidate must change the search/formulation
principle, not just remove or restore obvious capabilities.

Use the already available player-days. Freeze episode-level development/held-out
membership before tuning; no days from one episode may cross that boundary.
Reference and candidate receive exactly the same initial state and Plan. Report
semantic fulfillment and resulting state before efficiency, with runtime and
failure diagnostics. Do not imitate reference actions or test selected cases only.

OR-Tools CP-SAT is an optional research dependency, isolated from the pinned game
environment and current submission. Reference quality does not by itself establish
that solver startup, solve time or packaging satisfies Kaggle's action budget.

## Reproduction

`requirements-planning.txt` pins the optional overlay. Run its focused tests with
`.cache/planner-runtime/bin/python -m unittest tests.test_temporal_model`. Run the
full canonical suite with `.venv/bin/python`: its strict distribution-lock test
correctly rejects the deliberately different solver overlay. Do not weaken the
base lock to make the optional environment pass that identity assertion.

The implementation uses the [versioned CP-SAT source](https://github.com/google/or-tools/tree/v9.14/ortools/sat)
and [official solver API](https://developers.google.com/optimization/cp/cp_solver).
`scripts/benchmark_realizations.py` takes an audited shard directory and a new
output directory under `runs/`. It retains source snapshots, runtime/input hashes,
settings, official trajectories and per-goal results. Its development selection
never includes held-out episodes. Budget settings do not define different model
philosophies.

Score both the recorded reference and its opponent-PASS controlled replay. Flag
control-induced changes in fulfilled work. Fewer worker turns with unfulfilled
goals is not an efficiency win; retain terminal cash, overflow, surviving assets
and maintenance diagnostics. Current measured limitations belong to `STATE.md`.
