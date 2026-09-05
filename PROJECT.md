# Kaggriculture Project

## Confirmed objective

Build a strong, measurable agent for Kaggle's two-player `kaggriculture` simulation. Each agent operates its own farm while interacting through a shared market and town demand. The competitive objective is to finish the episode with more bank money than the opponent. Unsold inventory has no terminal value.

Development starts with exact environment understanding, reproducible local evaluation, replay diagnosis, and deterministic or planning-based baselines. Learning-based methods are introduced only if evidence identifies a limitation they are suited to resolve.

This document describes the full `kaggriculture` environment, not `kaggriculture_beginner`.

## Official environment identity

The maintained local contract is:

- environment: full `kaggriculture`;
- package: `kaggle-environments==1.32.7` from PyPI;
- schema version: `0.1.0`;
- matching release source commit: `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c` in Kaggle's official [`Kaggle/kaggle-environments`](https://github.com/Kaggle/kaggle-environments) repository;
- contract sources: `kaggriculture.json`, `kaggriculture.py`, `README.md`, and the environment's `AGENTS.md`.

The key contract files in the release wheel were byte-identical to current upstream commit `bbda347572cf5134e56f0eb49e8058e2560f9844` when rechecked on 2026-09-04. Exact wheel, source, and file hashes are maintained in `references/official/kaggriculture-environment-1.32.7.json`.

The executable environment and schema take precedence over summaries. An environment update is a contract change until compatibility has been checked and the identity in `EVALUATION.md` and current status in `STATE.md` have been updated.

## Agent and submission contract

The submitted root `main.py` must expose an `agent(observation)` function. A multi-file submission is a `.tar.gz` with `main.py` at its root; Kaggle notebook submission is also supported. During remote execution submission files are under `/kaggle_simulations/agent/`, so file-relative imports must tolerate that location. Competition metadata permits uploads up to 20,480 MB, but a practical submission should be substantially smaller and self-contained because an episode has no network ingress or egress.

The official environment schema sets a one-second per-action timeout and 60 seconds of overage time. Public pages inspected on 2026-09-04 did not expose stable numeric RAM, disk, or vCPU values, and Kaggle does not publish an immutable deployed evaluator image identity. Local success therefore checks the executable game contract, not exact remote resource equivalence.

The agent receives a JSON-like observation and returns a JSON-safe action dictionary:

```python
{
    "farmer": [operation, *arguments],
    "hands": [[operation, *arguments], ...],
    "market": [[operation, *arguments], ...],
}
```

There is one action for the main farmer and one for each currently hired hand. Omitted hands pass. Most extra hand actions reach a nonexistent unit and no-op, but every supplied `PLANT` action is counted during the earlier atomic seed-demand check—even one beyond the hired-hand count—and can therefore veto otherwise valid planting. At most ten market-order entries are processed per player per turn under the default configuration; later entries are discarded, although a valid entry may request multiple units.

Well-formed but illegal operations generally no-op, so legality and resource checks belong in the agent rather than being delegated to environment errors. Malformed actions are not uniformly safe: notably, a non-integer explicit quantity for `PICKUP` or shed `PLACE` raises during interpreter conversion and can invalidate execution. If supplied unit actions request more seeds of one crop than the player owns, all `PLANT` requests for that crop are rejected atomically for that turn.

Farmer and hand operations comprise movement, `PASS`, shed pickup/drop/placement, crop planting/watering/harvesting/fertilizing, coop or pasture construction, animal placement/feeding/care/harvest/fertilizer collection, and digging. Market operations comprise seed, selected product, and animal purchases; sales; daily hand hiring; and ordered land unlocks.

The observation exposes:

- both players' public money, tiles, main-farmer and hand positions, unlocked quadrants, and daily hire count;
- only the acting player's private shed, seed holdings, and per-unit carried inventories;
- shared market inventory and current prices;
- shared unlocked town-shop instances;
- zero-indexed step, day, hour, and player identity, plus the framework's remaining overage time.

The opponent's shed, seeds, and carried inventories are hidden. Opponent modeling must not assume access to them. The environment reward is each player's final money, while the competition episode result compares those values as a win, draw, or loss; the size of the money margin does not change an episode win.

## Stable game semantics

The verified default configuration is:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `episodeSteps` | 720 | Recorded states, including initial state |
| `actTimeout` | 1 second | Per-call allowance before overage is charged |
| initial `remainingOverageTime` | 60 seconds | Cumulative per-agent reserve |
| `boardSize` | 10 | Width and height of each separate player farm |
| `startingMoney` | $3,000 | Initial and terminally scored bank money |
| `maxMarketOrdersPerTurn` | 10 | Market-list entries retained per player |
| `turnsPerDay` | 24 | Day/hour conversion and refresh schedule |
| `shedCapacity` | 100 | Total non-seed items, including animals |
| `weedSpawnChance` | 0.005 | Per empty unlocked tile at daily refresh |
| `townShopUnlockInterval` | 3 days | One shop instance added on each eligible refresh |
| `townShopSellInterval` | 4 turns | Demand tick for every unlocked shop instance |
| `townCenterSellInterval` | 24 turns | Demand tick for the town center |
| `farmHandCostMult` | 1 | Multiplier on the daily Fibonacci hire sequence |

Most values are configurable in local environments. Competitive evidence must retain the resolved configuration and must not mix overridden games with default-contract games.

### Horizon, map, labor, and storage

The default episode is 720 recorded states over 24 turns per day and 30 days. The current interpreter processes the action at step 718, records the resulting step-719 state as `DONE`, and never requests an action at step 719. This boundary is enforced by the pinned-package contract tests.

Each player starts with $3,000, one farmer, and the northwest 5x5 quadrant of a 10x10 board. The other quadrants unlock in `NE`, `SW`, `SE` order for $1,000, $2,000, and $4,000. Locked tiles may be crossed but not farmed. The shed is not a board tile; its four center access positions work even when their quadrant is locked.

Hands are hired for the current day. The daily cost sequence is Fibonacci-scaled (`1, 1, 2, 3, 5, ...` at the default multiplier), and hands disappear after the end-of-day inventory drop. The non-seed shed capacity is 100 by default; overflow is discarded. Seeds are stored separately and are consumed directly by planting.

Movement is orthogonal by one tile and cannot leave the board. Locked tiles may be crossed, but tile mutations on them no-op. Every worker has an unbounded carried inventory in the current implementation. `DROP` empties a worker's entire carried inventory into the shed up to capacity and discards overflow; `PLACE item [n]` can instead deposit a selected item. At day end every worker inventory is dropped under the same capacity rule before hands disappear and the main farmer returns to the northwest shed-access tile.

Workers do not block one another and may occupy the same tile. Their actions resolve in list order—main farmer, then hands—so later actions see earlier same-player tile changes. Building an empty coop or pasture has no money or material cost, each structure holds at most one animal, and `DIG` removes weeds, plants, and empty structures but not a structure containing an animal. Empty unlocked tiles independently receive weeds at the default probability `0.005` on each daily refresh.

### Crops and animals

The crop set is wheat, carrot, tomato, strawberry, and melon. Wheat, carrot, and melon are one-time crops; tomato and strawberry produce on schedules but have a capped number of production events and then decay. Watering status is daily. A new plant begins with one missed-water count, so a plant not watered on its planting day becomes a weed at that day's refresh. Two consecutive missed refreshes otherwise turn a plant into a weed.

| Crop | Seed cost | First yield age | Production / bonus-watering ages | Yield cap |
| --- | ---: | ---: | --- | ---: |
| Wheat | $10 | 2 days | one-time; water at ages 2–4 adds yield | 6 |
| Carrot | $20 | 2 days | one-time; water at ages 2–3 adds yield | 4 |
| Tomato | $50 | 8 days | daily from age 8, 4 production events | 4 |
| Strawberry | $100 | 10 days | every 2 days from age 10, 4 production events | 4 |
| Melon | $80 | 10 days | one-time; water at ages 6–12 adds yield | 6 |

One-time crops start with one harvestable unit but cannot be harvested before first-yield age. Their yield increases when watered in the crop-specific bonus window and is capped; fertilizer doubles the daily bonus while active. Ongoing crop production occurs on its schedule and doubles only when the crop is both watered and fertilized for the relevant day. Mature plant yield decays by one every other turn after the crop-specific lifespan boundary until the tile becomes a weed. Fertilizing consumes one carried fertilizer and remains active for the current day plus the next two days.

Geese require coops and produce eggs; cows and sheep require pastures and produce milk and wool. An animal is purchased into the shed, carried to the matching structure, and placed. Animals consume carried wheat when fed. A newly placed animal survives its first unfed day, but two consecutive unfed refreshes make it escape while leaving the structure. A surviving animal can produce its base unit on its first unfed refresh; feeding is required to survive beyond it and to cash a pending care bonus. A day on which the animal is both fed and cared for banks one bonus unit for the next fed production event. Each surviving animal makes one non-accumulating fertilizer unit available per day.

| Animal | Purchase cost | Structure | Product | First yield age | Interval | Unharvested cap |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Goose | $300 | Coop | Egg | 4 days | 1 day | 4 |
| Cow | $400 | Pasture | Milk | 8 days | 2 days | 6 |
| Sheep | $500 | Pasture | Wool | 6 days | 3 days | 6 |

Exact crop timing, costs, yield caps, animal intervals, and market curve parameters belong to the pinned official implementation. Agent code must represent them through one verified rule/economics owner rather than copying constants across policies.

### Shared market and town

Seeds and animals have fixed purchase costs. Purchases land seeds in separate seed storage and animals in the capacity-limited shed. Products are held in the private shed and explicitly sold; only wheat and fertilizer can be bought back as products, and those purchases also obey shed capacity. Product prices are deterministic functions of shared per-product market inventory, with resource-specific curves, nearest-dollar rounding, and a $1 floor. A sale is quoted on pre-sale inventory; a product purchase is quoted on post-purchase inventory. A sale at the $1 floor pays $1 but does not add supply.

At the shared starting inventory `I0 = 10,000`, base product prices are wheat $25, carrot $35, tomato $60, strawberry $120, melon $250, egg $50, milk $160, wool $200, and fertilizer $100. Away from `I0`, the official model applies a resource- and side-specific shape to the inventory displacement, scaled by a throughput anchor and target price move. The exact parameter table and price function must remain owned by the verified rule/economics model.

| Product | Base | Throughput `T` | Below `I0`: shape, target | Above `I0`: shape, target |
| --- | ---: | ---: | --- | --- |
| Wheat | $25 | 400 | square root, 0.80 | natural log, 0.20 |
| Carrot | $35 | 450 | hinge, 1.00 | square root, 0.70 |
| Tomato | $60 | 200 | hinge, 0.40 | square root, 0.60 |
| Strawberry | $120 | 100 | square root, 0.70 | linear, 1.60 |
| Melon | $250 | 300 | natural log, 0.20 | square, 3.60 |
| Egg | $50 | 332 | hinge, 0.40 | natural log, 0.20 |
| Milk | $160 | 122 | square root, 0.60 | linear, 1.60 |
| Wool | $200 | 105 | natural log, 0.20 | square, 3.20 |
| Fertilizer | $100 | 200 | linear, 0.40 | linear, 0.40 |

The target is the fraction of base price moved at displacement `T`; the implementation derives an amplitude for the selected shape. The hinge is linear through its knee at `T` and adds a gain-8 quadratic term beyond it. Code must use one verified implementation of this formula rather than treating this table alone as executable truth.

At each market-list position, the two players' eligible product orders advance one unit at a time from the same pre-commit inventory, after which prices refresh. This lockstep processing makes order position, quantity, opponent orders, and shared inventory strategically relevant. Atomic `HIRE` and `BUY_LAND` orders at a position are handled once before per-unit product orders.

The town center consumes one of every non-fertilizer product every 24 interpreted turns by default, including the first actionable turn where the interpreter's step is zero. Every three days, up to eight shop instances unlock with replacement. Each instance consumes its product basket every four interpreted turns by default; a single-product shop consumes twice the quantity. Duplicate shops therefore create duplicate demand.

Town consumption directly subtracts market inventory and is not limited by available stock. Players interact only through this shared market and visible state; they cannot enter or mutate the opponent's farm.

The shop baskets are: bakery (egg, wheat), pizza shop (milk, tomato, wheat), brunch spot (egg, wheat, strawberry), yarn store (wool), ice-cream shop (strawberry, milk, wheat), pet cafe (carrot), smoothie shop (strawberry, milk), and farmers market (wheat, carrot, tomato, strawberry). Yarn stores and pet cafes receive the single-product 2x consumption rule.

### Stochasticity and turn order

The environment seed controls weed generation and town-shop unlocks. The resolved seed is stored in `env.info["seed"]` and replay metadata and cleared to `None` in the configuration visible to agents. Each end-of-day generator is derived from the resolved seed and day. Its draw count still depends on which tiles are empty before the shop draw, so equal seeds across different farm trajectories do not guarantee identical realized weeds or shop sequences.

Within an interpreted turn, unit actions are applied, market orders are resolved, town demand consumes market inventory, plant decay is applied, and then any end-of-day refresh runs. The refresh updates plant and animal needs and production, spawns weeds, drops carried inventory into the shed, resets labor, and may unlock a shop. Decisions that depend on same-turn availability or terminal liquidation must be verified against this order.

## Reproducible local environment

Python `3.12.3`, its verified Conda build, pip `25.2`, and the direct official dependency are pinned in `.python-version`, `environment.yml`, and `requirements.txt`. `requirements.lock` records the exact resolved Python distribution set used for the 2026-09-04 verification. The Conda build pins and Python wheels currently target Linux x86_64, matching the verified local platform and Kaggle's Linux execution model; another platform requires a separately recorded compatible lock. Create or refresh the isolated environment from the repository root with:

```bash
conda env create --prefix .venv --file environment.yml
conda run --prefix .venv python scripts/verify_environment.py
```

The `.venv/` environment and generated match artifacts are local cache/data, not project body. A dependency update must deliberately regenerate the lock, re-run the contract verification, compare official source changes, and update the environment identities in this document, `STATE.md`, and `EVALUATION.md` before evidence is combined.

## Conceptual agent decomposition

The maintained agent should have one owner for each of these concerns:

1. **Contract adapter:** normalize observations and emit valid JSON-safe actions through `main.py`.
2. **State and rule model:** represent public/private state, legal actions, timing, inventory flow, production, and market mechanics exactly as verified against the environment.
3. **Decision core:** choose economic commitments, production targets, labor allocation, routes, and market timing. Deterministic rules or planning are the initial default, not a permanent restriction.
4. **Execution controller:** translate decisions into per-unit actions, handle blocked or stochastic state, and maintain invariants without hiding errors.
5. **Packaging boundary:** assemble the self-contained submission without importing local evaluation or analysis machinery.

Opponent inference may be used by the decision core, but only from observable state. It is not a separate authority for game rules.

The decision core represents every economic commitment through six semantic dimensions `(C, T, L, A, Q, R)`: cash commitments and timing; relevant time structure and horizon; land occupancy over time; dated service work plus an explicit travel approximation; physical inputs and outputs; and realizable revenue or terminal-cash effects. These dimensions retain schedules, intervals, and structured records where the game requires them. Terminal profit, profit per action, and profit per tile-day are derived comparisons after feasibility checks, not replacements for the underlying record or an arbitrary weighted utility.

The same representation covers crop and animal production, fertilizer allocation, daily hiring, land expansion, and liquidation. An already-purchased seed, animal, crop, structure, or inventory item is a sunk commitment: planning records its historical cost for diagnosis but evaluates only the marginal cash, work, timing, storage, and realizable value of maintaining, moving, harvesting, using, or liquidating it.

Economic planning has a daily operating cadence. At the first observation of each day, the decision core forms that day's production commitments, maintenance work, required inputs, staffing target, and land decision from the dated work already carried by `(C, T, L, A, Q, R)`. Intraday observations normally reuse that plan while the execution controller adapts tasks, routes, carried inventory, harvesting, transport, and current-market sales. At most one intraday economic repair is permitted when a physical premise is materially invalidated or the remaining same-day work has become certainly infeasible; ordinary price movement alone is not a replan trigger. The next day's post-refresh observation starts a fresh plan after automatic inventory drop, hand removal, and farmer reset.

Local environment adapters, arenas, opponent loaders, replay parsers, statistics, and reports are evaluation infrastructure rather than submission-policy components. They must be able to compare an unchanged packaged agent without importing private implementation hooks.

## Stable architectural boundaries

- `main.py` is the Kaggle entrypoint and eventual packaged runtime boundary. It should be thin unless self-containment requires generated or vendored code.
- `src/kaggriculture_agent/` will own reusable observation, rule, planning, and execution mechanisms.
- `src/kaggriculture_eval/` owns reusable local evaluation and replay mechanisms and must not be imported by the submission policy.
- `scripts/` will contain thin local-match, league, replay-analysis, and packaging commands that call maintained owners.
- `tests/` will own contract, determinism, integration, packaging, and regression checks.
- Evaluation instances will be declarative where practical. Generated matches, replays, reports, downloaded opponents, and build artifacts remain outside the project body and authoritative record.
- Strategy variants must share mechanisms and differ through explicit policy/configuration choices. Market analysis, scheduling, pathfinding, and opponent modeling are code concerns, not separate agent-procedure skills.

This project contract intentionally does not select an accepted competitive strategy or baseline.
