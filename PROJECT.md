# Kaggriculture Project

## Confirmed objective

Build a strong, measurable agent for Kaggle's two-player `kaggriculture` simulation. Each agent operates its own farm while interacting through a shared market and town demand. The competitive objective is to finish the episode with more bank money than the opponent. Unsold inventory has no terminal value.

Development starts with exact environment understanding, reproducible local evaluation, replay diagnosis, and deterministic or planning-based baselines. Learning-based methods are introduced only if evidence identifies a limitation they are suited to resolve.

This document describes the full `kaggriculture` environment, not `kaggriculture_beginner`.

## Official environment identity

The initial contract was inspected on 2026-09-04 from Kaggle's official [`Kaggle/kaggle-environments`](https://github.com/Kaggle/kaggle-environments) repository:

- environment: `kaggriculture`;
- schema version: `0.1.0`;
- upstream commit: `bbda347572cf5134e56f0eb49e8058e2560f9844`;
- contract sources: `kaggriculture.json`, `kaggriculture.py`, `README.md`, and the environment's `AGENTS.md`.

The executable environment and schema take precedence over summaries. An environment update is a contract change until compatibility has been checked and the identity in `EVALUATION.md` and current status in `STATE.md` have been updated.

## Agent and submission contract

The submitted root `main.py` must expose an `agent(observation)` function. A multi-file submission is a `.tar.gz` with `main.py` at its root. The official schema sets a one-second per-action timeout and 60 seconds of overage time; packaging and runtime dependencies must therefore remain deliberate and bounded.

The agent receives a JSON-like observation and returns a JSON-safe action dictionary:

```python
{
    "farmer": [operation, *arguments],
    "hands": [[operation, *arguments], ...],
    "market": [[operation, *arguments], ...],
}
```

There is one action for the main farmer and one for each currently hired hand. At most ten market orders are processed per player per turn under the default configuration; later orders are discarded. Invalid or illegal actions are silent no-ops, so legality and resource checks belong in the agent rather than being delegated to environment errors.

Farmer and hand operations comprise movement, `PASS`, shed pickup/drop/placement, crop planting/watering/harvesting/fertilizing, coop or pasture construction, animal placement/feeding/care/harvest/fertilizer collection, and digging. Market operations comprise seed, selected product, and animal purchases; sales; daily hand hiring; and ordered land unlocks.

The observation exposes:

- both players' public money, tiles, main-farmer and hand positions, unlocked quadrants, and daily hire count;
- only the acting player's private shed, seed holdings, and per-unit carried inventories;
- shared market inventory and current prices;
- shared unlocked town-shop instances;
- zero-indexed step, day, hour, and player identity.

The opponent's shed, seeds, and carried inventories are hidden. Opponent modeling must not assume access to them.

## Stable game semantics

### Horizon, map, labor, and storage

The default episode is 720 recorded states over 24 turns per day and 30 days. The current interpreter processes the action at step 718 and then marks both agents `DONE`; code must not assume a usable step 719. This boundary must receive a local contract test against the pinned package.

Each player starts with $3,000, one farmer, and the northwest 5x5 quadrant of a 10x10 board. The other quadrants unlock in `NE`, `SW`, `SE` order for $1,000, $2,000, and $4,000. Locked tiles may be crossed but not farmed. The shed is not a board tile; its four center access positions work even when their quadrant is locked.

Hands are hired for the current day. The daily cost sequence is Fibonacci-scaled (`1, 1, 2, 3, 5, ...` at the default multiplier), and hands disappear after the end-of-day inventory drop. The non-seed shed capacity is 100 by default; overflow is discarded. Seeds are stored separately and are consumed directly by planting.

### Crops and animals

The crop set is wheat, carrot, tomato, strawberry, and melon. Wheat, carrot, and melon are one-time crops; tomato and strawberry produce on schedules but have a capped number of production events and then decay. Watering status is daily. A new plant begins with one missed-water count, so a plant not watered on its planting day becomes a weed at that day's refresh. Two consecutive missed refreshes otherwise turn a plant into a weed.

| Crop | Seed cost | First yield age | Schedule/window boundary | Yield cap |
| --- | ---: | ---: | --- | ---: |
| Wheat | $10 | 2 days | one-time; bonus through age 4 | 6 |
| Carrot | $20 | 2 days | one-time; bonus through age 3 | 4 |
| Tomato | $50 | 8 days | daily, 4 production events | 4 |
| Strawberry | $100 | 10 days | every 2 days, 4 production events | 4 |
| Melon | $80 | 10 days | one-time; bonus through age 12 | 6 |

One-time crop yield increases when watered in its crop-specific bonus window and is capped; fertilizer doubles the daily bonus while active. Ongoing crop production occurs on its schedule and doubles only when the crop is both watered and fertilized for the relevant day. Mature plant yield decays by one every other turn after the crop-specific lifespan boundary until the tile becomes a weed.

Geese require coops and produce eggs; cows and sheep require pastures and produce milk and wool. An animal is purchased into the shed, carried to the matching structure, and placed. Animals consume carried wheat when fed. A newly placed animal survives its first unfed day, but two consecutive unfed refreshes make it escape while leaving the structure. Fed-and-cared days bank a bonus for the next eligible production. Each surviving animal makes at most one unaccumulated fertilizer unit available per day.

| Animal | Purchase cost | Structure | Product | First yield age | Interval | Unharvested cap |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Goose | $300 | Coop | Egg | 4 days | 1 day | 4 |
| Cow | $400 | Pasture | Milk | 8 days | 2 days | 6 |
| Sheep | $500 | Pasture | Wool | 6 days | 3 days | 6 |

Exact crop timing, costs, yield caps, animal intervals, and market curve parameters belong to the pinned official implementation. Agent code must represent them through one verified rule/economics owner rather than copying constants across policies.

### Shared market and town

Seeds and animals have fixed purchase costs. Products are held in the private shed and explicitly sold; only wheat and fertilizer can be bought back as products. Product prices are deterministic functions of shared per-product market inventory, with resource-specific curves, nearest-dollar rounding, and a $1 floor. A sale is quoted on pre-sale inventory; a product purchase is quoted on post-purchase inventory. A sale at the $1 floor pays $1 but does not add supply.

At the shared starting inventory `I0 = 10,000`, base product prices are wheat $25, carrot $35, tomato $60, strawberry $120, melon $250, egg $50, milk $160, wool $200, and fertilizer $100. Away from `I0`, the official model applies a resource- and side-specific shape to the inventory displacement, scaled by a throughput anchor and target price move. The exact parameter table and price function must remain owned by the verified rule/economics model.

At each market-list position, the two players' eligible product orders advance one unit at a time from the same pre-commit inventory, after which prices refresh. This lockstep processing makes order position, quantity, opponent orders, and shared inventory strategically relevant. Atomic `HIRE` and `BUY_LAND` orders at a position are handled once before per-unit product orders.

The town center consumes one of every non-fertilizer product every 24 turns by default. Every three days, up to eight shop instances unlock with replacement. Each instance consumes its product basket every four turns by default; a single-product shop consumes twice the quantity. Duplicate shops therefore create duplicate demand.

The shop baskets are: bakery (egg, wheat), pizza shop (milk, tomato, wheat), brunch spot (egg, wheat, strawberry), yarn store (wool), ice-cream shop (strawberry, milk, wheat), pet cafe (carrot), smoothie shop (strawberry, milk), and farmers market (wheat, carrot, tomato, strawberry). Yarn stores and pet cafes receive the single-product 2x consumption rule.

### Stochasticity and turn order

The environment seed controls weed generation and town-shop unlocks. The resolved seed is stored in replay metadata and removed from the observation. End-of-day randomness is reproducible for a fixed seed and trajectory, but the random stream is consumed conditionally on farm state; equal seeds across different policies do not guarantee identical realized weeds or shop sequences.

Within an interpreted turn, unit actions are applied, market orders are resolved, town demand consumes market inventory, plant decay is applied, and then any end-of-day refresh runs. The refresh updates plant and animal needs and production, spawns weeds, drops carried inventory into the shed, resets labor, and may unlock a shop. Decisions that depend on same-turn availability or terminal liquidation must be verified against this order.

## Conceptual agent decomposition

The maintained agent should have one owner for each of these concerns:

1. **Contract adapter:** normalize observations and emit valid JSON-safe actions through `main.py`.
2. **State and rule model:** represent public/private state, legal actions, timing, inventory flow, production, and market mechanics exactly as verified against the environment.
3. **Decision core:** choose economic commitments, production targets, labor allocation, routes, and market timing. Deterministic rules or planning are the initial default, not a permanent restriction.
4. **Execution controller:** translate decisions into per-unit actions, handle blocked or stochastic state, and maintain invariants without hiding errors.
5. **Packaging boundary:** assemble the self-contained submission without importing local evaluation or analysis machinery.

Opponent inference may be used by the decision core, but only from observable state. It is not a separate authority for game rules.

Local environment adapters, arenas, opponent loaders, replay parsers, statistics, and reports are evaluation infrastructure rather than submission-policy components. They must be able to compare an unchanged packaged agent without importing private implementation hooks.

## Stable architectural boundaries

- `main.py` is the Kaggle entrypoint and eventual packaged runtime boundary. It should be thin unless self-containment requires generated or vendored code.
- `src/kaggriculture_agent/` will own reusable observation, rule, planning, and execution mechanisms.
- `scripts/` will contain thin local-match, league, replay-analysis, and packaging commands that call maintained owners.
- `tests/` will own contract, determinism, integration, packaging, and regression checks.
- Evaluation instances will be declarative where practical. Generated matches, replays, reports, downloaded opponents, and build artifacts remain outside the project body and authoritative record.
- Strategy variants must share mechanisms and differ through explicit policy/configuration choices. Market analysis, scheduling, pathfinding, and opponent modeling are code concerns, not separate agent-procedure skills.

No competitive strategy or accepted baseline is defined by this initial harness.
