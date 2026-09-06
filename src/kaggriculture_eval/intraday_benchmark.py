"""Deterministic execution challenges; development evidence, not a league.

Both controllers receive identical daily intent, inventory, market and staffing
limits. Their trajectories are replayed through the official interpreter against
PASS. Random weeds are disabled to isolate execution; random shop unlocks are
irrelevant until after the measured horizon. No policy logic lives here.
"""
from copy import deepcopy
from dataclasses import replace
from collections import Counter

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.execution import execute
from src.kaggriculture_agent.intraday import EndValue, farm_key, search_day
from src.kaggriculture_agent.planner import PlannerConfig, make_plan
from src.kaggriculture_agent.realization import ExecutionChoices, legacy_choices
from src.kaggriculture_agent.state import TileState, WorkerState, reconstruct


def oracle_environment(state):
    env = make("kaggriculture", configuration={"seed": 41, "weedSpawnChance": 0})
    obs = env.state[0].observation
    farm = obs.farms[0]
    farm.update(money=state.money, farmer=list(state.workers[0].position),
        hands=[list(w.position) for w in state.workers[1:]], hires_today=state.hires_today,
        unlocked_quadrants=list(state.unlocked_quadrants),
        tiles=[[deepcopy(state.tile_at((x, y)).raw) for x in range(10)] for y in range(10)])
    obs.private.update(shed=dict(state.shed), seeds=dict(state.seeds),
        inventories=[dict(w.inventory) for w in state.workers])
    obs.market.inventory = dict(state.market_inventory)
    obs.market.prices = dict(state.market_prices)
    obs.town.unlocked_shops = list(state.unlocked_shops)
    for s in env.state:
        s.observation.step, s.observation.day, s.observation.hour = state.step, state.day, state.hour
        s.observation.farms = obs.farms
        s.observation.market = obs.market
        s.observation.town = obs.town
    return env


def oracle_step(env, actions, orders=()):
    step = env.state[0].observation.step
    env.state[0].action = {"farmer": list(actions[0]) if actions else ["PASS"],
        "hands": [list(a) for a in actions[1:]], "market": [list(a) for a in orders]}
    env.state[1].action = {"farmer": ["PASS"], "hands": [], "market": []}
    official.interpreter(env.state, env)
    for s in env.state:
        s.observation.step = step + 1
    return reconstruct(env.state[0].observation)


def scenarios():
    initial = reconstruct(make("kaggriculture", configuration={"seed": 41}).reset(2)[0].observation)
    initial = replace(initial, tiles=tuple(TileState(t.position, None) for t in initial.tiles),
        unlocked_quadrants=("NW", "NE", "SW", "SE"), money=3000, shed={}, seeds={},
        step=12 * 24, day=12, hour=0)
    def build(name, crops=(), animals=(), workers=(((4, 4), {}),), hour=0, shed=None, day=12, max_hands=0):
        tiles = list(initial.tiles)
        for pos in crops:
            raw = official._new_plant("MELON", day - 12, 24)
            raw.update(yield_units=3, consecutive_unwatered=0)
            tiles[pos[1] * 10 + pos[0]] = TileState(pos, raw)
        for pos in animals:
            raw = official._new_animal("GOOSE", day - 4)
            raw.update(yield_units=3, fertilizer_available=True, pending_care_bonus=1)
            tiles[pos[1] * 10 + pos[0]] = TileState(pos, raw)
        state = replace(initial, tiles=tuple(tiles), day=day, step=day * 24 + hour, hour=hour,
            workers=tuple(WorkerState(i, pos, inv) for i, (pos, inv) in enumerate(workers)),
            hires_today=len(workers) - 1, shed=shed or {})
        plan = make_plan(state, PlannerConfig(cash_reserve=10**9, max_daily_hands=max_hands))
        return name, state, plan
    # 21 turns exactly: buy/pickup 2 + shortest covering route 9 +
    # two fertilizer actions + four water/harvest pairs. The 20-turn control
    # below cannot realize every obligation under this fixed fertilizer intent.
    yield build("clustered-harvest", crops=((0, 0), (1, 0), (0, 1), (1, 1)), hour=3)
    yield build("clustered-capacity-limit", crops=((0, 0), (1, 0), (0, 1), (1, 1)), hour=4)
    yield build("dispersed-maintenance", crops=((0, 0), (4, 0), (0, 4), (8, 8)),
        workers=(((4, 4), {}), ((5, 4), {})))
    yield build("competing-routes", crops=((0, 0), (0, 1), (0, 2)), animals=((4, 2), (5, 2)),
        workers=(((0, 3), {}), ((4, 4), {"WHEAT": 2})), shed={"WHEAT": 2})
    yield build("mixed-maintenance", crops=((2, 2), (3, 2), (4, 2)), animals=((2, 3), (3, 3), (4, 3)),
        workers=(((4, 4), {}), ((5, 4), {})), shed={"WHEAT": 6})
    yield build("carry-chains", animals=((0, 0), (0, 1), (4, 2), (4, 3)),
        workers=(((0, 0), {"WHEAT": 2}), ((4, 4), {})), shed={"WHEAT": 4}, hour=8)
    yield build("terminal-sale", crops=((0, 4), (3, 3)), animals=((4, 2),),
        workers=(((4, 4), {}), ((0, 4), {})), day=29, hour=11)
    yield build("hiring-unlocks-work", crops=((0, 0), (0, 1), (0, 2), (4, 0), (4, 1), (8, 8)),
        max_hands=3)
    # New persistent assets compete with crops for convenient service locations.
    name, state, _ = build("persistent-placement", workers=(((0, 0), {}), ((4, 4), {})))
    state = replace(state, shed={"GOOSE": 1}, seeds={"MELON": 3}, day=0, step=0)
    plan = make_plan(state, PlannerConfig(cash_reserve=10**9, max_daily_hands=1))
    yield name, state, plan


def run_scenario(name, state, plan):
    searched = search_day(state, plan)
    results = {}
    for label in ("greedy", "search"):
        env = oracle_environment(state)
        current = state
        choices = legacy_choices(state, plan) if label == "greedy" else searched.choices
        operations = Counter()
        for i in range(min(state.turns_left_today, state.turns_left)):
            action = execute(current, plan, choices) if label == "greedy" else searched.executions[i]
            predicted = rules.advance_owned(current, action.worker_actions, action.market_orders)
            actual = oracle_step(env, action.worker_actions, action.market_orders)
            if farm_key(actual) != farm_key(predicted) or actual.money != predicted.money or actual.market_inventory != predicted.market_inventory:
                raise AssertionError(f"transition divergence: {name}/{label}/{current.step}")
            operations.update(a[0] for a in action.worker_actions)
            current = actual
        results[label] = {"economic_value": EndValue(state)(current)[0], "cash": current.money,
            "carried": sum(w.carried for w in current.workers), "shed_units": current.shed_used,
            "crops": len(current.crop_tiles()), "animals": len(current.animal_tiles()),
            "maintenance_debt": sum(t.raw.get("consecutive_unwatered", t.raw.get("consecutive_unfed", 0))
                for t in current.tiles if isinstance(t.raw, dict)),
            "operations": dict(operations),
            "movement": sum(operations[op] for op in ("NORTH", "SOUTH", "EAST", "WEST")),
            "logistics": sum(operations[op] for op in ("PICKUP", "DROP", "PLACE"))}
    return {"name": name, **results, "search_diagnostics": searched.diagnostics}
