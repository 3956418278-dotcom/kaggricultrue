"""Fixed economic intent -> semantic service requirements, not worker routes.

Local legal service orders come from the maintained unit transition. This
compiler never reads demonstrated worker actions or rewrites the Daily Plan.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Mapping

from . import rules
from .state import TileState, WorkerState


@dataclass(frozen=True)
class ServiceGoal:
    identifier: str
    entity: str
    operation: tuple | None
    required_effect: Mapping
    minimum_output: Mapping
    deadline: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServiceEvent:
    goal: str
    action: tuple
    before: object
    after: object
    delta: Mapping


@dataclass(frozen=True)
class EntityWork:
    identifier: str
    positions: tuple
    opening: object
    existing: bool
    goals: tuple[ServiceGoal, ...]
    paths: tuple[tuple[ServiceEvent, ...], ...]
    future_service_days: int = 0


@dataclass(frozen=True)
class IntentProblem:
    state: object
    plan: object
    entities: tuple[EntityWork, ...]
    land: tuple[str, ...]

    @property
    def goals(self):
        return tuple(g for e in self.entities for g in e.goals)


@dataclass(frozen=True)
class IntentProgress:
    """Execution history, never an edited economic Plan."""
    completed: frozenset[str] = frozenset()
    placements: Mapping = field(default_factory=dict)


def tile_key(raw):
    return tuple(sorted(raw.items())) if isinstance(raw, dict) else raw


def _stock(state):
    total = Counter(state.shed)
    for w in state.workers:
        total.update(w.inventory)
    total.update({f"{c}_SEED": n for c, n in state.seeds.items()})
    return total


def unit_event(before_state, worker, action):
    """Actual ordered primitive effect, excluding market/decay/daily refresh."""
    actions = tuple(["PASS"] if i != worker else list(action) for i in range(len(before_state.workers)))
    after_state = rules.advance_owned(before_state, actions, unit_only=True)
    pos = before_state.workers[worker].position
    before, after = before_state.tile_at(pos).raw, after_state.tile_at(pos).raw
    a, b = _stock(before_state), _stock(after_state)
    delta = {k: b[k] - a[k] for k in sorted(a.keys() | b.keys()) if a[k] != b[k]}
    return after_state, before, after, delta


def matches(goal, before, after, delta, action):
    if before == after:
        return False
    if goal.operation is not None and tuple(action) != goal.operation:
        return False
    fields = after if isinstance(after, dict) else {"$tile": after}
    for name, condition in goal.required_effect.items():
        if (name in fields) != condition["after_present"]:
            return False
        if not condition["after_present"]:
            continue
        expected, actual = condition["after"], fields[name]
        # More physical yield or fertilizer lifetime is not worse fulfillment.
        if name in ("yield_units", "fertilized_until_day") and isinstance(expected, int) and expected > 0:
            if actual < expected:
                return False
        elif actual != expected:
            return False
    return all(delta.get(item, 0) >= n for item, n in goal.minimum_output.items())


def work_alphabet():
    """Legal service primitives, not a taxonomy of reconstructed intentions."""
    return (("WATER",), ("FERTILIZE",), ("HARVEST",), ("FEED",), ("CARE",),
            ("COLLECT_FERTILIZER",), ("DIG",), ("BUILD_COOP",), ("BUILD_PASTURE",),
            *(("PLANT", c) for c in rules.CROPS), *(("PLACE", a) for a in rules.ANIMALS))


def service_paths(state, opening, goals):
    """Enumerate every legal local service order, including explicit shortfalls.

    This is local causal structure, not route templates: global worker/time/
    location/material variables can interleave these events in any realization.
    """
    stock = {i: 10000 for i in (*rules.PRODUCTS, *rules.ANIMALS)}
    template = replace(state, workers=(WorkerState(0, (0, 0), stock),), shed={},
                       seeds={c: 10000 for c in rules.CROPS})
    cache, paths = {}, []

    def probe(raw, action):
        key = (tile_key(raw), action)
        if key not in cache:
            tiles = (TileState((0, 0), raw), *template.tiles[1:])
            _, before, after, delta = unit_event(replace(template, tiles=tiles), 0, action)
            cache[key] = (before, after, delta)
        return cache[key]

    def visit(raw, remaining, events):
        paths.append(events)
        for goal in remaining:
            for action in ((goal.operation,) if goal.operation else work_alphabet()):
                before, after, delta = probe(raw, action)
                if matches(goal, before, after, delta, action):
                    event = ServiceEvent(goal.identifier, action, before, after, delta)
                    visit(after, tuple(g for g in remaining if g.identifier != goal.identifier), (*events, event))
    visit(opening, goals, ())
    return tuple(paths)


def compile_intent(state, plan, progress=None):
    """Compile both maintained economic commitments and semantic replay Plans."""
    progress = progress or IntentProgress()
    groups, domains, openings, existing, future = defaultdict(list), {}, {}, {}, Counter()
    end = min((state.day + 1) * 24 - 1, rules.TERMINAL_ACTION_STEP)
    for p in (*plan.obligations, *plan.selected, *plan.support):
        if p.kind == "LAND":
            continue
        if p.kind == "STATE_EFFECT":
            entity = p.metadata["entity"]
            positions = (p.target,) if p.target is not None else plan.placement_domains.get(p.identifier, ())
            requirements = [(p.identifier, None, p.metadata["required_effect"],
                             {q.item: q.quantity for q in p.physical.outputs}, end)]
            if p.metadata.get("demonstrated_count", 1) > 1:
                requirements *= p.metadata["demonstrated_count"]
        else:
            entity = f"existing:{p.target[0]}:{p.target[1]}" if p.target is not None else f"new:{p.identifier}"
            positions = (p.target,) if p.target is not None else plan.placement_domains.get(p.identifier, ())
            requirements = []
            aliases = {"PICKUP_PLACE": ("PLACE",), "HARVEST_TRANSPORT": ("HARVEST",),
                       "WATER_HARVEST_TRANSPORT": ("WATER", "HARVEST")}
            for index, work in enumerate(p.actions.work):
                if work.day != state.day:
                    continue
                for kind in aliases.get(work.kind, (work.kind,)):
                    operation = (("PLANT", p.metadata["crop"]) if kind == "PLANT" else
                                 ("PLACE", p.metadata["animal"]) if kind == "PLACE" else
                                 ("BUILD_" + p.metadata["structure"],) if kind == "BUILD" else (kind,))
                    if operation not in work_alphabet():
                        raise ValueError(f"unsupported daily service requirement: {kind}")
                    # Logistics counts in WorkAmount are not extra economic
                    # tasks. Pickup/distribution remains a realization choice.
                    requirements.append((f"{p.identifier}:{index}:{kind}", operation, {}, {},
                                         min(end, work.deadline_step if work.deadline_step is not None else end)))
            future[entity] += len({w.day for w in p.actions.work if w.day > state.day})
        if not requirements:
            continue
        if entity in progress.placements:
            positions = (progress.placements[entity],)
        positions = tuple(tuple(pos) for pos in positions)
        if not positions:
            # Preserve an impossible objective as a domain-empty requirement;
            # never silently replace its intent with another production goal.
            opening = None
        else:
            opening = state.tile_at(positions[0]).raw
            if opening == "LOCKED" and not p.existing:
                opening = None  # authorized land unlock is a separate constraint
            if entity not in progress.placements and p.target is None and (p.kind == "CROP" or p.metadata.get("requires_construction")):
                opening = None
            if entity not in progress.placements and p.kind == "STATE_EFFECT" and p.target is None:
                tile_before = p.metadata["required_effect"].get("$tile", {})
                if tile_before.get("before_present") and tile_before.get("before") is None:
                    opening = None
        if entity in domains:
            domains[entity] = tuple(pos for pos in domains[entity] if pos in positions)
            if opening is None:
                openings[entity] = None
        else:
            domains[entity], openings[entity], existing[entity] = positions, opening, p.target is not None or entity in progress.placements
        for n, (identifier, op, fields, outputs, deadline) in enumerate(requirements):
            identifier = f"{identifier}:copy:{n}" if p.metadata.get("demonstrated_count", 1) > 1 else identifier
            if identifier in progress.completed:
                continue
            goal = ServiceGoal(identifier, entity, op, fields, outputs, deadline)
            if op and isinstance(opening, dict):
                satisfied = {"WATER": opening.get("watered_today"), "FEED": opening.get("fed_today"),
                             "CARE": opening.get("cared_today"),
                             "FERTILIZE": opening.get("fertilized_until_day", -1) >= state.day + 2}
                if satisfied.get(op[0], False):
                    continue  # state-proven residual completion, not Plan editing
            groups[entity].append(goal)
    entities = []
    for key, goals in sorted(groups.items()):
        # Separate native commitments can name the same physical daily action.
        unique = {}
        for goal in goals:
            signature = goal.operation if goal.operation else goal.identifier
            if signature in unique:
                prior = unique[signature]
                unique[signature] = replace(prior, aliases=(*prior.aliases, goal.identifier))
            else:
                unique[signature] = goal
        goals = tuple(unique.values())
        paths = service_paths(state, openings[key], goals)
        if not existing[key] and not future[key]:
            # A servicing-distance diagnostic derived from the created asset's
            # rules, not the demonstrator's route or a hidden strategy target.
            for path in paths:
                raw = path[-1].after if path else None
                if isinstance(raw, dict) and "animal" in raw:
                    future[key] = max(future[key], 29 - state.day)
                elif isinstance(raw, dict) and raw.get("kind") == "PLANT":
                    crop = rules.CROPS[raw["crop"]]
                    duration = crop.first_yield_day + ((crop.max_yield-1)*crop.interval if crop.ongoing else 0)
                    future[key] = max(future[key], min(29-state.day, duration))
        entities.append(EntityWork(key, domains[key], openings[key], existing[key], goals, paths, future[key]))
    land = tuple(p.metadata["quadrant"] for p in plan.support if p.kind == "LAND"
                 and p.metadata["quadrant"] not in state.unlocked_quadrants)
    return IntentProblem(state, plan, tuple(entities), land)
