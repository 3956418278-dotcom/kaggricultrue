"""Construct a feasible starting witness for the joint constraint optimizer.

This is neither a competing planner nor the definition of its search space.
Every hinted hire, assignment, quantity, order and placement stays a free
variable in the optimizer. Diagnostics distinguish an unchanged starting
witness from a search-improved solution.
"""
from collections import Counter
from math import ceil

from . import rules
from .intent import matches, unit_event
from .market import realization_orders


def starting_witness(model):
    state, problem, H = model.state, model.problem, model.H
    if not model.goals or H < 4:
        return None
    paths = {e.identifier: max(e.paths, key=lambda p: (len(p),
        sum(max(0, n)*state.market_prices.get(i, 0) for event in p for i, n in event.delta.items())))
        for e in problem.entities}
    bill = sum(max(0, model.need[i] - model.output[i] - (state.seeds.get(i[:-5], 0) if i.endswith("_SEED") else state.owned_total(i))) *
        (rules.CROPS[i[:-5]].seed_cost if i.endswith("_SEED") else rules.ANIMALS[i].cost if i in rules.ANIMALS else
         rules.market_price(i, state.market_inventory[i] - model.need[i])) for i in model.need)
    bill += sum(rules.LAND_PRICES[rules.LAND_ORDER.index(q)] for q in problem.land)
    available = state.money - bill + sum(rules.projected_sale_revenue(i,
        max(0, state.shed.get(i, 0)+model.output[i]-model.need[i]), state.market_inventory[i], state.step, state.step, ()) for i in rules.PRODUCTS)
    largest = len(state.workers)
    while largest < model.W and rules.fibonacci_hire_cost(state.hires_today + largest-len(state.workers)) <= available:
        available -= rules.fibonacci_hire_cost(state.hires_today + largest-len(state.workers))
        largest += 1
    minimum = max(len(state.workers), ceil(sum(map(len, paths.values())) / max(1, H-2)))
    best = None
    for workforce in range(min(minimum, largest), largest+1):
        witness = _construct(model, paths, workforce)
        if best is None or len(witness["completed"]) > len(best["completed"]):
            best = witness
        if len(witness["completed"]) == len(model.goals):
            break
    return best


def _construct(model, paths, workforce):
    state, H = model.state, model.H
    entities = {e.identifier: e for e in model.problem.entities}
    goals = {g.identifier: g for g in model.goals}
    performed = {e.identifier: () for e in entities.values()}
    bindings = {e.identifier: e.positions[0] for e in entities.values() if e.existing and e.positions}
    leased, done = {}, set()
    history, actions, assignments, orders, microstates = [state], [], {}, [], []
    remaining = Counter(model.need)
    future_output = Counter(model.output)
    def next_events(key, micro, worker):
        prefix = performed[key]
        suffixes = [p[len(prefix):] for p in entities[key].paths if tuple(v.goal for v in p[:len(prefix)]) == prefix]
        if not suffixes:
            return []
        best = max((len(p), sum(max(0, n)*state.market_prices.get(i, 0) for e in p for i, n in e.delta.items())) for p in suffixes)
        unique = {p[0].goal: p[0] for p in suffixes if p and (len(p), sum(max(0, n)*state.market_prices.get(i, 0)
                  for e in p for i, n in e.delta.items())) == best}
        def rank(event):
            missing = sum(max(0, -n - worker.inventory.get(i, 0)) for i, n in event.delta.items()
                          if n < 0 and not i.endswith("_SEED"))
            critical = sum(min(n, max(0, remaining[i] - micro.owned_total(i))) for i, n in event.delta.items() if n > 0)
            return missing, -critical, event.goal
        return sorted(unique.values(), key=rank)

    for t in range(H):
        unit_actions = [["PASS"] for _ in state.workers]
        micro = state
        for worker in state.workers:
            w, pos = worker.index, worker.position
            key = leased.get(w)
            if key is not None and len(performed[key]) >= len(paths[key]):
                leased.pop(w)
                key = None
            if key is not None:
                event = next_events(key, micro, micro.workers[w])[0]
                starved = any(n < 0 and not i.endswith("_SEED") and micro.owned_total(i) == 0
                              and future_output[i] > 0 for i, n in event.delta.items())
                if starved:
                    # Do not lease a worker indefinitely to a consumer while
                    # reachable harvesting/collection could supply its input.
                    leased.pop(w)
                    key = None
            if key is None:
                candidates = []
                busy = set(leased.values())
                occupied = {bindings[k] for k in leased.values()}
                for e in entities.values():
                    if e.identifier in busy or len(performed[e.identifier]) >= len(paths[e.identifier]):
                        continue
                    options = next_events(e.identifier, micro, micro.workers[w])
                    if not options:
                        continue
                    event = options[0]
                    if any(n < 0 and not i.endswith("_SEED") and micro.owned_total(i) == 0
                           and future_output[i] > 0 for i, n in event.delta.items()):
                        continue
                    locations = (bindings[e.identifier],) if e.identifier in bindings else (
                        p for p in e.positions if p not in occupied and micro.tile_at(p).raw == e.opening)
                    for p in locations:
                        missing = [i for i, n in event.delta.items() if n < 0 and not i.endswith("_SEED") and micro.workers[w].inventory.get(i, 0) < -n]
                        travel = (rules.distance_to_shed(pos) + rules.distance_to_shed(p) + len(missing)
                                  if missing else rules.manhattan(pos, p))
                        candidates.append((goals[event.goal].deadline, -e.future_service_days if not e.existing else 0,
                                           travel + len(paths[e.identifier])-len(performed[e.identifier]), p, e.identifier))
                if candidates:
                    _, _, _, target, key = min(candidates)
                    leased[w], bindings[key] = key, target
            if key is None:
                if worker.carried and (state.day == 29 or t < H-1):
                    supply = any(n and remaining[i] > micro.owned_total(i) - micro.workers[w].inventory.get(i, 0)
                                 for i, n in micro.workers[w].inventory.items())
                    overflow = sum(micro.shed.values()) + sum(v.carried for v in micro.workers) + sum(future_output.values()) - sum(remaining.values()) > rules.SHED_CAPACITY
                    if pos in rules.shed_access():
                        unit_actions[w] = ["DROP"]
                    elif state.day == 29 or supply or overflow:
                        unit_actions[w] = _move(pos, min(rules.shed_access(), key=lambda p: rules.manhattan(pos, p)))
                micro = rules.advance_owned(micro, tuple(a if i == w else ["PASS"] for i, a in enumerate(unit_actions)), unit_only=True)
                continue
            event = next_events(key, micro, micro.workers[w])[0]
            missing = [i for i, n in event.delta.items() if n < 0 and not i.endswith("_SEED") and micro.workers[w].inventory.get(i, 0) < -n]
            if missing:
                if pos not in rules.shed_access():
                    action = _move(pos, min(rules.shed_access(), key=lambda p: rules.manhattan(pos, p)))
                else:
                    item = missing[0]
                    quantity = min(micro.shed.get(item, 0), max(1, ceil(remaining[item] / len(state.workers))))
                    returning_inputs = any(n and remaining[i] > micro.shed.get(i, 0)
                                           for i, n in micro.workers[w].inventory.items())
                    action = (["PICKUP", item, quantity] if quantity else
                              ["DROP"] if returning_inputs else ["PASS"])
            elif pos != bindings[key]:
                action = _move(pos, bindings[key])
            else:
                after, before_tile, after_tile, delta = unit_event(micro, w, event.action)
                if matches(goals[event.goal], before_tile, after_tile, delta, event.action):
                    action = list(event.action)
                    performed[key] += (event.goal,)
                    assignments[t, w] = event.goal
                    done.add(event.goal)
                    remaining.subtract({i: -n for i, n in delta.items() if n < 0})
                    future_output.subtract({i: hi for i, (_, hi) in model.goal_deltas[event.goal].items() if hi > 0})
                    if len(performed[key]) >= len(paths[key]):
                        leased.pop(w, None)
                else:
                    action = ["PASS"]
            unit_actions[w] = action
            micro = rules.advance_owned(micro, tuple(a if i == w else ["PASS"] for i, a in enumerate(unit_actions)), unit_only=True)
        purchases = []
        if t < H-1:
            for item in sorted(remaining):
                owned = micro.seeds.get(item[:-5], 0) if item.endswith("_SEED") else micro.owned_total(item)
                quantity = max(0, remaining[item] - owned - future_output[item])
                if quantity:
                    purchases.append(["BUY_SEED", item[:-5], quantity] if item.endswith("_SEED") else
                                     ["BUY_ANIMAL" if item in rules.ANIMALS else "BUY_PRODUCT", item, quantity])
        land = sum(q not in state.unlocked_quadrants for q in model.problem.land) if t < H-1 else 0
        purchase_slots = max(0, rules.MAX_MARKET_ORDERS - land)
        purchases = purchases[:purchase_slots]
        hires = min(max(0, workforce-len(state.workers)), max(0, purchase_slots-len(purchases))) if t < H-1 else 0
        market = realization_orders(state, tuple(unit_actions), purchases, hires, land, remaining)
        if t < H-1:
            market = _affordable(micro, market)
            market = realization_orders(state, tuple(unit_actions),
                [o for o in market if o[0].startswith("BUY_") and o[0] != "BUY_LAND"],
                sum(o[0] == "HIRE" for o in market), sum(o[0] == "BUY_LAND" for o in market), remaining)
        actions.append(unit_actions)
        orders.append(market)
        microstates.append(micro)
        state = rules.advance_owned(state, unit_actions, market)
        history.append(state)
    return dict(states=history, microstates=microstates, actions=actions, assignments=assignments,
                orders=orders, bindings=bindings, completed=done)


def _affordable(after_units, orders):
    """Normalize requests to actual quantities with the maintained transition."""
    accepted = []
    previous = after_units
    passes = tuple(["PASS"] for _ in after_units.workers)
    for order in orders:
        following = rules.advance_owned(after_units, passes, (*accepted, order))
        op = order[0]
        if op == "HIRE":
            if following.hires_today > previous.hires_today:
                accepted.append(order)
        elif op == "BUY_LAND":
            if len(following.unlocked_quadrants) > len(previous.unlocked_quadrants):
                accepted.append(order)
        elif op.startswith("BUY_"):
            before = previous.seeds if op == "BUY_SEED" else previous.shed
            after = following.seeds if op == "BUY_SEED" else following.shed
            quantity = after.get(order[1], 0) - before.get(order[1], 0)
            if quantity:
                accepted.append([op, order[1], quantity])
        else:
            accepted.append(order)
        previous = following
    return tuple(accepted)


def _move(origin, target):
    if origin[0] != target[0]:
        return ["EAST" if origin[0] < target[0] else "WEST"]
    if origin[1] != target[1]:
        return ["SOUTH" if origin[1] < target[1] else "NORTH"]
    return ["PASS"]


def hint_witness(model, witness):
    """Hints supply an incumbent, never restrictions on joint optimization."""
    m, H, W = model.model, model.H, model.W
    m.clear_hints()
    if witness is None:
        return
    # Respect the model's label symmetry without changing a physical action.
    # Equivalent assets may be renamed; different work or terrain may not.
    import json
    def semantic(g):
        return json.dumps([g.operation, g.required_effect, g.minimum_output, g.deadline], sort_keys=True)
    rename, goal_rename = {}, {}
    bindings = witness["bindings"]
    for group in model.equivalent_entities:
        ordered = sorted(group, key=lambda e: (bindings.get(e.identifier, (-1, -1))[1],
                                               bindings.get(e.identifier, (-1, -1))[0], e.identifier))
        for source, target in zip(ordered, group):
            rename[source.identifier] = target.identifier
            for a, b in zip(sorted(source.goals, key=semantic), sorted(target.goals, key=semantic)):
                goal_rename[a.identifier] = b.identifier
    bindings = {rename.get(k, k): v for k, v in bindings.items()}
    assigned = {slot: goal_rename.get(g, g) for slot, g in witness["assignments"].items()}
    for e in model.problem.entities:
        executed = tuple(g for (t, w), g in sorted(assigned.items()) if goals_entity(model, g) == e.identifier)
        path_index = next((i for i, p in enumerate(e.paths) if tuple(event.goal for event in p) == executed), None)
        if path_index is None:
            m.clear_hints()
            return
        for i, var in enumerate(model.path_choices[e.identifier]):
            m.add_hint(var, int(i == path_index))
        position = bindings.get(e.identifier, e.positions[0] if e.positions else (0, 0))
        m.add_hint(model.cells[e.identifier], position[1]*10+position[0])
    for (g, w, t), fire in model.fires.items():
        m.add_hint(fire, int(assigned.get((t, w)) == g))
    for w in range(W):
        hire = -1 if w < len(model.state.workers) else next((t-1 for t, s in enumerate(witness["states"][1:], 1)
            if len(s.workers) > w), H)
        m.add_hint(model.hire[w], hire)
        for t in range(H+1):
            s = witness["microstates"][-1] if t == H and model.state.day != 29 else witness["states"][t]
            pos = s.workers[w].position if w < len(s.workers) else rules.shed_access()[0]
            m.add_hint(model.x[w, t], pos[0])
            m.add_hint(model.y[w, t], pos[1])
        for t in range(H):
            a = witness["actions"][t][w] if w < len(witness["actions"][t]) else ["PASS"]
            m.add_hint(model.drop[w, t], int(a[0] == "DROP"))
            for item in model.items:
                m.add_hint(model.pickup[w, t, item], a[2] if a[0] == "PICKUP" and a[1] == item else 0)
                m.add_hint(model.deposit[w, t, item], a[2] if a[0] == "PLACE" and len(a) > 2 and a[1] == item and (t, w) not in assigned else 0)
    for (item, t), var in model.buy.items():
        quantity = sum(o[2] for o in witness["orders"][t] if o[0].startswith("BUY_") and len(o) > 2
                       and (o[1]+"_SEED" if o[0] == "BUY_SEED" else o[1]) == item)
        m.add_hint(var, quantity)
    for q, var in model.land_unlock.items():
        m.add_hint(var, next((t for t in range(H) if q in witness["states"][t+1].unlocked_quadrants), H))


def goals_entity(model, identifier):
    return next(g.entity for g in model.goals if g.identifier == identifier)
