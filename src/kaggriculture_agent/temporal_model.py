"""Joint temporal/resource constraint realization (optional research solver).

No route templates or preassigned worker regions. Service order, locations,
staffing, pickup quantities, carrying and movement are simultaneous variables.
The submission does not import OR-Tools unless this model is explicitly used.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from . import rules
from .intent import compile_intent
from .intent import matches, unit_event
from .execution import Execution
from .market import realization_orders


@dataclass(frozen=True)
class TemporalConfig:
    deterministic_time: float = 2.0
    refinement_limit: int = 8
    random_seed: int = 0
    neighborhoods: int = 6


@dataclass(frozen=True)
class TemporalResult:
    executions: tuple
    final_state: object
    placements: dict
    completed: tuple[str, ...]
    unfulfilled: tuple[str, ...]
    diagnostics: dict
    expected: tuple = ()
    remaining_inputs: tuple = ()


class TemporalModel:
    """Finite-horizon model, with solver-independent intent as its only target."""

    def __init__(self, state, plan, progress=None):
        from ortools.sat.python import cp_model
        self.cp = cp_model
        self.model = cp_model.CpModel()
        self.state, self.plan = state, plan
        if state.board_size != rules.BOARD_SIZE:
            raise ValueError("temporal model requires the pinned default board")
        self.problem = compile_intent(state, plan, progress)
        self.goals = self.problem.goals
        self.H = min(state.turns_left, state.turns_left_today)
        self.need, self.output = Counter(), Counter()
        self.goal_deltas = {}
        for entity in self.problem.entities:
            for goal in entity.goals:
                deltas = [event.delta for path in entity.paths for event in path if event.goal == goal.identifier]
                items = {item for d in deltas for item in d}
                self.goal_deltas[goal.identifier] = {item: (min([0, *(d.get(item, 0) for d in deltas)]),
                    max([0, *(d.get(item, 0) for d in deltas)])) for item in sorted(items)}
                for item, (lo, hi) in self.goal_deltas[goal.identifier].items():
                    self.need[item] += -lo
                    self.output[item] += hi
        initial = Counter(state.shed)
        for worker in state.workers:
            initial.update(worker.inventory)
        self.items = tuple(sorted({*(i for i, n in initial.items() if n),
            *(i for i, n in self.need.items() if n and not i.endswith("_SEED")),
            *(i for i, n in self.output.items() if n)}))
        self.items = tuple(i for i in self.items if not i.endswith("_SEED"))
        self.bounds = {i: max(1, initial[i] + self.need[i] + self.output[i]) for i in self.items}
        self.credit = state.money
        for item in rules.PRODUCTS:
            low_inventory = state.market_inventory[item] - self.need[item] - 2 * self.H * max(1, len(state.unlocked_shops))
            self.credit += (initial[item] + self.output[item]) * rules.market_price(item, low_inventory)
        budget, hands = max(0, self.credit), state.hires_today
        # Derived useful-worker and affordability bounds, not a policy hire cap.
        useful = len(self.goals) + len(self.items)
        while hands < useful and rules.fibonacci_hire_cost(hands) <= budget:
            budget -= rules.fibonacci_hire_cost(hands)
            hands += 1
        self.W = max(len(state.workers), hands + 1)
        self.M = self.W * self.H
        self.decisions = []
        self.service = defaultdict(list)
        self.flows = defaultdict(list)
        self.seed_use = defaultdict(list)
        self.path_choices, self.fires, self.cells, self.present, self.event_times = {}, {}, {}, {}, {}
        self.nonpass = []
        self.asset_values = []
        self.asset_value_bound = 0
        self.settlement_tables = {}
        self._workers()
        self._services()
        self._symmetry()
        self._materials()
        self._markets()
        self._objective()

    def _symmetry(self):
        """Permutable new-asset identities need only one spatial labeling.

        This removes duplicate labelings, not layouts, routes or worker regions.
        Existing assets and transient tile reuse are deliberately not conflated.
        """
        import json
        groups = defaultdict(list)
        for e in self.problem.entities:
            if e.existing or any(path and path[-1].after is None for path in e.paths):
                continue
            semantics = sorted(json.dumps([g.operation, g.required_effect, g.minimum_output, g.deadline], sort_keys=True)
                               for g in e.goals)
            key = json.dumps([e.positions, e.opening, semantics, e.future_service_days], sort_keys=True)
            groups[key].append(e)
        self.equivalent_entities = tuple(tuple(g) for g in groups.values())
        for equivalent in groups.values():
            for a, b in zip(equivalent, equivalent[1:]):
                # When either is absent its unused cell has no physical effect.
                active_a = self.maximum([self.present[g.identifier] for g in a.goals], 0, 1, f"sym-active:{a.identifier}")
                active_b = self.maximum([self.present[g.identifier] for g in b.goals], 0, 1, f"sym-active:{b.identifier}")
                both = self.boolean(f"sym-both:{a.identifier}:{b.identifier}")
                self.model.add_min_equality(both, [active_a, active_b])
                self.model.add(self.cells[a.identifier] <= self.cells[b.identifier]).only_enforce_if(both)

    def integer(self, lo, hi, name):
        return self.model.new_int_var(int(lo), int(hi), name)

    def boolean(self, name):
        return self.model.new_bool_var(name)

    def equal(self, a, b, name):
        result = self.boolean(name)
        self.model.add(a == b).only_enforce_if(result)
        self.model.add(a != b).only_enforce_if(result.Not())
        return result

    def positive(self, amount, name):
        result = self.boolean(name)
        self.model.add(amount >= 1).only_enforce_if(result)
        self.model.add(amount == 0).only_enforce_if(result.Not())
        return result

    def product(self, a, b, lo, hi, name):
        result = self.integer(lo, hi, name)
        self.model.add_multiplication_equality(result, [a, b])
        return result

    def maximum(self, expressions, lo, hi, name):
        result = self.integer(lo, hi, name)
        self.model.add_max_equality(result, expressions)
        return result

    def minimum(self, expressions, lo, hi, name):
        result = self.integer(lo, hi, name)
        self.model.add_min_equality(result, expressions)
        return result

    def _workers(self):
        m, state, H, W = self.model, self.state, self.H, self.W
        self.x, self.y, self.alive, self.birth, self.hire = {}, {}, {}, {}, {}
        self.move, self.shed_site = {}, {}
        spawn = rules.shed_access(state.board_size)[0]
        current = len(state.workers)
        for w in range(W):
            hire = self.integer(-1 if w < current else 0, -1 if w < current else H, f"hire:{w}")
            self.hire[w] = hire
            self.decisions.append(hire)
            if w >= current + 1:
                m.add(hire >= self.hire[w - 1])
            for t in range(H + 1):
                a = self.boolean(f"alive:{w}:{t}")
                m.add(hire < t).only_enforce_if(a)
                m.add(hire >= t).only_enforce_if(a.Not())
                self.alive[w, t] = a
                self.x[w, t] = self.integer(0, state.board_size - 1, f"x:{w}:{t}")
                self.y[w, t] = self.integer(0, state.board_size - 1, f"y:{w}:{t}")
                self.decisions.extend((self.x[w, t], self.y[w, t]))
            start = state.workers[w].position if w < current else spawn
            m.add(self.x[w, 0] == start[0])
            m.add(self.y[w, 0] == start[1])
            for t in range(H):
                born = self.equal(hire, t, f"born:{w}:{t}") if w >= current else 0
                self.birth[w, t] = born
                dx, dy = self.integer(0, 9, f"dx:{w}:{t}"), self.integer(0, 9, f"dy:{w}:{t}")
                m.add_abs_equality(dx, self.x[w, t+1] - self.x[w, t])
                m.add_abs_equality(dy, self.y[w, t+1] - self.y[w, t])
                m.add(dx + dy <= self.alive[w, t] + 2 * born)
                moving = self.boolean(f"move:{w}:{t}")
                m.add(dx + dy == moving).only_enforce_if(self.alive[w, t])
                m.add(moving <= self.alive[w, t])
                self.move[w, t] = moving
                self.nonpass.append(moving)
                site = self.boolean(f"at-shed:{w}:{t}")
                m.add_allowed_assignments([self.x[w, t], self.y[w, t]], rules.shed_access(state.board_size)).only_enforce_if(site)
                self.shed_site[w, t] = site
        # Exact post-unit, prefix-ordered least-occupied shed spawn rule.
        occupancy = {}
        for t in range(1, H + 1):
            for w in range(W):
                for s, (x, y) in enumerate(rules.shed_access(state.board_size)):
                    same_x = self.equal(self.x[w, t], x, f"spawn-x:{w}:{t}:{s}")
                    same_y = self.equal(self.y[w, t], y, f"spawn-y:{w}:{t}:{s}")
                    present = self.boolean(f"spawn-present:{w}:{t}:{s}")
                    m.add_min_equality(present, [same_x, same_y, self.alive[w, t]])
                    occupancy[w, t, s] = present
            for w in range(current, W):
                scores = [4 * sum(occupancy[u, t, s] for u in range(w)) + s for s in range(4)]
                best = self.minimum(scores, 0, 4 * W + 3, f"spawn-best:{w}:{t}")
                alternatives = []
                for s, (x, y) in enumerate(rules.shed_access(state.board_size)):
                    choose = self.boolean(f"spawn-choice:{w}:{t}:{s}")
                    alternatives.append(choose)
                    m.add(scores[s] == best).only_enforce_if(choose)
                    m.add(self.x[w, t] == x).only_enforce_if(choose)
                    m.add(self.y[w, t] == y).only_enforce_if(choose)
                m.add(sum(alternatives) == self.birth[w, t-1])

    def _services(self):
        m, state, H, W = self.model, self.state, self.H, self.W
        spatial, temporal = [], []
        self.goal_actions, self.goal_quantity = {}, {}
        for entity in self.problem.entities:
            cells = [y * state.board_size + x for x, y in entity.positions]
            cell = m.new_int_var_from_domain(self.cp.Domain.from_values(cells or [0]), f"cell:{entity.identifier}")
            self.cells[entity.identifier] = cell
            self.decisions.append(cell)
            ex, ey = self.integer(0, 9, f"ex:{entity.identifier}"), self.integer(0, 9, f"ey:{entity.identifier}")
            m.add_modulo_equality(ex, cell, state.board_size)
            m.add_division_equality(ey, cell, state.board_size)
            choices = [self.boolean(f"path:{entity.identifier}:{i}") for i in range(len(entity.paths))]
            m.add_exactly_one(choices)
            self.path_choices[entity.identifier] = choices
            self.decisions.extend(choices)
            for goal in entity.goals:
                appearances = [(i, event) for i, path in enumerate(entity.paths) for event in path if event.goal == goal.identifier]
                present = self.boolean(f"fulfilled:{goal.identifier}")
                m.add(present == sum(choices[i] for i, _ in appearances))
                self.present[goal.identifier] = present
                quantities, dynamic = {}, set()
                for item, (lo, hi) in self.goal_deltas[goal.identifier].items():
                    q = self.integer(lo, hi, f"q:{goal.identifier}:{item}")
                    if any(event.action[0] == "HARVEST" and event.delta.get(item, 0) > 0 for _, event in appearances):
                        dynamic.add(item)
                    else:
                        m.add(q == sum(event.delta.get(item, 0) * choices[i] for i, event in appearances))
                    quantities[item] = q
                self.goal_quantity[goal.identifier] = quantities
                self.goal_actions[goal.identifier] = appearances
                assigned, ticks = [], []
                for w in range(W):
                    origin = state.workers[w].position if w < len(state.workers) else None
                    travel = min((rules.manhattan(origin, p) if origin else rules.distance_to_shed(p) + 1
                                  for p in entity.positions), default=H)
                    for t in range(travel, min(H, goal.deadline - state.step + 1)):
                        fire = self.boolean(f"do:{goal.identifier}:{w}:{t}")
                        self.fires[goal.identifier, w, t] = fire
                        self.service[w, t].append(fire)
                        assigned.append(fire)
                        ticks.append((t * W + w) * fire)
                        m.add(fire <= self.alive[w, t])
                        m.add(self.x[w, t] == ex).only_enforce_if(fire)
                        m.add(self.y[w, t] == ey).only_enforce_if(fire)
                        for item, q in quantities.items():
                            lo, hi = self.goal_deltas[goal.identifier][item]
                            values = {event.delta.get(item, 0) for _, event in appearances}
                            if len(values) == 1 and item not in dynamic:
                                amount = next(iter(values)) * fire
                            else:
                                amount = self.product(q, fire, lo, hi, f"flow:{goal.identifier}:{w}:{t}:{item}")
                            if item.endswith("_SEED"):
                                self.seed_use[item, t].append(-amount)
                            else:
                                self.flows[w, t, item].append(amount)
                m.add(sum(assigned) == present)
                tick = self.integer(0, self.M, f"tick:{goal.identifier}")
                m.add(tick == sum(ticks))
                self.event_times[goal.identifier] = tick
            self._temporal_yields(entity, choices)
            active = self.boolean(f"entity-active:{entity.identifier}")
            m.add_max_equality(active, [self.present[g.identifier] for g in entity.goals])
            start, finish = self.integer(0, self.M, f"start:{entity.identifier}"), self.integer(0, self.M, f"end:{entity.identifier}")
            for choice, path in zip(choices, entity.paths):
                for a, b in zip(path, path[1:]):
                    m.add(self.event_times[a.goal] + 1 <= self.event_times[b.goal]).only_enforce_if(choice)
                beginning = 0 if entity.opening is not None or not path else self.event_times[path[0].goal]
                ending = self.event_times[path[-1].goal] + 1 if path and path[-1].after is None else self.M
                m.add(start == beginning).only_enforce_if(choice)
                m.add(finish == ending).only_enforce_if(choice)
            duration = self.integer(0, self.M, f"duration:{entity.identifier}")
            m.add(duration == finish - start)
            # Existing occupied sites remain blocked even if no service is done.
            occupied = 1 if entity.existing and entity.opening is not None else active
            spatial.append(m.new_optional_fixed_size_interval_var(cell, 1, occupied, f"space:{entity.identifier}"))
            temporal.append(m.new_optional_interval_var(start, duration, finish, occupied, f"occupancy:{entity.identifier}"))
        if spatial:
            m.add_no_overlap_2d(spatial, temporal)

    def _temporal_yields(self, entity, choices):
        """Clock-aware local dynamics: water/fertilizer/harvest around decay.

        No earliest-harvest heuristic or fixed +1 fertilizer approximation.
        Local paths choose causal order; these equations choose actual timing
        and physical yield within that order, including same-turn unit order.
        """
        m, H, W = self.model, self.H, self.W
        frames = {}
        for goal in entity.goals:
            frame = self.integer(0, H - 1, f"frame:{goal.identifier}")
            m.add_division_equality(frame, self.event_times[goal.identifier], W)
            frames[goal.identifier] = frame
        varying = defaultdict(list)
        for index, (choice, path) in enumerate(zip(choices, entity.paths)):
            raw = entity.opening
            quantity = raw.get("yield_units", 0) if isinstance(raw, dict) else 0
            previous = 0
            for event in path:
                frame, operation = frames[event.goal], event.action[0]
                label = f"{entity.identifier}:{index}:{event.goal}"
                if isinstance(raw, dict) and raw.get("kind") == "PLANT":
                    lifespan = raw.get("max_lifespan_step", -1)
                    if lifespan >= 0:
                        counts = [sum(s >= lifespan and (s-lifespan) % 2 == 0
                                      for s in range(self.state.step, self.state.step+t)) for t in range(H)]
                        now_count, prior_count = self.integer(0, H, f"decay-now:{label}"), self.integer(0, H, f"decay-prior:{label}")
                        m.add_element(frame, counts, now_count)
                        m.add_element(previous, counts, prior_count)
                        lost = self.integer(0, H, f"decayed:{label}")
                        m.add(lost == now_count - prior_count).only_enforce_if(choice)
                        before = self.integer(-H, 100, f"yield-before:{label}")
                        m.add(before == quantity - lost).only_enforce_if(choice)
                        has_decay = self.positive(lost, f"has-decay:{label}")
                        if operation != "DIG":
                            m.add(before >= 1).only_enforce_if([choice, has_decay])
                            m.add(before >= 0).only_enforce_if(choice)
                        quantity = before
                if operation == "WATER" and isinstance(raw, dict):
                    crop = rules.CROPS[raw["crop"]]
                    gain = rules.one_time_water_gain(raw["crop"], planted_day=raw["planted_day"],
                        day=self.state.day, yield_units=0, fertilized_until_day=raw.get("fertilized_until_day", -1))
                    if gain:
                        space = self.maximum([0, crop.max_yield - quantity], 0, crop.max_yield + H, f"yield-space:{label}")
                        increase = self.minimum([gain, space], 0, gain, f"water-gain:{label}")
                        after = self.integer(0, 100, f"yield-after:{label}")
                        m.add(after == quantity + increase).only_enforce_if(choice)
                        quantity = after
                if operation == "HARVEST":
                    m.add(quantity >= 1).only_enforce_if(choice)
                    goal = next(g for g in entity.goals if g.identifier == event.goal)
                    for item, amount in event.delta.items():
                        if amount > 0:
                            bound = self.goal_deltas[event.goal][item][1]
                            produced = self.integer(0, bound, f"harvest:{label}:{item}")
                            m.add(produced == quantity).only_enforce_if(choice)
                            m.add(produced == 0).only_enforce_if(choice.Not())
                            m.add(produced >= goal.minimum_output.get(item, 0)).only_enforce_if(choice)
                            varying[event.goal, item].append(produced)
                    quantity = 0
                elif operation in ("PLANT", "PLACE", "BUILD_COOP", "BUILD_PASTURE", "DIG"):
                    quantity = event.after.get("yield_units", 0) if isinstance(event.after, dict) else 0
                goal = next(g for g in entity.goals if g.identifier == event.goal)
                expected = goal.required_effect.get("yield_units", {}).get("after")
                if isinstance(expected, int) and expected > 0:
                    m.add(quantity >= expected).only_enforce_if(choice)
                raw, previous = event.after, frame
            self._settled_asset_value(entity, index, choice, raw, quantity, previous)
        for (goal, item), quantities in varying.items():
            m.add(self.goal_quantity[goal][item] == sum(quantities))

    def _settled_asset_value(self, entity, index, choice, raw, quantity, previous):
        """Value the state actually surviving decay/refresh, not serviced flags.

        Surviving capital is an acquisition-cost option value, NOT terminal
        cash. Held products and banked care bonuses use opening price quotes.
        At the terminal boundary only bank money has value. The actual final
        tile is still checked by the trajectory replay.
        """
        if self.state.day == 29 or not isinstance(raw, dict):
            return
        from .state import TileState, WorkerState
        m, H = self.model, self.H
        label = f"settle:{entity.identifier}:{index}"
        lifespan = raw.get("max_lifespan_step", -1)
        upper = (max(raw.get("yield_units", 0), rules.CROPS[raw["crop"]].max_yield)
                 if raw.get("kind") == "PLANT" else
                 max(raw.get("yield_units", 0), rules.ANIMALS[raw["animal"]].max_held) if "animal" in raw else 0)
        if raw.get("kind") == "PLANT" and lifespan >= 0:
            # Final turn's decay is executed by the rule-owner lookup below.
            counts = [sum(s >= lifespan and (s-lifespan) % 2 == 0
                          for s in range(self.state.step, self.state.step+t)) for t in range(H)]
            earlier = self.integer(0, H, label+":prior")
            m.add_element(previous, counts, earlier)
            q = self.integer(-H, upper, label+":yield")
            m.add(q == quantity - counts[-1] + earlier).only_enforce_if(choice)
            lost = self.integer(0, H, label+":lost")
            m.add(lost == counts[-1] - earlier)
            decayed = self.positive(lost, label+":decayed")
            empty = self.boolean(label+":empty")
            m.add(q <= 0).only_enforce_if(empty)
            m.add(q >= 1).only_enforce_if(empty.Not())
            dead = self.boolean(label+":dead")
            m.add_min_equality(dead, [decayed, empty])
            clamped = self.maximum([-1, q], -1, upper, label+":clamped")
            lookup_quantity = self.integer(-1, upper, label+":lookup-yield")
            m.add(lookup_quantity == -1).only_enforce_if(dead)
            m.add(lookup_quantity == clamped).only_enforce_if(dead.Not())
        else:
            lookup_quantity = quantity
        table_key = (tuple(sorted((k, v) for k, v in raw.items() if k != "yield_units")), upper)
        values = self.settlement_tables.get(table_key, [])
        for amount in (range(-1, upper + 1) if not values else ()):
            final_raw = dict(raw, yield_units=amount) if "yield_units" in raw else dict(raw)
            if amount < 0:
                final_raw = {"kind": "WEED"}
            probe = replace(self.state, step=self.state.step+H-1, hour=23,
                tiles=(TileState((0, 0), final_raw), *self.state.tiles[1:]),
                workers=(WorkerState(0, (0, 0), {}),), shed={}, seeds={})
            settled = rules.advance_owned(probe, (["PASS"],)).tiles[0].raw
            value = 0
            if isinstance(settled, dict) and settled.get("kind") == "PLANT":
                c = settled["crop"]
                value = rules.CROPS[c].seed_cost + settled.get("yield_units", 0) * self.state.market_prices[c]
            elif isinstance(settled, dict) and "animal" in settled:
                animal = rules.ANIMALS[settled["animal"]]
                value = animal.cost + (settled.get("yield_units", 0) + settled.get("pending_care_bonus", 0)) * self.state.market_prices[animal.product]
                value += int(settled.get("fertilizer_available", False)) * self.state.market_prices["FERTILIZER"]
            values.append(value)
        self.settlement_tables[table_key] = values
        index_var = self.integer(0, len(values)-1, label+":index")
        m.add(index_var == lookup_quantity + 1)
        worth = self.integer(0, max(values), label+":worth")
        m.add_element(index_var, values, worth)
        chosen = self.product(choice, worth, 0, max(values), label+":chosen")
        self.asset_values.append(chosen)
        self.asset_value_bound += max(values)

    def _drop_amounts(self, inventory, rank, shed, label):
        """Exact insertion-order overflow for a whole-inventory DROP."""
        if sum(self.bounds.values()) <= rules.SHED_CAPACITY:
            # Global physical upper bound includes every initial item, every
            # permitted purchase and every possible output. Overflow is then
            # impossible in any reachable state, not just a guessed trajectory.
            return dict(inventory)
        result, total_bound = {}, sum(self.bounds.values())
        room = self.maximum([0, rules.SHED_CAPACITY - sum(shed.values())], 0, rules.SHED_CAPACITY, f"room:{label}")
        for item in self.items:
            prefix = []
            for other in self.items:
                if other == item:
                    continue
                earlier = self.boolean(f"earlier:{label}:{other}:{item}")
                self.model.add(rank[other] < rank[item]).only_enforce_if(earlier)
                self.model.add(rank[other] >= rank[item]).only_enforce_if(earlier.Not())
                prefix.append(self.product(earlier, inventory[other], 0, self.bounds[other], f"prefix:{label}:{other}:{item}"))
            available = self.maximum([0, room - sum(prefix)], 0, rules.SHED_CAPACITY, f"available:{label}:{item}")
            result[item] = self.minimum([inventory[item], available], 0, self.bounds[item], f"drop-kept:{label}:{item}")
        return result

    def _materials(self):
        m, state, H, W = self.model, self.state, self.H, self.W
        self.inventory, self.rank, self.shed = {}, {}, {}
        self.pickup, self.deposit, self.drop = {}, {}, {}
        self.pickup_flags, self.deposit_flags = {}, {}
        for item in self.items:
            for t in range(H + 1):
                self.shed[item, t, 0] = self.integer(0, rules.SHED_CAPACITY, f"shed:{item}:{t}:0")
            m.add(self.shed[item, 0, 0] == state.shed.get(item, 0))
        for w in range(W):
            initial = state.workers[w].inventory if w < len(state.workers) else {}
            initial_order = {item: n + 1 for n, item in enumerate(initial)}
            for item in self.items:
                for t in range(H + 1):
                    self.inventory[w, t, item] = self.integer(0, self.bounds[item], f"carry:{w}:{t}:{item}")
                    self.rank[w, t, item] = self.integer(0, len(self.items) + H + 1, f"rank:{w}:{t}:{item}")
                m.add(self.inventory[w, 0, item] == initial.get(item, 0))
                m.add(self.rank[w, 0, item] == initial_order.get(item, 0))
        for t in range(H):
            for w in range(W):
                drop = self.boolean(f"drop:{w}:{t}")
                self.drop[w, t] = drop
                self.decisions.append(drop)
                m.add(drop <= self.shed_site[w, t])
                m.add(sum(self.inventory[w, t, i] for i in self.items) >= 1).only_enforce_if(drop)
                inv = {i: self.inventory[w, t, i] for i in self.items}
                rank = {i: self.rank[w, t, i] for i in self.items}
                stock = {i: self.shed[i, t, w] for i in self.items}
                kept = self._drop_amounts(inv, rank, stock, f"{w}:{t}")
                actions = [drop, self.move[w, t], *self.service[w, t]]
                for item in self.items:
                    bound = self.bounds[item]
                    pickup = self.integer(0, bound, f"pickup:{w}:{t}:{item}")
                    deposit = self.integer(0, bound, f"deposit:{w}:{t}:{item}")
                    self.pickup[w, t, item], self.deposit[w, t, item] = pickup, deposit
                    self.decisions.extend((pickup, deposit))
                    taking, putting = self.positive(pickup, f"taking:{w}:{t}:{item}"), self.positive(deposit, f"putting:{w}:{t}:{item}")
                    self.pickup_flags[w, t, item], self.deposit_flags[w, t, item] = taking, putting
                    m.add(taking <= self.shed_site[w, t])
                    m.add(putting <= self.shed_site[w, t])
                    m.add(pickup <= stock[item])
                    m.add(deposit <= inv[item])
                    actions.extend((taking, putting))
                    dropped = self.product(drop, inv[item], 0, bound, f"dropped:{w}:{t}:{item}")
                    delivered = self.product(drop, kept[item], 0, bound, f"delivered:{w}:{t}:{item}")
                    after = self.inventory[w, t + 1, item]
                    m.add(after == inv[item] + pickup - deposit - dropped + sum(self.flows[w, t, item]))
                    next_shed = self.integer(0, rules.SHED_CAPACITY, f"shed:{item}:{t}:{w+1}")
                    self.shed[item, t, w + 1] = next_shed
                    m.add(next_shed == stock[item] - pickup + deposit + delivered)
                    remains = self.positive(after, f"has:{w}:{t+1}:{item}")
                    had = self.positive(inv[item], f"had:{w}:{t}:{item}")
                    m.add(self.rank[w, t+1, item] == 0).only_enforce_if(remains.Not())
                    m.add(self.rank[w, t+1, item] == rank[item]).only_enforce_if([remains, had])
                    m.add(self.rank[w, t+1, item] == len(self.items) + t + 1).only_enforce_if([remains, had.Not()])
                m.add(sum(actions) <= self.alive[w, t])
                self.nonpass.extend([drop, *(self.pickup_flags[w, t, i] for i in self.items),
                                     *(self.deposit_flags[w, t, i] for i in self.items), *self.service[w, t]])
                m.add(sum(self.shed[i, t, w+1] for i in self.items) <= rules.SHED_CAPACITY)

    def _markets(self):
        m, state, H, W = self.model, self.state, self.H, self.W
        self.buy, self.sell, self.cash, self.land_unlock = {}, {}, {}, {}
        self.ending_market, self.market_tables = {}, {}
        self.cash[0] = state.money
        purchase_items = tuple(sorted(i for i, n in self.need.items() if n))
        buy_flags, sell_flags = defaultdict(list), defaultdict(list)
        eligible, eligibility, quotes = {}, {}, {}
        for item in purchase_items:
            for t in range(H):
                quantity = self.integer(0, self.need[item], f"buy:{item}:{t}")
                self.buy[item, t] = quantity
                self.decisions.append(quantity)
                buy_flags[t].append(self.positive(quantity, f"buying:{item}:{t}"))
            m.add(sum(self.buy[item, t] for t in range(H)) <= self.need[item])
            # Requirements bound acquisitions, but a bought input is not proof
            # of completed work. Partial realizations can legitimately end with
            # unused inputs; do not exclude those reachable states.
        for crop in rules.CROPS:
            item = crop + "_SEED"
            amount = state.seeds.get(crop, 0)
            for t in range(H):
                consumed = sum(self.seed_use[item, t])
                m.add(consumed <= amount)  # atomic pre-market availability
                following = self.integer(0, state.seeds.get(crop, 0) + self.need[item], f"seeds:{crop}:{t+1}")
                m.add(following == amount - consumed + self.buy.get((item, t), 0))
                amount = following
            if not hasattr(self, "final_seeds"):
                self.final_seeds = {}
            self.final_seeds[crop] = amount
        land_cost = defaultdict(list)
        for q in self.problem.land:
            at = self.integer(0, H, f"land:{q}")
            self.land_unlock[q] = at
            self.decisions.append(at)
            for t in range(H):
                unlocked = self.equal(at, t, f"unlock:{q}:{t}")
                buy_flags[t].append(unlocked)
                land_cost[t].append(rules.LAND_PRICES[rules.LAND_ORDER.index(q)] * unlocked)
        for a, b in zip(self.problem.land, self.problem.land[1:]):
            m.add(self.land_unlock[a] <= self.land_unlock[b])
        for entity in self.problem.entities:
            for q in rules.LAND_ORDER:
                if q in state.unlocked_quadrants:
                    continue
                locations = [y * 10 + x for x, y in entity.positions if rules.quadrant((x, y)) == q]
                for cell in locations:
                    here = self.equal(self.cells[entity.identifier], cell, f"locked-cell:{entity.identifier}:{cell}")
                    for goal in entity.goals:
                        if q not in self.land_unlock:
                            m.add(self.present[goal.identifier] == 0).only_enforce_if(here)
                        else:
                            m.add(self.event_times[goal.identifier] >= (self.land_unlock[q] + 1) * W).only_enforce_if([here, self.present[goal.identifier]])
        self.remaining = {}
        revenue, product_cost = defaultdict(list), defaultdict(list)
        for item in self.items:
            remaining = self.need[item]
            if item in rules.PRODUCTS:
                low = state.market_inventory[item] - self.need[item] - 2 * H * max(1, len(state.unlocked_shops)) - H
                high = state.market_inventory[item] + self.bounds[item] + self.output[item] + 1
                prices = [rules.market_price(item, n) for n in range(low, high + 1)]
                prefix = [0]
                for price in prices:
                    prefix.append(prefix[-1] + price)
                market = state.market_inventory[item]
                floor = next((low + n for n, price in enumerate(prices) if price == 1), high + 1)
            for t in range(H):
                # Input needs remain fixed even when the solver cannot schedule
                # every goal. Only actual scheduled consumption reduces them.
                if item in self.need:
                    consumed = []
                    for goal in self.goals:
                        if self.goal_deltas[goal.identifier].get(item, (0, 0))[0] < 0:
                            for w in range(W):
                                fire = self.fires.get((goal.identifier, w, t))
                                if fire is not None:
                                    consumed.append(fire)
                    remaining = remaining - sum(consumed)
                self.remaining[item, t] = remaining
                if item in rules.PRODUCTS:
                    carried = sum(self.inventory[w, t+1, item] for w in range(W))
                    reserve = self.maximum([0, remaining - carried], 0, max(0, self.need[item]), f"reserve:{item}:{t}")
                    potential = self.maximum([0, self.shed[item, t, W] - reserve], 0, rules.SHED_CAPACITY, f"sale-potential:{item}:{t}")
                    eligible[item, t] = potential
                    eligibility[item, t] = self.positive(potential, f"sale-eligible:{item}:{t}")
                    sale = self.integer(0, rules.SHED_CAPACITY, f"sale:{item}:{t}")
                    self.sell[item, t] = sale
                    sell_flags[t].append(self.positive(sale, f"selling:{item}:{t}"))
                    quote_index = self.integer(0, len(prices), f"quote-index:{item}:{t}")
                    m.add(quote_index == market - low)
                    quotes[item, t] = self.integer(1, max(prices), f"quote:{item}:{t}")
                    m.add_element(quote_index, [*prices, rules.market_price(item, high+1)], quotes[item, t])
                    ceiling = self.maximum([market, floor], low, high + 1, f"floor-ceiling:{item}:{t}")
                    effective = self.minimum([market + sale, ceiling], low, high + 1, f"after-sale:{item}:{t}")
                    bought = self.buy.get((item, t), 0)
                    after_buy = self.integer(low, high + 1, f"after-buy:{item}:{t}")
                    m.add(after_buy == effective - bought)
                    values = []
                    for name, point in (("pre", market), ("sold", effective), ("bought", after_buy)):
                        index = self.integer(0, len(prefix)-1, f"price-index:{item}:{t}:{name}")
                        m.add(index == point - low)
                        value = self.integer(0, prefix[-1], f"price-integral:{item}:{t}:{name}")
                        m.add_element(index, prefix, value)
                        values.append(value)
                    revenue[t].append(values[1] - values[0] + sale - (effective - market))
                    product_cost[t].append(values[1] - values[2])
                    demand = (sum((2 if len(rules.SHOPS[s]) == 1 else 1) for s in state.unlocked_shops if item in rules.SHOPS[s])
                              if (state.step + t) % 4 == 0 else 0)
                    demand += int(item != "FERTILIZER" and (state.step + t) % 24 == 0)
                    market = self.integer(low, high + 1, f"market:{item}:{t+1}")
                    m.add(market == after_buy - demand)
                else:
                    sale = 0
                m.add(self.shed[item, t+1, 0] == self.shed[item, t, W] - sale + self.buy.get((item, t), 0))
            if item in rules.PRODUCTS:
                self.ending_market[item] = market
                self.market_tables[item] = (low, high, prefix, floor)
        for t in range(H):
            births = [self.birth[w, t] for w in range(len(state.workers), W)]
            sale_slots = rules.MAX_MARKET_ORDERS - sum(births) - sum(buy_flags[t])
            alphabet = sorted(rules.PRODUCTS)
            products = [i for i in self.items if i in rules.PRODUCTS]
            for item in products:
                prior = []
                for other in products:
                    if other == item:
                        continue
                    higher = self.boolean(f"sale-higher:{other}:{item}:{t}")
                    difference = len(alphabet) * (quotes[other, t] - quotes[item, t]) + alphabet.index(item) - alphabet.index(other)
                    m.add(difference > 0).only_enforce_if(higher)
                    m.add(difference <= 0).only_enforce_if(higher.Not())
                    precedes = self.boolean(f"sale-prior:{other}:{item}:{t}")
                    m.add_min_equality(precedes, [higher, eligibility[other, t]])
                    prior.append(precedes)
                chosen = self.positive(self.sell[item, t], f"sale-chosen:{item}:{t}")
                m.add(chosen <= eligibility[item, t])
                m.add(sale_slots > sum(prior)).only_enforce_if(chosen)
                m.add(sale_slots <= sum(prior)).only_enforce_if([eligibility[item, t], chosen.Not()])
                m.add(self.sell[item, t] == eligible[item, t]).only_enforce_if(chosen)
                m.add(self.sell[item, t] == 0).only_enforce_if(chosen.Not())
            m.add(sum(births) + sum(buy_flags[t]) + sum(sell_flags[t]) <= rules.MAX_MARKET_ORDERS)
            fixed_cost = [rules.fibonacci_hire_cost(state.hires_today + w - len(state.workers)) * self.birth[w, t]
                          for w in range(len(state.workers), W)]
            fixed_cost += [rules.CROPS[i[:-5]].seed_cost * self.buy[i, t] for i in purchase_items if i.endswith("_SEED")]
            fixed_cost += [rules.ANIMALS[i].cost * self.buy[i, t] for i in purchase_items if i in rules.ANIMALS]
            self.cash[t+1] = self.integer(0, max(1, self.credit), f"cash:{t+1}")
            m.add(self.cash[t+1] == self.cash[t] + sum(revenue[t]) - sum(product_cost[t]) - sum(fixed_cost) - sum(land_cost[t]))
            m.add(sum(self.shed[i, t+1, 0] for i in self.items) <= rules.SHED_CAPACITY)

    def _objective(self):
        # All integer coefficients below implement a lexicographic order using
        # proven finite bounds, not a weighted economic/land/labor utility.
        m, H, W = self.model, self.H, self.W
        stock = {i: self.shed[i, H, 0] for i in self.items}
        if self.state.day != 29:
            for w in range(W):
                kept = self._drop_amounts({i: self.inventory[w, H, i] for i in self.items},
                                         {i: self.rank[w, H, i] for i in self.items}, stock, f"refresh:{w}")
                stock = {i: stock[i] + kept[i] for i in self.items}
        self.end_stock = stock
        quotes = {**self.state.market_prices, **{a: r.cost for a, r in rules.ANIMALS.items()}}
        inventory_values = []
        if self.state.day != 29:
            for item in self.items:
                if item in rules.ANIMALS:
                    inventory_values.append(stock[item] * rules.ANIMALS[item].cost)
                    continue
                if item not in rules.PRODUCTS:
                    continue
                low, high, prefix, floor = self.market_tables[item]
                before = self.ending_market[item]
                ceiling = self.maximum([before, floor], low, high+1, f"end-floor:{item}")
                after = self.minimum([before + stock[item], ceiling], low, high+1, f"end-sale:{item}")
                integrals = []
                for name, point in (("before", before), ("after", after)):
                    index = self.integer(0, len(prefix)-1, f"end-index:{item}:{name}")
                    m.add(index == point-low)
                    integral = self.integer(0, prefix[-1], f"end-integral:{item}:{name}")
                    m.add_element(index, prefix, integral)
                    integrals.append(integral)
                inventory_values.append(integrals[1]-integrals[0] + stock[item] - (after-before))
        inventory_value = sum(inventory_values)
        seed_value = sum(self.final_seeds[c] * r.seed_cost for c, r in rules.CROPS.items()) if self.state.day != 29 else 0
        seed_bound = sum((self.state.seeds.get(c, 0) + self.need[c+"_SEED"]) * r.seed_cost for c, r in rules.CROPS.items())
        stock_quote_bound = max([*quotes.values(), *(rules.market_price(i, table[0]) for i, table in self.market_tables.items())])
        worth_bound = max(1, self.credit + rules.SHED_CAPACITY * stock_quote_bound + seed_bound + self.asset_value_bound)
        future = []
        future_bound = 0
        for e in self.problem.entities:
            if e.existing or not e.positions or not e.future_service_days:
                continue
            distance = self.integer(0, 18, f"future-distance:{e.identifier}")
            self.model.add_allowed_assignments([self.cells[e.identifier], distance],
                [(y * 10 + x, rules.distance_to_shed((x, y))) for x, y in e.positions])
            persistent = self.boolean(f"persistent:{e.identifier}")
            m.add(persistent == sum(choice for choice, path in zip(self.path_choices[e.identifier], e.paths)
                                   if path and isinstance(path[-1].after, dict)))
            servicing = self.product(distance, persistent, 0, 18, f"future-service:{e.identifier}")
            future.append(servicing * e.future_service_days)
            future_bound += 18 * e.future_service_days
        turns_bound = H * W + future_bound
        self.fulfillment = sum(self.present.values()) + sum(self.equal(at, H, f"land-unmet:{q}").Not()
                                                         for q, at in self.land_unlock.items())
        self.worth = self.cash[H] + inventory_value + seed_value + sum(self.asset_values)
        self.capacity_used = sum(self.nonpass) + sum(future)
        self.objective = (self.fulfillment * (worth_bound + 1) * (turns_bound + 1)
                          + self.worth * (turns_bound + 1) - self.capacity_used)
        m.maximize(self.objective)

    def decode(self, solver):
        """Decode decisions; the transition validator still owns reachability."""
        actions = [[['PASS'] for _ in range(self.W)] for _ in range(self.H)]
        goals = {}
        placements = {key: (solver.value(cell) % 10, solver.value(cell) // 10) for key, cell in self.cells.items()}
        events = {}
        for entity in self.problem.entities:
            chosen = next(i for i, var in enumerate(self.path_choices[entity.identifier]) if solver.boolean_value(var))
            events.update({event.goal: event for event in entity.paths[chosen]})
        for (identifier, w, t), var in self.fires.items():
            if solver.boolean_value(var):
                actions[t][w] = list(events[identifier].action)
                goals[t, w] = identifier
        for t in range(self.H):
            for w in range(self.W):
                if (t, w) in goals:
                    continue
                if solver.boolean_value(self.drop[w, t]):
                    actions[t][w] = ['DROP']
                for item in self.items:
                    quantity = solver.value(self.pickup[w, t, item])
                    if quantity:
                        actions[t][w] = ['PICKUP', item, quantity]
                    quantity = solver.value(self.deposit[w, t, item])
                    if quantity:
                        actions[t][w] = ['PLACE', item, quantity]
                if solver.boolean_value(self.move[w, t]):
                    dx = solver.value(self.x[w, t+1]) - solver.value(self.x[w, t])
                    dy = solver.value(self.y[w, t+1]) - solver.value(self.y[w, t])
                    actions[t][w] = [{(1, 0): 'EAST', (-1, 0): 'WEST', (0, 1): 'SOUTH', (0, -1): 'NORTH'}[dx, dy]]
        purchases, hires, land, remaining = [], [], [], []
        for t in range(self.H):
            orders = []
            for item in sorted(self.need):
                quantity = solver.value(self.buy[item, t]) if (item, t) in self.buy else 0
                if quantity:
                    orders.append(['BUY_SEED', item[:-5], quantity] if item.endswith('_SEED') else
                                  ['BUY_ANIMAL' if item in rules.ANIMALS else 'BUY_PRODUCT', item, quantity])
            purchases.append(orders)
            hires.append(sum(solver.value(self.birth[w, t]) for w in range(len(self.state.workers), self.W)))
            land.append(sum(solver.value(at) == t for at in self.land_unlock.values()))
            remaining.append({i: solver.value(self.remaining[i, t]) for i in self.items})
        return dict(actions=actions, goals=goals, placements=placements, purchases=purchases,
                    hires=hires, land=land, remaining=remaining, events=events)

    def exclude_solution(self, solver):
        """Exact no-good over all realization decisions; not a route-template cut."""
        equalities = [self.equal(variable, solver.value(variable), f"nogood:{n}:{len(self.model.proto.constraints)}")
                      for n, variable in enumerate(self.decisions)]
        selected_services = [fire for fire in self.fires.values() if solver.boolean_value(fire)]
        self.model.add_bool_or([literal.Not() for literal in (*equalities, *selected_services)])


def validate_temporal_solution(model, solver, proposal):
    """Reachability, not surrogate objective, decides whether a solution exists."""
    state = model.state
    trace, completed = [], set()
    expected = [state]
    goals = {g.identifier: g for g in model.goals}
    counters = Counter()
    for t in range(model.H):
        actions = proposal["actions"][t][:len(state.workers)]
        if any((t, w) in proposal["goals"] for w in range(len(state.workers), model.W)):
            return None, {"reason": "service assigned to absent worker", "turn": t}
        plants = Counter(a[1] for a in actions if a and a[0] == "PLANT")
        if any(n > state.seeds.get(c, 0) for c, n in plants.items()):
            return None, {"reason": "atomic seed shortfall", "turn": t}
        micro = state
        assignments = {}
        for w, action in enumerate(actions):
            micro, before, after, delta = unit_event(micro, w, action)
            goal = proposal["goals"].get((t, w))
            if goal:
                if not matches(goals[goal], before, after, delta, action):
                    return None, {"reason": "service effect mismatch", "turn": t, "goal": goal, "operation": action[0]}
                predicted = {i: solver.value(q) for i, q in model.goal_quantity[goal].items() if solver.value(q)}
                if delta != predicted:
                    return None, {"reason": "physical service flow mismatch", "turn": t, "goal": goal}
                completed.add(goal)
                assignments[w] = goal
            elif before != after:
                return None, {"reason": "logistics changed an unassigned asset", "turn": t, "worker": w}
        orders = realization_orders(state, tuple(actions), proposal["purchases"][t],
            proposal["hires"][t], proposal["land"][t], proposal["remaining"][t])
        following = rules.advance_owned(state, actions, orders)
        if following.money != solver.value(model.cash[t+1]):
            return None, {"reason": "market cash mismatch", "turn": t}
        for item in model.items:
            expected_stock = (model.end_stock[item] if t == model.H-1 and state.hour == 23 else model.shed[item, t+1, 0])
            if following.shed.get(item, 0) != solver.value(expected_stock):
                return None, {"reason": "shed/overflow mismatch", "turn": t, "item": item}
        if state.hour != 23:
            active = sum(solver.value(model.alive[w, t+1]) for w in range(model.W))
            if len(following.workers) != active:
                return None, {"reason": "staffing mismatch", "turn": t}
            for worker in following.workers:
                w = worker.index
                pos = (solver.value(model.x[w, t+1]), solver.value(model.y[w, t+1]))
                if worker.position != pos:
                    return None, {"reason": "position/spawn mismatch", "turn": t, "worker": w}
                if any(worker.inventory.get(i, 0) != solver.value(model.inventory[w, t+1, i]) for i in model.items):
                    return None, {"reason": "carried resource mismatch", "turn": t, "worker": w}
        trace.append(Execution(tuple(actions), (), assignments, tuple(orders)))
        counters.update(a[0] for a in actions)
        state = following
        expected.append(state)
    unfulfilled = tuple(sorted(goals.keys() - completed))
    completed.update(f"land:{q}" for q in model.problem.land if q in state.unlocked_quadrants)
    unfulfilled += tuple(f"land:{q}" for q in model.problem.land if q not in state.unlocked_quadrants)
    return TemporalResult(tuple(trace), state, proposal["placements"], tuple(sorted(completed)), unfulfilled,
        {"operations": dict(counters), "worker_actions_used": sum(n for op, n in counters.items() if op != "PASS"),
         "goal_count": len(goals) + len(model.problem.land), "completed_count": len(completed), "unfulfilled": list(unfulfilled),
         "hired_hands": sum(proposal["hires"]), "model_fulfillment": solver.value(model.fulfillment),
         "reachable": True}, tuple(expected), tuple(proposal["remaining"])), None


def solve_temporal(state, plan, config=None, progress=None):
    """Search a joint model; never return an unrealizable surrogate trajectory."""
    config = config or TemporalConfig()
    if state.turns_left == 0:
        return TemporalResult((), state, {}, (), (), {"model": "joint-temporal-resource-constraints", "terminal": True}, (state,), ())
    begun = perf_counter()
    model = TemporalModel(state, plan, progress)
    built = perf_counter()
    failures, status_name, bound = [], "NOT_SOLVED", None
    result = None
    spent = 0.0
    phases = []
    full = model.boolean("require-complete-plan")
    model.model.add(model.fulfillment == len(model.goals) + len(model.problem.land)).only_enforce_if(full)
    require_full = all(any(len(path) == len(e.goals) for path in e.paths) for e in model.problem.entities)
    from .temporal_start import starting_witness, hint_witness
    witness = starting_witness(model)
    hint_witness(model, witness)
    initialization = {"supplied": witness is not None, "validated": False}
    incumbent_value = float("-inf")
    incumbent_solver = None
    if witness is not None and model.model.proto.solution_hint.vars:
        seed_solver = model.cp.CpSolver()
        seed_solver.parameters.num_search_workers = 1
        seed_solver.parameters.fix_variables_to_their_hinted_value = True
        seed_solver.parameters.max_deterministic_time = min(0.5, config.deterministic_time * 0.2)
        seed_status = seed_solver.solve(model.model)
        spent += seed_solver.response_proto.deterministic_time
        initialization.update(status=seed_solver.status_name(seed_status),
                              constructed_goals=len(witness["completed"]))
        if seed_status in (model.cp.OPTIMAL, model.cp.FEASIBLE):
            result, failure = validate_temporal_solution(model, seed_solver, model.decode(seed_solver))
            initialization.update(validated=result is not None, failure=failure)
            if result is not None:
                incumbent_value = seed_solver.objective_value
                incumbent_solver = seed_solver
                initialization.update(completed=result.diagnostics["completed_count"], objective=incumbent_value)
                model.model.clear_hints()
                for n in range(len(model.model.proto.variables)):
                    if n != full.index:
                        v = model.model.get_int_var_from_proto_index(n)
                        model.model.add_hint(v, seed_solver.value(v))
        if result is None:
            model.model.clear_hints()
    improved = False
    # Constraint neighborhoods free whole worker trajectories together with ALL
    # material/market decisions and open service/placement choices. They do not
    # enumerate route templates. A subsequent unrestricted solve retains the
    # entire joint feasible space. Bounds from a neighborhood are never global
    # optimality certificates.
    if incumbent_solver is not None:
        import random
        rng = random.Random(config.random_seed + state.step)
        for iteration in range(config.neighborhoods):
            if spent >= config.deterministic_time * 0.7:
                break
            active = [w for w in range(model.W) if incumbent_solver.value(model.hire[w]) < model.H]
            rng.shuffle(active)
            radius = min(len(active), 1 << (iteration % max(1, len(active).bit_length())))
            released = set(active[:radius])
            # Make hiring part of the same subproblem, not a staffing sweep.
            inactive = [w for w in range(model.W) if incumbent_solver.value(model.hire[w]) == model.H]
            released.update(inactive[:radius])
            neighborhood = model.model.clone()
            neighborhood.clear_assumptions()
            for w in range(model.W):
                if w in released:
                    continue
                neighborhood.add(model.hire[w] == incumbent_solver.value(model.hire[w]))
                for t in range(model.H+1):
                    for v in (model.x[w, t], model.y[w, t]):
                        neighborhood.add(v == incumbent_solver.value(v))
            for (g, w, t), fire in model.fires.items():
                if w not in released and incumbent_solver.boolean_value(fire):
                    neighborhood.add(fire == 1)
            neighborhood.add(model.objective >= round(incumbent_value) + 1)
            solver = model.cp.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = config.random_seed
            solver.parameters.cp_model_probing_level = 0
            solver.parameters.max_presolve_iterations = 1
            solver.parameters.symmetry_level = 0
            solver.parameters.linearization_level = 0
            solver.parameters.max_deterministic_time = max(0.001,
                (config.deterministic_time * 0.7 - spent) / max(1, config.neighborhoods-iteration))
            status = solver.solve(neighborhood)
            spent += solver.response_proto.deterministic_time
            phase = {"kind": "joint-constraint-neighborhood", "released_workers": sorted(released),
                "status": solver.status_name(status), "branches": solver.num_branches,
                "deterministic_time": solver.response_proto.deterministic_time, "improved": False}
            if status in (model.cp.OPTIMAL, model.cp.FEASIBLE):
                proposed, failure = validate_temporal_solution(model, solver, model.decode(solver))
                if proposed is not None and solver.objective_value > incumbent_value:
                    result, incumbent_solver, incumbent_value = proposed, solver, solver.objective_value
                    improved, phase["improved"] = True, True
                    phase["completed"] = result.diagnostics["completed_count"]
                    model.model.clear_hints()
                    for n in range(len(model.model.proto.variables)):
                        if n != full.index:
                            v = model.model.get_int_var_from_proto_index(n)
                            model.model.add_hint(v, solver.value(v))
                elif failure:
                    phase["transition_failure"] = failure
            phases.append(phase)
    for refinement in range(config.refinement_limit):
        model.model.clear_assumptions()
        model.model.add_assumption(full if require_full else full.Not())
        solver = model.cp.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = config.random_seed
        # The previous dense encoding exhausted the whole action budget in
        # probing, before making a single search branch. Keep presolve bounded;
        # explicit asset symmetry is already imposed in the formulation.
        solver.parameters.cp_model_probing_level = 0
        solver.parameters.max_presolve_iterations = 1
        solver.parameters.symmetry_level = 0
        solver.parameters.linearization_level = 0
        allowance = config.deterministic_time - spent
        if require_full and result is None:
            allowance *= 0.65
        solver.parameters.max_deterministic_time = max(0.001, allowance)
        status = solver.solve(model.model)
        status_name = solver.status_name(status)
        spent += solver.response_proto.deterministic_time
        phases.append({"require_full_plan": require_full, "status": status_name,
                       "branches": solver.num_branches, "conflicts": solver.num_conflicts,
                       "deterministic_time": solver.response_proto.deterministic_time})
        if status not in (model.cp.OPTIMAL, model.cp.FEASIBLE):
            if require_full and spent < config.deterministic_time:
                require_full = False
                continue
            break
        bound = solver.best_objective_bound
        proposal = model.decode(solver)
        proposed, failure = validate_temporal_solution(model, solver, proposal)
        if proposed is not None:
            if solver.objective_value > incumbent_value:
                result, improved = proposed, True
            break
        failures.append(failure)
        if spent >= config.deterministic_time:
            break
        model.exclude_solution(solver)
    diagnostics = {"model": "joint-temporal-resource-constraints", "status": status_name,
        "build_seconds": built - begun, "seconds": perf_counter() - begun,
        "deterministic_time": spent, "refinement_failures": failures,
        "solve_phases": phases,
        "initialization": initialization, "search_improved_start": improved,
        "solver_objective_bound": bound, "worker_domain": model.W,
        "variables": len(model.model.proto.variables), "constraints": len(model.model.proto.constraints)}
    if result is None:
        # Explicit failure, not a silently weakened economic Plan or a claim of
        # successful planning. Callers must not promote this to a benchmark win.
        current, trace, expected = state, [], [state]
        for _ in range(model.H):
            actions = tuple(["PASS"] for _ in current.workers)
            orders = realization_orders(current, actions, remaining_inputs=model.need)
            trace.append(Execution(actions, (), {}, orders))
            current = rules.advance_owned(current, actions, orders)
            expected.append(current)
        unmet = tuple(g.identifier for g in model.goals) + tuple(f"land:{q}" for q in model.problem.land)
        result = TemporalResult(tuple(trace), current, {}, (), unmet,
            {"reachable": True, "planning_failure": True, "goal_count": len(unmet), "completed_count": 0},
            tuple(expected), tuple(dict(model.need) for _ in range(model.H)))
    return replace(result, diagnostics={**result.diagnostics, **diagnostics})
