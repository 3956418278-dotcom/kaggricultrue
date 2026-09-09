"""Production intraday solver.

The public boundary is deliberately only ``OwnedState + Plan -> Realization``.
Everything below is private CP-SAT machinery: local rule-derived event choices,
worker trajectories, placement, resources, market cash and bounded workforce.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter

from . import rules
from .execution import execute_realization
from .market import realization_orders
from .realization import PlanningFailure, Realization, TurnDecision


@dataclass(frozen=True)
class _SolverConfig:
    deterministic_time: float = 0.3
    refinement_limit: int = 8
    random_seed: int = 0
    neighborhoods: int = 6


@dataclass(frozen=True)
class _Candidate:
    turns: tuple[TurnDecision, ...]
    final_state: object
    placements: dict
    completed: tuple[str, ...]
    unfulfilled: tuple[str, ...]
    diagnostics: dict
    expected: tuple = ()
    remaining_inputs: tuple = ()


class _CPModel:
    """Finite-horizon model, with solver-independent intent as its only target."""

    def __init__(self, state, plan, workforce, progress=None, problem=None):
        from ortools.sat.python import cp_model
        self.cp = cp_model
        self.model = cp_model.CpModel()
        self.state, self.plan = state, plan
        if state.board_size != rules.BOARD_SIZE:
            raise ValueError("temporal model requires the pinned default board")
        self.problem = problem or _compile_plan(state, plan, progress)
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
        self.W = workforce
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
            if w >= current:
                # A fixed-workforce solve means this many workers are actually
                # hired, while CP-SAT still owns their economically feasible timing.
                m.add(hire < H)
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


def _validate_solution(model, solver, proposal):
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
            micro, before, after, delta = _unit_event(micro, w, action)
            goal = proposal["goals"].get((t, w))
            if goal:
                if not _matches(goals[goal], before, after, delta, action):
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
        trace.append(TurnDecision(tuple(tuple(action) for action in actions),
                                  tuple(tuple(order) for order in orders)))
        counters.update(a[0] for a in actions)
        state = following
        expected.append(state)
    unfulfilled = tuple(sorted(goals.keys() - completed))
    completed.update(f"land:{q}" for q in model.problem.land if q in state.unlocked_quadrants)
    unfulfilled += tuple(f"land:{q}" for q in model.problem.land if q not in state.unlocked_quadrants)
    return _Candidate(tuple(trace), state, proposal["placements"], tuple(sorted(completed)), unfulfilled,
        {"operations": dict(counters), "worker_actions_used": sum(n for op, n in counters.items() if op != "PASS"),
         "goal_count": len(goals) + len(model.problem.land), "completed_count": len(completed), "unfulfilled": list(unfulfilled),
         "hired_hands": sum(proposal["hires"]), "model_fulfillment": solver.value(model.fulfillment),
         "reachable": True}, tuple(expected), tuple(proposal["remaining"])), None


def _solve_workforce(state, plan, workforce, config, problem=None):
    """One complete fixed-workforce solve; no partial objective or fallback."""
    begun = perf_counter()
    model = _CPModel(state, plan, workforce, problem=problem)
    required = len(model.goals) + len(model.problem.land)
    if not all(any(len(path) == len(entity.goals) for path in entity.paths)
               for entity in model.problem.entities):
        return None, {"workforce": workforce, "status": "INFEASIBLE",
                      "reason": "no legal local action chain",
                      "seconds": perf_counter() - begun}
    model.model.add(model.fulfillment == required)

    # A constructive witness is only a CP hint. It neither removes variables
    # nor changes the mandatory feasible set.
    witness = _starting_witness(model)
    _hint_witness(model, witness)
    solve_model = model.model
    neighborhood_workers = []
    fixed_incumbent_events = 0
    if witness and 0 < len(witness["completed"]) < len(model.goals):
        # Keep the exact reached incumbent services, but leave every unused
        # event/worker/turn variable free. This is a constrained CP completion
        # neighborhood, not greedy reinsertion and not a second planner.
        neighborhood = model.model.clone()
        for (turn, worker), identifier in witness["assignments"].items():
            neighborhood.add(model.fires[identifier, worker, turn] == 1)
            fixed_incumbent_events += 1
        solve_model = neighborhood
    solver = model.cp.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = config.random_seed
    solver.parameters.cp_model_probing_level = 0
    solver.parameters.max_presolve_iterations = 1
    solver.parameters.symmetry_level = 0
    solver.parameters.linearization_level = 0
    solver.parameters.max_deterministic_time = config.deterministic_time
    status = solver.solve(solve_model)
    diagnostics = {
        "workforce": workforce,
        "status": solver.status_name(status),
        "seconds": perf_counter() - begun,
        "deterministic_time": solver.response_proto.deterministic_time,
        "branches": solver.num_branches,
        "conflicts": solver.num_conflicts,
        "variables": len(model.model.proto.variables),
        "constraints": len(model.model.proto.constraints),
        "hint_complete": bool(witness and len(witness["completed"]) == len(model.goals)),
        "hint_completed": len(witness["completed"]) if witness else 0,
        "neighborhood_workers": neighborhood_workers,
        "fixed_incumbent_events": fixed_incumbent_events,
    }
    if status not in (model.cp.OPTIMAL, model.cp.FEASIBLE):
        return None, diagnostics
    candidate, failure = _validate_solution(model, solver, model.decode(solver))
    if candidate is None or candidate.unfulfilled:
        return None, {**diagnostics, "exact_failure": failure,
                      "unfulfilled": list(candidate.unfulfilled) if candidate else []}
    return candidate, diagnostics


def _placement_map(plan, internal):
    placements = {}
    for project in (*plan.obligations, *plan.selected, *plan.support):
        if project.target is not None:
            continue
        entity = (str(project.metadata["entity"]) if project.kind == "STATE_EFFECT"
                  else f"new:{project.identifier}")
        if entity in internal:
            placements[project.identifier] = internal[entity]
    return placements


def _maximum_workforce(state, plan, goal_count):
    if plan.max_hands is not None:
        return max(len(state.workers), int(plan.max_hands) + 1)
    money = state.money
    hands = state.hires_today
    useful_hands = min(goal_count, max(0, state.turns_left_today - 1))
    while hands < useful_hands:
        cost = rules.fibonacci_hire_cost(hands)
        if cost > money:
            break
        money -= cost
        hands += 1
    return max(len(state.workers), hands + 1)


def solve_intraday(state, plan):
    """Solve the fixed Plan and return only a complete exact ``Realization``."""
    if state.day != plan.day:
        raise PlanningFailure("Plan day does not match OwnedState")
    if min(state.turns_left, state.turns_left_today) <= 0:
        empty = Realization({}, ())
        try:
            execute_realization(state, plan, empty)
        except Exception as error:
            raise PlanningFailure("no turns remain for the mandatory Daily Plan") from error
        return empty

    problem = _compile_plan(state, plan)
    goal_count = len(problem.goals) + len(problem.land)
    if goal_count == 0:
        return Realization({}, tuple(
            TurnDecision(tuple(("PASS",) for _ in state.workers))
            for _ in range(min(state.turns_left, state.turns_left_today))
        ))
    maximum = _maximum_workforce(state, plan, goal_count)
    horizon = min(state.turns_left, state.turns_left_today)
    minimum = max(len(state.workers), (goal_count + horizon - 1) // horizon)
    config = _SolverConfig()
    candidates = []
    attempts = []
    first_complete = None
    best_value = None
    for workforce in range(minimum, maximum + 1):
        candidate, diagnostics = _solve_workforce(
            state, plan, workforce, config, problem=problem
        )
        attempts.append(diagnostics)
        if candidate is None:
            continue
        realization = Realization(_placement_map(plan, candidate.placements), candidate.turns)
        try:
            final_state = execute_realization(state, plan, realization)
        except Exception as error:
            attempts[-1] = {**diagnostics, "exact_validation": repr(error)}
            continue
        from .valuation import end_value
        value = end_value(state, final_state)
        candidates.append((value, -workforce, realization))
        if first_complete is None:
            first_complete, best_value = workforce, value
        elif workforce == first_complete + 1:
            if value <= best_value:
                break
            best_value = value
    if not candidates:
        raise PlanningFailure("no complete intraday realization found",
                              diagnostics={"workforces": attempts})
    return max(candidates, key=lambda item: item[:2])[2]


# Private Plan requirement expansion. It derives local legal action chains from
# the exact rule transition and never leaves this solver module.
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Mapping

from . import rules
from .state import TileState, WorkerState


@dataclass(frozen=True)
class _Goal:
    identifier: str
    entity: str
    operation: tuple | None
    required_effect: Mapping
    minimum_output: Mapping
    deadline: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Event:
    goal: str
    action: tuple
    before: object
    after: object
    delta: Mapping


@dataclass(frozen=True)
class _Entity:
    identifier: str
    positions: tuple
    opening: object
    existing: bool
    goals: tuple[_Goal, ...]
    paths: tuple[tuple[_Event, ...], ...]
    future_service_days: int = 0


@dataclass(frozen=True)
class _Problem:
    state: object
    plan: object
    entities: tuple[_Entity, ...]
    land: tuple[str, ...]

    @property
    def goals(self):
        return tuple(g for e in self.entities for g in e.goals)


@dataclass(frozen=True)
class _Progress:
    """Execution history, never an edited economic Plan."""
    completed: frozenset[str] = frozenset()
    placements: Mapping = field(default_factory=dict)


def _tile_key(raw):
    return tuple(sorted(raw.items())) if isinstance(raw, dict) else raw


def _stock(state):
    total = Counter(state.shed)
    for w in state.workers:
        total.update(w.inventory)
    total.update({f"{c}_SEED": n for c, n in state.seeds.items()})
    return total


def _unit_event(before_state, worker, action):
    """Actual ordered primitive effect, excluding market/decay/daily refresh."""
    actions = tuple(["PASS"] if i != worker else list(action) for i in range(len(before_state.workers)))
    after_state = rules.advance_owned(before_state, actions, unit_only=True)
    pos = before_state.workers[worker].position
    before, after = before_state.tile_at(pos).raw, after_state.tile_at(pos).raw
    a, b = _stock(before_state), _stock(after_state)
    delta = {k: b[k] - a[k] for k in sorted(a.keys() | b.keys()) if a[k] != b[k]}
    return after_state, before, after, delta


def _matches(goal, before, after, delta, action):
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


def _work_alphabet():
    """Legal service primitives, not a taxonomy of reconstructed intentions."""
    return (("WATER",), ("FERTILIZE",), ("HARVEST",), ("FEED",), ("CARE",),
            ("COLLECT_FERTILIZER",), ("DIG",), ("BUILD_COOP",), ("BUILD_PASTURE",),
            *(("PLANT", c) for c in rules.CROPS), *(("PLACE", a) for a in rules.ANIMALS))


def _service_paths(state, opening, goals):
    """Enumerate every legal local service order, including explicit shortfalls.

    This is local causal structure, not route templates: global worker/time/
    location/material variables can interleave these events in any realization.
    """
    stock = {i: 10000 for i in (*rules.PRODUCTS, *rules.ANIMALS)}
    template = replace(state, workers=(WorkerState(0, (0, 0), stock),), shed={},
                       seeds={c: 10000 for c in rules.CROPS})
    cache, paths = {}, []

    def probe(raw, action):
        key = (_tile_key(raw), action)
        if key not in cache:
            tiles = (TileState((0, 0), raw), *template.tiles[1:])
            _, before, after, delta = _unit_event(replace(template, tiles=tiles), 0, action)
            cache[key] = (before, after, delta)
        return cache[key]

    def visit(raw, remaining, events):
        paths.append(events)
        for goal in remaining:
            for action in ((goal.operation,) if goal.operation else _work_alphabet()):
                before, after, delta = probe(raw, action)
                if _matches(goal, before, after, delta, action):
                    event = _Event(goal.identifier, action, before, after, delta)
                    visit(after, tuple(g for g in remaining if g.identifier != goal.identifier), (*events, event))
    visit(opening, goals, ())
    return tuple(paths)


def _compile_plan(state, plan, progress=None):
    """Compile both maintained economic commitments and semantic replay Plans."""
    progress = progress or _Progress()
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
                    if operation not in _work_alphabet():
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
            goal = _Goal(identifier, entity, op, fields, outputs, deadline)
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
        paths = _service_paths(state, openings[key], goals)
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
        entities.append(_Entity(key, domains[key], openings[key], existing[key], goals, paths, future[key]))
    land = tuple(p.metadata["quadrant"] for p in plan.support if p.kind == "LAND"
                 and p.metadata["quadrant"] not in state.unlocked_quadrants)
    return _Problem(state, plan, tuple(entities), land)


# Private standard routing incumbent and CP hint construction.
from collections import Counter
from math import ceil

from . import rules


def _routing_seed(model, paths, workforce):
    """Standard capacitated multi-depot routing seed for the joint CP model.

    Each entity is one routing visit with its complete local chain as service
    duration.  This only constructs a hint; CP-SAT remains free to interleave
    those events, change assignment/order and change every open placement.
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    state = model.state
    entities = tuple(model.problem.entities)
    bindings = {entity.identifier: entity.positions[0] for entity in entities
                if entity.existing and entity.positions}
    open_entities = [entity for entity in entities if not entity.existing]
    if open_entities:
        from ortools.sat.python import cp_model
        placement = cp_model.CpModel()
        cells = {
            entity.identifier: placement.new_int_var_from_domain(
                cp_model.Domain.from_values([y * state.board_size + x for x, y in entity.positions]),
                f"seed-place:{entity.identifier}",
            )
            for entity in open_entities if entity.positions
        }
        if len(cells) != len(open_entities):
            return None, bindings
        placement.add_all_different(cells.values())
        costs = []
        for entity in open_entities:
            cost = placement.new_int_var(0, 18, f"seed-distance:{entity.identifier}")
            placement.add_allowed_assignments(
                [cells[entity.identifier], cost],
                [(y * state.board_size + x, rules.distance_to_shed((x, y)))
                 for x, y in entity.positions],
            )
            costs.append(cost)
        placement.minimize(sum(costs))
        placement_solver = cp_model.CpSolver()
        placement_solver.parameters.num_search_workers = 1
        placement_solver.parameters.max_time_in_seconds = 0.1
        status = placement_solver.solve(placement)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, bindings
        for entity in open_entities:
            cell = placement_solver.value(cells[entity.identifier])
            bindings[entity.identifier] = (cell % state.board_size, cell // state.board_size)

    starts = [worker.position for worker in state.workers]
    starts.extend(rules.shed_access()[index % len(rules.shed_access())]
                  for index in range(workforce - len(starts)))
    positions = [*starts, *(bindings.get(entity.identifier, entity.positions[0] if entity.positions else (0, 0))
                              for entity in entities)]
    manager = pywrapcp.RoutingIndexManager(len(positions), workforce,
                                           list(range(workforce)), list(range(workforce)))
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index, to_index):
        source, target = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        if target < workforce:
            return 0
        entity = entities[target - workforce]
        return rules.manhattan(positions[source], positions[target]) + len(paths[entity.identifier])

    callback = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(callback)
    routing.AddDimension(callback, 0, max(1, model.H - 2), True, "turns")
    routing.GetDimensionOrDie("turns").SetGlobalSpanCostCoefficient(100)
    delayed_assets = []
    for offset, entity in enumerate(entities, start=workforce):
        first = paths[entity.identifier][0] if paths[entity.identifier] else None
        if first and any(item in rules.ANIMALS and quantity < 0
                         and state.owned_total(item) < -quantity
                         for item, quantity in first.delta.items()):
            delayed_assets.append(offset)
    for worker, node in enumerate(delayed_assets, start=1):
        if worker >= workforce:
            break
        index = manager.NodeToIndex(node)
        routing.solver().Add(routing.VehicleVar(index) == worker)
        routing.solver().Add(routing.NextVar(routing.Start(worker)) == index)
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.seconds = 1
    parameters.log_search = False
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        return None, bindings
    routes = [[] for _ in range(workforce)]
    for worker in range(workforce):
        index = routing.Start(worker)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= workforce:
                routes[worker].append(entities[node - workforce].identifier)
            index = solution.Value(routing.NextVar(index))
    empty_workers = [worker for worker, route in enumerate(routes) if not route]
    while empty_workers:
        donor = max(range(workforce), key=lambda worker: len(routes[worker]))
        if len(routes[donor]) < 2:
            break
        receiver = empty_workers.pop(0)
        split = len(routes[donor]) // 2
        routes[receiver] = routes[donor][split:]
        routes[donor] = routes[donor][:split]
    return routes, bindings


def _starting_witness(model):
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
    # The caller owns the workforce outer loop.  This routine only provides a
    # feasible hint for that exact model size.
    return _construct(model, paths, model.W)


def _construct(model, paths, workforce):
    state, H = model.state, model.H
    entities = {e.identifier: e for e in model.problem.entities}
    goals = {g.identifier: g for g in model.goals}
    performed = {e.identifier: () for e in entities.values()}
    planned, bindings = _routing_seed(model, paths, workforce)
    if planned is None:
        planned = [[] for _ in range(workforce)]
    planned = [list(route) for route in planned]
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
            purchase_shortfall = any(
                remaining[item] > (
                    micro.seeds.get(item[:-5], 0) if item.endswith("_SEED")
                    else micro.owned_total(item) + future_output[item]
                )
                for item in remaining
            )
            liquidate = purchase_shortfall and any(
                quantity > 0 and item in rules.SELLABLE_PRODUCTS and remaining[item] <= 0
                for item, quantity in worker.inventory.items()
            )
            if liquidate:
                target = min(rules.shed_access(), key=lambda p: rules.manhattan(pos, p))
                unit_actions[w] = ["DROP"] if pos in rules.shed_access() else _move(pos, target)
                micro = rules.advance_owned(
                    micro,
                    tuple(unit_actions[w] if index == w else ["PASS"]
                          for index in range(len(micro.workers))),
                    unit_only=True,
                )
                continue
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
            if key is None and w < len(planned):
                for _ in range(len(planned[w])):
                    candidate = planned[w][0]
                    if len(performed[candidate]) >= len(paths[candidate]):
                        planned[w].pop(0)
                        continue
                    event = next_events(candidate, micro, micro.workers[w])[0]
                    unavailable = any(
                        quantity < 0 and not item.endswith("_SEED")
                        and micro.owned_total(item) < -quantity
                        for item, quantity in event.delta.items()
                    ) or any(
                        quantity < 0 and item.endswith("_SEED")
                        and micro.seeds.get(item[:-5], 0) < -quantity
                        for item, quantity in event.delta.items()
                    )
                    if unavailable and len(planned[w]) > 1:
                        planned[w].append(planned[w].pop(0))
                        continue
                    key = candidate
                    leased[w] = key
                    break
            if key is None:
                # A completed route does not strand untouched event blocks.
                # Reassignment is only an incumbent construction choice; the
                # joint CP model remains free to assign every event differently.
                available = [
                    (rules.manhattan(pos, bindings[candidate]), owner, candidate)
                    for owner, route in enumerate(planned)
                    for candidate in route
                    if candidate not in leased.values()
                    and len(performed[candidate]) < len(paths[candidate])
                ]
                if available:
                    _, owner, key = min(available)
                    planned[owner].remove(key)
                    leased[w] = key
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
                after, before_tile, after_tile, delta = _unit_event(micro, w, event.action)
                if _matches(goals[event.goal], before_tile, after_tile, delta, event.action):
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
        entry_slots = max(0, rules.MAX_MARKET_ORDERS - land)
        hires = min(max(0, workforce-len(state.workers)), entry_slots) if t < H-1 else 0
        purchases = purchases[:max(0, entry_slots-hires)]
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


def _hint_witness(model, witness):
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
        executed = tuple(g for (t, w), g in sorted(assigned.items()) if _goals_entity(model, g) == e.identifier)
        path_index = next((i for i, p in enumerate(e.paths)
                           if tuple(event.goal for event in p[:len(executed)]) == executed), None)
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


def _goals_entity(model, identifier):
    return next(g.entity for g in model.goals if g.identifier == identifier)
