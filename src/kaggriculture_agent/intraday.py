"""Fixed-job intraday routing for one canonical Daily Plan.

The Plan owns economic outcomes, exact farm placements, and any exceptional
economic windows. This module derives primitive local work through the real
rule transition, then schedules only worker assignment, route order, and sparse
resource logistics. It never reads ActionDimension as an execution prescription
and never creates a turn-expanded farm/inventory state lattice.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping

from . import rules
from .execution import execute_realization
from .market import realization_orders
from .realization import PlanningFailure, Realization, TurnDecision
from .state import OwnedState, TileState, WorkerState


@dataclass(frozen=True)
class _Event:
    identifier: str
    project: str
    action: tuple[object, ...]
    delta: Mapping[str, int]
    position: tuple[int, int]
    deadline: int


@dataclass(frozen=True)
class _Project:
    identifier: str
    position: tuple[int, int]
    opening: object
    ending: object
    events: tuple[_Event, ...]


@dataclass(frozen=True)
class _Node:
    identifier: str
    kind: str
    position: tuple[int, int] | None = None
    event: str | None = None
    item: str | None = None
    quantity: int = 0
    producer: str | None = None


def _stock(state: OwnedState) -> Counter:
    result = Counter(state.shed)
    for worker in state.workers:
        result.update(worker.inventory)
    result.update({f"{crop}_SEED": quantity
                   for crop, quantity in state.seeds.items()})
    return result


def _unit_effect(state: OwnedState, raw: object, action: tuple[object, ...]):
    local = replace(state, tiles=(TileState((0, 0), raw), *state.tiles[1:]))
    after = rules.advance_owned(local, (action,), unit_only=True)
    before_stock, after_stock = _stock(local), _stock(after)
    delta = {
        item: after_stock[item] - before_stock[item]
        for item in sorted(before_stock.keys() | after_stock.keys())
        if before_stock[item] != after_stock[item]
    }
    return after.tile_at((0, 0)).raw, delta


def _actions() -> tuple[tuple[object, ...], ...]:
    # Only a deterministic tie-break among equally short local chains.
    return (
        ("DIG",), ("BUILD_COOP",), ("BUILD_PASTURE",),
        *(("PLANT", crop) for crop in sorted(rules.CROPS)),
        *(("PLACE", animal) for animal in sorted(rules.ANIMALS)),
        ("WATER",), ("FERTILIZE",), ("HARVEST",), ("FEED",),
        ("CARE",), ("COLLECT_FERTILIZER",),
    )


def _state_matches(raw: object, required: Mapping[str, object]) -> bool:
    if "$tile" in required:
        return raw == required["$tile"]
    if not isinstance(raw, Mapping):
        return False
    for field, expected in required.items():
        actual = raw.get(field)
        if isinstance(expected, Mapping) and set(expected) == {"at_least"}:
            if not isinstance(actual, int) or actual < int(expected["at_least"]):
                return False
        elif field not in raw or actual != expected:
            return False
    return True


def _outputs_satisfy(produced: Mapping[str, int], required: Mapping[str, int]) -> bool:
    return all(produced.get(item, 0) >= quantity
               for item, quantity in required.items())


def _derive_local_chain(state, opening, required_state, required_outputs):
    """Return one shortest legal local chain using only the real unit rules."""
    abundant = {item: 10_000 for item in (*rules.PRODUCTS, *rules.ANIMALS)}
    template = replace(
        state,
        workers=(WorkerState(0, (0, 0), abundant),),
        shed={}, seeds={crop: 10_000 for crop in rules.CROPS},
    )
    required_outputs = {str(item): max(0, int(quantity))
                        for item, quantity in required_outputs.items()}
    queue = deque([(opening, Counter(), ())])
    visited = set()
    while queue:
        raw, produced, chain = queue.popleft()
        if (_state_matches(raw, required_state)
                and _outputs_satisfy(produced, required_outputs)):
            return chain
        if len(chain) >= 10:
            continue
        capped = tuple(sorted(
            (item, min(quantity, required_outputs.get(item, 0)))
            for item, quantity in produced.items() if item in required_outputs
        ))
        signature = (repr(raw), capped)
        if signature in visited:
            continue
        visited.add(signature)
        for action in _actions():
            after, delta = _unit_effect(template, raw, action)
            if after == raw and not delta:
                continue
            following = produced.copy()
            following.update({item: quantity for item, quantity in delta.items()
                              if quantity > 0})
            queue.append((after, following, (*chain, (action, delta))))
    raise PlanningFailure("Plan outcome has no legal local action chain")


def _deadline(state, project, horizon_end):
    relevant = tuple(int(value) for value in project.time.deadlines
                     if state.step <= int(value) <= horizon_end)
    return min(relevant, default=horizon_end)


def _expand_plan(state, plan):
    """Derive fixed-position primitive jobs from canonical Plan outcomes."""
    horizon_end = min((state.day + 1) * rules.TURNS_PER_DAY - 1,
                      rules.TERMINAL_ACTION_STEP)
    projects, land, placements = [], [], {}
    identifiers = set()
    chain_cache = {}
    commitments = (*plan.obligations, *plan.selected, *plan.support)
    planned_land = {
        str(commitment.metadata["quadrant"])
        for commitment in commitments
        if commitment.kind == "LAND"
        and str(commitment.metadata["quadrant"])
        not in state.unlocked_quadrants
    }
    for commitment in commitments:
        if commitment.identifier in identifiers:
            raise PlanningFailure(f"duplicate Plan commitment {commitment.identifier!r}")
        identifiers.add(commitment.identifier)
        if commitment.kind == "LAND":
            quadrant = str(commitment.metadata["quadrant"])
            if quadrant not in state.unlocked_quadrants:
                land.append(quadrant)
            continue
        if not commitment.required_state and not commitment.required_outputs:
            continue
        if commitment.target is None:
            raise PlanningFailure(
                f"Plan commitment {commitment.identifier!r} has no fixed placement"
            )
        position = tuple(commitment.target)
        placements[commitment.identifier] = position
        opening = state.tile_at(position).raw
        if (
            opening == "LOCKED"
            and rules.quadrant(position, state.board_size) in planned_land
        ):
            # BUY_LAND changes every tile in that quadrant to empty.  Local
            # action-chain derivation therefore starts from the deterministic
            # post-purchase tile state; route timing below still forbids use
            # until the Plan-specified purchase turn has completed.
            opening = None
        chain_key = (
            repr(opening), repr(dict(commitment.required_state)),
            tuple(sorted(commitment.required_outputs.items())),
        )
        if chain_key not in chain_cache:
            chain_cache[chain_key] = _derive_local_chain(
                state, opening, commitment.required_state,
                commitment.required_outputs,
            )
        chain = chain_cache[chain_key]
        raw, events = opening, []
        deadline = _deadline(state, commitment, horizon_end)
        abundant = replace(
            state,
            workers=(WorkerState(0, (0, 0), {
                item: 10_000 for item in (*rules.PRODUCTS, *rules.ANIMALS)
            }),),
            shed={}, seeds={crop: 10_000 for crop in rules.CROPS},
        )
        for index, (action, delta) in enumerate(chain):
            identifier = f"{commitment.identifier}:{index}:{action[0]}"
            events.append(_Event(identifier, commitment.identifier,
                                 tuple(action), dict(delta), position, deadline))
            raw, _ = _unit_effect(abundant, raw, tuple(action))
        if events:
            projects.append(_Project(commitment.identifier, position, opening,
                                     raw, tuple(events)))
    return tuple(projects), tuple(land), placements


class _CompactModel:
    """Fixed jobs + sparse resources + multi-worker route order."""

    def __init__(self, state, plan, projects, land, placements, workforce):
        from ortools.sat.python import cp_model
        self.cp = cp_model
        self.model = cp_model.CpModel()
        self.state, self.plan = state, plan
        self._read_window_market()
        self.projects, self.land = projects, land
        self.placements, self.W = placements, workforce
        self.H = min(state.turns_left, state.turns_left_today)
        self.events = {event.identifier: event
                       for project in projects for event in project.events}
        self.nodes = []
        self.node_time, self.node_active = {}, {}
        self.node_assign, self.node_x, self.node_y = {}, {}, {}
        self.purchase_quantity = {}
        self.purchase_sources = defaultdict(list)
        self.hire_time, self.land_time = {}, {}
        self._producer_uses = defaultdict(list)
        self._carry_uses = defaultdict(list)
        self._opening_uses = defaultdict(list)
        self._travel_terms = []
        self._build_services()
        self._build_sparse_resources()
        self._build_window_deliveries()
        self._add_resource_capacities()
        self._build_market()
        self._add_window_purchase_dependencies()
        self._build_routes()
        self._build_tile_dependencies()
        self._build_objective()

    def _read_window_market(self):
        """Index only Plan-provided purchases and their fixed availability."""

        quantities = Counter()
        turns = defaultdict(list)
        hire_turns, land_turns = [], []
        for window in self.plan.economic_windows:
            offset = max(0, int(window.end_turn) - self.state.hour)
            for order in window.market_orders:
                if not order:
                    continue
                operation = str(order[0])
                quantity = int(order[2]) if len(order) > 2 else 1
                if operation == "BUY_SEED":
                    item = f"{order[1]}_SEED"
                elif operation in ("BUY_ANIMAL", "BUY_PRODUCT"):
                    item = str(order[1])
                elif operation == "HIRE":
                    hire_turns.extend((offset,) * quantity)
                    continue
                elif operation == "BUY_LAND":
                    land_turns.extend((offset,) * quantity)
                    continue
                else:
                    continue
                quantities[item] += quantity
                turns[item].extend((offset,) * quantity)
        self.window_purchase_quantity = dict(quantities)
        # Requiring use after the latest same-item purchase is conservative and
        # prevents a later window's quantity being credited early.
        self.window_purchase_turn = {
            item: max(values) for item, values in turns.items()
        }
        self.window_hire_turns = tuple(sorted(hire_turns))
        self.window_land_turns = tuple(sorted(land_turns))

    def _bool_and(self, name, literals):
        literals = tuple(literals)
        value = self.model.new_bool_var(name)
        for literal in literals:
            self.model.add(value <= literal)
        self.model.add(value >= sum(literals) - len(literals) + 1)
        return value

    def _rank(self, event):
        node = f"service:{event}"
        worker = sum(index * self.node_assign[node, index]
                     for index in range(self.W))
        return self.node_time[node] * self.W + worker

    def _service_node(self, event):
        identifier = f"service:{event.identifier}"
        node = _Node(identifier, "service", event.position, event=event.identifier)
        self.nodes.append(node)
        self.node_active[identifier] = self.model.new_constant(1)
        self.node_time[identifier] = self.model.new_int_var(
            0, self.H - 1, f"time:{identifier}")
        assignments = []
        for worker in range(self.W):
            assignment = self.model.new_bool_var(
                f"assign:{event.identifier}:{worker}")
            self.node_assign[identifier, worker] = assignment
            assignments.append(assignment)
        self.model.add_exactly_one(assignments)
        self.node_x[identifier] = self.model.new_constant(event.position[0])
        self.node_y[identifier] = self.model.new_constant(event.position[1])
        self.model.add(self.node_time[identifier] <= max(
            0, min(self.H - 1, event.deadline - self.state.step)))

    def _build_services(self):
        for project in self.projects:
            for event in project.events:
                self._service_node(event)
            for before, after in zip(project.events, project.events[1:]):
                self.model.add(self._rank(before.identifier)
                               < self._rank(after.identifier))

    def _shed_location(self, identifier):
        positions = rules.shed_access(self.state.board_size)
        x = self.model.new_int_var(0, self.state.board_size - 1,
                                   f"x:{identifier}")
        y = self.model.new_int_var(0, self.state.board_size - 1,
                                   f"y:{identifier}")
        self.model.add_allowed_assignments([x, y], positions)
        return x, y

    def _support_node(self, identifier, kind, active, item, quantity, related_event):
        node = _Node(identifier, kind, event=related_event,
                     item=item, quantity=quantity)
        self.nodes.append(node)
        self.node_active[identifier] = active
        self.node_time[identifier] = self.model.new_int_var(
            0, self.H - 1, f"time:{identifier}")
        self.node_x[identifier], self.node_y[identifier] = self._shed_location(identifier)
        assignments = []
        service = f"service:{related_event}"
        for worker in range(self.W):
            assignment = self._bool_and(
                f"assign:{identifier}:{worker}",
                (active, self.node_assign[service, worker]))
            self.node_assign[identifier, worker] = assignment
            assignments.append(assignment)
        self.model.add(sum(assignments) == active)
        return node

    def _fixed_worker_drop(self, identifier, active, item, worker, deadline):
        node = _Node(identifier, "drop", item=item)
        self.nodes.append(node)
        self.node_active[identifier] = active
        self.node_time[identifier] = self.model.new_int_var(
            0, max(0, deadline), f"time:{identifier}")
        self.node_x[identifier], self.node_y[identifier] = self._shed_location(identifier)
        for candidate in range(self.W):
            self.node_assign[identifier, candidate] = (
                active if candidate == worker else self.model.new_constant(0))
        return node

    def _producer_drop(self, producer, item, active, *, suffix="", deadline=None):
        identifier = f"drop:{producer}:{item}{suffix}"
        node = self._support_node(identifier, "drop", active, item, 0, producer)
        node = replace(node, producer=producer)
        self.nodes[-1] = node
        self.model.add(self.node_time[identifier]
                       > self.node_time[f"service:{producer}"]).only_enforce_if(active)
        if deadline is not None:
            self.model.add(self.node_time[identifier] <= deadline).only_enforce_if(active)
        return node

    def _event_deltas(self):
        return {event.identifier: Counter(event.delta)
                for event in self.events.values()}

    def _build_sparse_resources(self):
        delta = self._event_deltas()
        producers, consumers, seed_needs = defaultdict(list), [], Counter()
        seed_consumers = defaultdict(list)
        for event, values in delta.items():
            for item, quantity in values.items():
                if quantity < 0:
                    if item.endswith("_SEED"):
                        seed_needs[item] += -quantity
                        seed_consumers[item].append(event)
                    else:
                        consumers.append((event, item, -quantity))
                elif quantity > 0 and not item.endswith("_SEED"):
                    producers[item].append((event, quantity))
        self._producers = producers
        opening_carry = {(worker.index, item): quantity
                         for worker in self.state.workers
                         for item, quantity in worker.inventory.items()
                         if quantity > 0}
        for item in tuple(seed_needs):
            seed_needs[item] = max(0, seed_needs[item]
                - self.state.seeds.get(item.removesuffix("_SEED"), 0))

        transfer_uses = defaultdict(list)
        for event, item, quantity in consumers:
            sources = []
            for worker in range(len(self.state.workers)):
                if opening_carry.get((worker, item), 0) < quantity:
                    continue
                literal = self.model.new_bool_var(
                    f"source:carry:{worker}:{item}:{event}")
                self.model.add(literal <= self.node_assign[f"service:{event}", worker])
                self._carry_uses[worker, item].append((quantity, literal))
                sources.append(literal)
            for producer, available in producers.get(item, ()):
                if producer == event or available < quantity:
                    continue
                per_worker = []
                for worker in range(self.W):
                    literal = self.model.new_bool_var(
                        f"source:direct:{producer}:{event}:{worker}")
                    self.model.add(literal <= self.node_assign[f"service:{producer}", worker])
                    self.model.add(literal <= self.node_assign[f"service:{event}", worker])
                    self.model.add(self._rank(producer) < self._rank(event)).only_enforce_if(literal)
                    per_worker.append(literal)
                direct = self.model.new_bool_var(f"source:direct:{producer}:{event}")
                self.model.add(sum(per_worker) == direct)
                self._producer_uses[producer, item].append((quantity, direct))
                sources.append(direct)
            opening = self.model.new_bool_var(f"source:opening-shed:{item}:{event}")
            self._opening_uses[item].append((quantity, opening))
            sources.append(opening)
            purchase = None
            if item in rules.ANIMALS or item in ("WHEAT", "FERTILIZER"):
                purchase = self.model.new_bool_var(f"source:purchase:{item}:{event}")
                self.purchase_sources[item].append((event, quantity, purchase))
                sources.append(purchase)
            transfers = []
            for producer, available in producers.get(item, ()):
                if producer == event or available < quantity:
                    continue
                literal = self.model.new_bool_var(
                    f"source:transfer:{producer}:{event}:{item}")
                self._producer_uses[producer, item].append((quantity, literal))
                transfer_uses[producer, item].append((event, literal))
                transfers.append(literal)
                sources.append(literal)
            self.model.add_exactly_one(sources)
            pickup_active = self.model.new_bool_var(f"pickup-active:{event}:{item}")
            self.model.add(pickup_active == opening
                           + (purchase if purchase is not None else 0)
                           + sum(transfers))
            pickup = self._support_node(f"pickup:{event}:{item}", "pickup",
                                        pickup_active, item, quantity, event)
            self.model.add(self.node_time[pickup.identifier]
                           < self.node_time[f"service:{event}"]).only_enforce_if(pickup_active)
            if purchase is not None:
                purchase_turn = self.window_purchase_turn.get(item, 0)
                self.model.add(
                    self.node_time[pickup.identifier] > purchase_turn
                ).only_enforce_if(purchase)

        for (producer, item), transfers in transfer_uses.items():
            active = self.model.new_bool_var(f"drop-active:{producer}:{item}")
            for _, literal in transfers:
                self.model.add(active >= literal)
            self.model.add(active <= sum(literal for _, literal in transfers))
            drop = self._producer_drop(producer, item, active)
            for event, literal in transfers:
                self.model.add(self.node_time[drop.identifier]
                               < self.node_time[f"pickup:{event}:{item}"
                               ]).only_enforce_if(literal)

        self._opening_carry = opening_carry

        for item in sorted(set(self.purchase_sources) | set(seed_needs)):
            upper = seed_needs[item] + sum(
                quantity for _, quantity, _ in self.purchase_sources[item])
            quantity = self.model.new_int_var(0, upper,
                                               f"purchase-quantity:{item}")
            self.model.add(quantity == seed_needs[item] + sum(
                amount * literal for _, amount, literal
                in self.purchase_sources[item]))
            if item in self.window_purchase_quantity:
                self.model.add(
                    quantity <= self.window_purchase_quantity[item]
                )
            if item.endswith("_SEED") and seed_needs[item] > 0:
                purchase_turn = self.window_purchase_turn.get(item, 0)
                for event in seed_consumers[item]:
                    self.model.add(
                        self.node_time[f"service:{event}"] > purchase_turn
                    )
            self.purchase_quantity[item] = quantity

    def _add_resource_capacities(self):
        for key, uses in self._carry_uses.items():
            self.model.add(sum(q * lit for q, lit in uses)
                           <= self._opening_carry.get(key, 0))
        for item, uses in self._opening_uses.items():
            self.model.add(sum(q * lit for q, lit in uses)
                           <= self.state.shed.get(item, 0))
        for (producer, item), uses in self._producer_uses.items():
            self.model.add(sum(q * lit for q, lit in uses)
                           <= dict(self._producers[item]).get(producer, 0))

    def _window_requirements(self):
        requirements = defaultdict(int)
        for window in self.plan.economic_windows:
            for item, quantity in window.required_shed.items():
                requirements[item] = max(requirements[item], int(quantity))
            for order in window.market_orders:
                if order and order[0] == "SELL":
                    requirements[str(order[1])] = max(
                        requirements[str(order[1])],
                        int(order[2]) if len(order) > 2 else 1)
        return requirements

    def _build_window_deliveries(self):
        requirements = self._window_requirements()
        if not requirements:
            return
        for item, required in requirements.items():
            matching = tuple(window for window in self.plan.economic_windows
                if item in window.required_shed or any(
                    order and order[0] == "SELL" and str(order[1]) == item
                    for order in window.market_orders))
            deadline = max(0, min(window.end_turn for window in matching)
                           - self.state.hour)
            shortage = max(0, required - self.state.shed.get(item, 0))
            if not shortage:
                continue
            candidates = []
            for worker in self.state.workers:
                available = worker.inventory.get(item, 0)
                if not available:
                    continue
                active = self.model.new_bool_var(f"window-carry:{worker.index}:{item}")
                self._carry_uses[worker.index, item].append(
                    (min(shortage, available), active))
                self._fixed_worker_drop(
                    f"window-drop:carry:{worker.index}:{item}", active,
                    item, worker.index, deadline)
                candidates.append((available, active))
            for producer, available in self._producers.get(item, ()):
                active = self.model.new_bool_var(f"window-producer:{producer}:{item}")
                self._producer_uses[producer, item].append(
                    (min(shortage, available), active))
                self._producer_drop(producer, item, active,
                                    suffix=":window", deadline=deadline)
                candidates.append((available, active))
            if not candidates:
                self.model.add(0 == 1)
            else:
                self.model.add(sum(q * lit for q, lit in candidates) >= shortage)

    def _build_market(self):
        existing = len(self.state.workers)
        new_workers = tuple(range(existing, self.W))
        if self.window_hire_turns:
            if len(self.window_hire_turns) != len(new_workers):
                self.model.add(0 == 1)
            self.hire_time = {
                worker: self.window_hire_turns[index]
                for index, worker in enumerate(new_workers)
                if index < len(self.window_hire_turns)
            }
            self.ordinary_hire_workers = ()
        else:
            self.hire_time = {worker: 0 for worker in new_workers}
            self.ordinary_hire_workers = new_workers
        self.land_time = {
            quadrant: (self.window_land_turns[index]
                       if index < len(self.window_land_turns) else 0)
            for index, quadrant in enumerate(self.land)
        }
        self.ordinary_land_quadrants = tuple(
            quadrant for quadrant in self.land
            if self.land_time[quadrant] == 0 and not self.window_land_turns
        )
        services = [node for node in self.nodes if node.kind == "service"]
        for worker in range(existing, self.W):
            self.model.add(sum(self.node_assign[node.identifier, worker]
                               for node in services) >= 1)
        for node in self.nodes:
            for worker in range(existing, self.W):
                self.model.add(
                    self.node_time[node.identifier] > self.hire_time.get(worker, 0)
                ).only_enforce_if(self.node_assign[node.identifier, worker])
        planned_land = set(self.land) | set(self.state.unlocked_quadrants)
        for event in self.events.values():
            quadrant = rules.quadrant(event.position, self.state.board_size)
            if quadrant not in self.state.unlocked_quadrants:
                if quadrant not in planned_land:
                    self.model.add(0 == 1)
                else:
                    self.model.add(
                        self.node_time[f"service:{event.identifier}"]
                        > self.land_time.get(quadrant, 0)
                    )
        ordinary_purchase_items = tuple(
            item for item in self.purchase_quantity
            if item not in self.window_purchase_quantity
        )
        ordinary = (len(self.ordinary_hire_workers)
                    + len(self.ordinary_land_quadrants)
                    + len(ordinary_purchase_items))
        window_zero = sum(len(window.market_orders)
            for window in self.plan.economic_windows
            if window.start_turn <= self.state.hour <= window.end_turn)
        if ordinary + window_zero > rules.MAX_MARKET_ORDERS:
            self.model.add(0 == 1)

    def _add_window_purchase_dependencies(self):
        """Apply the narrow D5-D10 purchase-before-use timing contract."""

        animal_place_deadline = max(
            0, min(self.H - 1, 19 - self.state.hour)
        )
        crop_deadline = max(0, min(self.H - 1, 20 - self.state.hour))
        for project in self.projects:
            ending = project.ending if isinstance(project.ending, Mapping) else {}
            animal = str(ending.get("animal", ""))
            if animal in self.window_purchase_turn:
                purchase_turn = self.window_purchase_turn[animal]
                for event in project.events:
                    node = f"service:{event.identifier}"
                    operation = event.action[0]
                    if operation in ("BUILD_COOP", "BUILD_PASTURE"):
                        self.model.add(
                            self.node_time[node] <= max(0, purchase_turn - 1)
                        )
                    elif operation == "PLACE":
                        self.model.add(self.node_time[node] > purchase_turn)
                        self.model.add(
                            self.node_time[node] <= animal_place_deadline
                        )
            crop = str(ending.get("crop", ""))
            seed = f"{crop}_SEED"
            if seed in self.window_purchase_turn:
                for event in project.events:
                    if event.action[0] in ("PLANT", "WATER"):
                        self.model.add(
                            self.node_time[f"service:{event.identifier}"]
                            <= crop_deadline
                        )

    def _distance(self, left, right):
        if left.position is not None and right.position is not None:
            return rules.manhattan(left.position, right.position)
        dx = self.model.new_int_var(0, self.state.board_size - 1,
                                    f"dx:{left.identifier}:{right.identifier}")
        dy = self.model.new_int_var(0, self.state.board_size - 1,
                                    f"dy:{left.identifier}:{right.identifier}")
        self.model.add_abs_equality(
            dx, self.node_x[left.identifier] - self.node_x[right.identifier])
        self.model.add_abs_equality(
            dy, self.node_y[left.identifier] - self.node_y[right.identifier])
        return dx + dy

    def _travel_on_arc(self, name, arc, distance):
        if isinstance(distance, int):
            self._travel_terms.append(distance * arc)
            return
        value = self.model.new_int_var(0, 2 * self.state.board_size,
                                       f"travel:{name}")
        self.model.add(value == distance).only_enforce_if(arc)
        self.model.add(value == 0).only_enforce_if(arc.Not())
        self._travel_terms.append(value)

    def _build_routes(self):
        existing = len(self.state.workers)
        for worker in range(self.W):
            arcs = []
            empty = self.model.new_bool_var(f"route-empty:{worker}")
            arcs.append((0, 0, empty))
            for index, node in enumerate(self.nodes, start=1):
                assigned = self.node_assign[node.identifier, worker]
                arcs.append((index, index, assigned.Not()))
                first = self.model.new_bool_var(f"route-start:{worker}:{node.identifier}")
                last = self.model.new_bool_var(f"route-end:{worker}:{node.identifier}")
                arcs.extend(((0, index, first), (index, 0, last)))
                self.model.add(first <= assigned)
                self.model.add(last <= assigned)
                if worker < existing:
                    origin = self.state.workers[worker].position
                    if node.position is not None:
                        distance = rules.manhattan(origin, node.position)
                    else:
                        dx = self.model.new_int_var(0, self.state.board_size - 1,
                            f"start-dx:{worker}:{node.identifier}")
                        dy = self.model.new_int_var(0, self.state.board_size - 1,
                            f"start-dy:{worker}:{node.identifier}")
                        self.model.add_abs_equality(dx, self.node_x[node.identifier] - origin[0])
                        self.model.add_abs_equality(dy, self.node_y[node.identifier] - origin[1])
                        distance = dx + dy
                    self.model.add(self.node_time[node.identifier] >= distance).only_enforce_if(first)
                    self._travel_on_arc(f"start:{worker}:{node.identifier}", first, distance)
                else:
                    distance = self.model.new_int_var(0, 2 * self.state.board_size,
                        f"spawn-distance:{worker}:{node.identifier}")
                    self.model.add_allowed_assignments(
                        [self.node_x[node.identifier], self.node_y[node.identifier], distance],
                        [(x, y, rules.distance_to_shed((x, y), self.state.board_size))
                         for x in range(self.state.board_size)
                         for y in range(self.state.board_size)])
                    self.model.add(
                        self.node_time[node.identifier]
                        >= self.hire_time.get(worker, 0) + 1 + distance
                    ).only_enforce_if(first)
                    self._travel_on_arc(f"spawn:{worker}:{node.identifier}", first, distance)
            for left_index, left in enumerate(self.nodes, start=1):
                for right_index, right in enumerate(self.nodes, start=1):
                    if left_index == right_index:
                        continue
                    arc = self.model.new_bool_var(
                        f"route-arc:{worker}:{left.identifier}:{right.identifier}")
                    arcs.append((left_index, right_index, arc))
                    distance = self._distance(left, right)
                    self.model.add(self.node_time[right.identifier]
                        >= self.node_time[left.identifier] + 1 + distance).only_enforce_if(arc)
                    self._travel_on_arc(f"{worker}:{left.identifier}:{right.identifier}",
                                        arc, distance)
            self.model.add_circuit(arcs)

    def _build_tile_dependencies(self):
        for index, left in enumerate(self.projects):
            for right in self.projects[index + 1:]:
                if left.position != right.position:
                    continue
                if left.ending == right.opening and left.opening != left.ending:
                    self.model.add(self._rank(left.events[-1].identifier)
                                   < self._rank(right.events[0].identifier))
                elif right.ending == left.opening and right.opening != right.ending:
                    self.model.add(self._rank(right.events[-1].identifier)
                                   < self._rank(left.events[0].identifier))

    def _build_objective(self):
        movement = sum(self._travel_terms)
        logistics = sum(self.node_active[node.identifier]
                        for node in self.nodes if node.kind in ("pickup", "drop"))
        purchases = sum(self.purchase_quantity.values())
        makespan = self.model.new_int_var(0, self.H - 1, "makespan")
        self.model.add_max_equality(makespan,
            [self.node_time[node.identifier] for node in self.nodes]
            or [self.model.new_constant(0)])
        # Bounded lexicographic encoding of transparent execution metrics.
        bound = self.H * self.W + 1
        total_actions = movement + logistics + len(self.events)
        objective = purchases
        objective = objective * bound + total_actions
        objective = objective * bound + movement
        objective = objective * bound + logistics
        objective = objective * (self.H + 1) + makespan
        self.model.minimize(objective)

    def decode(self, solver):
        schedule = defaultdict(list)
        for node in self.nodes:
            if not solver.boolean_value(self.node_active[node.identifier]):
                continue
            worker = next(worker for worker in range(self.W)
                if solver.boolean_value(self.node_assign[node.identifier, worker]))
            schedule[worker].append((solver.value(self.node_time[node.identifier]), node))
        for route in schedule.values():
            route.sort(key=lambda entry: (entry[0], entry[1].identifier))
        positions = {node.identifier: (
                solver.value(self.node_x[node.identifier]),
                solver.value(self.node_y[node.identifier]))
            for node in self.nodes
            if solver.boolean_value(self.node_active[node.identifier])}
        purchases = {item: solver.value(quantity)
                     for item, quantity in self.purchase_quantity.items()
                     if solver.value(quantity) > 0
                     and item not in self.window_purchase_quantity}
        return schedule, positions, purchases

    def exclude(self, solver):
        equalities = []
        for node in self.nodes:
            if not solver.boolean_value(self.node_active[node.identifier]):
                continue
            value = solver.value(self.node_time[node.identifier])
            equality = self.model.new_bool_var(f"nogood-time:{node.identifier}:{value}")
            self.model.add(self.node_time[node.identifier] == value).only_enforce_if(equality)
            self.model.add(self.node_time[node.identifier] != value).only_enforce_if(equality.Not())
            equalities.extend((equality, next(
                self.node_assign[node.identifier, worker] for worker in range(self.W)
                if solver.boolean_value(self.node_assign[node.identifier, worker]))))
        if equalities:
            self.model.add_bool_or([literal.Not() for literal in equalities])


def _move(origin, target):
    if origin[0] != target[0]:
        return ("EAST",) if origin[0] < target[0] else ("WEST",)
    if origin[1] != target[1]:
        return ("SOUTH",) if origin[1] < target[1] else ("NORTH",)
    return ("PASS",)


def _window_ready(after_units, window):
    return (after_units.money >= window.minimum_cash
        and all(after_units.shed.get(item, 0) >= quantity
                for item, quantity in window.required_shed.items())
        and all(not order or order[0] != "SELL"
                or after_units.shed.get(str(order[1]), 0)
                >= (int(order[2]) if len(order) > 2 else 1)
                for order in window.market_orders))


def _materialize(model, decoded):
    schedule, node_positions, purchases = decoded
    routes = {worker: list(route) for worker, route in schedule.items()}
    current, turns = model.state, []
    completed_windows = set()
    windows = tuple(enumerate(model.plan.economic_windows))
    for offset in range(model.H):
        actions = [("PASS",) for _ in current.workers]
        for worker in current.workers:
            route = routes.get(worker.index, ())
            if route and route[0][0] < offset:
                return None, {"reason": "missed-event-time",
                    "event": route[0][1].identifier, "turn": offset}
            if not route:
                continue
            event_time, node = route[0]
            target = node_positions[node.identifier]
            if event_time == offset:
                if worker.position != target:
                    return None, {"reason": "route-arrival-mismatch",
                        "event": node.identifier, "turn": offset,
                        "worker": worker.index}
                if node.kind == "service":
                    actions[worker.index] = model.events[node.event].action
                elif node.kind == "pickup":
                    actions[worker.index] = ("PICKUP", node.item, node.quantity)
                else:
                    actions[worker.index] = ("DROP",)
                route.pop(0)
            else:
                actions[worker.index] = _move(worker.position, target)
        after_units = rules.advance_owned(current, tuple(actions), unit_only=True)
        window_orders = []
        for index, window in windows:
            if index not in completed_windows and window.start_turn <= current.hour <= window.end_turn \
                    and _window_ready(after_units, window):
                window_orders.extend(window.market_orders)
                completed_windows.add(index)
        ordinary_orders = []
        if offset == 0:
            ordinary_orders.extend(("HIRE",)
                                   for _ in model.ordinary_hire_workers)
            ordinary_orders.extend(("BUY_LAND",)
                                   for _ in model.ordinary_land_quadrants)
            for item, quantity in purchases.items():
                if item.endswith("_SEED"):
                    ordinary_orders.append(
                        ("BUY_SEED", item.removesuffix("_SEED"), quantity))
                elif item in rules.ANIMALS:
                    ordinary_orders.append(("BUY_ANIMAL", item, quantity))
                else:
                    ordinary_orders.append(("BUY_PRODUCT", item, quantity))
        if offset == 0 and ordinary_orders:
            # OPENING_FINANCE is the sole mixed case: Plan-owned opening sales
            # must fund solver-selected hires, while Plan-owned emergency buys
            # remain after those hires.
            split = 0
            while (split < len(window_orders)
                   and window_orders[split]
                   and window_orders[split][0] == "SELL"):
                split += 1
            required_orders = [
                *window_orders[:split],
                *ordinary_orders,
                *window_orders[split:],
            ]
        else:
            required_orders = [*ordinary_orders, *window_orders]
        try:
            orders = realization_orders(current, tuple(actions),
                                         required_sequence=tuple(required_orders))
            following = rules.advance_owned(current, tuple(actions), orders)
        except Exception as error:
            return None, {"reason": "materialization-error", "turn": offset,
                          "error": repr(error)}
        turns.append(TurnDecision(tuple(actions), orders))
        current = following
    missing = [index for index, _ in windows if index not in completed_windows]
    if missing:
        return None, {"reason": "economic-window-unmet", "windows": missing}
    realization = Realization(dict(model.placements), tuple(turns))
    try:
        ending = execute_realization(model.state, model.plan, realization)
    except Exception as error:
        return None, {"reason": "exact-execution", "error": repr(error)}
    return (realization, ending), None


def _maximum_workforce(state, plan, event_count):
    extra = max((window.extra_hands_allowed for window in plan.economic_windows),
                default=0)
    if plan.max_hands is not None:
        return max(len(state.workers), int(plan.max_hands) + extra + 1)
    money, hires, maximum = state.money, state.hires_today, len(state.workers)
    target = event_count + extra
    while maximum < target:
        cost = rules.fibonacci_hire_cost(hires)
        if cost > money:
            break
        money -= cost
        hires += 1
        maximum += 1
    return maximum


def _solve_fixed(state, plan, projects, land, placements, workforce):
    begun = perf_counter()
    model = _CompactModel(state, plan, projects, land, placements, workforce)
    failures = []
    for attempt in range(3):
        solver = model.cp.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = attempt
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.solve(model.model)
        if status not in (model.cp.OPTIMAL, model.cp.FEASIBLE):
            return None, {"workforce": workforce,
                "status": solver.status_name(status),
                "seconds": perf_counter() - begun,
                "variables": len(model.model.proto.variables),
                "constraints": len(model.model.proto.constraints),
                "materialization_failures": failures}
        candidate, failure = _materialize(model, model.decode(solver))
        if candidate is not None:
            realization, _ = candidate
            return realization, {"workforce": workforce,
                "status": solver.status_name(status),
                "seconds": perf_counter() - begun,
                "variables": len(model.model.proto.variables),
                "constraints": len(model.model.proto.constraints),
                "exact_attempts": attempt + 1}
        failures.append(failure)
        model.exclude(solver)
    return None, {"workforce": workforce, "status": "SEARCH_EXHAUSTED",
        "seconds": perf_counter() - begun,
        "variables": len(model.model.proto.variables),
        "constraints": len(model.model.proto.constraints),
        "materialization_failures": failures}


def solve_intraday(state: OwnedState, plan) -> Realization:
    """Return a complete exact Realization or raise PlanningFailure."""
    if state.day != plan.day:
        raise PlanningFailure("Plan day does not match OwnedState")
    for window in plan.economic_windows:
        if not (0 <= window.start_turn <= window.end_turn < rules.TURNS_PER_DAY):
            raise PlanningFailure("invalid economic window")
        if window.end_turn < state.hour:
            raise PlanningFailure("economic window has already expired")
    projects, land, placements = _expand_plan(state, plan)
    event_count = sum(len(project.events) for project in projects)
    horizon = min(state.turns_left, state.turns_left_today)
    if not event_count and not land and not plan.economic_windows:
        return Realization(dict(placements), tuple(
            TurnDecision(tuple(("PASS",) for _ in state.workers))
            for _ in range(horizon)))
    if horizon <= 0:
        raise PlanningFailure("no turns remain for the mandatory Daily Plan")
    minimum = max(len(state.workers), (event_count + horizon - 1) // horizon)
    maximum = _maximum_workforce(state, plan, event_count)
    attempts = []
    for workforce in range(minimum, maximum + 1):
        candidate, diagnostics = _solve_fixed(
            state, plan, projects, land, placements, workforce)
        attempts.append(diagnostics)
        if candidate is not None:
            return candidate
    raise PlanningFailure("no complete fixed-job realization found",
                          diagnostics={"workforces": attempts})
