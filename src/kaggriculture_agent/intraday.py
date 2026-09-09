"""Compact event-based intraday optimization.

The only public boundary is OwnedState + Plan -> Realization. The solver never
expands worker/item state over every remaining turn. Time exists only on
mandatory service and sparse logistics/market events; primitive movement is
materialized after an event schedule has been selected.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping

from . import rules
from .execution import execute_realization
from .market import realization_orders
from .realization import PlanningFailure, Realization, TurnDecision
from .state import OwnedState, TileState, WorkerState
from .valuation import end_value


@dataclass(frozen=True)
class _Requirement:
    identifier: str
    project: str
    operation: tuple[object, ...] | None
    effect: Mapping[str, object]
    minimum_output: Mapping[str, int]
    deadline: int


@dataclass(frozen=True)
class _Event:
    requirement: str
    action: tuple[object, ...]
    delta: Mapping[str, int]


@dataclass(frozen=True)
class _Project:
    identifier: str
    positions: tuple[tuple[int, int], ...]
    opening: object
    existing: bool
    requirements: tuple[_Requirement, ...]
    chains: tuple[tuple[_Event, ...], ...]
    placement_keys: tuple[str, ...]


@dataclass(frozen=True)
class _Node:
    identifier: str
    kind: str
    project: str | None = None
    requirement: str | None = None
    item: str | None = None
    quantity: int = 0
    producer: str | None = None


def _stock(state):
    result = Counter(state.shed)
    for worker in state.workers:
        result.update(worker.inventory)
    result.update({f"{crop}_SEED": quantity
                   for crop, quantity in state.seeds.items()})
    return result


def _unit_effect(state, action):
    after = rules.advance_owned(state, (tuple(action),), unit_only=True)
    before_tile = state.tile_at((0, 0)).raw
    after_tile = after.tile_at((0, 0)).raw
    before_stock, after_stock = _stock(state), _stock(after)
    delta = {
        item: after_stock[item] - before_stock[item]
        for item in sorted(before_stock.keys() | after_stock.keys())
        if before_stock[item] != after_stock[item]
    }
    return before_tile, after_tile, delta


def _matches(requirement, before, after, delta, action):
    if before == after:
        return False
    if requirement.operation is not None and tuple(action) != requirement.operation:
        return False
    fields = after if isinstance(after, Mapping) else {"$tile": after}
    for name, condition in requirement.effect.items():
        if (name in fields) != bool(condition["after_present"]):
            return False
        if not condition["after_present"]:
            continue
        expected, actual = condition.get("after"), fields[name]
        if name in ("yield_units", "fertilized_until_day") and isinstance(expected, int) and expected > 0:
            if not isinstance(actual, int) or actual < expected:
                return False
        elif actual != expected:
            return False
    return all(delta.get(item, 0) >= quantity
               for item, quantity in requirement.minimum_output.items())


def _actions():
    return (
        ("WATER",), ("FERTILIZE",), ("HARVEST",), ("FEED",), ("CARE",),
        ("COLLECT_FERTILIZER",), ("DIG",), ("BUILD_COOP",),
        ("BUILD_PASTURE",),
        *(("PLANT", crop) for crop in rules.CROPS),
        *(("PLACE", animal) for animal in rules.ANIMALS),
    )


def _local_chains(state, opening, requirements):
    """Enumerate complete legal action chains for one physical project."""
    abundant = {item: 10_000 for item in (*rules.PRODUCTS, *rules.ANIMALS)}
    template = replace(
        state,
        workers=(WorkerState(0, (0, 0), abundant),),
        shed={},
        seeds={crop: 10_000 for crop in rules.CROPS},
        tiles=(TileState((0, 0), opening), *state.tiles[1:]),
    )
    cache, complete = {}, []

    def probe(raw, action):
        key = (repr(raw), action)
        if key not in cache:
            local = replace(
                template,
                tiles=(TileState((0, 0), raw), *template.tiles[1:]),
            )
            cache[key] = _unit_effect(local, action)
        return cache[key]

    def visit(raw, remaining, events):
        if not remaining:
            complete.append(events)
            return
        for requirement in remaining:
            alphabet = ((requirement.operation,)
                        if requirement.operation is not None else _actions())
            for action in alphabet:
                before, after, delta = probe(raw, action)
                if _matches(requirement, before, after, delta, action):
                    visit(
                        after,
                        tuple(value for value in remaining
                              if value.identifier != requirement.identifier),
                        (*events, _Event(requirement.identifier,
                                         tuple(action), delta)),
                    )

    visit(opening, requirements, ())
    unique = {}
    for chain in complete:
        signature = tuple((event.requirement, event.action) for event in chain)
        unique[signature] = chain
    return tuple(unique.values())


def _operation(project, kind):
    if kind == "PLANT":
        return ("PLANT", project.metadata["crop"])
    if kind == "PLACE":
        return ("PLACE", project.metadata["animal"])
    if kind == "BUILD":
        return ("BUILD_" + str(project.metadata["structure"]),)
    return (kind,)


def _expand_plan(state, plan):
    """Mechanically derive solver-private events from the fixed Plan."""
    grouped = defaultdict(list)
    domains, openings, existing = {}, {}, {}
    placement_keys = defaultdict(list)
    horizon_end = min((state.day + 1) * rules.TURNS_PER_DAY - 1,
                      rules.TERMINAL_ACTION_STEP)
    aliases = {
        "PICKUP_PLACE": ("PLACE",),
        "HARVEST_TRANSPORT": ("HARVEST",),
        "WATER_HARVEST_TRANSPORT": ("WATER", "HARVEST"),
    }
    land = []
    for project in (*plan.obligations, *plan.selected, *plan.support):
        if project.kind == "LAND":
            quadrant = str(project.metadata["quadrant"])
            if quadrant not in state.unlocked_quadrants:
                land.append(quadrant)
            continue
        if project.kind == "STATE_EFFECT":
            entity = str(project.metadata["entity"])
            positions = ((project.target,) if project.target is not None else
                         tuple(plan.placement_domains.get(project.identifier, ())))
            count = max(1, int(project.metadata.get("demonstrated_count", 1)))
            entries = [
                (
                    project.identifier if count == 1
                    else f"{project.identifier}:copy:{index}",
                    None,
                    project.metadata["required_effect"],
                    {amount.item: amount.quantity
                     for amount in project.physical.outputs},
                    horizon_end,
                )
                for index in range(count)
            ]
        else:
            entity = (f"existing:{project.target[0]}:{project.target[1]}"
                      if project.target is not None
                      else f"new:{project.identifier}")
            positions = ((project.target,) if project.target is not None else
                         tuple(plan.placement_domains.get(project.identifier, ())))
            entries = []
            for index, work in enumerate(project.actions.work):
                if work.day != state.day:
                    continue
                for kind in aliases.get(work.kind, (work.kind,)):
                    entries.append((
                        f"{project.identifier}:{index}:{kind}",
                        _operation(project, kind), {}, {},
                        min(horizon_end, work.deadline_step
                            if work.deadline_step is not None else horizon_end),
                    ))
        positions = tuple(tuple(position) for position in positions)
        placement_keys[entity].append(project.identifier)
        if entity in domains:
            domains[entity] = tuple(position for position in domains[entity]
                                    if position in positions)
        else:
            domains[entity] = positions
            openings[entity] = (
                state.tile_at(positions[0]).raw if positions else None
            )
            existing[entity] = project.target is not None
        if project.target is None and (
            project.kind == "CROP"
            or project.metadata.get("requires_construction")
            or (
                project.kind == "STATE_EFFECT"
                and project.metadata["required_effect"].get(
                    "$tile", {}
                ).get("before_present", False)
                and project.metadata["required_effect"]["$tile"].get(
                    "before"
                ) is None
            )
        ):
            openings[entity] = None
        for identifier, operation, effect, outputs, deadline in entries:
            grouped[entity].append(_Requirement(
                identifier, entity, operation, effect, outputs, deadline
            ))

    projects = []
    for identifier in sorted(grouped):
        requirements = tuple(grouped[identifier])
        if not domains.get(identifier):
            raise PlanningFailure(
                f"Plan project {identifier!r} has no legal placement"
            )
        chains = _local_chains(state, openings[identifier], requirements)
        if not chains:
            raise PlanningFailure(
                f"Plan project {identifier!r} has no complete local action chain"
            )
        projects.append(_Project(
            identifier, domains[identifier], openings[identifier],
            existing[identifier], requirements, chains,
            tuple(placement_keys[identifier]),
        ))
    return tuple(projects), tuple(land)


class _CompactModel:
    """Event, route-order, placement and sparse source-consumer formulation."""

    def __init__(self, state, plan, projects, land, workforce):
        from ortools.sat.python import cp_model

        self.cp = cp_model
        self.model = cp_model.CpModel()
        self.state, self.plan = state, plan
        self.projects, self.land, self.W = projects, land, workforce
        self.H = min(state.turns_left, state.turns_left_today)
        self.project_by_id = {project.identifier: project
                              for project in projects}
        self.requirements = {
            requirement.identifier: requirement
            for project in projects for requirement in project.requirements
        }
        self.requirement_project = {
            requirement.identifier: project.identifier
            for project in projects for requirement in project.requirements
        }
        self.cell, self.project_x, self.project_y = {}, {}, {}
        self.path_choice = {}
        self.nodes = []
        self.node_time, self.node_active = {}, {}
        self.node_assign, self.node_x, self.node_y = {}, {}, {}
        self.purchase_quantity, self.purchase_time = {}, {}
        self.purchase_sources = defaultdict(list)
        self.hire_time, self.land_time = {}, {}
        self._build_placements_and_services()
        self._build_sparse_resources()
        self._build_market()
        self._build_routes()
        self._build_tile_lifecycles()
        self.model.minimize(
            sum(self.node_time[f"service:{identifier}"]
                for identifier in self.requirements)
            + 10 * sum(self.purchase_quantity.values())
            + sum(self.hire_time.values())
            + sum(self.land_time.values())
            + sum(self.purchase_time.values())
        )

    def _bool_and(self, name, literals):
        value = self.model.new_bool_var(name)
        for literal in literals:
            self.model.add(value <= literal)
        self.model.add(value >= sum(literals) - len(literals) + 1)
        return value

    def _location(self, key, positions):
        cells = sorted({y * self.state.board_size + x for x, y in positions})
        cell = self.model.new_int_var_from_domain(
            self.cp.Domain.from_values(cells), f"cell:{key}"
        )
        x = self.model.new_int_var(0, self.state.board_size - 1, f"x:{key}")
        y = self.model.new_int_var(0, self.state.board_size - 1, f"y:{key}")
        self.model.add_allowed_assignments(
            [cell, x, y],
            [(value, value % self.state.board_size,
              value // self.state.board_size) for value in cells],
        )
        return cell, x, y

    def _rank(self, requirement):
        node = f"service:{requirement}"
        worker = sum(index * self.node_assign[node, index]
                     for index in range(self.W))
        return self.node_time[node] * self.W + worker

    def _service_node(self, requirement):
        identifier = f"service:{requirement.identifier}"
        node = _Node(identifier, "service",
                     project=requirement.project,
                     requirement=requirement.identifier)
        self.nodes.append(node)
        self.node_active[identifier] = self.model.new_constant(1)
        self.node_time[identifier] = self.model.new_int_var(
            0, self.H - 1, f"time:{identifier}"
        )
        assignments = []
        for worker in range(self.W):
            assignment = self.model.new_bool_var(
                f"assign:{requirement.identifier}:{worker}"
            )
            self.node_assign[identifier, worker] = assignment
            assignments.append(assignment)
        self.model.add_exactly_one(assignments)
        self.node_x[identifier] = self.project_x[requirement.project]
        self.node_y[identifier] = self.project_y[requirement.project]
        deadline = min(self.H - 1, requirement.deadline - self.state.step)
        self.model.add(self.node_time[identifier] <= max(0, deadline))

    def _build_placements_and_services(self):
        for project in self.projects:
            cell, x, y = self._location(
                f"project:{project.identifier}", project.positions
            )
            self.cell[project.identifier] = cell
            self.project_x[project.identifier] = x
            self.project_y[project.identifier] = y
            choices = tuple(
                self.model.new_bool_var(
                    f"path:{project.identifier}:{index}"
                )
                for index in range(len(project.chains))
            )
            self.model.add_exactly_one(choices)
            self.path_choice[project.identifier] = choices
            for requirement in project.requirements:
                self._service_node(requirement)
            for choice, chain in zip(choices, project.chains):
                for before, after in zip(chain, chain[1:]):
                    self.model.add(
                        self._rank(before.requirement)
                        < self._rank(after.requirement)
                    ).only_enforce_if(choice)

    def _event_deltas(self):
        alternatives = defaultdict(list)
        for project in self.projects:
            for chain in project.chains:
                for event in chain:
                    alternatives[event.requirement].append(Counter(event.delta))
        result = {}
        for identifier, choices in alternatives.items():
            items = set().union(*(choice.keys() for choice in choices))
            result[identifier] = {
                item: (
                    min(choice.get(item, 0) for choice in choices),
                    max(choice.get(item, 0) for choice in choices),
                )
                for item in items
            }
        return result

    def _support_node(self, identifier, kind, active, item, quantity,
                      related_requirement):
        node = _Node(
            identifier, kind,
            project=self.requirement_project[related_requirement],
            requirement=related_requirement,
            item=item, quantity=quantity,
        )
        self.nodes.append(node)
        self.node_active[identifier] = active
        self.node_time[identifier] = self.model.new_int_var(
            0, self.H - 1, f"time:{identifier}"
        )
        _, x, y = self._location(identifier, rules.shed_access())
        self.node_x[identifier], self.node_y[identifier] = x, y
        assignments = []
        service = f"service:{related_requirement}"
        for worker in range(self.W):
            assignment = self._bool_and(
                f"assign:{identifier}:{worker}",
                (active, self.node_assign[service, worker]),
            )
            self.node_assign[identifier, worker] = assignment
            assignments.append(assignment)
        self.model.add(sum(assignments) == active)
        return node

    def _drop_node(self, producer, item, active):
        identifier = f"drop:{producer}:{item}"
        node = _Node(
            identifier, "drop",
            project=self.requirement_project[producer],
            item=item, producer=producer,
        )
        self.nodes.append(node)
        self.node_active[identifier] = active
        self.node_time[identifier] = self.model.new_int_var(
            0, self.H - 1, f"time:{identifier}"
        )
        _, x, y = self._location(identifier, rules.shed_access())
        self.node_x[identifier], self.node_y[identifier] = x, y
        assignments = []
        service = f"service:{producer}"
        for worker in range(self.W):
            assignment = self._bool_and(
                f"assign:{identifier}:{worker}",
                (active, self.node_assign[service, worker]),
            )
            self.node_assign[identifier, worker] = assignment
            assignments.append(assignment)
        self.model.add(sum(assignments) == active)
        self.model.add(
            self.node_time[identifier] > self.node_time[service]
        ).only_enforce_if(active)
        return node

    def _build_sparse_resources(self):
        delta = self._event_deltas()
        consumers = []
        producers = defaultdict(list)
        seed_needs = Counter()
        for requirement, values in delta.items():
            for item, (minimum, maximum) in values.items():
                if minimum < 0:
                    quantity = -minimum
                    if item.endswith("_SEED"):
                        seed_needs[item] += quantity
                    else:
                        consumers.append((requirement, item, quantity))
                if maximum > 0 and not item.endswith("_SEED"):
                    producers[item].append((requirement, maximum))

        opening_carry = {
            (worker.index, item): quantity
            for worker in self.state.workers
            for item, quantity in worker.inventory.items()
            if quantity > 0
        }
        carry_uses, opening_uses = defaultdict(list), defaultdict(list)
        producer_uses, transfer_uses = defaultdict(list), defaultdict(list)
        for item in tuple(seed_needs):
            seed_needs[item] = max(
                0,
                seed_needs[item]
                - self.state.seeds.get(item.removesuffix("_SEED"), 0),
            )

        for requirement, item, quantity in consumers:
            sources = []
            for worker in range(len(self.state.workers)):
                if opening_carry.get((worker, item), 0) < quantity:
                    continue
                literal = self.model.new_bool_var(
                    f"source:carry:{worker}:{item}:{requirement}"
                )
                self.model.add(
                    literal <= self.node_assign[
                        f"service:{requirement}", worker
                    ]
                )
                carry_uses[worker, item].append((quantity, literal))
                sources.append(literal)

            for producer, available in producers.get(item, ()):
                if producer == requirement or available < quantity:
                    continue
                per_worker = []
                for worker in range(self.W):
                    literal = self.model.new_bool_var(
                        f"source:direct:{producer}:{requirement}:{worker}"
                    )
                    self.model.add(
                        literal <= self.node_assign[
                            f"service:{producer}", worker
                        ]
                    )
                    self.model.add(
                        literal <= self.node_assign[
                            f"service:{requirement}", worker
                        ]
                    )
                    self.model.add(
                        self._rank(producer) < self._rank(requirement)
                    ).only_enforce_if(literal)
                    per_worker.append(literal)
                direct = self.model.new_bool_var(
                    f"source:direct:{producer}:{requirement}"
                )
                self.model.add(sum(per_worker) == direct)
                producer_uses[producer, item].append((quantity, direct))
                sources.append(direct)

            opening = self.model.new_bool_var(
                f"source:opening-shed:{item}:{requirement}"
            )
            purchase = self.model.new_bool_var(
                f"source:purchase:{item}:{requirement}"
            )
            opening_uses[item].append((quantity, opening))
            self.purchase_sources[item].append(
                (requirement, quantity, purchase)
            )
            sources.extend((opening, purchase))
            transfers = []
            for producer, available in producers.get(item, ()):
                if producer == requirement or available < quantity:
                    continue
                literal = self.model.new_bool_var(
                    f"source:transfer:{producer}:{requirement}:{item}"
                )
                producer_uses[producer, item].append((quantity, literal))
                transfer_uses[producer, item].append((requirement, literal))
                transfers.append(literal)
                sources.append(literal)
            self.model.add_exactly_one(sources)
            pickup_active = self.model.new_bool_var(
                f"pickup-active:{requirement}:{item}"
            )
            self.model.add(
                pickup_active == opening + purchase + sum(transfers)
            )
            pickup = self._support_node(
                f"pickup:{requirement}:{item}", "pickup",
                pickup_active, item, quantity, requirement,
            )
            self.model.add(
                self.node_time[pickup.identifier]
                < self.node_time[f"service:{requirement}"]
            ).only_enforce_if(pickup_active)

        for key, uses in carry_uses.items():
            self.model.add(
                sum(quantity * literal for quantity, literal in uses)
                <= opening_carry[key]
            )
        for item, uses in opening_uses.items():
            self.model.add(
                sum(quantity * literal for quantity, literal in uses)
                <= self.state.shed.get(item, 0)
            )
        for key, uses in producer_uses.items():
            producer, item = key
            self.model.add(
                sum(quantity * literal for quantity, literal in uses)
                <= dict(producers[item])[producer]
            )

        for (producer, item), transfers in transfer_uses.items():
            active = self.model.new_bool_var(
                f"drop-active:{producer}:{item}"
            )
            for _, literal in transfers:
                self.model.add(active >= literal)
            self.model.add(
                active <= sum(literal for _, literal in transfers)
            )
            drop = self._drop_node(producer, item, active)
            for requirement, literal in transfers:
                self.model.add(
                    self.node_time[drop.identifier]
                    < self.node_time[f"pickup:{requirement}:{item}"]
                ).only_enforce_if(literal)

        for item in sorted(set(self.purchase_sources) | set(seed_needs)):
            upper = seed_needs[item] + sum(
                quantity for _, quantity, _ in self.purchase_sources[item]
            )
            quantity = self.model.new_int_var(
                0, upper, f"purchase-quantity:{item}"
            )
            self.model.add(
                quantity == seed_needs[item] + sum(
                    amount * literal
                    for _, amount, literal in self.purchase_sources[item]
                )
            )
            self.purchase_quantity[item] = quantity
            self.purchase_time[item] = self.model.new_int_var(
                0, max(0, self.H - 2), f"purchase-time:{item}"
            )
            for requirement, _, literal in self.purchase_sources[item]:
                self.model.add(
                    self.purchase_time[item]
                    < self.node_time[f"pickup:{requirement}:{item}"]
                ).only_enforce_if(literal)
            if seed_needs[item]:
                for requirement, values in delta.items():
                    if values.get(item, (0, 0))[0] < 0:
                        self.model.add(
                            self.purchase_time[item]
                            < self.node_time[f"service:{requirement}"]
                        )

    def _build_market(self):
        intervals, demands = [], []
        existing = len(self.state.workers)
        for worker in range(existing, self.W):
            time = self.model.new_int_var(
                0, max(0, self.H - 2), f"hire-time:{worker}"
            )
            self.hire_time[worker] = time
            intervals.append(self.model.new_fixed_size_interval_var(
                time, 1, f"hire-entry:{worker}"
            ))
            demands.append(1)
            if worker > existing:
                self.model.add(time >= self.hire_time[worker - 1])
        first_service = {}
        service_nodes = [
            node for node in self.nodes if node.kind == "service"
        ]
        for worker in range(existing, self.W):
            self.model.add(
                sum(self.node_assign[node.identifier, worker]
                    for node in service_nodes) >= 1
            )
            candidates = []
            for node in service_nodes:
                candidate = self.model.new_int_var(
                    0, 2 * self.H, f"first-candidate:{worker}:{node.identifier}"
                )
                self.model.add(
                    candidate == self.node_time[node.identifier]
                ).only_enforce_if(
                    self.node_assign[node.identifier, worker]
                )
                self.model.add(candidate == 2 * self.H).only_enforce_if(
                    self.node_assign[node.identifier, worker].Not()
                )
                candidates.append(candidate)
            first = self.model.new_int_var(
                0, self.H - 1, f"first-service:{worker}"
            )
            self.model.add_min_equality(first, candidates)
            first_service[worker] = first
            if worker > existing:
                self.model.add(first_service[worker - 1] <= first)
        for index, quadrant in enumerate(self.land):
            time = self.model.new_int_var(
                0, max(0, self.H - 2), f"land-time:{quadrant}"
            )
            self.land_time[quadrant] = time
            intervals.append(self.model.new_fixed_size_interval_var(
                time, 1, f"land-entry:{quadrant}"
            ))
            demands.append(1)
            if index:
                self.model.add(
                    time >= self.land_time[self.land[index - 1]]
                )
        for item, quantity in self.purchase_quantity.items():
            active = self.model.new_bool_var(f"purchase-active:{item}")
            self.model.add(quantity > 0).only_enforce_if(active)
            self.model.add(quantity == 0).only_enforce_if(active.Not())
            intervals.append(
                self.model.new_optional_fixed_size_interval_var(
                    self.purchase_time[item], 1, active,
                    f"purchase-entry:{item}",
                )
            )
            demands.append(1)
        if intervals:
            self.model.add_cumulative(
                intervals, demands, rules.MAX_MARKET_ORDERS
            )

        for node in self.nodes:
            for worker in range(existing, self.W):
                self.model.add(
                    self.node_time[node.identifier] > self.hire_time[worker]
                ).only_enforce_if(
                    self.node_assign[node.identifier, worker]
                )

        hire_cost = rules.hire_expenditure(
            self.state.hires_today, self.W - existing
        )
        land_cost = sum(
            rules.LAND_PRICES[rules.LAND_ORDER.index(quadrant)]
            for quadrant in self.land
        )
        purchase_cost = []
        for item, quantity in self.purchase_quantity.items():
            if item.endswith("_SEED"):
                price = rules.CROPS[item.removesuffix("_SEED")].seed_cost
            elif item in rules.ANIMALS:
                price = rules.ANIMALS[item].cost
            else:
                price = max(1, self.state.market_prices.get(item, 1))
            purchase_cost.append(price * quantity)
        potential_sales = sum(
            quantity * self.state.market_prices.get(item, 0)
            for item, quantity in self.state.shed.items()
            if item in rules.SELLABLE_PRODUCTS
        )
        for values in self._event_deltas().values():
            potential_sales += sum(
                maximum * self.state.market_prices.get(item, 0)
                for item, (_, maximum) in values.items()
                if maximum > 0 and item in rules.SELLABLE_PRODUCTS
            )
        self.model.add(
            hire_cost + land_cost + sum(purchase_cost)
            <= self.state.money + potential_sales
        )

    def _distance(self, left, right):
        key = tuple(sorted((left.identifier, right.identifier)))
        if not hasattr(self, "_distance_cache"):
            self._distance_cache = {}
        if key in self._distance_cache:
            return self._distance_cache[key]
        if (left.kind == right.kind == "service"
                and left.project == right.project):
            return 0
        dx = self.model.new_int_var(
            0, self.state.board_size - 1,
            f"dx:{left.identifier}:{right.identifier}",
        )
        dy = self.model.new_int_var(
            0, self.state.board_size - 1,
            f"dy:{left.identifier}:{right.identifier}",
        )
        self.model.add_abs_equality(
            dx,
            self.node_x[left.identifier] - self.node_x[right.identifier],
        )
        self.model.add_abs_equality(
            dy,
            self.node_y[left.identifier] - self.node_y[right.identifier],
        )
        distance = dx + dy
        self._distance_cache[key] = distance
        return distance

    def _build_routes(self):
        existing = len(self.state.workers)
        for worker in range(self.W):
            arcs = []
            empty = self.model.new_bool_var(f"route-empty:{worker}")
            arcs.append((0, 0, empty))
            for index, node in enumerate(self.nodes, start=1):
                assigned = self.node_assign[node.identifier, worker]
                arcs.append((index, index, assigned.Not()))
                first = self.model.new_bool_var(
                    f"route-start:{worker}:{node.identifier}"
                )
                last = self.model.new_bool_var(
                    f"route-end:{worker}:{node.identifier}"
                )
                arcs.extend(((0, index, first), (index, 0, last)))
                self.model.add(first <= assigned)
                self.model.add(last <= assigned)
                if worker < existing:
                    origin = self.state.workers[worker].position
                    dx = self.model.new_int_var(
                        0, self.state.board_size - 1,
                        f"start-dx:{worker}:{node.identifier}"
                    )
                    dy = self.model.new_int_var(
                        0, self.state.board_size - 1,
                        f"start-dy:{worker}:{node.identifier}"
                    )
                    self.model.add_abs_equality(
                        dx, self.node_x[node.identifier] - origin[0]
                    )
                    self.model.add_abs_equality(
                        dy, self.node_y[node.identifier] - origin[1]
                    )
                    self.model.add(
                        self.node_time[node.identifier] >= dx + dy
                    ).only_enforce_if(first)
                else:
                    distance = self.model.new_int_var(
                        0, 18,
                        f"spawn-distance:{worker}:{node.identifier}"
                    )
                    self.model.add_allowed_assignments(
                        [
                            self.node_x[node.identifier],
                            self.node_y[node.identifier],
                            distance,
                        ],
                        [
                            (x, y, rules.distance_to_shed((x, y)))
                            for x in range(self.state.board_size)
                            for y in range(self.state.board_size)
                        ],
                    )
                    self.model.add(
                        self.node_time[node.identifier]
                        >= self.hire_time[worker] + 1 + distance
                    ).only_enforce_if(first)
            for left_index, left in enumerate(self.nodes, start=1):
                for right_index, right in enumerate(self.nodes, start=1):
                    if left_index == right_index:
                        continue
                    arc = self.model.new_bool_var(
                        f"route-arc:{worker}:{left.identifier}:{right.identifier}"
                    )
                    arcs.append((left_index, right_index, arc))
                    self.model.add(
                        self.node_time[right.identifier]
                        >= self.node_time[left.identifier]
                        + 1 + self._distance(left, right)
                    ).only_enforce_if(arc)
            self.model.add_circuit(arcs)

    def _build_tile_lifecycles(self):
        bound = self.H * self.W - 1
        starts, ends = {}, {}
        for project in self.projects:
            ranks = [
                self._rank(requirement.identifier)
                for requirement in project.requirements
            ]
            start = self.model.new_int_var(
                0, bound, f"lifecycle-start:{project.identifier}"
            )
            end = self.model.new_int_var(
                0, bound, f"lifecycle-end:{project.identifier}"
            )
            self.model.add_min_equality(start, ranks)
            self.model.add_max_equality(end, ranks)
            starts[project.identifier], ends[project.identifier] = start, end
        for index, left in enumerate(self.projects):
            for right in self.projects[index + 1:]:
                if not set(left.positions) & set(right.positions):
                    continue
                same = self.model.new_bool_var(
                    f"same-cell:{left.identifier}:{right.identifier}"
                )
                self.model.add(
                    self.cell[left.identifier] == self.cell[right.identifier]
                ).only_enforce_if(same)
                self.model.add(
                    self.cell[left.identifier] != self.cell[right.identifier]
                ).only_enforce_if(same.Not())
                before = self.model.new_bool_var(
                    f"lifecycle-order:{left.identifier}:{right.identifier}"
                )
                self.model.add(
                    ends[left.identifier] < starts[right.identifier]
                ).only_enforce_if((same, before))
                self.model.add(
                    ends[right.identifier] < starts[left.identifier]
                ).only_enforce_if((same, before.Not()))

    def decode(self, solver):
        placements = {
            project.identifier: (
                solver.value(self.project_x[project.identifier]),
                solver.value(self.project_y[project.identifier]),
            )
            for project in self.projects
        }
        selected = {}
        for project in self.projects:
            chain_index = next(
                index for index, choice in enumerate(
                    self.path_choice[project.identifier]
                ) if solver.boolean_value(choice)
            )
            selected.update({
                event.requirement: event
                for event in project.chains[chain_index]
            })
        schedule = defaultdict(list)
        for node in self.nodes:
            if not solver.boolean_value(self.node_active[node.identifier]):
                continue
            worker = next(
                worker for worker in range(self.W)
                if solver.boolean_value(
                    self.node_assign[node.identifier, worker]
                )
            )
            schedule[worker].append(
                (solver.value(self.node_time[node.identifier]), node)
            )
        for route in schedule.values():
            route.sort(key=lambda entry: (entry[0], entry[1].identifier))
        purchases = {
            item: (
                solver.value(quantity),
                solver.value(self.purchase_time[item]),
            )
            for item, quantity in self.purchase_quantity.items()
            if solver.value(quantity) > 0
        }
        hires = {
            worker: solver.value(time)
            for worker, time in self.hire_time.items()
        }
        land = {
            quadrant: solver.value(time)
            for quadrant, time in self.land_time.items()
        }
        node_positions = {
            node.identifier: (
                solver.value(self.node_x[node.identifier]),
                solver.value(self.node_y[node.identifier]),
            )
            for node in self.nodes
            if solver.boolean_value(self.node_active[node.identifier])
        }
        return (
            placements, selected, schedule, node_positions,
            purchases, hires, land,
        )

    def exclude(self, solver):
        equalities = []
        for choices in self.path_choice.values():
            equalities.append(next(
                choice for choice in choices if solver.boolean_value(choice)
            ))
        for node in self.nodes:
            if not solver.boolean_value(self.node_active[node.identifier]):
                continue
            value = solver.value(self.node_time[node.identifier])
            equality = self.model.new_bool_var(
                f"nogood-time:{node.identifier}:{value}"
            )
            self.model.add(
                self.node_time[node.identifier] == value
            ).only_enforce_if(equality)
            self.model.add(
                self.node_time[node.identifier] != value
            ).only_enforce_if(equality.Not())
            equalities.append(equality)
            equalities.append(next(
                self.node_assign[node.identifier, worker]
                for worker in range(self.W)
                if solver.boolean_value(
                    self.node_assign[node.identifier, worker]
                )
            ))
        self.model.add_bool_or(
            [literal.Not() for literal in equalities]
        )


def _move(origin, target):
    if origin[0] != target[0]:
        return ("EAST",) if origin[0] < target[0] else ("WEST",)
    if origin[1] != target[1]:
        return ("SOUTH",) if origin[1] < target[1] else ("NORTH",)
    return ("PASS",)


def _materialize(model, decoded):
    (
        placements, selected, schedule, node_positions,
        purchases, hires, land,
    ) = decoded
    routes = {worker: list(route) for worker, route in schedule.items()}
    current = model.state
    turns = []
    remaining = Counter()
    for event in selected.values():
        remaining.update({
            item: -quantity
            for item, quantity in event.delta.items()
            if quantity < 0 and not item.endswith("_SEED")
        })

    for offset in range(model.H):
        actions = [("PASS",) for _ in current.workers]
        for worker in current.workers:
            route = routes.get(worker.index, ())
            if route and route[0][0] < offset:
                return None, {
                    "reason": "missed-event-time",
                    "event": route[0][1].identifier,
                    "turn": offset,
                }
            if not route:
                continue
            event_time, node = route[0]
            target = node_positions[node.identifier]
            if event_time == offset:
                if worker.position != target:
                    return None, {
                        "reason": "route-arrival-mismatch",
                        "event": node.identifier,
                        "turn": offset,
                        "worker": worker.index,
                    }
                if node.kind == "service":
                    event = selected[node.requirement]
                    actions[worker.index] = event.action
                    remaining.subtract({
                        item: -quantity
                        for item, quantity in event.delta.items()
                        if quantity < 0 and not item.endswith("_SEED")
                    })
                elif node.kind == "pickup":
                    actions[worker.index] = (
                        "PICKUP", node.item, node.quantity
                    )
                else:
                    actions[worker.index] = ("DROP",)
                route.pop(0)
            else:
                actions[worker.index] = _move(worker.position, target)

        required_purchases = []
        for item, (quantity, time) in purchases.items():
            if time != offset:
                continue
            if item.endswith("_SEED"):
                required_purchases.append(
                    ("BUY_SEED", item.removesuffix("_SEED"), quantity)
                )
            elif item in rules.ANIMALS:
                required_purchases.append(
                    ("BUY_ANIMAL", item, quantity)
                )
            else:
                required_purchases.append(
                    ("BUY_PRODUCT", item, quantity)
                )
        hire_count = sum(time == offset for time in hires.values())
        land_count = sum(time == offset for time in land.values())
        try:
            orders = realization_orders(
                current, tuple(actions), required_purchases,
                hire_count, land_count, remaining,
            )
            following = rules.advance_owned(current, tuple(actions), orders)
        except Exception as error:
            return None, {
                "reason": "materialization-error",
                "turn": offset,
                "error": repr(error),
            }
        turns.append(TurnDecision(
            tuple(tuple(action) for action in actions),
            tuple(tuple(order) for order in orders),
        ))
        current = following

    commitments = (
        *model.plan.obligations,
        *model.plan.selected,
        *model.plan.support,
    )
    realization = Realization(
        {
            key: placements[project.identifier]
            for project in model.projects
            for key in project.placement_keys
            if any(commitment.identifier == key
                   and commitment.target is None
                   for commitment in commitments)
        },
        tuple(turns),
    )
    try:
        ending = execute_realization(model.state, model.plan, realization)
    except Exception as error:
        return None, {
            "reason": "exact-execution",
            "error": repr(error),
        }
    return (realization, ending), None


def _maximum_workforce(state, plan, event_count):
    if plan.max_hands is not None:
        return max(len(state.workers), int(plan.max_hands) + 1)
    money, hires = state.money, state.hires_today
    maximum = len(state.workers)
    while maximum < event_count:
        cost = rules.fibonacci_hire_cost(hires)
        if cost > money:
            break
        money -= cost
        hires += 1
        maximum += 1
    return maximum


def _solve_fixed(state, plan, projects, land, workforce):
    begun = perf_counter()
    model = _CompactModel(state, plan, projects, land, workforce)
    failures = []
    for attempt in range(3):
        solver = model.cp.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = attempt
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.solve(model.model)
        if status not in (model.cp.OPTIMAL, model.cp.FEASIBLE):
            return None, {
                "workforce": workforce,
                "status": solver.status_name(status),
                "seconds": perf_counter() - begun,
                "variables": len(model.model.proto.variables),
                "constraints": len(model.model.proto.constraints),
                "materialization_failures": failures,
            }
        candidate, failure = _materialize(model, model.decode(solver))
        if candidate is not None:
            realization, ending = candidate
            return (end_value(state, ending), realization), {
                "workforce": workforce,
                "status": solver.status_name(status),
                "seconds": perf_counter() - begun,
                "variables": len(model.model.proto.variables),
                "constraints": len(model.model.proto.constraints),
                "exact_attempts": attempt + 1,
            }
        failures.append(failure)
        model.exclude(solver)
    return None, {
        "workforce": workforce,
        "status": "SEARCH_EXHAUSTED",
        "seconds": perf_counter() - begun,
        "variables": len(model.model.proto.variables),
        "constraints": len(model.model.proto.constraints),
        "materialization_failures": failures,
    }


def solve_intraday(state: OwnedState, plan) -> Realization:
    """Return a complete exact Realization or raise PlanningFailure."""
    if state.day != plan.day:
        raise PlanningFailure("Plan day does not match OwnedState")
    projects, land = _expand_plan(state, plan)
    event_count = sum(len(project.requirements) for project in projects)
    horizon = min(state.turns_left, state.turns_left_today)
    if not event_count and not land:
        return Realization({}, tuple(
            TurnDecision(tuple(("PASS",) for _ in state.workers))
            for _ in range(horizon)
        ))
    if horizon <= 0:
        raise PlanningFailure("no turns remain for the mandatory Daily Plan")
    minimum = max(
        len(state.workers),
        (event_count + horizon - 1) // horizon,
    )
    maximum = _maximum_workforce(state, plan, event_count)
    attempts, complete = [], []
    first_workforce = None
    best_value = None
    for workforce in range(minimum, maximum + 1):
        candidate, diagnostics = _solve_fixed(
            state, plan, projects, land, workforce
        )
        attempts.append(diagnostics)
        if candidate is None:
            continue
        value, realization = candidate
        complete.append((value, -workforce, realization))
        if first_workforce is None:
            first_workforce, best_value = workforce, value
            continue
        if workforce == first_workforce + 1 and value <= best_value:
            break
        best_value = max(best_value, value)
    if not complete:
        raise PlanningFailure(
            "no complete compact event realization found",
            diagnostics={"workforces": attempts},
        )
    return max(complete, key=lambda item: item[:2])[2]
