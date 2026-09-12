"""The single end-to-end Programme simulator.

All primitive farm and market transitions go through ``rules.advance_owned``;
this module only assembles the frozen programme and records separated flows.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Mapping

from . import rules
from .market import opponent_pressure
from .programme import AssetProgramme, Programme, ProgrammeEvent
from .state import State, TileState, WorkerState


@dataclass(frozen=True)
class SimulationResult:
    terminal_cash: int
    final_state: State
    feasible: bool
    farm_output: Mapping[int, Mapping[str, int]]
    field_stock: Mapping[int, Mapping[str, int]]
    worker_stock: Mapping[int, Mapping[str, int]]
    shed_stock: Mapping[int, Mapping[str, int]]
    planned_sale: Mapping[int, Mapping[str, int]]
    market_inventory: Mapping[int, Mapping[str, int]]
    failure: str | None = None


def _actions(programme: Programme, step: int, worker_count: int):
    result = [("PASS",) for _ in range(worker_count)]
    for route in programme.routes:
        action = route.actions.get(step)
        if action is not None and route.worker < worker_count:
            result[route.worker] = tuple(action)
    return tuple(result)


def _orders(programme: Programme, state: State):
    orders = [
        ("SELL", item, quantity)
        for item, quantity in sorted(programme.planned_sale.get(state.step, {}).items())
        if quantity
    ]
    for event in programme.events_at(state.step):
        if event.kind in {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT"}:
            orders.append((event.kind, event.item, event.quantity))
        elif event.kind == "BUY_LAND":
            orders.append(("BUY_LAND",))
        elif event.kind == "HIRE":
            orders.extend(("HIRE",) for _ in range(event.quantity))
    return tuple(orders[:rules.MAX_MARKET_ORDERS])


def drop_arrivals(state: State, programme: Programme):
    """Return newly produced stock at the turn it actually reaches the shed.

    This is a logistics projection over the already-built routes.  PICKUP stock
    is deliberately tracked separately: dropping an item previously taken from
    the shed is not a new arrival.  The pinned end-of-day worker reset is also a
    real drop, recorded at the following state step rather than assigned to a
    generic daily close in advance.
    """
    events_by_tile_day = defaultdict(list)
    for event in programme.events:
        if event.tile is not None and event.action and event.action[0] in {
            "HARVEST", "COLLECT_FERTILIZER"
        }:
            events_by_tile_day[event.step // rules.TURNS_PER_DAY, event.tile,
                               str(event.action[0])].append(event)
    for events in events_by_tile_day.values():
        events.sort(key=lambda event: (event.step, event.event_id))

    routes = {(route.day, route.worker): route for route in programme.routes}
    arrivals = defaultdict(Counter)
    access = set(rules.shed_access(state.board_size))
    for day in range(state.day, rules.TERMINAL_ACTION_STEP // rules.TURNS_PER_DAY + 1):
        first = max(state.step, day * rules.TURNS_PER_DAY)
        last = min((day + 1) * rules.TURNS_PER_DAY - 1,
                   rules.TERMINAL_ACTION_STEP)
        worker_ids = sorted(worker for route_day, worker in routes if route_day == day)
        if day == state.day:
            worker_ids = sorted(set(worker_ids) | set(range(len(state.workers))))
        for worker in worker_ids:
            route = routes.get((day, worker))
            position = (state.workers[worker].position
                        if day == state.day and worker < len(state.workers)
                        else rules.shed_access(state.board_size)[worker % len(access)])
            carried = Counter(state.workers[worker].inventory
                              if day == state.day and worker < len(state.workers) else {})
            # ``fresh`` is the subset not already counted in opening shed stock.
            fresh = Counter(carried)
            used_events = set()
            for step in range(first, last + 1):
                action = route.actions.get(step, ("PASS",)) if route else ("PASS",)
                op = str(action[0]) if action else "PASS"
                if op in {"NORTH", "SOUTH", "EAST", "WEST"}:
                    dx, dy = {"NORTH": (0, -1), "SOUTH": (0, 1),
                              "EAST": (1, 0), "WEST": (-1, 0)}[op]
                    position = (position[0] + dx, position[1] + dy)
                elif op == "PICKUP":
                    carried[str(action[1])] += int(action[2]) if len(action) > 2 else 1
                elif op in {"FEED", "FERTILIZE", "PLACE"}:
                    item = ("WHEAT" if op == "FEED" else
                            "FERTILIZER" if op == "FERTILIZE" else str(action[1]))
                    if carried[item]:
                        carried[item] -= 1
                    elif fresh[item]:
                        fresh[item] -= 1
                    fresh[item] = min(fresh[item], carried[item])
                elif op in {"HARVEST", "COLLECT_FERTILIZER"}:
                    key = (day, position, op)
                    candidates = events_by_tile_day.get(key, ())
                    event = next((candidate for candidate in candidates
                                  if candidate.event_id not in used_events), None)
                    if event is not None:
                        used_events.add(event.event_id)
                        item = event.item or "FERTILIZER"
                        quantity = event.quantity or 1
                        carried[item] += quantity
                        fresh[item] += quantity
                elif op == "DROP" and position in access:
                    for item, quantity in fresh.items():
                        if quantity > 0:
                            arrivals[step][item] += quantity
                    carried.clear(); fresh.clear()
            # The official refresh drops every remaining worker inventory.
            if fresh:
                arrival_step = min(last + 1, rules.TERMINAL_ACTION_STEP)
                for item, quantity in fresh.items():
                    if quantity > 0:
                        arrivals[arrival_step][item] += quantity
    return {step: dict(amounts) for step, amounts in sorted(arrivals.items())}


def project_asset_transitions(state: State, asset: AssetProgramme) -> AssetProgramme:
    """Derive one asset's output and harvest quantities through pinned rules.

    Movement and stock sourcing belong to intraday planning, so this shadow
    replay places one supplied worker on the exact tile.  Every biological
    change itself is made exclusively by ``rules.advance_owned``.
    """
    tile_index = asset.tile[1] * state.board_size + asset.tile[0]
    raw_tiles = [tile.raw for tile in state.tiles]
    if not asset.existing:
        # BUY_LAND candidates may still be LOCKED in the observation; the land
        # event makes the tile empty before the asset's first field action.
        raw_tiles[tile_index] = None
    shadow_tiles = tuple(TileState(tile.position,
                                  dict(raw) if isinstance(raw, dict) else raw)
                         for tile, raw in zip(state.tiles, raw_tiles))
    worker = WorkerState(0, asset.tile, {})
    own = replace(state.own, workers=(worker,), worker_positions=(asset.tile,),
                  worker_inventory=({},), tiles=shadow_tiles,
                  animals=tuple(a for a in state.own.animals
                                if a.position == asset.tile and asset.existing),
                  crops=tuple(a for a in state.own.crops
                              if a.position == asset.tile and asset.existing),
                  shed_inventory={}, seeds={})
    current = replace(state, own=own)

    scheduled = sorted((*asset.service_schedule, *asset.harvest_schedule),
                       key=lambda event: (event.step, event.priority, event.event_id))
    by_step = defaultdict(list)
    cursor_by_day = defaultdict(int)
    for event in scheduled:
        day = event.step // rules.TURNS_PER_DAY
        cursor = max(event.step, cursor_by_day[day],
                     state.step if day == state.day else day * rules.TURNS_PER_DAY)
        deadline = event.deadline if event.deadline is not None else (day + 1) * 24 - 1
        if cursor <= min(deadline, rules.TERMINAL_ACTION_STEP):
            by_step[cursor].append(event)
            cursor_by_day[day] = cursor + 1

    output = []
    actual_harvest = {}
    while current.step <= rules.TERMINAL_ACTION_STEP:
        action = ("PASS",)
        event = by_step.get(current.step, [None])[0]
        if event is not None:
            action = tuple(event.action)
            inventory = dict(current.workers[0].inventory)
            seeds = dict(current.seeds)
            op = str(action[0])
            if op == "FEED": inventory["WHEAT"] = inventory.get("WHEAT", 0) + 1
            elif op == "FERTILIZE": inventory["FERTILIZER"] = inventory.get("FERTILIZER", 0) + 1
            elif op == "PLACE": inventory[str(action[1])] = inventory.get(str(action[1]), 0) + 1
            elif op == "PLANT": seeds[str(action[1])] = seeds.get(str(action[1]), 0) + 1
            supplied = replace(current.workers[0], position=asset.tile, inventory=inventory)
            current = replace(current, own=replace(
                current.own, workers=(supplied,), worker_positions=(asset.tile,),
                worker_inventory=(inventory,), seeds=seeds))

        before = current.tile_at(asset.tile).raw
        before_yield = int(before.get("yield_units", 0)) if isinstance(before, dict) else 0
        before_f = bool(before.get("fertilizer_available", False)) if isinstance(before, dict) else False
        before_inventory = dict(current.workers[0].inventory)
        after = rules.advance_owned(current, (action,))
        after_raw = after.tile_at(asset.tile).raw
        after_yield = int(after_raw.get("yield_units", 0)) if isinstance(after_raw, dict) else 0
        after_f = bool(after_raw.get("fertilizer_available", False)) if isinstance(after_raw, dict) else False

        if after_yield > before_yield:
            item = (asset.asset_type if asset.asset_type in rules.CROPS
                    else rules.ANIMALS[asset.asset_type].product)
            output.append(ProgrammeEvent(
                after.step, 0, f"{asset.asset_id}:official-output:{after.step}:{len(output)}",
                "OUTPUT", asset.tile, asset.asset_id, item,
                after_yield - before_yield, mandatory=False))
        if after_f and not before_f:
            output.append(ProgrammeEvent(
                after.step, 0, f"{asset.asset_id}:official-output:F:{after.step}",
                "OUTPUT", asset.tile, asset.asset_id, "FERTILIZER", 1,
                mandatory=False))
        if event is not None and event.kind == "HARVEST":
            item = event.item or asset.asset_type
            after_inventory = (after.workers[0].inventory
                               if after.workers else {})
            quantity = max(0, after_inventory.get(item, 0) - before_inventory.get(item, 0))
            actual_harvest[event.event_id] = replace(event, quantity=quantity)
        current = after

    harvest = tuple(actual_harvest.get(event.event_id, event)
                    for event in asset.harvest_schedule)
    return replace(asset, output_schedule=tuple(output), harvest_schedule=harvest)


def _inject_pressure(state: State, amounts: Mapping[str, int]) -> State:
    if not amounts:
        return state
    inventory = dict(state.market.inventory)
    for item, quantity in amounts.items():
        inventory[item] += quantity
    market = replace(state.market, inventory=inventory,
                     price={item: rules.market_price(item, value) for item, value in inventory.items()})
    return replace(state, market=market)


def _tile_yield(state: State):
    result = {}
    for tile in state.tiles:
        raw = tile.raw
        if isinstance(raw, dict) and raw.get("yield_units", 0):
            item = raw.get("crop")
            if "animal" in raw:
                item = rules.ANIMALS[str(raw["animal"])].product
            result[tile.position] = (str(item), int(raw["yield_units"]))
    return result


def _fertilizer_tiles(state: State):
    return {tile.position for tile in state.tiles if isinstance(tile.raw, dict)
            and "animal" in tile.raw and bool(tile.raw.get("fertilizer_available",False))}


def simulate_programme(state: State, programme: Programme,
                       *, pressure_events=None) -> SimulationResult:
    pressure_events = pressure_events or opponent_pressure(state)
    pressure_by_step = defaultdict(dict)
    for item, events in pressure_events.items():
        for step, quantity in events:
            pressure_by_step[step][item] = pressure_by_step[step].get(item, 0) + quantity

    current = state
    farm_output, field_stock, worker_stock, shed_stock, market_inventory = {}, {}, {}, {}, {}
    failure = None
    while current.step <= rules.TERMINAL_ACTION_STEP:
        current = _inject_pressure(current, pressure_by_step.get(current.step, {}))
        before_yield = _tile_yield(current)
        before_fertilizer = _fertilizer_tiles(current)
        actions = _actions(programme, current.step, len(current.workers))
        orders = _orders(programme, current)
        if len(orders) > rules.MAX_MARKET_ORDERS:
            failure = f"step {current.step}: market order limit exceeded"
            break
        after_units=rules.advance_owned(current,actions,unit_only=True)
        for item,quantity in programme.planned_sale.get(current.step,{}).items():
            if after_units.shed.get(item,0)<quantity:
                failure=f"step {current.step}: planned {item} sale lacks stock"
                break
        if failure is not None: break
        after = rules.advance_owned(current, actions, orders)
        after_yield = _tile_yield(after)
        after_fertilizer = _fertilizer_tiles(after)
        produced = defaultdict(int)
        for position, (item, quantity) in after_yield.items():
            old_item, old_quantity = before_yield.get(position, (item, 0))
            if item == old_item and quantity > old_quantity:
                produced[item] += quantity - old_quantity
        if produced:
            farm_output[after.step] = dict(produced)
        new_fertilizer=len(after_fertilizer-before_fertilizer)
        if new_fertilizer:
            farm_output.setdefault(after.step,{})["FERTILIZER"]=new_fertilizer
        field = defaultdict(int)
        for item, quantity in after_yield.values():
            field[item] += quantity
        field_stock[after.step] = dict(field)
        carried = defaultdict(int)
        for worker in after.workers:
            for item, quantity in worker.inventory.items():
                carried[item] += quantity
        worker_stock[after.step] = dict(carried)
        shed_stock[after.step] = dict(after.shed)
        market_inventory[after.step] = dict(after.market.inventory)
        if sum(after.shed.values()) > rules.SHED_CAPACITY:
            failure = f"step {current.step}: shed capacity exceeded"
            break
        current = after

    return SimulationResult(
        terminal_cash=current.money,
        final_state=current,
        feasible=failure is None,
        farm_output=farm_output,
        field_stock=field_stock,
        worker_stock=worker_stock,
        shed_stock=shed_stock,
        planned_sale=programme.planned_sale,
        market_inventory=market_inventory,
        failure=failure,
    )


def programme_value(state: State, programme: Programme) -> int:
    result = simulate_programme(state, programme)
    return result.terminal_cash if result.feasible else -(10**18)


def cash_projection(state: State, programme: Programme, pressure_events, *, details=False):
    """Exact cash/market projection after route feasibility is established.

    Unit actions never change cash in the pinned engine. This is therefore an
    equivalent value path for candidate comparisons; committed programmes are
    still physically replayed by ``simulate_programme``.
    """
    money=state.money; market=dict(state.market.inventory); hires=state.hires_today
    pressure=defaultdict(lambda:defaultdict(int))
    for item,events in pressure_events.items():
        for step,quantity in events: pressure[step][item]+=quantity
    events=defaultdict(list)
    for event in programme.events:
        if event.kind in {"BUY_SEED","BUY_ANIMAL","BUY_PRODUCT","BUY_LAND","HIRE"}:
            events[event.step].append(event)
    for step in range(state.step,rules.TERMINAL_ACTION_STEP+1):
        if step%24==0 and step!=state.step: hires=0
        for item,quantity in pressure.get(step,{}).items(): market[item]+=quantity
        orders=[]
        for item,quantity in sorted(programme.planned_sale.get(step,{}).items()):
            if quantity: orders.append(("SELL",item,quantity))
        for event in sorted(events.get(step,()),key=lambda e:(e.priority,e.event_id)):
            if event.kind=="HIRE": orders.extend(("HIRE",None,1) for _ in range(event.quantity))
            else: orders.append((event.kind,event.item,event.quantity))
        if len(orders)>rules.MAX_MARKET_ORDERS:
            return (-(10**18),False,step) if details else (-(10**18),False)
        for kind,item,quantity in orders:
            if kind=="HIRE":
                cost=rules.fibonacci_hire_cost(hires)
                if money<cost: return (money,False,step) if details else (-(10**18),False)
                money-=cost; hires+=1
            elif kind=="BUY_LAND":
                count=sum(e.kind=="BUY_LAND" and e.step<=step for es in events.values() for e in es)
                index=len(state.own.owned_land)-1+count-1
                cost=rules.LAND_PRICES[index]
                if money<cost:return (money,False,step) if details else (-(10**18),False)
                money-=cost
            elif kind=="BUY_SEED":
                cost=rules.CROPS[item].seed_cost*quantity
                if money<cost:return (money,False,step) if details else (-(10**18),False)
                money-=cost
            elif kind=="BUY_ANIMAL":
                cost=rules.ANIMALS[item].cost*quantity
                if money<cost:return (money,False,step) if details else (-(10**18),False)
                money-=cost
            elif kind=="BUY_PRODUCT":
                for _ in range(quantity):
                    cost=rules.market_price(item,market[item]-1)
                    if money<cost:return (money,False,step) if details else (-(10**18),False)
                    money-=cost; market[item]-=1
            elif kind=="SELL":
                for _ in range(quantity):
                    price=rules.market_price(item,market[item]); money+=price
                    if price>rules.PRICE_FLOOR: market[item]+=1
        if step%4==0:
            for shop in state.shops:
                basket=rules.SHOPS[shop]
                for item in basket: market[item]-=2 if len(basket)==1 else 1
        if step%24==0:
            for item in rules.PRODUCTS:
                if item!="FERTILIZER":market[item]-=1
    return (money,True,None) if details else (money,True)
