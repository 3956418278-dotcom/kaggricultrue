"""The single end-to-end Programme simulator.

All primitive farm and market transitions go through ``rules.advance_owned``;
this module only assembles the frozen programme and records separated flows.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping

from . import rules
from .market import opponent_pressure
from .programme import Programme
from .state import State


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
