"""Observation-driven midgame planner; it may attach on any game day."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable, Mapping

from . import rules
from .current_assets import (AnimalDecisionInputs, animal_locality_bonus,
                             current_asset_programmes, exit_current_asset,
                             read_current_assets)
from .intraday import PlanningFailure, clear_route_cache, solve_intraday
from .market import clear_sale_cache, known_demand_events, next_reveal, opponent_pressure, optimize_sales
from .midgame_config import coerce_midgame_parameters
from .programme import (AssetProgramme, CurrentAssetState,
                        FertilizerProgramme, LandProgramme, Programme,
                        ProgrammeEvent)
from .simulation import (cash_projection, drop_arrivals,
                         project_asset_transitions, simulate_programme)
from .state import AssetState, Position, State

LONG_ASSETS = ("COW", "SHEEP", "GOOSE", "STRAWBERRY", "TOMATO", "MELON")


def _pe(step, priority, identifier, kind, *, tile=None, asset=None, item=None,
        quantity=0, action=("PASS",), mandatory=True, deadline=None, source=None):
    return ProgrammeEvent(step, priority, identifier, kind, tile, asset, item,
                          quantity, tuple(action), mandatory, deadline, None, source)


def _minimal_service_days(raw: Mapping[str, object], start_day: int, last_day: int,
                          *, animal: bool):
    key = "consecutive_unfed" if animal else "consecutive_unwatered"
    done = "fed_today" if animal else "watered_today"
    consecutive = int(raw.get(key, 0))
    result = set()
    for day in range(start_day, last_day + 1):
        if day == start_day and bool(raw.get(done, False)):
            consecutive = 0
        elif consecutive >= 1:
            result.add(day); consecutive = 0
        else:
            consecutive += 1
    return result


def _animal_programme(state: State, asset_type: str, tile: Position, *, existing: bool,
                      raw=None, start_step=None, purpose="LONG") -> AssetProgramme:
    if existing:
        raise ValueError("existing animals must be read through CurrentAssetState")
    start_step = state.step if start_step is None else start_step
    start_day = start_step // 24
    identifier = f"{'existing' if existing else 'new'}:{asset_type}:{tile[0]}:{tile[1]}"
    rule = rules.ANIMALS[asset_type]
    raw = dict(raw or {"kind": rule.structure, "animal": asset_type,
        "placed_day": start_day, "yield_units": 0, "consecutive_unfed": 0,
        "fed_today": False, "cared_today": False, "fertilizer_available": False,
        "pending_care_bonus": 0})
    service, output, harvest = [], [], []
    observed = state.tile_at(tile)
    if start_step == state.step:
        opening_kind = observed.kind
    elif observed.is_locked or observed.crop is not None:
        opening_kind = None
    else:
        # A future start on today's animal tile is only reachable through an
        # EXIT release and therefore retains the official structure.
        opening_kind = observed.kind
    cursor=start_step
    if opening_kind == "WEED" or opening_kind in {"COOP", "PASTURE"} and opening_kind != rule.structure:
        service.append(_pe(cursor,9,identifier+":dig","DIG",tile=tile,asset=identifier,
                           action=("DIG",),deadline=min(start_day*24+21,718)));cursor+=1
        opening_kind = None
    if opening_kind != rule.structure:
        service.append(_pe(cursor,10,identifier+":build","BUILD",tile=tile,asset=identifier,
            action=("BUILD_"+rule.structure,),deadline=min(start_day*24+22,718)));cursor+=1
    service.append(
        _pe(cursor, 11, identifier+":place", "PLACE", tile=tile, asset=identifier,
            item=asset_type, quantity=1, action=("PLACE", asset_type),
            deadline=min(start_day*24+23, rules.TERMINAL_ACTION_STEP)))
    feed_days = _minimal_service_days(raw, start_day, 29, animal=True)
    if int(raw.get("pending_care_bonus", 0)):
        for day in range(start_day, 30):
            since = day + 1 - int(raw["placed_day"]) - rule.first_yield_day
            if since >= 0 and since % rule.interval == 0:
                feed_days.add(day); break
    for day in sorted(feed_days):
        release = max(start_step, day * 24)
        if release <= rules.TERMINAL_ACTION_STEP:
            service.append(_pe(release, 20, f"{identifier}:feed:{day}", "FEED", tile=tile,
                asset=identifier, item="WHEAT", quantity=1, action=("FEED",), deadline=min(day*24+23, 718)))
    held = int(raw.get("yield_units", 0))
    if held:
        harvest.append(_pe(start_step, 30, f"{identifier}:harvest:opening", "HARVEST", tile=tile,
            asset=identifier, item=rule.product, quantity=held, action=("HARVEST",), deadline=min(start_day*24+23,718)))
    for day in range(start_day, 30):
        next_day = day + 1
        since = next_day - int(raw["placed_day"]) - rule.first_yield_day
        step = next_day * 24
        if since >= 0 and since % rule.interval == 0 and step <= 718:
            quantity=1
            if int(raw.get("pending_care_bonus",0)) and day in feed_days:
                quantity+=int(raw.get("pending_care_bonus",0)); raw["pending_care_bonus"]=0
            out = _pe(step, 0, f"{identifier}:output:{next_day}", "OUTPUT", tile=tile,
                      asset=identifier, item=rule.product, quantity=quantity, mandatory=False)
            output.append(out)
            harvest.append(_pe(step, 30, f"{identifier}:harvest:{next_day}", "HARVEST", tile=tile,
                asset=identifier, item=rule.product, quantity=quantity, action=("HARVEST",),
                deadline=min(next_day*24+23,718)))
    if not bool(raw.get("fertilizer_available",False)) and start_day < 30:
        step=min((start_day+1)*24,718)
        output.append(_pe(step,0,f"{identifier}:output:F:{start_day+1}","OUTPUT",tile=tile,
            asset=identifier,item="FERTILIZER",quantity=1,mandatory=False))
    return AssetProgramme(identifier, asset_type, tile, "NEW", False,
                          purpose, None, tuple(service), tuple(output), tuple(harvest))


def _crop_programme(state: State, crop: str, tile: Position, *, existing: bool,
                    raw=None, start_step=None, purpose="LONG", release_step=None):
    if existing:
        raise ValueError("existing crops must be read through CurrentAssetState")
    start_step = state.step if start_step is None else start_step
    start_day = start_step // 24
    identifier = f"{'existing' if existing else 'new'}:{crop}:{tile[0]}:{tile[1]}:{start_step}"
    rule = rules.CROPS[crop]
    raw = dict(raw or {"kind":"PLANT", "crop":crop, "planted_day":start_day,
        "watered_today":False, "consecutive_unwatered":1,
        "yield_units":0 if rule.ongoing else 1,
        "max_lifespan_step":-1 if rule.ongoing else (start_day+rule.max_yield_day+1)*24,
        "fertilized_until_day":-1})
    service, output, harvest = [], [], []
    observed = state.tile_at(tile)
    if observed.is_locked or start_step > state.step and observed.crop is not None:
        opening_kind = None
    else:
        opening_kind = observed.kind
    cursor=start_step+1
    if opening_kind in {"WEED", "COOP", "PASTURE"}:
        service.append(_pe(cursor,9,identifier+":dig","DIG",tile=tile,asset=identifier,
            action=("DIG",),deadline=min(start_day*24+20,718)));cursor+=1
    service.extend([
        _pe(cursor, 10, identifier+":plant", "PLANT", tile=tile, asset=identifier,
            item=crop, quantity=1, action=("PLANT", crop), deadline=min(start_day*24+21,718), source=purpose),
        _pe(cursor+1, 11, identifier+":water:plant", "WATER", tile=tile, asset=identifier,
            action=("WATER",), deadline=min(start_day*24+22,718), source=purpose),
    ])
    water_days = _minimal_service_days(raw, start_day, 29, animal=False)
    water_days.discard(start_day)
    for day in sorted(water_days):
        release = max(start_step, day*24)
        if release <= 718:
            service.append(_pe(release, 20, f"{identifier}:water:{day}", "WATER", tile=tile,
                asset=identifier, action=("WATER",), deadline=min(day*24+23,718), source=purpose))
    held = int(raw.get("yield_units", 0))
    planted = int(raw["planted_day"])
    mature_step = max(start_step, (planted + rule.first_yield_day) * 24)
    if not rule.ongoing:
        harvest_step = release_step if release_step is not None else mature_step
        harvest_step = max(mature_step, min(harvest_step, 718))
        if harvest_step <= 718:
            quantity = max(1, held)
            fertilized_until = int(raw.get("fertilized_until_day", -1))
            for water in sorted(e for e in service if e.kind == "WATER"):
                if water.step > harvest_step:
                    continue
                quantity += rules.one_time_water_gain(
                    crop, planted_day=planted, day=water.step // 24,
                    yield_units=quantity, fertilized_until_day=fertilized_until,
                )
            harvest.append(_pe(harvest_step, 30, identifier+":harvest", "HARVEST", tile=tile,
                asset=identifier, item=crop, quantity=quantity, action=("HARVEST",),
                deadline=min(harvest_step//24*24+23,718), source=purpose))
    else:
        water_days={e.step//24 for e in service if e.kind=="WATER"}
        if bool(raw.get("watered_today",False)): water_days.add(state.day)
        for day in range(start_day, 30):
            next_day = day + 1
            since = next_day - planted - rule.first_yield_day
            step = next_day*24
            count = since // rule.interval + 1 if since >= 0 else 0
            if since >= 0 and since % rule.interval == 0 and count <= rule.max_yield and step <= 718:
                quantity=2 if day in water_days and int(raw.get("fertilized_until_day",-1))>=day else 1
                output.append(_pe(step, 0, f"{identifier}:output:{next_day}", "OUTPUT", tile=tile,
                    asset=identifier, item=crop, quantity=quantity, mandatory=False))
                harvest.append(_pe(step, 30, f"{identifier}:harvest:{next_day}", "HARVEST", tile=tile,
                    asset=identifier, item=crop, quantity=quantity, action=("HARVEST",), deadline=min(next_day*24+23,718)))
    return AssetProgramme(identifier, crop, tile, "NEW", False,
                          purpose, release_step, tuple(service), tuple(output), tuple(harvest))


def _purchase_event(state: State, asset: AssetProgramme, step: int):
    if asset.asset_type in rules.ANIMALS:
        return _pe(step, -20, asset.asset_id+":buy", "BUY_ANIMAL", asset=asset.asset_id,
                   item=asset.asset_type, quantity=1)
    return _pe(step, -20, asset.asset_id+":buy", "BUY_SEED", asset=asset.asset_id,
               item=asset.asset_type, quantity=1)


def _programme_from_assets(state: State, assets: Iterable[AssetProgramme],
                           extra_events=(), land=(),
                           current_assets: Iterable[CurrentAssetState] = ()):
    assets = tuple(assets)
    events = [*extra_events]
    for asset in assets:
        events.extend(asset.service_schedule); events.extend(asset.harvest_schedule)
        if not asset.existing:
            start = min((e.step for e in asset.service_schedule), default=state.step)
            events.append(_purchase_event(state, asset, max(state.step, start-1)))
    unique = {event.event_id: event for event in events}
    return Programme(state.step, state.day, state.shops, assets,
                     tuple(sorted(unique.values())), land=tuple(land),
                     current_assets=tuple(current_assets))


def _arrivals(state: State, programme: Programme):
    if not programme.routes:
        from collections import defaultdict
        optimistic_drops = defaultdict(lambda: defaultdict(int))
        for asset in programme.assets:
            for ev in asset.harvest_schedule:
                item = ev.item or (asset.asset_type if asset.asset_type in rules.CROPS else rules.ANIMALS[asset.asset_type].product)
                optimistic_drops[ev.step][item] += ev.quantity
        return {s: dict(a) for s, a in optimistic_drops.items()}
    from .simulation import drop_arrivals
    return drop_arrivals(state, programme)


_VALUE_CACHE={}
_SALE_CACHE={}
_ECONOMIC_CACHE={}


def _economic_core(programme: Programme):
    """Cache tile-independent production and fixed cash commitments first."""
    key = tuple(sorted((
        asset.asset_type, asset.existing, asset.purpose, asset.release_turn,
        tuple((event.step, event.kind, event.item, event.quantity, event.mandatory)
              for event in asset.service_schedule),
        tuple((event.step, event.item, event.quantity)
              for event in asset.output_schedule),
        tuple((event.step, event.item, event.quantity)
              for event in asset.harvest_schedule),
    ) for asset in programme.assets))
    cached = _ECONOMIC_CACHE.get(key)
    if cached is None:
        output = tuple(sorted((event.step, event.item, event.quantity)
                              for asset in programme.assets
                              for event in asset.output_schedule))
        harvest = tuple(sorted((event.step, event.item, event.quantity)
                               for asset in programme.assets
                               for event in asset.harvest_schedule))
        commitments = tuple(sorted((event.step, event.kind, event.item, event.quantity)
                                   for event in programme.events
                                   if event.kind in {"BUY_SEED", "BUY_ANIMAL",
                                                     "BUY_PRODUCT", "BUY_LAND"}))
        cached = (output, harvest, commitments)
        _ECONOMIC_CACHE[key] = cached
    return cached


def _feed_sale_inputs(state: State, programme: Programme, arrivals):
    """Reserve dated wheat needed by FEED before exposing stock to SELL DP."""
    feeds=sorted((event.deadline if event.deadline is not None else event.step)
                 for event in programme.events if event.kind=="FEED")
    if not feeds:
        return state,arrivals
    adjusted={step:dict(amounts) for step,amounts in arrivals.items()}
    opening=state.shed.get("WHEAT",0)
    opening_reserved=0
    arrival_records=[[step,amounts.get("WHEAT",0)]
                     for step,amounts in sorted(adjusted.items())
                     if amounts.get("WHEAT",0)]
    buys=sorted(event.step+1 for event in programme.events
                if event.kind=="BUY_PRODUCT" and event.item=="WHEAT")
    used_buys=0
    for deadline in feeds:
        if used_buys<len(buys) and buys[used_buys]<=deadline:
            used_buys+=1
            continue
        if opening_reserved<opening:
            opening_reserved+=1
            continue
        source=next((record for record in arrival_records
                     if record[0]<=deadline and record[1]>0),None)
        if source is not None:
            source[1]-=1
    for step,remaining in arrival_records:
        adjusted[step]["WHEAT"]=remaining
        if not remaining:
            del adjusted[step]["WHEAT"]
    adjusted={step:amounts for step,amounts in adjusted.items() if amounts}
    shed=dict(state.shed)
    if opening_reserved:
        shed["WHEAT"]=opening-opening_reserved
    sale_state=replace(state,own=replace(state.own,shed_inventory=shed))
    return sale_state,adjusted


def _economic_signature(state: State, programme: Programme):
    assets=tuple(sorted((a.asset_type,a.decision,a.existing,a.purpose,a.release_turn,
        tuple((e.step,e.kind,e.item,e.quantity,e.action,e.mandatory,e.deadline) for e in a.service_schedule if e.kind not in {"PICKUP","DROP"}),
        tuple((e.step,e.item,e.quantity) for e in a.output_schedule),
        tuple((e.step,e.item,e.quantity) for e in a.harvest_schedule)) for a in programme.assets))
    events=tuple(sorted(((e.step,e.kind,e.item,e.quantity,e.source) for e in programme.events
                        if e.kind not in {"HIRE"} and e.kind not in {"PICKUP","DROP"}), key=repr))
    return (state.step,state.money,tuple(sorted(state.shed.items())),tuple(sorted(state.seeds.items())),
            tuple(sorted(state.market.inventory.items())),assets,events,
            tuple((land.quadrant,land.buy_step,land.cost,land.use) for land in programme.land))


def _economic_valuation(state: State, programme: Programme, demand, pressure):
    signature = _economic_signature(state, programme)
    cached = _ECONOMIC_CACHE.get(signature)
    if cached is not None:
        return cached

    import time
    t0 = time.perf_counter()
    estimated_arrivals = defaultdict(lambda: defaultdict(int))
    for asset in programme.assets:
        for event in asset.harvest_schedule:
            item = event.item or (asset.asset_type if asset.asset_type in rules.CROPS else rules.ANIMALS[asset.asset_type].product)
            estimated_arrivals[event.step][item] += event.quantity

    sale_state, sale_arrivals = _feed_sale_inputs(state, programme, estimated_arrivals)
    sale_key = (tuple((step, tuple(sorted(amounts.items())))
                      for step, amounts in sorted(sale_arrivals.items())),
                tuple(sorted(sale_state.shed.items())),
                tuple(sorted(state.market.inventory.items())),
                tuple((e.step, e.kind, e.item, e.quantity) for e in programme.events if e.kind in {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "BUY_LAND", "HIRE"}))
    sales = _SALE_CACHE.get(sale_key)
    if sales is None:
        sales = optimize_sales(sale_state, sale_arrivals, demand, pressure, commitments=programme.events)
        _SALE_CACHE[sale_key] = sales
    updated = replace(programme, planned_sale=sales.planned_sale, market_inventory=sales.market_inventory)
    cash, feasible = cash_projection(state, updated, pressure)
    result = (cash, feasible, sales)
    _ECONOMIC_CACHE[signature] = result
    return result


def _finalize(state: State, programme: Programme, demand, pressure, *, force=False,
              ensure_feed=True):
    if ensure_feed:
        programme = _ensure_feed_supply(state, programme)
    
    cash, feasible, sales = _economic_valuation(state, programme, demand, pressure)
    if not feasible:
        return replace(programme, feasible=False, terminal_cash=-(10**18),
                       diagnostics={"failure": "cash/market infeasible"})

    updated = replace(programme, planned_sale=sales.planned_sale, market_inventory=sales.market_inventory)
    from .simulation import drop_arrivals

    try:
        routed = solve_intraday(state, programme, complete_actions=force)
    except PlanningFailure as exc:
        return replace(programme, feasible=False, terminal_cash=-(10**18),
                       diagnostics={"failure": str(exc)})
    
    arrivals = drop_arrivals(state, routed)
    sale_state, sale_arrivals = _feed_sale_inputs(state, routed, arrivals)
    
    sale_key = (tuple((step, tuple(sorted(amounts.items()))) for step, amounts in sorted(sale_arrivals.items())),
                tuple(sorted(sale_state.shed.items())), tuple(sorted(state.market.inventory.items())),
                tuple((e.step, e.kind, e.item, e.quantity) for e in routed.events if e.kind in {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "BUY_LAND", "HIRE"}))
    sales = _SALE_CACHE.get(sale_key)
    if sales is None:
        sales = optimize_sales(sale_state, sale_arrivals, demand, pressure, commitments=routed.events)
        _SALE_CACHE[sale_key] = sales
        
    updated = replace(routed, planned_sale=sales.planned_sale, market_inventory=sales.market_inventory)

    cash, feasible = cash_projection(state, updated, pressure)
    if not feasible:
        return replace(programme, feasible=False, terminal_cash=-(10**18),
                       diagnostics={"failure": "cash/market infeasible after routing"})

    if not force:
        return replace(updated, terminal_cash=cash, feasible=True)

    result = simulate_programme(state, updated, pressure_events=pressure)
    if not result.feasible or result.terminal_cash != cash:
        result = replace(result, feasible=False, failure=result.failure or "physical replay did not realize projected cash")
    return replace(updated, terminal_cash=result.terminal_cash, feasible=result.feasible,
        farm_output=result.farm_output, field_stock=result.field_stock,
        worker_stock=result.worker_stock, shed_stock=result.shed_stock,
        planned_sale=result.planned_sale, market_inventory=result.market_inventory,
        diagnostics={} if result.feasible else {"failure": result.failure})


def _attach_asset_flows(programme: Programme, arrivals):
    queues=defaultdict(list)
    available_arrivals={item:[[step,quantity] for step,amounts in sorted(arrivals.items())
                              if (quantity:=amounts.get(item,0))]
                        for item in rules.PRODUCTS}
    assets=[]
    for asset in programme.assets:
        core_service=tuple(e for e in asset.service_schedule if e.kind not in {"PICKUP","DROP"})
        stock=[]
        logistics=[]
        for event in asset.harvest_schedule:
            arrival=next((record[0] for record in available_arrivals.get(event.item,())
                          if record[1]>0 and record[0]>=event.step),
                         rules.TERMINAL_ACTION_STEP)
            left=event.quantity
            for record in available_arrivals.get(event.item,()):
                if record[0]!=arrival or left<=0: continue
                take=min(left,record[1]); record[1]-=take; left-=take
            stock.append(_pe(arrival,0,event.event_id+":stock","STOCK",tile=asset.tile,
                asset=asset.asset_id,item=event.item,quantity=event.quantity,mandatory=False))
            queues[event.item].append([arrival,event.quantity,asset.asset_id,asset.tile])
            logistics.append(_pe(arrival,90,event.event_id+":drop","DROP",tile=asset.tile,
                asset=asset.asset_id,item=event.item,quantity=event.quantity,mandatory=True))
        for event in core_service:
            if event.action and event.action[0] in {"FEED","FERTILIZE","PLACE"}:
                item=("WHEAT" if event.action[0]=="FEED" else "FERTILIZER" if event.action[0]=="FERTILIZE" else str(event.action[1]))
                logistics.append(_pe(event.step,5,event.event_id+":pickup","PICKUP",tile=asset.tile,
                    asset=asset.asset_id,item=item,quantity=1,mandatory=True))
        assets.append(replace(asset,service_schedule=tuple(sorted((*core_service,*logistics))),
                              stock_schedule=tuple(stock),sale_schedule=()))
    sales=defaultdict(list)
    for step,amounts in sorted(programme.planned_sale.items()):
        for item,quantity in amounts.items():
            left=quantity
            for record in queues[item]:
                if record[0]>step or record[1]<=0: continue
                take=min(left,record[1]); record[1]-=take; left-=take
                if take:
                    sales[record[2]].append(_pe(step,0,f"{record[2]}:sale:{step}:{take}","SALE",
                        tile=record[3],asset=record[2],item=item,quantity=take,mandatory=False))
                if left==0: break
    return replace(programme,assets=tuple(replace(a,sale_schedule=tuple(sales[a.asset_id])) for a in assets))


def _ensure_feed_supply(state: State, programme: Programme):
    """Attach dated BUY_W only where committed feed otherwise lacks wheat.

    This is the provisional BUY alternative used while valuing KEEP/optional/
    long candidates. The explicit W_FEED phase may replace it with planted wheat.
    """
    feeds = sorted(e for e in programme.events if e.kind == "FEED")
    incoming = defaultdict(int)
    for asset in programme.assets:
        if asset.asset_type == "WHEAT":
            for event in asset.harvest_schedule:
                incoming[min((event.step//24+1)*24,718)] += event.quantity
    available = state.owned_total("WHEAT")
    prior = [e for e in programme.events if not e.event_id.startswith("feed-wheat:auto:")]
    shifted = {}
    buys = []
    last = state.step
    for index, feed in enumerate(feeds):
        for step in sorted(s for s in incoming if last < s <= feed.step):
            available += incoming[step]
        last = feed.step
        if available <= 0:
            buy_step = max(state.step, (feed.step//24)*24)
            buys.append(_pe(buy_step,-15,f"feed-wheat:auto:{index}","BUY_PRODUCT",
                            item="WHEAT",quantity=1,source="W_FEED"))
            if feed.step <= buy_step:
                shifted[feed.event_id] = replace(feed, step=buy_step+1)
            available += 1
        available -= 1
    if not buys and not shifted:
        return programme
    events = [shifted.get(e.event_id,e) for e in prior]
    events.extend(buys)
    assets=[]
    for asset in programme.assets:
        service=tuple(shifted.get(e.event_id,e) for e in asset.service_schedule)
        assets.append(replace(asset,service_schedule=service))
    return replace(programme,assets=tuple(assets),events=tuple(sorted({e.event_id:e for e in events}.values())))


def _initial_assets(state: State, prior: Programme | None = None,
                    inputs: AnimalDecisionInputs | None = None):
    """Read existing assets once and create only today's executable shells."""
    current = read_current_assets(
        state,
        prior_assets=prior.current_assets if prior is not None else (),
        prior_shops=prior.shops if prior is not None else (),
        inputs=inputs)
    return current_asset_programmes(current), current


def _buffer(state: State, tile: Position, crop: str, start_step=None):
    start_step = state.step if start_step is None else start_step
    reveal = next_reveal(start_step // 24)
    if reveal == 31: return None
    release = reveal * 24
    if start_step // 24 + rules.CROPS[crop].first_yield_day > reveal:
        return None
    return _crop_programme(state, crop, tile, existing=False, start_step=start_step,
                           purpose=f"{crop[0]}_BUFFER", release_step=release)


def _exit_programme(state: State, programme: Programme,
                    asset: AssetProgramme):
    """Replace economic ownership with the bounded official escape tail.

    Tile release is metadata, not an immediate vacancy, and DIG is not committed
    until a later selected use actually needs the released structure removed.
    """
    observed = next((animal for animal in state.own.animals
                     if animal.position == asset.tile and
                     animal.asset_type == asset.asset_type), None)
    if observed is None:
        return programme
    current = exit_current_asset(state, observed)
    replacement = current_asset_programmes((current,))[0]
    assets = tuple(replacement if old.asset_id == asset.asset_id else old
                   for old in programme.assets)
    events = tuple(event for event in programme.events
                   if event.asset_id != asset.asset_id)
    updated_current = tuple(current if old.asset_id == asset.asset_id else old
                            for old in programme.current_assets)
    if not any(old.asset_id == asset.asset_id
               for old in programme.current_assets):
        updated_current = (*updated_current, current)
    return _programme_from_assets(state, assets, events, programme.land,
                                  updated_current)


def _optional_events(state: State, programme: Programme, asset_ids=None):
    result = []
    for asset in programme.assets:
        if asset.decision == "EXIT": continue
        # Existing assets are governed by the lightweight current-state machine.
        # In particular, it emits at most one joint FEED+CARE decision and never
        # enumerates future daily CARE/F collections.
        if asset.existing: continue
        if asset_ids is not None and asset.asset_id not in asset_ids: continue
        if asset.asset_type in rules.ANIMALS:
            feed_days = {e.step//24 for e in asset.service_schedule if e.kind == "FEED"}
            for day in feed_days:
                step = max(state.step, day*24)
                result.append(_pe(step, 21, f"{asset.asset_id}:care:{day}", "CARE", tile=asset.tile,
                    asset=asset.asset_id, action=("CARE",), mandatory=False, deadline=min(day*24+23,718)))
            raw = state.tile_at(asset.tile).raw if asset.existing else {}
            for day in range(state.day, 30):
                if day == state.day and not raw.get("fertilizer_available", False): continue
                step=max(state.step,day*24)
                result.append(_pe(step, 35, f"{asset.asset_id}:collect:{day}", "COLLECT_F", tile=asset.tile,
                    asset=asset.asset_id, item="FERTILIZER", quantity=1,
                    action=("COLLECT_FERTILIZER",), mandatory=False, deadline=min(day*24+23,718)))
        elif asset.asset_type in rules.CROPS:
            rule=rules.CROPS[asset.asset_type]
            if not rule.ongoing:
                planted = int((state.tile_at(asset.tile).raw or {}).get("planted_day", state.day)) if asset.existing else asset.service_schedule[0].step//24
                existing_days={e.step//24 for e in asset.service_schedule if e.kind=="WATER"}
                for day in range(max(state.day, planted+(rule.max_yield_day+1)//2), min(29,planted+rule.max_yield_day)+1):
                    if day not in existing_days:
                        result.append(_pe(max(state.step,day*24), 22, f"{asset.asset_id}:extra-water:{day}", "WATER",
                            tile=asset.tile, asset=asset.asset_id, action=("WATER",), mandatory=False,
                            deadline=min(day*24+23,718)))
            covered = int((state.tile_at(asset.tile).raw or {}).get("fertilized_until_day", -1)) if asset.existing else -1
            for day in range(state.day, 30):
                if day <= covered:
                    continue
                relevant = (not rule.ongoing and any(e.kind == "WATER" and day <= e.step//24 <= day+2 for e in asset.service_schedule)
                            or rule.ongoing and any(day <= e.step//24-1 <= day+2 for e in asset.output_schedule))
                if relevant:
                    result.append(_pe(max(state.step,day*24), 19, f"{asset.asset_id}:fertilize:{day}",
                        "FERTILIZE", tile=asset.tile, asset=asset.asset_id, item="FERTILIZER", quantity=1,
                        action=("FERTILIZE",), mandatory=False, deadline=min(day*24+23,718)))
    return result


def _add_event(state: State, programme: Programme, event: ProgrammeEvent):
    assets=[]
    for asset in programme.assets:
        if asset.asset_id == event.asset_id:
            assets.append(replace(asset,
                service_schedule=tuple(sorted((*asset.service_schedule,event)))))
        else: assets.append(asset)
    events=[*programme.events,event]
    if event.kind=="FERTILIZE":
        available=sum(programme.shed_stock.get(event.step,{}).get("FERTILIZER",0) for _ in (0,))
        if available<=0:
            buy_step=max(programme.formed_step,event.step)
            buy=_pe(buy_step,-16,event.event_id+":buy","BUY_PRODUCT",item="FERTILIZER",quantity=1,source="FERTILIZE")
            events.append(buy)
            if event.step<=buy_step:
                shifted=replace(event,step=buy_step+1)
                events=[shifted if e.event_id==event.event_id else e for e in events]
                assets=[replace(a,service_schedule=tuple(shifted if e.event_id==event.event_id else e for e in a.service_schedule)) if a.asset_id==event.asset_id else a for a in assets]
    assets=tuple(project_asset_transitions(state,asset)
                 if asset.asset_id==event.asset_id else asset for asset in assets)
    projected={asset.asset_id:asset for asset in assets}
    rebuilt=[]
    for old in events:
        if old.asset_id==event.asset_id and old.kind=="HARVEST":
            replacement_event=next((harvest for harvest in projected[event.asset_id].harvest_schedule
                                    if harvest.event_id==old.event_id),old)
            rebuilt.append(replacement_event)
        else:
            rebuilt.append(old)
    return replace(programme, assets=assets,
                   events=tuple(sorted({e.event_id:e for e in rebuilt}.values())))


_OPTIONAL_KINDS={"CARE","WATER","FERTILIZE","COLLECT_F"}


def _without_optional(state: State, programme: Programme, asset_ids=None):
    optional_ids={event.event_id for event in programme.events
                  if not event.mandatory and event.kind in _OPTIONAL_KINDS
                  and (asset_ids is None or getattr(event, "asset_id", None) in asset_ids)}
    if not optional_ids:
        return programme
    assets=[]
    for asset in programme.assets:
        service=tuple(event for event in asset.service_schedule
                      if event.event_id not in optional_ids)
        stripped=replace(asset,service_schedule=service)
        assets.append(stripped if asset.existing else
                      project_asset_transitions(state,stripped))
    events=tuple(event for event in programme.events
                 if event.event_id not in optional_ids and
                 not (event.source=="FERTILIZE" and event.kind=="BUY_PRODUCT"))
    return replace(programme,assets=tuple(assets),events=events,
                   routes=(),planned_sale={},market_inventory={})


def _optimize_optional(state: State, programme: Programme, demand, pressure, asset_ids=None, force=False, finalize=True):
    """Rebuild optional service against the current programme."""
    programme = _without_optional(state, programme, asset_ids)
    candidates = [e for e in _optional_events(state, programme, asset_ids=asset_ids)
                  if all(old.event_id != e.event_id for old in programme.events)]
    
    if not candidates:
        return _finalize(state, programme, demand, pressure, force=force) if finalize else programme
        
    individual_gains = []
    for event in candidates:
        trial_prog = _add_event(state, programme, event)
        cash, feasible, _ = _economic_valuation(state, trial_prog, demand, pressure)
        if feasible:
            gain = cash - programme.terminal_cash
            if gain > 0:
                individual_gains.append((gain, event))
                
    individual_gains.sort(key=lambda x: (-x[0], x[1].event_id))
    
    for _, event in individual_gains:
        trial_prog = _add_event(state, programme, event)
        cash, feasible, _ = _economic_valuation(state, trial_prog, demand, pressure)
        if feasible and cash > programme.terminal_cash:
            programme = replace(trial_prog, terminal_cash=cash)

    return _finalize(state, programme, demand, pressure, force=force) if finalize else programme


def _first_feed_deficit(state: State, programme: Programme):
    arrivals=_arrivals(state,programme)
    available=state.owned_total("WHEAT")
    arrival_steps=sorted((step,amounts.get("WHEAT",0))
                         for step,amounts in arrivals.items()
                         if amounts.get("WHEAT",0))
    buys=sorted(event.step+1 for event in programme.events
                if event.kind=="BUY_PRODUCT" and event.item=="WHEAT")
    cursor=buy_cursor=0
    feeds=sorted(event.deadline if event.deadline is not None else event.step
                 for event in programme.events if event.kind=="FEED")
    for deadline in feeds:
        while cursor<len(arrival_steps) and arrival_steps[cursor][0]<=deadline:
            available+=arrival_steps[cursor][1];cursor+=1
        while buy_cursor<len(buys) and buys[buy_cursor]<=deadline:
            available+=1;buy_cursor+=1
        available-=1
        if available<0:
            return deadline,-available
    return None


_VALUE_CACHE = {}
_SALE_CACHE = {}
_ECONOMIC_CACHE = {}
_FEED_CACHE = {}


def _feed_signature(state: State, programme: Programme, force: bool):
    return _economic_signature(state, programme)


def _strip_auto_feed(programme: Programme):
    return replace(programme,events=tuple(e for e in programme.events if not e.event_id.startswith("feed-wheat:auto:")))


def _resolve_feed(state: State, programme: Programme, demand, pressure, force=False, finalize=True):
    if not programme.feasible:
        return programme
    base=_strip_auto_feed(programme)
    sig = _feed_signature(state, base, force)
    cached = _FEED_CACHE.get(sig)
    if cached is not None:
        if not cached:
            return replace(programme, feasible=False, terminal_cash=-(10**18))
        added_assets, added_events = cached
        restored = _programme_from_assets(state, (*base.assets, *added_assets),
                                          (*base.events, *added_events), base.land,
                                          base.current_assets)
        return _finalize(state, restored, demand, pressure, ensure_feed=False, force=force) if finalize else restored

    if _first_feed_deficit(state, base) is None:
        if finalize:
            res = _finalize(state, base, demand, pressure, ensure_feed=False, force=force)
        else:
            cash, feasible, sales = _economic_valuation(state, base, demand, pressure)
            if not feasible:
                res = replace(base, feasible=False, terminal_cash=-(10**18))
            else:
                res = replace(base, planned_sale=sales.planned_sale, market_inventory=sales.market_inventory, terminal_cash=cash, feasible=True)
        if res.feasible:
            _FEED_CACHE[sig] = ((), ())
        else: _FEED_CACHE[sig] = False
        return res
    memo={}

    def source_key(current):
        return (tuple(sorted(asset.tile for asset in current.assets
                             if asset.purpose=="W_FEED")),
                tuple(sorted(event.step for event in current.events
                             if event.kind=="BUY_PRODUCT" and
                             event.item=="WHEAT")))

    def search(current):
        key=source_key(current)
        if key in memo:
            return memo[key]
        deficit=_first_feed_deficit(state,current)
        if deficit is None:
            cash, feasible, sales = _economic_valuation(state, current, demand, pressure)
            if not feasible:
                result = replace(current, feasible=False, terminal_cash=-(10**18))
            else:
                result = replace(current, planned_sale=sales.planned_sale, market_inventory=sales.market_inventory, terminal_cash=cash, feasible=True)
            memo[key] = result
            return result
        deadline,gap=deficit
        def progresses(next_deficit):
            return (next_deficit is None or next_deficit[0]>deadline or
                    (next_deficit[0]==deadline and next_deficit[1]<gap))
        choices=[]
        ordinal=sum(event.kind=="BUY_PRODUCT" and event.item=="WHEAT"
                    for event in current.events)

        # 1. Candidate BUY step: max(state.step, (deadline // 24) * 24)
        buy_step = max(state.step, (deadline // 24) * 24)
        if buy_step < deadline:
            buy=_pe(buy_step,-15,f"feed-wheat:mixed:{ordinal}:{buy_step}",
                    "BUY_PRODUCT",item="WHEAT",quantity=1,source="W_FEED")
            trial_prog = replace(current, events=tuple(sorted((*current.events, buy))))
            next_deficit=_first_feed_deficit(state, trial_prog)
            if progresses(next_deficit):
                choices.append(search(trial_prog))

        # 2. Candidate PLANT tile
        best_tiles = {}
        for tile in _unused_tiles(state, current):
            tile_state = state.tile_at(tile)
            if tile_state.is_locked: continue
            key = "EMPTY" if tile_state.is_empty else tile_state.kind
            start = _candidate_start(state, current, tile)
            mature = max(start, (start // 24 + rules.CROPS["WHEAT"].first_yield_day) * 24)
            if mature <= deadline:
                dist = rules.distance_to_shed(tile, state.board_size)
                if key not in best_tiles or dist < best_tiles[key][0]:
                    best_tiles[key] = (dist, tile, start, mature)
        
        valid_tiles = list(best_tiles.values())

        for _, tile, start, mature in valid_tiles:
            wheat=_crop_programme(state,"WHEAT",tile,existing=False,
                                  start_step=start,purpose="W_FEED",
                                  release_step=mature)
            trial_prog = _programme_from_assets(state, (*current.assets, wheat),
                                                current.events, current.land,
                                                current.current_assets)
            next_deficit=_first_feed_deficit(state, trial_prog)
            if progresses(next_deficit):
                choices.append(search(trial_prog))

        feasible=[choice for choice in choices if choice.feasible]
        result=(max(feasible,key=lambda choice:choice.terminal_cash)
                if feasible else replace(current,feasible=False,
                                         terminal_cash=-(10**18),
                                         diagnostics={"failure":
                                             f"WHEAT unavailable by feed deadline {deadline}"}))
        memo[key]=result
        return result

    best_econ = search(base)
    if not best_econ.feasible:
        _FEED_CACHE[sig] = False
        return best_econ
    
    base_asset_ids = {a.asset_id for a in base.assets}
    base_event_ids = {e.event_id for e in base.events}
    added_assets = tuple(a for a in best_econ.assets if a.asset_id not in base_asset_ids)
    added_events = tuple(e for e in best_econ.events if e.event_id not in base_event_ids)
    _FEED_CACHE[sig] = (added_assets, added_events)
    return _finalize(state, best_econ, demand, pressure, ensure_feed=False, force=force) if finalize else best_econ


def _unused_tiles(state: State, programme: Programme):
    occupied={a.tile for a in programme.assets
              if a.decision != "EXIT" and a.release_turn is None}
    unlocked=set(state.own.owned_land)|{land.quadrant for land in programme.land}
    return tuple(t.position for t in state.tiles if t.position not in occupied and
                 rules.quadrant(t.position,state.board_size) in unlocked and
                 (t.is_locked or t.is_empty or t.kind=="WEED" or
                  t.kind in {"COOP","PASTURE"} and t.animal is None))


def _candidate_start(state: State, programme: Programme, tile: Position):
    releases = [asset.release_turn for asset in programme.assets
                if asset.tile == tile and asset.release_turn is not None]
    release_start = max(releases, default=state.step - 1) + 1
    if not state.tile_at(tile).is_locked:
        return max(state.step, release_start)
    buy=min((land.buy_step for land in programme.land
             if land.quadrant==rules.quadrant(tile,state.board_size)),
            default=state.step)
    return max(buy + 1, release_start)


def _tile_legal_for_kind(state: State, programme: Programme, tile: Position,
                         kind: str) -> bool:
    tile_state = state.tile_at(tile)
    bought = any(land.quadrant == rules.quadrant(tile, state.board_size)
                 for land in programme.land)
    if tile_state.is_locked:
        return bought
    releasing = next((asset for asset in programme.assets
                      if asset.tile == tile and asset.release_turn is not None),
                     None)
    if releasing is not None:
        # An escaped animal leaves its structure. It is immediately reusable by
        # the matching species; other uses may DIG only when actually selected.
        if kind in rules.ANIMALS and releasing.asset_type in rules.ANIMALS:
            return rules.ANIMALS[kind].structure == rules.ANIMALS[
                releasing.asset_type].structure
        return True
    if kind in rules.ANIMALS:
        return (tile_state.is_empty or tile_state.kind == "WEED" or
                tile_state.kind == rules.ANIMALS[kind].structure)
    return tile_state.is_empty or tile_state.kind == "WEED"


def _representative_tile(state: State, tiles: Iterable[Position]) -> Position | None:
    return min(tiles, key=lambda tile: (rules.distance_to_shed(
        tile, state.board_size), tile[1], tile[0]), default=None)


def _candidate_can_start_today(state: State, kind: str, tile: Position,
                               candidate: AssetProgramme, land_price: int) -> bool:
    asset_cost = (rules.ANIMALS[kind].cost if kind in rules.ANIMALS else
                  rules.CROPS[kind].seed_cost)
    if state.money < land_price + asset_cost:
        return False
    day_end = min((state.day + 1) * 24 - 1, rules.TERMINAL_ACTION_STEP)
    field_actions = len(candidate.service_schedule)
    if kind in rules.ANIMALS:
        access = rules.shed_access(state.board_size)
        lower_bound = min(
            rules.manhattan(worker.position, shed) + 1 +
            rules.manhattan(shed, tile) + field_actions
            for worker in state.workers for shed in access
        ) if state.workers else rules.TERMINAL_ACTION_STEP
    else:
        lower_bound = min((rules.manhattan(worker.position, tile) + field_actions
                           for worker in state.workers),
                          default=rules.TERMINAL_ACTION_STEP)
    return state.step + lower_bound <= day_end


def _blocked_long_kinds(state: State, programme: Programme) -> tuple[str, ...]:
    usable = tuple(tile for tile in _unused_tiles(state, programme)
                   if not state.tile_at(tile).is_locked and
                   _candidate_start(state, programme, tile) <= state.step)
    return tuple(kind for kind in LONG_ASSETS
                 if not any(_tile_legal_for_kind(state, programme, tile, kind)
                            for tile in usable))


def _maybe_buy_land_before_expansion(state: State, programme: Programme,
                                     demand, pressure) -> Programme:
    """Buy only for a positive long candidate blocked by current legal tiles.

    This probes one deterministic representative tile in the next quadrant and
    compares starting now with the earliest physical release. It never runs a
    25-tile post-purchase optimizer.
    """
    if len(state.own.owned_land) >= 4:
        return programme
    blocked = _blocked_long_kinds(state, programme)
    if not blocked:
        return programme
    land_index = len(state.own.owned_land) - 1
    quadrant = rules.LAND_ORDER[land_index]
    price = rules.LAND_PRICES[land_index]
    if state.money < price:
        return programme
    locked = tuple(tile.position for tile in state.tiles
                   if tile.is_locked and
                   rules.quadrant(tile.position, state.board_size) == quadrant)
    tile = _representative_tile(state, locked)
    if tile is None:
        return programme

    base_cash, base_feasible, _ = _economic_valuation(
        state, programme, demand, pressure)
    if not base_feasible:
        return programme
    releases = sorted((asset.release_turn, asset.tile)
                      for asset in programme.assets
                      if asset.release_turn is not None and
                      asset.release_turn <= rules.TERMINAL_ACTION_STEP)
    probes = []
    for kind in blocked:
        start = state.step + 1
        now_asset = (_animal_programme(state, kind, tile, existing=False,
                                       start_step=start)
                     if kind in rules.ANIMALS else
                     _crop_programme(state, kind, tile, existing=False,
                                     start_step=start))
        if not _candidate_can_start_today(state, kind, tile, now_asset, price):
            continue
        now = _programme_from_assets(state, (*programme.assets, now_asset),
                                     programme.events, programme.land,
                                     programme.current_assets)
        now_cash, now_feasible, _ = _economic_valuation(state, now, demand,
                                                        pressure)
        if not now_feasible or now_cash <= base_cash:
            continue
        delayed_cash = base_cash
        compatible_release = next(((step, old_tile) for step, old_tile in releases
                                   if _tile_legal_for_kind(state, programme,
                                                           old_tile, kind)), None)
        if compatible_release is not None:
            release_step, old_tile = compatible_release
            delayed_asset = (_animal_programme(
                state, kind, old_tile, existing=False,
                start_step=min(release_step + 1, rules.TERMINAL_ACTION_STEP))
                if kind in rules.ANIMALS else _crop_programme(
                    state, kind, old_tile, existing=False,
                    start_step=min(release_step + 1,
                                   rules.TERMINAL_ACTION_STEP)))
            delayed = _programme_from_assets(
                state, (*programme.assets, delayed_asset), programme.events,
                programme.land, programme.current_assets)
            value, feasible, _ = _economic_valuation(state, delayed, demand,
                                                      pressure)
            if feasible:
                delayed_cash = value
        advance_gain = now_cash - delayed_cash
        if advance_gain > price:
            probes.append((advance_gain, -rules.distance_to_shed(
                tile, state.board_size), kind, tile))
    if not probes:
        return programme
    advance_gain, _, kind, use_tile = max(probes)
    land = LandProgramme(quadrant, state.step, price, use_tile,
                         f"UNBLOCK:{kind}")
    event = _pe(state.step, -30, f"land:{quadrant}", "BUY_LAND")
    diagnostics = dict(programme.diagnostics)
    diagnostics["land_probe"] = {
        "quadrant": quadrant, "blocked_asset": kind,
        "representative_tile": use_tile, "advance_gain": advance_gain,
        "price": price,
    }
    bought = _programme_from_assets(
        state, programme.assets, (*programme.events, event),
        (*programme.land, land), programme.current_assets)
    return replace(bought, diagnostics=diagnostics,
                   terminal_cash=programme.terminal_cash,
                   feasible=programme.feasible)


def _purchased_land_used_today(state: State, programme: Programme,
                               original_land_count: int) -> bool:
    new_quadrants = {land.quadrant for land in
                     programme.land[original_land_count:]}
    if not new_quadrants:
        return True
    day_end = min((state.day + 1) * 24 - 1, rules.TERMINAL_ACTION_STEP)
    return any(
        asset.decision == "NEW" and
        rules.quadrant(asset.tile, state.board_size) in new_quadrants and
        any(event.kind in {"PLACE", "PLANT"} and event.step <= day_end
            for event in asset.service_schedule)
        for asset in programme.assets
    )


def _long_candidate_loop(state: State, programme: Programme, demand, pressure,
                         params=None):
    """Run the exact-tile greedy loop for the currently usable land."""
    params = coerce_midgame_parameters(params)
    import time
    loop_start = time.perf_counter()
    iter_count = 0
    while True:
        iter_count += 1
        ranked = []
        unused = _unused_tiles(state, programme)
        tiles_to_check = {}
        for tile in unused:
            tile_state = state.tile_at(tile)
            if tile_state.is_locked and not any(
                    land.quadrant == rules.quadrant(tile, state.board_size)
                    for land in programme.land):
                continue
            key = "EMPTY" if tile_state.is_empty or tile_state.is_locked else tile_state.kind
            dist = rules.distance_to_shed(tile, state.board_size)
            if key not in tiles_to_check or dist < tiles_to_check[key][0]:
                tiles_to_check[key] = (dist, tile)
                
        for idx, tile in enumerate(t for _, t in tiles_to_check.values()):
            t_tile = time.perf_counter()
            start = _candidate_start(state, programme, tile)
            baselines = [programme]
            for crop in ("WHEAT", "CARROT"):
                buffer = _buffer(state, tile, crop, start)
                if buffer:
                    trial_b = _programme_from_assets(state, (*programme.assets, buffer),
                                                     programme.events, programme.land,
                                                     programme.current_assets)
                    now_b = _resolve_feed(state, trial_b, demand, pressure)
                    buffer_dependent = {buffer.asset_id}
                    if buffer.asset_type in rules.CROPS:
                        buffer_dependent.update(a.asset_id for a in now_b.assets if a.asset_type in rules.CROPS)
                    now_b = _optimize_optional(state, now_b, demand, pressure, asset_ids=buffer_dependent, finalize=False)
                    if now_b.feasible:
                        baselines.append(now_b)
            baseline = max(enumerate(baselines), key=lambda value: (value[1].terminal_cash, -value[0]))[1]
            for kind in LONG_ASSETS:
                tile_state = state.tile_at(tile)
                if not tile_state.is_locked:
                    if kind in rules.ANIMALS and not (
                            tile_state.is_empty or tile_state.kind == "WEED" or
                            tile_state.kind == rules.ANIMALS[kind].structure):
                        continue
                    if kind in rules.CROPS and not (
                            tile_state.is_empty or tile_state.kind == "WEED"):
                        continue
                candidate = (_animal_programme(state, kind, tile, existing=False, start_step=start)
                             if kind in rules.ANIMALS else
                             _crop_programme(state, kind, tile, existing=False, start_step=start))
                trial = _programme_from_assets(state, (*programme.assets, candidate),
                                               programme.events, programme.land,
                                               programme.current_assets)
                now = _resolve_feed(state, trial, demand, pressure)
                if not now.feasible:
                    continue
                dependent = {candidate.asset_id}
                if (candidate.asset_type in rules.ANIMALS and rules.ANIMALS[candidate.asset_type].product == "FERTILIZER") or candidate.asset_type in rules.CROPS:
                    dependent.update(a.asset_id for a in now.assets if a.asset_type in rules.CROPS)
                now = _optimize_optional(state, now, demand, pressure, asset_ids=dependent, finalize=False)
                if not now.feasible:
                    continue
                advantage = now.terminal_cash - baseline.terminal_cash
                if advantage <= 0:
                    continue
                reveal = next_reveal(start // 24)
                wait_value = baseline.terminal_cash
                if reveal < 31:
                    delayed = (_animal_programme(state, kind, tile, existing=False, start_step=reveal * 24 + 1)
                               if kind in rules.ANIMALS else
                               _crop_programme(state, kind, tile, existing=False, start_step=reveal * 24 + 1))
                    wait = _programme_from_assets(state, (*baseline.assets, delayed),
                                                  baseline.events, baseline.land,
                                                  baseline.current_assets)
                    wait = _resolve_feed(state, wait, demand, pressure)
                    delayed_dependent = {delayed.asset_id}
                    if (delayed.asset_type in rules.ANIMALS and rules.ANIMALS[delayed.asset_type].product == "FERTILIZER") or delayed.asset_type in rules.CROPS:
                        delayed_dependent.update(a.asset_id for a in wait.assets if a.asset_type in rules.CROPS)
                    wait = _optimize_optional(state, wait, demand, pressure, asset_ids=delayed_dependent, finalize=False)
                    if wait.feasible:
                        wait_value = wait.terminal_cash
                if now.terminal_cash <= wait_value:
                    continue
                locality = animal_locality_bonus(
                    kind, tile, state.day, state.board_size, params)
                placement_score = advantage * (1.0 + locality)
                key = (placement_score, advantage,
                       -rules.distance_to_shed(tile, state.board_size),
                       -tile[1], -tile[0], kind)
                ranked.append((key, now))
        if not ranked:
            return programme
        committed = None
        for _, candidate in sorted(ranked, key=lambda value: value[0], reverse=True):
            checked = _finalize(state, candidate, demand, pressure, force=True)
            if checked.feasible:
                committed = checked
                break
        if committed is None:
            return programme
        programme = _optimize_optional(state, committed, demand, pressure)


def _fill_buffers(state: State, programme: Programme, demand, pressure):
    for tile in _unused_tiles(state,programme):
        start=_candidate_start(state,programme,tile)
        options=[programme]
        for crop in ("WHEAT","CARROT"):
            asset=_buffer(state,tile,crop,start)
            if asset:
                options.append(_finalize(state,_programme_from_assets(
                    state,(*programme.assets,asset),programme.events,programme.land,
                    programme.current_assets),
                    demand,pressure))
        programme=max(enumerate(options),
                      key=lambda value:(value[1].terminal_cash,-value[0]))[1]
    return programme


def make_plan(
    state: State,
    config=None,
    *,
    prior_programme: Programme | None = None,
    animal_inputs: AnimalDecisionInputs | None = None,
    replacement_advantages: Mapping[Position, int] | None = None,
    base_incremental_hire_costs: Mapping[str, int] | None = None,
    care_incremental_hire_costs: Mapping[str, int] | None = None,
    liquidation_incremental_hire_costs: Mapping[str, int] | None = None,
    **_ignored,
) -> Programme:
    """Build a midgame programme directly from any legal current observation."""
    params = coerce_midgame_parameters(config)
    prior = prior_programme
    if animal_inputs is None:
        animal_inputs = AnimalDecisionInputs(
            replacement_advantages=replacement_advantages,
            base_incremental_hire_costs=base_incremental_hire_costs,
            care_incremental_hire_costs=care_incremental_hire_costs,
            liquidation_incremental_hire_costs=(
                liquidation_incremental_hire_costs))
    _VALUE_CACHE.clear()
    _SALE_CACHE.clear()
    _ECONOMIC_CACHE.clear()
    _FEED_CACHE.clear()
    clear_route_cache()
    clear_sale_cache()
    demand=known_demand_events(state); pressure=opponent_pressure(state)
    existing, current = _initial_assets(state, prior, animal_inputs)
    programme=_finalize(
        state,
        _programme_from_assets(state, existing, current_assets=current),
        demand, pressure, force=True)

    # Existing animals already carry their frozen MAINTAIN/EXIT and care-cycle
    # decisions. Optional search now applies only to newly committed assets.
    programme=_optimize_optional(state,programme,demand,pressure)

    # Feed wheat is segregated before ordinary buffers.
    programme=_finalize(state,_resolve_feed(state,programme,demand,pressure),demand,pressure,force=True)

    # Land is decided before any long placement commits. An accepted quadrant
    # and all existing tiles then enter exactly one shared placement loop.
    before_land = programme
    programme=_maybe_buy_land_before_expansion(state, programme, demand,
                                               pressure)
    land_count = len(before_land.land)
    programme=_long_candidate_loop(state,programme,demand,pressure,params)
    if not _purchased_land_used_today(state, programme, land_count):
        # The probe justified an immediate use, but earlier commits changed its
        # marginal value. Do not keep an unused purchase; expand once without it.
        programme=_long_candidate_loop(state,before_land,demand,pressure,params)
    programme=_fill_buffers(state,programme,demand,pressure)

    wheat_feed=defaultdict(list); wheat_buffer=defaultdict(list); carrot_buffer=defaultdict(list)
    for asset in programme.assets:
        target=(wheat_feed if asset.purpose=="W_FEED" else wheat_buffer if asset.purpose=="W_BUFFER" else carrot_buffer if asset.purpose=="C_BUFFER" else None)
        if target is not None: target[asset.tile].extend(e.step for e in asset.harvest_schedule)
    produced,bought,used,sold=defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)
    for event in programme.events:
        if event.kind=="COLLECT_F": produced[event.step]+=event.quantity
        elif event.kind=="BUY_PRODUCT" and event.item=="FERTILIZER": bought[event.step]+=event.quantity
        elif event.kind=="FERTILIZE": used[event.step]+=event.quantity
    for step,amounts in programme.planned_sale.items(): sold[step]+=amounts.get("FERTILIZER",0)
    programme=_finalize(state,programme,demand,pressure,force=True)
    return replace(programme,wheat_feed={k:tuple(v) for k,v in wheat_feed.items()},
        wheat_buffer={k:tuple(v) for k,v in wheat_buffer.items()},carrot_buffer={k:tuple(v) for k,v in carrot_buffer.items()},
        fertilizer=FertilizerProgramme(dict(produced),dict(bought),dict(used),dict(sold)))
