"""D11+ macro economic planner specified by programme-value comparisons."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable, Mapping

from . import rules
from .intraday import PlanningFailure, solve_intraday
from .market import clear_sale_cache, known_demand_events, next_reveal, opponent_pressure, optimize_sales
from .programme import AssetProgramme, FertilizerProgramme, LandProgramme, Programme, ProgrammeEvent
from .simulation import cash_projection, simulate_programme
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
    start_step = state.step if start_step is None else start_step
    start_day = start_step // 24
    identifier = f"{'existing' if existing else 'new'}:{asset_type}:{tile[0]}:{tile[1]}"
    rule = rules.ANIMALS[asset_type]
    raw = dict(raw or {"kind": rule.structure, "animal": asset_type,
        "placed_day": start_day, "yield_units": 0, "consecutive_unfed": 0,
        "fed_today": False, "cared_today": False, "fertilizer_available": False,
        "pending_care_bonus": 0})
    service, output, harvest = [], [], []
    if not existing:
        opening=state.tile_at(tile) if start_step==state.step else None
        cursor=start_step
        if opening is not None and opening.kind=="WEED":
            service.append(_pe(cursor,9,identifier+":dig","DIG",tile=tile,asset=identifier,
                               action=("DIG",),deadline=min(start_day*24+21,718)));cursor+=1
        if opening is None or opening.kind!=rule.structure:
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
    if not bool(raw.get("fertilizer_available",False)) and (existing or start_day < 30):
        step=min((start_day+1)*24,718)
        output.append(_pe(step,0,f"{identifier}:output:F:{start_day+1}","OUTPUT",tile=tile,
            asset=identifier,item="FERTILIZER",quantity=1,mandatory=False))
    return AssetProgramme(identifier, asset_type, tile, "KEEP" if existing else "NEW", existing,
                          purpose, None, tuple(service), tuple(output), tuple(harvest))


def _crop_programme(state: State, crop: str, tile: Position, *, existing: bool,
                    raw=None, start_step=None, purpose="LONG", release_step=None):
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
    if not existing:
        cursor=start_step+1
        if start_step==state.step and state.tile_at(tile).kind=="WEED":
            service.append(_pe(cursor,9,identifier+":dig","DIG",tile=tile,asset=identifier,
                action=("DIG",),deadline=min(start_day*24+20,718)));cursor+=1
        service.extend([
            _pe(cursor, 10, identifier+":plant", "PLANT", tile=tile, asset=identifier,
                item=crop, quantity=1, action=("PLANT", crop), deadline=min(start_day*24+21,718), source=purpose),
            _pe(cursor+1, 11, identifier+":water:plant", "WATER", tile=tile, asset=identifier,
                action=("WATER",), deadline=min(start_day*24+22,718), source=purpose),
        ])
    water_days = _minimal_service_days(raw, start_day, 29, animal=False)
    if not existing:
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
        if held and state.day - planted >= rule.first_yield_day:
            harvest.append(_pe(start_step, 30, identifier+":harvest:opening", "HARVEST", tile=tile,
                asset=identifier, item=crop, quantity=held, action=("HARVEST",), deadline=min(start_day*24+23,718)))
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
    return AssetProgramme(identifier, crop, tile, "KEEP" if existing else "NEW", existing,
                          purpose, release_step, tuple(service), tuple(output), tuple(harvest))


def _purchase_event(state: State, asset: AssetProgramme, step: int):
    if asset.asset_type in rules.ANIMALS:
        return _pe(step, -20, asset.asset_id+":buy", "BUY_ANIMAL", asset=asset.asset_id,
                   item=asset.asset_type, quantity=1)
    return _pe(step, -20, asset.asset_id+":buy", "BUY_SEED", asset=asset.asset_id,
               item=asset.asset_type, quantity=1)


def _programme_from_assets(state: State, assets: Iterable[AssetProgramme], extra_events=(), land=()):
    assets = tuple(assets)
    events = [*extra_events]
    for asset in assets:
        events.extend(asset.service_schedule); events.extend(asset.harvest_schedule)
        if not asset.existing:
            start = min((e.step for e in asset.service_schedule), default=state.step)
            events.append(_purchase_event(state, asset, max(state.step, start-1)))
    unique = {event.event_id: event for event in events}
    return Programme(state.step, state.day, state.shops, assets, tuple(sorted(unique.values())), land=tuple(land))


def _arrivals(state: State, programme: Programme):
    arrivals = defaultdict(lambda: defaultdict(int))
    # Carried goods reach the shed at the next daily close if no earlier route drop exists.
    close = min((state.day + 1) * 24, 718)
    for worker in state.workers:
        for item, quantity in worker.inventory.items(): arrivals[close][item] += quantity
    for asset in programme.assets:
        if asset.purpose=="W_FEED":
            continue
        for event in asset.harvest_schedule:
            arrival = min((event.step // 24 + 1) * 24, 718)
            arrivals[arrival][event.item] += event.quantity
        for event in asset.service_schedule:
            if event.action and event.action[0] == "COLLECT_FERTILIZER":
                arrivals[min((event.step//24+1)*24,718)]["FERTILIZER"] += 1
    return {step: dict(amounts) for step, amounts in arrivals.items()}


_VALUE_CACHE={}
_SALE_CACHE={}


def _economic_signature(state: State, programme: Programme):
    assets=tuple(sorted((a.asset_type,a.decision,a.existing,a.purpose,a.release_turn,
        tuple((e.step,e.kind,e.item,e.quantity,e.action,e.mandatory,e.deadline) for e in a.service_schedule if e.kind not in {"PICKUP","DROP"}),
        tuple((e.step,e.item,e.quantity) for e in a.output_schedule),
        tuple((e.step,e.item,e.quantity) for e in a.harvest_schedule)) for a in programme.assets))
    events=tuple(sorted(((e.step,e.kind,e.item,e.quantity,e.source) for e in programme.events
                        if e.kind not in {"HIRE"} and e.kind not in {"PICKUP","DROP"}), key=repr))
    sales=tuple((step,tuple(sorted(amounts.items()))) for step,amounts in sorted(programme.planned_sale.items()))
    return (state.step,state.money,tuple(sorted(state.shed.items())),tuple(sorted(state.seeds.items())),
            tuple(sorted(state.market.inventory.items())),assets,events,sales,programme.worker_count,
            tuple((land.quadrant,land.buy_step,land.cost,land.use) for land in programme.land))


def _finalize(state: State, programme: Programme, demand, pressure, *, force=False):
    programme = _ensure_feed_supply(state, programme)
    arrivals=_arrivals(state,programme)
    sale_key=(tuple((step,tuple(sorted(amounts.items()))) for step,amounts in sorted(arrivals.items())),
              tuple(sorted(state.shed.items())),tuple(sorted(state.market.inventory.items())))
    sales=_SALE_CACHE.get(sale_key)
    if sales is None:
        sales=optimize_sales(state,arrivals,demand,pressure); _SALE_CACHE[sale_key]=sales
    programme = replace(programme, planned_sale=sales.planned_sale,
                        market_inventory=sales.market_inventory)
    programme=_finance_sales(state,programme,arrivals,pressure)
    try:
        programme = solve_intraday(state, programme, complete_actions=force)
    except PlanningFailure as exc:
        return replace(programme, feasible=False, terminal_cash=-(10**18), diagnostics={"failure":str(exc)})
    signature=_economic_signature(state,programme)
    if not force and signature in _VALUE_CACHE:
        cash,feasible,failure=_VALUE_CACHE[signature]
        return replace(_attach_asset_flows(programme),terminal_cash=cash,feasible=feasible,
                       diagnostics={} if feasible else {"failure":failure})
    if not force:
        cash,feasible=cash_projection(state,programme,pressure)
        _VALUE_CACHE[signature]=(cash,feasible,None if feasible else "cash/market infeasible")
        return replace(_attach_asset_flows(programme),terminal_cash=cash,feasible=feasible,
                       diagnostics={} if feasible else {"failure":"cash/market infeasible"})
    result = simulate_programme(state, programme, pressure_events=pressure)
    projected_cash,projected_feasible=cash_projection(state,programme,pressure)
    if not projected_feasible or result.terminal_cash!=projected_cash:
        result=replace(result,feasible=False,
                       failure=result.failure or "physical replay did not realize projected cash")
    _VALUE_CACHE[signature]=(result.terminal_cash,result.feasible,result.failure)
    programme = _attach_asset_flows(programme)
    return replace(programme, terminal_cash=result.terminal_cash, feasible=result.feasible,
        farm_output=result.farm_output, field_stock=result.field_stock,
        worker_stock=result.worker_stock, shed_stock=result.shed_stock,
        market_inventory=result.market_inventory,
        diagnostics={} if result.feasible else {"failure":result.failure})


def _finance_sales(state: State, programme: Programme, arrivals, pressure):
    """Move only the minimum already-planned stock needed to finance commitments."""
    for _ in range(rules.SHED_CAPACITY+1):
        _,feasible,step=cash_projection(state,programme,pressure,details=True)
        if feasible:return programme
        available={p:state.shed.get(p,0)+sum(a.get(p,0) for s,a in arrivals.items() if s<=step)
                   -sum(a.get(p,0) for s,a in programme.planned_sale.items() if s<=step)
                   for p in rules.PRODUCTS}
        choices=[]
        for product,stock in available.items():
            if stock<=0:continue
            future=next((s for s in sorted(programme.planned_sale) if s>step and programme.planned_sale[s].get(product,0)>0),None)
            if future is None:continue
            now_inventory=programme.market_inventory.get(step,state.market.inventory).get(product,state.market.inventory[product])
            future_inventory=programme.market_inventory.get(future,state.market.inventory).get(product,state.market.inventory[product])
            loss=rules.market_price(product,future_inventory)-rules.market_price(product,now_inventory)
            choices.append((loss,product,future))
        if not choices:return programme
        _,product,future=min(choices)
        sales={s:dict(a) for s,a in programme.planned_sale.items()}
        sales[future][product]-=1
        if not sales[future][product]:del sales[future][product]
        sales.setdefault(step,{})[product]=sales.get(step,{}).get(product,0)+1
        programme=replace(programme,planned_sale={s:a for s,a in sales.items() if a})
    return programme


def _attach_asset_flows(programme: Programme):
    queues=defaultdict(list)
    assets=[]
    for asset in programme.assets:
        core_service=tuple(e for e in asset.service_schedule if e.kind not in {"PICKUP","DROP"})
        stock=[]
        logistics=[]
        for event in asset.harvest_schedule:
            arrival=min((event.step//24+1)*24,718)
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


def _initial_assets(state: State):
    assets = []
    for animal in state.own.animals:
        assets.append(_animal_programme(state, animal.asset_type, animal.position,
                                       existing=True, raw=animal.official))
    for crop in state.own.crops:
        assets.append(_crop_programme(state, crop.asset_type, crop.position,
                                     existing=True, raw=crop.official))
    return assets


def _buffer(state: State, tile: Position, crop: str, start_step=None):
    start_step = state.step if start_step is None else start_step
    reveal = next_reveal(start_step // 24)
    if reveal == 31: return None
    release = reveal * 24
    if start_step // 24 + rules.CROPS[crop].first_yield_day > reveal:
        return None
    return _crop_programme(state, crop, tile, existing=False, start_step=start_step,
                           purpose=f"{crop[0]}_BUFFER", release_step=release)


def _exit_programme(state: State, programme: Programme, asset: AssetProgramme):
    raw = state.tile_at(asset.tile).raw
    escape_days = 2 if bool(raw.get("fed_today", False)) or int(raw.get("consecutive_unfed",0)) == 0 else 1
    free_step = min((state.day + escape_days) * 24, 718)
    replacement = replace(asset, decision="EXIT", release_turn=free_step,
                          service_schedule=(), output_schedule=(), harvest_schedule=(), sale_schedule=())
    events = [event for event in programme.events if event.asset_id != asset.asset_id]
    events.append(_pe(free_step, 40, asset.asset_id+":dig", "DIG", tile=asset.tile,
                      asset=asset.asset_id, action=("DIG",), deadline=min(free_step//24*24+23,718)))
    assets = [replacement if a.asset_id == asset.asset_id else a for a in programme.assets]
    buffer = _buffer(state, asset.tile, "WHEAT", free_step+1)
    if buffer is not None:
        assets.append(buffer)
    return _programme_from_assets(state, assets, events)


def _optional_events(state: State, programme: Programme):
    result = []
    for asset in programme.assets:
        if asset.decision == "EXIT": continue
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


def _add_event(programme: Programme, event: ProgrammeEvent):
    assets=[]
    for asset in programme.assets:
        if asset.asset_id == event.asset_id:
            output=list(asset.output_schedule); harvest=list(asset.harvest_schedule)
            if event.kind == "CARE":
                care_day=event.step//24
                feed_days={e.step//24 for e in asset.service_schedule if e.kind=="FEED"}
                for index,out in enumerate(output):
                    production_day=out.step//24-1
                    if production_day>care_day and production_day in feed_days:
                        output[index]=replace(out,quantity=out.quantity+1)
                        harvest=[replace(h,quantity=h.quantity+1) if h.step==out.step else h for h in harvest]
                        break
            elif event.kind == "WATER" and asset.asset_type in rules.CROPS and not rules.CROPS[asset.asset_type].ongoing:
                harvest=[replace(h,quantity=min(rules.CROPS[asset.asset_type].max_yield,h.quantity+1)) for h in harvest]
            elif event.kind == "FERTILIZE" and asset.asset_type in rules.CROPS:
                day=event.step//24; rule=rules.CROPS[asset.asset_type]
                if rule.ongoing:
                    water_days={e.step//24 for e in asset.service_schedule if e.kind=="WATER"}
                    changed=set()
                    for index,out in enumerate(output):
                        production_day=out.step//24-1
                        if day<=production_day<=day+2 and production_day in water_days:
                            output[index]=replace(out,quantity=min(2,out.quantity+1)); changed.add(out.step)
                    harvest=[replace(h,quantity=h.quantity+1) if h.step in changed else h for h in harvest]
                else:
                    water_days=sum(day<=e.step//24<=day+2 for e in asset.service_schedule if e.kind=="WATER")
                    harvest=[replace(h,quantity=min(rule.max_yield,h.quantity+water_days)) for h in harvest]
            elif event.kind == "COLLECT_F":
                step=min((event.step//24+1)*24,718)
                if step>event.step:
                    output.append(_pe(step,0,event.event_id+":next-output","OUTPUT",tile=asset.tile,
                        asset=asset.asset_id,item="FERTILIZER",quantity=1,mandatory=False))
            assets.append(replace(asset, service_schedule=tuple(sorted((*asset.service_schedule,event))),
                                  output_schedule=tuple(output),harvest_schedule=tuple(harvest)))
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
    return replace(programme, assets=tuple(assets), events=tuple(sorted({e.event_id:e for e in events}.values())))


def _feed_gap(state: State, programme: Programme):
    deadline_day=next_reveal(state.day)
    required=sum(1 for e in programme.events if e.kind=="FEED" and e.step//24 < deadline_day)
    available=state.owned_total("WHEAT")
    for asset in programme.assets:
        if asset.asset_type=="WHEAT":
            available += sum(e.quantity for e in asset.harvest_schedule if e.step//24 < deadline_day)
    return max(0, required-available)


def _strip_auto_feed(programme: Programme):
    return replace(programme,events=tuple(e for e in programme.events if not e.event_id.startswith("feed-wheat:auto:")))


def _resolve_feed(state: State, programme: Programme, demand, pressure):
    base=_strip_auto_feed(programme)
    gap=_feed_gap(state,base)
    if gap<=0: return _finalize(state,base,demand,pressure)
    buy=_finalize(state,base,demand,pressure)
    order=list(_unused_tiles(state,base))
    inner=[p for p in order if p in programme.inner]
    outer=[p for p in order if p not in programme.inner]
    order=sorted(inner,key=lambda p:(-rules.distance_to_shed(p,state.board_size),p[1],p[0]))+sorted(
        outer,key=lambda p:(rules.distance_to_shed(p,state.board_size),p[1],p[0]))
    planted=[]; supplied=0
    for tile in order:
        wheat=_buffer(state,tile,"WHEAT")
        if wheat is None: continue
        wheat=replace(wheat,purpose="W_FEED"); planted.append(wheat)
        supplied+=sum(e.quantity for e in wheat.harvest_schedule)
        if supplied>=gap: break
    if supplied<gap: return buy
    plant=_finalize(state,_programme_from_assets(state,(*base.assets,*planted),base.events),demand,pressure)
    return max((buy,plant),key=lambda p:p.terminal_cash)


def _unused_tiles(state: State, programme: Programme):
    occupied={a.tile for a in programme.assets if a.decision != "EXIT"}
    return tuple(t.position for t in state.tiles if t.position not in occupied and
                 (t.is_empty or t.kind=="WEED" or t.kind in {"COOP","PASTURE"} and t.animal is None))


def make_plan(state: State, config=None, **_ignored) -> Programme:
    """Run the prescribed greedy programme construction from the real state."""
    del config
    _VALUE_CACHE.clear()
    _SALE_CACHE.clear()
    clear_sale_cache()
    demand=known_demand_events(state); pressure=opponent_pressure(state)
    programme=_finalize(state, _programme_from_assets(state,_initial_assets(state)), demand, pressure, force=True)

    # Existing animals start KEEP; accept one best positive EXIT and recompute.
    while True:
        choices=[]
        for asset in programme.assets:
            if asset.existing and asset.asset_type in rules.ANIMALS and asset.decision=="KEEP":
                trial=_finalize(state,_exit_programme(state,programme,asset),demand,pressure)
                choices.append((trial.terminal_cash-programme.terminal_cash,asset.tile,trial))
        if not choices: break
        gain,_,trial=max(choices,key=lambda x:(x[0],-x[1][1],-x[1][0]))
        if gain<=0: break
        committed=_finalize(state,trial,demand,pressure,force=True)
        if not committed.feasible: break
        programme=committed

    # Optional service: accept only the largest positive marginal event, then repeat.
    accepted_optional=set()
    while True:
        remaining={e.event_id:e for e in _optional_events(state,programme)
                   if e.event_id not in accepted_optional and all(old.event_id != e.event_id for old in programme.events)}
        if not remaining:
            break
        choices=[]
        for event in remaining.values():
            trial=_finalize(state,_add_event(programme,event),demand,pressure)
            choices.append((trial.terminal_cash-programme.terminal_cash,event.event_id,trial))
        gain,event_id,trial=max(choices,key=lambda x:(x[0],x[1]))
        if gain<=0: break
        committed=_finalize(state,trial,demand,pressure,force=True); accepted_optional.add(event_id)
        if committed.feasible: programme=committed

    # Feed wheat is segregated before ordinary buffers.
    programme=_finalize(state,_resolve_feed(state,programme,demand,pressure),demand,pressure,force=True)

    # Long assets are exact (type,tile) candidates with same-tile buffer baseline.
    while True:
        ranked=[]
        for tile in _unused_tiles(state,programme):
            baselines=[programme]
            for crop in ("WHEAT","CARROT"):
                buffer=_buffer(state,tile,crop)
                if buffer:
                    baselines.append(_finalize(state,_programme_from_assets(state,(*programme.assets,buffer),programme.events),demand,pressure))
            baseline=max(baselines,key=lambda p:p.terminal_cash)
            for kind in LONG_ASSETS:
                if kind in rules.ANIMALS and not (state.tile_at(tile).is_empty or state.tile_at(tile).kind=="WEED" or state.tile_at(tile).kind==rules.ANIMALS[kind].structure):
                    continue
                if kind in rules.CROPS and not (state.tile_at(tile).is_empty or state.tile_at(tile).kind=="WEED"):
                    continue
                candidate=(_animal_programme(state,kind,tile,existing=False) if kind in rules.ANIMALS
                           else _crop_programme(state,kind,tile,existing=False))
                now=_finalize(state,_programme_from_assets(state,(*programme.assets,candidate),programme.events),demand,pressure)
                advantage=now.terminal_cash-baseline.terminal_cash
                if advantage<=0:
                    continue
                # WAIT_REVEAL uses only known shops and the best legal buffer first.
                reveal=next_reveal(state.day)
                wait_value=baseline.terminal_cash
                if reveal<31:
                    delayed=(_animal_programme(state,kind,tile,existing=False,start_step=reveal*24+1) if kind in rules.ANIMALS
                             else _crop_programme(state,kind,tile,existing=False,start_step=reveal*24+1))
                    wait=_finalize(state,_programme_from_assets(state,(*baseline.assets,delayed),baseline.events),demand,pressure)
                    wait_value=wait.terminal_cash
                if now.terminal_cash<=wait_value: continue
                key=(advantage,-rules.distance_to_shed(tile,state.board_size),-tile[1],-tile[0],kind)
                ranked.append((key,now))
        if not ranked: break
        committed=None
        for _,candidate in sorted(ranked,key=lambda value:value[0],reverse=True):
            checked=_finalize(state,candidate,demand,pressure,force=True)
            if checked.feasible:
                committed=checked;break
        if committed is None:break
        programme=_finalize(state,_resolve_feed(state,committed,demand,pressure),demand,pressure,force=True)
        # Required feed is regenerated after every accepted long candidate.

    # Land is valued only together with an exact first use.
    if len(state.own.owned_land)<4:
        quadrant=rules.LAND_ORDER[len(state.own.owned_land)-1]
        price=rules.LAND_PRICES[len(state.own.owned_land)-1]
        locked=[t.position for t in state.tiles if t.is_locked and rules.quadrant(t.position,state.board_size)==quadrant]
        land_best=None
        for tile in locked:
            for kind in LONG_ASSETS:
                asset=(_animal_programme(state,kind,tile,existing=False,start_step=state.step+1) if kind in rules.ANIMALS
                       else _crop_programme(state,kind,tile,existing=False,start_step=state.step+1))
                land=LandProgramme(quadrant,state.step,price,tile,kind)
                event=_pe(state.step,-30,f"land:{quadrant}","BUY_LAND",tile=tile)
                trial=_finalize(state,_programme_from_assets(state,(*programme.assets,asset),(*programme.events,event),(land,)),demand,pressure)
                gain=trial.terminal_cash-programme.terminal_cash
                key=(gain,-rules.distance_to_shed(tile,state.board_size),-tile[1],-tile[0],kind)
                if land_best is None or key>land_best[0]: land_best=(key,trial)
        if land_best and land_best[0][0]>0: programme=_finalize(state,land_best[1],demand,pressure,force=True)

    # Final remaining exact tiles receive their best W/C/EMPTY use.
    for tile in _unused_tiles(state,programme):
        options=[programme]
        for crop in ("WHEAT","CARROT"):
            asset=_buffer(state,tile,crop)
            if asset: options.append(_finalize(state,_programme_from_assets(state,(*programme.assets,asset),programme.events),demand,pressure))
        # tuple order supplies the required W > C > EMPTY equality tie.
        programme=max(enumerate(options),key=lambda x:(x[1].terminal_cash,-x[0]))[1]

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
