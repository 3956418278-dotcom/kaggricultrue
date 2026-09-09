"""Loading of canonical Plans and explicit migration of old reference samples."""
from collections import Counter

from src.kaggriculture_agent import economics as e
from src.kaggriculture_agent.planner import EconomicWindow, Plan


def plan_from_dict(raw):
    def records(cls, values, positions=False):
        return tuple(cls(**{**v, **({"position": tuple(v["position"]) if v.get("position") is not None else None}
                                    if positions else {})}) for v in values)

    def commitment(p):
        return e.EconomicCommitment(**{**p,
            "target": tuple(p["target"]) if p["target"] is not None else None,
            "cash": e.CashDimension(**{**p["cash"], "scheduled": records(e.TimedCash, p["cash"]["scheduled"])}),
            "time": e.TimeDimension(**{**p["time"], "deadlines": tuple(p["time"]["deadlines"])}),
            "land": e.LandDimension(**{**p["land"], "intervals": records(e.OccupancyInterval, p["land"]["intervals"], True)}),
            "actions": e.ActionDimension(records(e.WorkAmount, p["actions"]["work"], True),
                {int(k): v for k, v in p["actions"]["capacity_supplied"].items()}),
            "physical": e.PhysicalDimension(**{**p["physical"],
                "inputs": records(e.TimedAmount, p["physical"]["inputs"]),
                "outputs": records(e.TimedAmount, p["physical"]["outputs"])}),
            "revenue": e.RevenueDimension(**{**p["revenue"],
                "projected_sales": records(e.TimedAmount, p["revenue"]["projected_sales"])}),
            "required_state": p.get("required_state", {}),
            "required_outputs": p.get("required_outputs", {})})
    values = dict(raw)
    # player-day-v2 carried placement freedom and action-shaped STATE_EFFECTs.
    # It is intentionally not accepted as a production Plan; callers with an
    # old frozen sample must first migrate it using the recorded reference.
    values.pop("placement_domains", None)
    windows = tuple(EconomicWindow(
        start_turn=int(window["start_turn"]),
        end_turn=int(window["end_turn"]),
        market_orders=tuple(tuple(order) for order in window.get("market_orders", ())),
        required_shed={str(k): int(v) for k, v in window.get("required_shed", {}).items()},
        minimum_cash=int(window.get("minimum_cash", 0)),
        extra_hands_allowed=int(window.get("extra_hands_allowed", 0)),
    ) for window in values.get("economic_windows", ()))
    return Plan(**{**values,
        **{key: tuple(commitment(p) for p in raw[key]) for key in ("obligations", "selected", "support")},
        "fertilize_targets": frozenset(tuple(p) for p in raw["fertilize_targets"]),
        "animal_purchases": tuple(raw["animal_purchases"]),
        "economic_windows": windows})


def plan_from_sample(sample):
    """Load a Plan, migrating player-day-v2 only inside the reference adapter.

    Old frozen samples left new placement open and stored one replay effect per
    commitment.  The production planner no longer accepts that representation;
    this adapter collapses the demonstrated effects of each farm entity into one
    fixed-place daily outcome before constructing ``Plan``.
    """
    raw = sample["plan"]
    projects = raw["obligations"] + raw["selected"]
    if "placement_domains" not in raw and all(
        "required_state" in project for project in projects
    ):
        return plan_from_dict(raw)

    from .player_days import digest

    by_entity = {}
    for turn in sample["demonstrated_realization"]:
        for event in turn["effects"]:
            if not event.get("achieved", False):
                continue
            record = by_entity.setdefault(event["entity"], {
                "position": tuple(event["position"]), "events": [],
            })
            record["events"].append(event)
    commitments = []
    start = int(sample["day"]) * 24
    end = start + int(sample["actionable_turns"])
    for entity, record in sorted(by_entity.items()):
        inputs, outputs = Counter(), Counter()
        for event in record["events"]:
            for item, quantity in event["physical_delta"].items():
                (outputs if quantity > 0 else inputs)[item] += abs(quantity)
        required_state = {"$tile": record["events"][-1]["after"]}
        identity = {"entity": entity, "position": record["position"],
                    "required_state": required_state,
                    "required_outputs": dict(sorted(outputs.items()))}
        commitments.append(e.EconomicCommitment(
            identifier=digest(identity), kind="STATE_EFFECT",
            target=record["position"], existing=entity.startswith("existing:"),
            cash=e.CashDimension(),
            time=e.TimeDimension(start, end - 1, end - 1, (end - 1,)),
            land=e.LandDimension((e.OccupancyInterval(
                record["position"], start, end - 1),)),
            actions=e.ActionDimension(),
            physical=e.PhysicalDimension(
                inputs=tuple(e.TimedAmount(start, item, quantity)
                             for item, quantity in sorted(inputs.items())),
                outputs=tuple(e.TimedAmount(end, item, quantity)
                              for item, quantity in sorted(outputs.items()))),
            revenue=e.RevenueDimension(), metadata={"entity": entity},
            required_state=required_state,
            required_outputs=dict(sorted(outputs.items())),
        ))
    land = plan_from_dict({**raw, "obligations": [], "selected": []}).support
    values = {key: raw[key] for key in (
        "rejected", "fertilize_targets", "animal_purchases", "buy_land",
        "feed_reserve", "fertilizer_reserve", "diagnostics",
        "starting_animals", "day", "formed_step", "revision",
        "replan_reason", "max_hands",
    ) if key in raw}
    values["fertilize_targets"] = frozenset(
        tuple(position) for position in values.get("fertilize_targets", ()))
    values["animal_purchases"] = tuple(values.get("animal_purchases", ()))
    return Plan(
        obligations=tuple(item for item in commitments if item.existing),
        selected=tuple(item for item in commitments if not item.existing),
        support=land,
        **values,
    )
