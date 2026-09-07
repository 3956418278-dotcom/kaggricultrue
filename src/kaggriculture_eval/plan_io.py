"""Lossless loading of the maintained structured Plan, without demonstration data."""
from src.kaggriculture_agent import economics as e
from src.kaggriculture_agent.planner import Plan


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
                "projected_sales": records(e.TimedAmount, p["revenue"]["projected_sales"])})})
    return Plan(**{**raw,
        **{key: tuple(commitment(p) for p in raw[key]) for key in ("obligations", "selected", "support")},
        "fertilize_targets": frozenset(tuple(p) for p in raw["fertilize_targets"]),
        "animal_purchases": tuple(raw["animal_purchases"]),
        "placement_domains": {k: tuple(tuple(p) for p in v) for k, v in raw["placement_domains"].items()}})
