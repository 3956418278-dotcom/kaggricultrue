"""Spatial binding of daily intent, shared by search and the greedy benchmark."""
from dataclasses import dataclass, field, fields, replace
from typing import Mapping
from . import rules
from .planner import Plan, hiring_commitments, unbind_project


@dataclass(frozen=True)
class Realization(Plan):
    """Execution-only spatial witness; never replaces the daily intent."""
    crop_targets: Mapping[tuple[int, int], str] = field(default_factory=dict)


def bind_plan(state, plan: Plan, variant: int = 0, hands: int | None = None) -> Realization:
    """A candidate placement, never written back into daily economic intent.

    Variants reserve low travel sites for recurrent animal service, cluster new
    work around current workers, or spread it among workers. Search compares the
    resulting reachable states, including future service distance.
    """
    if not isinstance(plan, Realization):
        plan = Realization(**{f.name: getattr(plan, f.name) for f in fields(Plan)})
    def repair(project):
        if project.kind not in ("CROP", "ANIMAL", "ANIMAL_PLACEMENT") or project.target is None:
            return project
        tile = state.tile_at(project.target)
        valid = tile.is_empty or (
            tile.kind == "PLANT" and tile.raw.get("crop") == project.metadata.get("crop")
            if project.kind == "CROP" else tile.kind == project.metadata.get("structure")
            and tile.animal in (None, project.metadata.get("animal")))
        return project if valid else unbind_project(project, state)
    plan = replace(plan, selected=tuple(repair(p) for p in plan.selected),
                   obligations=tuple(repair(p) for p in plan.obligations))
    occupied = {p.target for p in (*plan.obligations, *plan.selected) if p.target is not None}
    projects = [p for p in (*plan.obligations, *plan.selected) if p.metadata.get("placement_open")]
    projects.sort(key=lambda p: (p.kind == "CROP" if variant % 3 != 1 else p.kind != "CROP", p.identifier))
    bindings = {}
    for n, project in enumerate(projects):
        candidates = [t for t in state.tiles if t.position not in occupied and
            (t.is_empty or (project.kind != "CROP" and t.kind == project.metadata.get("structure") and not t.animal))]
        if not candidates:
            continue
        def location_key(tile):
            shed = rules.distance_to_shed(tile.position)
            route = min(rules.manhattan(w.position, tile.position) for w in state.workers)
            if variant % 3 == 2:
                route = rules.manhattan(state.workers[n % len(state.workers)].position, tile.position)
            return (not (tile.kind == project.metadata.get("structure") and not tile.is_empty),
                    shed if variant % 3 == 0 else route, route if variant % 3 == 0 else shed,
                    tile.position)
        target = min(candidates, key=location_key).position
        occupied.add(target)
        bindings[project.identifier] = replace(project, target=target,
            land=replace(project.land, intervals=tuple(replace(i, position=target) for i in project.land.intervals)),
            actions=replace(project.actions, work=tuple(replace(w, position=target) for w in project.actions.work)),
            metadata={**project.metadata, "placement_open": False})
    def bound(projects):
        return tuple(bindings.get(p.identifier, p) for p in projects)
    selected, obligations = bound(plan.selected), bound(plan.obligations)
    targets = {p.target: str(p.metadata["crop"]) for p in selected if p.kind == "CROP" and p.target is not None}
    hire_count = plan.hire_count if hands is None else hands
    return replace(plan, selected=selected, obligations=obligations,
        crop_targets=targets or plan.crop_targets, hire_count=hire_count,
        support=tuple(p for p in plan.support if p.kind != "HIRE") + hiring_commitments(state, hire_count))
