"""Execution decisions, separate from the fixed economic Plan.

The provisional legacy search still constructs a few placement candidates here.
This is a migration adapter, not the reference-driven planner redesign.
"""
from dataclasses import dataclass, field
from typing import Mapping
from . import rules
from .state import Position


@dataclass(frozen=True)
class ExecutionChoices:
    placements: Mapping[str, Position] = field(default_factory=dict)
    hire_count: int = 0

    def target(self, project):
        return project.target if project.target is not None else self.placements.get(project.identifier)

    def crop_targets(self, plan):
        return {self.target(p): str(p.metadata["crop"]) for p in plan.selected
                if p.kind == "CROP" and self.target(p) is not None}


def legacy_choices(state, plan, variant=0, hands=None, previous=None):
    """Concrete witness restricted to Plan domains; never edits commitments.

    Preserve completed and still-valid placement choices during trajectory
    repair. Missing witnesses stay missing and are reported as shortfall.
    """
    placements, occupied = {}, {p.target for p in (*plan.obligations, *plan.selected) if p.target is not None}
    projects = [p for p in (*plan.obligations, *plan.selected) if p.identifier in plan.placement_domains]
    projects.sort(key=lambda p: (p.kind == "CROP" if variant % 3 != 1 else p.kind != "CROP", p.identifier))
    for p in projects:
        pos = previous.placements.get(p.identifier) if previous else None
        if pos is None or pos not in plan.placement_domains[p.identifier]:
            continue
        tile = state.tile_at(pos)
        valid = tile.is_empty or (tile.kind == "PLANT" and tile.raw.get("crop") == p.metadata.get("crop")
            if p.kind == "CROP" else tile.kind == p.metadata.get("structure") and tile.animal in (None, p.metadata.get("animal")))
        if valid and pos not in occupied:
            placements[p.identifier] = pos
            occupied.add(pos)
    for n, p in enumerate(projects):
        if p.identifier in placements:
            continue
        candidates = [state.tile_at(pos) for pos in plan.placement_domains[p.identifier]
            if pos not in occupied and (state.tile_at(pos).is_empty or
                (p.kind != "CROP" and state.tile_at(pos).kind == p.metadata.get("structure") and not state.tile_at(pos).animal))]
        if not candidates:
            continue
        def key(tile):
            shed = rules.distance_to_shed(tile.position)
            route = min(rules.manhattan(w.position, tile.position) for w in state.workers)
            if variant % 3 == 2:
                route = rules.manhattan(state.workers[n % len(state.workers)].position, tile.position)
            return (shed if variant % 3 == 0 else route, route if variant % 3 == 0 else shed, tile.position)
        pos = min(candidates, key=key).position
        placements[p.identifier] = pos
        occupied.add(pos)
    if hands is None:
        # Weak benchmark's workload estimate; not an economic Plan decision.
        work = sum(w.actions + w.travel_actions for p in (*plan.obligations, *plan.selected, *plan.support)
                   for w in p.actions.work if w.day == state.day)
        hands = max(state.hires_today, min(plan.max_hands, max(0, (work + 23) // 24 - 1)))
    return ExecutionChoices(placements, hands)
