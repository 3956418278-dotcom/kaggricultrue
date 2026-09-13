#!/usr/bin/env python3
"""Print the hand-drawn zonal template catalogue; never generates templates."""
from __future__ import annotations

from src.kaggriculture_agent.zonal_templates import HAND_DRAWN_TEMPLATES


def main() -> None:
    for template in HAND_DRAWN_TEMPLATES:
        print(
            f"{template.template_id}  land={'+'.join(template.owned_land_mask)} "
            f"workers={template.worker_count} family={template.family} "
            f"phase={template.workload_phase}"
        )
        print(template.render())
        for zone in template.zones.values():
            print(
                f"  {zone.zone_id}: {zone.behavior:5s} tiles={len(zone.tiles):2d} "
                f"shed={zone.shed_access} adjacent={','.join(sorted(zone.adjacent_zones))}"
            )
        print(f"  why: {template.rationale}\n")


if __name__ == "__main__":
    main()

