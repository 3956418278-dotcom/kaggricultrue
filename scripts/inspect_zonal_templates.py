#!/usr/bin/env python3
"""Print the authored fixed-road catalogue; never generates partitions."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kaggriculture_agent.zonal_templates import HAND_AUTHORED_TEMPLATES


def main() -> None:
    for template in HAND_AUTHORED_TEMPLATES:
        print(
            f"{template.template_id}  land={'+'.join(template.owned_land_mask)} "
            f"workers={template.worker_count} family={template.family} "
            f"workload={template.workload_mode} "
            f"return={template.return_mode}"
        )
        print(template.render())
        for zone in template.zones.values():
            print(
                f"  {zone.zone_id}: tiles={len(zone.road):2d} "
                f"entry={zone.entry} shed={zone.shed_access} "
                f"returns={'yes' if zone.returns_to_shed else 'no'} "
                f"road={zone.road}"
            )
        print("  fixed return zones:", ",".join(
            zone.zone_id for zone in template.zones.values()
            if zone.returns_to_shed) or "none")
        print()


if __name__ == "__main__":
    main()
