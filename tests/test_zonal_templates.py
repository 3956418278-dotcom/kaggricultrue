from __future__ import annotations

import unittest

from src.kaggriculture_agent import rules
from src.kaggriculture_agent.zonal_templates import (
    FULL,
    HAND_DRAWN_TEMPLATES,
    NORTH,
    NW,
    THREE_QUADRANTS,
    templates_for,
)


class HandDrawnTemplateTests(unittest.TestCase):
    def test_every_map_exactly_covers_its_official_unlock_mask(self):
        for template in HAND_DRAWN_TEMPLATES:
            expected = {
                (x, y)
                for y in range(rules.BOARD_SIZE)
                for x in range(rules.BOARD_SIZE)
                if rules.quadrant((x, y)) in template.owned_land_mask
            }
            self.assertEqual(set(template.tile_to_zone), expected, template.template_id)
            self.assertEqual(len(template.zones), template.worker_count, template.template_id)

    def test_each_authored_worker_count_has_structurally_distinct_choices(self):
        for key, templates in {
            (template.owned_land_mask, template.worker_count): templates_for(
                template.owned_land_mask, template.worker_count
            )
            for template in HAND_DRAWN_TEMPLATES
        }.items():
            self.assertGreaterEqual(len(templates), 2, key)
            self.assertEqual(len({template.rows for template in templates}), len(templates), key)
            self.assertGreaterEqual(len({template.family for template in templates}), 2, key)

    def test_all_four_official_land_prefixes_are_authored(self):
        present = {template.owned_land_mask for template in HAND_DRAWN_TEMPLATES}
        self.assertEqual(present, {NW, NORTH, THREE_QUADRANTS, FULL})

    def test_three_quadrant_example_has_corner_block_edge_chain_and_one_inner_direction(self):
        template = next(t for t in HAND_DRAWN_TEMPLATES if t.template_id == "three-w6-corner-top")
        corner = template.zones["A"]
        self.assertEqual(set(corner.tiles), {(x, y) for x in range(3) for y in range(3)})
        self.assertEqual(corner.behavior, "OUTER")
        self.assertEqual(corner.adjacent_zones, frozenset(("B", "D")))
        self.assertEqual(template.zones["B"].behavior, "OUTER")
        self.assertEqual(template.zones["D"].behavior, "OUTER")
        self.assertEqual(template.zones["I"].behavior, "INNER")
        # The corner does not have a private INNER corridor in either direction.
        self.assertNotIn("I", corner.adjacent_zones)

    def test_fixed_metadata_is_complete_for_every_zone(self):
        for template in HAND_DRAWN_TEMPLATES:
            for zone in template.zones.values():
                self.assertEqual(
                    set(zone.cheap_sweeps),
                    {"clockwise", "counter_clockwise", "snake_a", "snake_b"},
                    (template.template_id, zone.zone_id),
                )
                for order in zone.cheap_sweeps.values():
                    self.assertEqual(set(order), set(zone.tiles))
                    self.assertEqual(len(order), len(zone.tiles))
                self.assertEqual(len(zone.tile_distances), len(zone.tiles))
                self.assertTrue(all(len(row) == len(zone.tiles) for row in zone.tile_distances))
                self.assertIn(zone.shed_access, rules.shed_access())

    def test_inner_skeletons_form_one_shed_connected_subgraph(self):
        accesses = set(rules.shed_access())
        for template in HAND_DRAWN_TEMPLATES:
            inner = {zone.zone_id for zone in template.zones.values() if zone.behavior == "INNER"}
            self.assertTrue(inner, template.template_id)
            reached = {
                zone_id for zone_id in inner
                if accesses.intersection(template.zones[zone_id].tiles)
            }
            self.assertTrue(reached, template.template_id)
            frontier = list(reached)
            while frontier:
                zone_id = frontier.pop()
                for neighbour in template.zones[zone_id].adjacent_zones & inner:
                    if neighbour not in reached:
                        reached.add(neighbour)
                        frontier.append(neighbour)
            self.assertEqual(reached, inner, template.template_id)


if __name__ == "__main__":
    unittest.main()
