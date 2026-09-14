from __future__ import annotations

from pathlib import Path
import unittest

from src.kaggriculture_agent import rules
from src.kaggriculture_agent import zonal_templates as zonal


class FixedRoadTemplateTests(unittest.TestCase):
    def test_library_contains_runtime_one_two_three_and_provisional_four_land(self):
        masks = {t.owned_land_mask for t in zonal.HAND_AUTHORED_TEMPLATES}
        self.assertEqual(masks, {
            zonal.NW, zonal.NORTH, zonal.THREE_LAND, zonal.FULL})
        expected = {
            zonal.NW: set(range(1, 13)),
            zonal.NORTH: set(range(3, 13)),
            zonal.THREE_LAND: set(range(5, 13)),
            zonal.FULL: set(range(6, 13)),
        }
        for mask, counts in expected.items():
            self.assertEqual({w for stored, w in zonal.TEMPLATE_LIBRARY
                              if stored == mask}, counts)

    def test_each_worker_count_has_distinct_spatial_road_layouts(self):
        for key, templates in zonal.TEMPLATE_LIBRARY.items():
            layouts = {template.road_layout_id for template in templates}
            self.assertGreaterEqual(len(layouts), 2, key)
            signatures = {
                (tuple(sorted(template.tile_to_zone.items())),
                 tuple(zone.road for zone in template.zones.values()))
                for template in templates
            }
            self.assertEqual(len(signatures), len(layouts), key)

    def test_each_layout_has_bounded_fixed_return_versions_not_all_masks(self):
        for templates in zonal.TEMPLATE_LIBRARY.values():
            by_layout = {}
            for template in templates:
                by_layout.setdefault(template.road_layout_id, []).append(template)
            for versions in by_layout.values():
                workers = versions[0].worker_count
                return_sets = {
                    frozenset(zone.zone_id for zone in version.zones.values()
                              if zone.returns_to_shed)
                    for version in versions
                }
                self.assertIn(frozenset(), return_sets)
                self.assertLessEqual(len(versions), 5)
                if workers > 1:
                    self.assertLess(len(versions), 1 << workers)

    def test_authored_roads_cover_land_without_repeat_or_jump(self):
        for template in zonal.HAND_AUTHORED_TEMPLATES:
            expected = {(x, y) for y in range(rules.BOARD_SIZE)
                        for x in range(rules.BOARD_SIZE)
                        if rules.quadrant((x, y)) in template.owned_land_mask}
            self.assertEqual(set(template.tile_to_zone), expected,
                             template.template_id)
            self.assertEqual(len(template.zones), template.worker_count)
            seen = set()
            for zone in template.zones.values():
                self.assertEqual(len(zone.road), len(set(zone.road)))
                self.assertFalse(seen.intersection(zone.road))
                seen.update(zone.road)
                self.assertTrue(all(rules.manhattan(a, b) == 1
                                    for a, b in zip(zone.road, zone.road[1:])))
                self.assertEqual(rules.distance_to_shed(zone.entry), min(
                    rules.distance_to_shed(tile) for tile in zone.road))
                self.assertEqual(set(zone.action_limit_by_tile), set(zone.road))
            self.assertEqual(seen, expected)

    def test_three_land_corner_is_a_whole_map_line_design(self):
        template = next(t for t in zonal.HAND_AUTHORED_TEMPLATES
                        if t.template_id == "three-w6-corner-eod")
        self.assertEqual(set(template.zones["A"].road),
                         {(x, y) for x in range(3) for y in range(3)})
        self.assertEqual(len(template.zones["A"].road), 9)
        self.assertEqual(len(template.zones["B"].road), 12)
        self.assertEqual(len(template.zones["I"].road), 12)
        self.assertEqual(template.tile_to_zone[(3, 0)], "B")
        self.assertEqual(template.tile_to_zone[(0, 3)], "D")

    def test_every_fixed_return_zone_reaches_shed_by_turn_22(self):
        for template in zonal.HAND_AUTHORED_TEMPLATES:
            for zone in template.zones.values():
                if not zone.returns_to_shed:
                    continue
                route = zonal.compile_zone_route(
                    template, zone.zone_id, zone.certified_return_workload)
                self.assertIsNotNone(route, template.template_id)
                self.assertLessEqual(route.drop_turn, 22, template.template_id)
                self.assertLessEqual(route.finish_turn, 23, template.template_id)
                self.assertGreater(route.returned_units, 0)
                repeated = {position for position in route.positions
                            if route.positions.count(position) > 1}
                self.assertLessEqual(len(repeated), 1)
                if repeated:
                    self.assertEqual(route.positions[0], route.positions[-1])
                    self.assertEqual(repeated, {route.positions[0]})

    def test_sparse_tasks_use_short_connectors_without_zone_repeat(self):
        template = next(t for t in zonal.templates_for(zonal.THREE_LAND, 6)
                        if t.template_id == "three-w6-corner-eod")
        zone = template.zones["I"]
        tasks = {zone.road[1]: zonal.TileWorkload(2),
                 zone.road[len(zone.road)//2]: zonal.TileWorkload(1),
                 zone.road[-2]: zonal.TileWorkload(2)}
        route = zonal.compile_zone_route(template, "I", tasks)
        self.assertIsNotNone(route)
        self.assertEqual(route.task_order,
                         tuple(tile for tile in zone.road if tile in tasks))
        inside = [p for p in route.positions if p in zone.road]
        self.assertEqual(len(inside), len(set(inside)))
        self.assertLess(route.movement_turns, len(zone.road) - 1
                        + rules.distance_to_shed(zone.entry))

    def test_exact_workload_decides_24_turn_feasibility(self):
        light = next(t for t in zonal.templates_for(zonal.THREE_LAND, 5)
                     if t.template_id == "three-w5-bands-eod")
        zone = light.zones["A"]
        self.assertIsNone(zonal.compile_zone_route(
            light, "A", {tile: zonal.TileWorkload(2) for tile in zone.road}))
        tile = zone.road[0]
        workload = {tile: zonal.TileWorkload(3)}
        self.assertIsNone(zonal.compile_zone_route(light, "A", workload))
        solution = zonal.select_minimum_workforce(
            zonal.THREE_LAND, workload, minimum_workers=5)
        self.assertIsNotNone(solution)
        self.assertEqual(solution.template.worker_count, 5)

    def test_two_and_three_land_have_cross_quadrant_road_families(self):
        for mask, minimum, family in (
                (zonal.NORTH, 3, "CROSS_SPINE"),
                (zonal.THREE_LAND, 5, "TOP_CROSS_SPINE")):
            for workers in range(minimum, 13):
                templates = zonal.templates_for(mask, workers)
                cross = [template for template in templates
                         if template.family == family]
                self.assertTrue(cross, (mask, workers))
                self.assertTrue(any(
                    len({rules.quadrant(tile) for tile in zone.road}) > 1
                    for template in cross for zone in template.zones.values()))

    def test_five_action_same_tile_is_supported_by_high_variance_layout(self):
        for mask in (zonal.NORTH, zonal.THREE_LAND, zonal.FULL):
            tile = min(zonal._owned_tiles(mask))
            solution = zonal.select_minimum_workforce(
                mask, {tile: zonal.TileWorkload(5)})
            self.assertIsNotNone(solution, mask)

    def test_return_requirement_switches_template_not_ownership(self):
        templates = zonal.templates_for(zonal.THREE_LAND, 7)
        base = next(t for t in templates if t.return_mode == "EOD")
        far = sorted(base.tile_to_zone,
                     key=lambda p: (-rules.distance_to_shed(p), p[1], p[0]))[:18]
        return_tiles = frozenset(far[-2:])
        workload = {
            tile: zonal.TileWorkload(
                1, 6, must_return=tile in return_tiles)
            for tile in far
        }
        eod = zonal.select_minimum_workforce(
            zonal.THREE_LAND, workload, minimum_workers=7)
        self.assertIsNotNone(eod)
        self.assertNotEqual(eod.template.return_mode, "EOD")
        self.assertTrue(all(
            eod.template.zones[eod.template.tile_to_zone[tile]].returns_to_shed
            for tile in return_tiles))
        self.assertLessEqual(max(r.drop_turn for r in eod.routes.values()
                                 if r.drop_turn is not None), 22)

    def test_multi_return_template_keeps_independent_one_return_routes(self):
        template = next(candidate for candidate in
                        zonal.templates_for(zonal.THREE_LAND, 7)
                        if candidate.road_layout_id == "three-w7-refined"
                        and candidate.return_mode == "WIDE_RETURN")
        return_zones = frozenset(
            zone.zone_id for zone in template.zones.values()
            if zone.returns_to_shed)
        workload = {
            tile: work
            for zone in template.zones.values()
            if zone.zone_id in return_zones
            for tile, work in zone.certified_return_workload.items()
        }
        solution = zonal.solve_template(template, workload)
        self.assertIsNotNone(solution)
        returning = [route for route in solution.routes.values()
                     if route.drop_turn is not None]
        self.assertEqual(len(returning), len(return_zones))
        for route in returning:
            self.assertLessEqual(route.drop_turn, 22)
            self.assertEqual(route.positions[0], route.positions[-1])
            self.assertEqual(route.positions.count(route.positions[0]), 2)

    def test_three_land_eighteen_melon_example_is_executable(self):
        owned = sorted(zonal._owned_tiles(zonal.THREE_LAND),
                       key=lambda p: (-rules.distance_to_shed(p), p[1], p[0]))
        melon_tiles = owned[:18]
        eod_tiles = frozenset(melon_tiles[:16])
        workload = {
            tile: zonal.TileWorkload(
                1, 6, must_return=tile not in eod_tiles)
            for tile in melon_tiles
        }
        solution = zonal.select_minimum_workforce(zonal.THREE_LAND, workload)
        self.assertIsNotNone(solution)
        self.assertGreaterEqual(solution.template.worker_count, 5)
        self.assertGreaterEqual(solution.returned_units, 12)
        self.assertLessEqual(108 - solution.returned_units, 100)
        self.assertLessEqual(solution.max_finish_turn, 23)
        self.assertTrue(all(route.drop_turn is None or route.drop_turn <= 22
                            for route in solution.routes.values()))

    def test_high_worker_corner_split_covers_32_far_harvest_tiles(self):
        for mask in (zonal.THREE_LAND,):
            tiles = sorted(
                zonal._owned_tiles(mask),
                key=lambda p: (-rules.distance_to_shed(p), p[1], p[0]))
            harvest = tiles[:32]
            eod_tiles = frozenset(harvest[:16])
            workload = {
                tile: zonal.TileWorkload(
                    1, 6, must_return=tile not in eod_tiles)
                for tile in harvest
            }
            solution = zonal.select_minimum_workforce(mask, workload)
            self.assertIsNotNone(solution, mask)
            self.assertLessEqual(solution.template.worker_count, 12)
            self.assertGreaterEqual(solution.returned_units, 96)
            self.assertLessEqual(solution.max_finish_turn, 22)

    def test_old_partition_and_pair_repair_api_is_gone(self):
        self.assertFalse(hasattr(zonal, "HAND_DRAWN_TEMPLATES"))
        self.assertFalse(hasattr(zonal, "THREE_QUADRANTS"))
        for template in zonal.HAND_AUTHORED_TEMPLATES:
            for zone in template.zones.values():
                self.assertFalse(hasattr(zone, "adjacent_zones"))
                self.assertFalse(hasattr(zone, "boundary_tiles"))
                self.assertFalse(hasattr(zone, "cheap_sweeps"))
                self.assertFalse(hasattr(zone, "behavior"))
        source = Path(zonal.__file__).read_text(encoding="utf-8")
        self.assertNotIn("repair_pair", source)
        self.assertNotIn("2-opt", source)
        self.assertNotIn("_RETURN_SCENARIOS", source)
        self.assertNotIn("select_return_mask", source)


if __name__ == "__main__":
    unittest.main()
