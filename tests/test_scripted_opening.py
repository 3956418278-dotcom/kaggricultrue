"""Fixed opening data and route-capacity invariants."""

import unittest

from src.kaggriculture_agent.scripted_opening import (
    ABANDONED_TILES,
    CARROT_TILES,
    DAY1_HANDS,
    DAY4_PASTURES,
    GOOSE_TILE,
    INNER_MELONS,
    MELON_TILES,
    OUTER_MELONS,
    WHEAT_TILES,
    choose_day4_animals,
    day1_required_hands,
    route_lengths,
)


class ScriptedOpeningTests(unittest.TestCase):
    def test_fixed_layout_and_route_capacity(self):
        self.assertEqual(len(MELON_TILES), 12)
        self.assertEqual(set(INNER_MELONS) | set(OUTER_MELONS), set(MELON_TILES))
        self.assertFalse(set(INNER_MELONS) & set(OUTER_MELONS))
        self.assertEqual(len(CARROT_TILES), 16)
        self.assertEqual(len(WHEAT_TILES), 15)
        occupied = (set(MELON_TILES) | set(CARROT_TILES)
                    | set(WHEAT_TILES) | {GOOSE_TILE})
        self.assertEqual(len(occupied), 44)
        self.assertEqual(
            occupied | set(ABANDONED_TILES),
            {(x, y) for y in range(5) for x in range(10)},
        )
        self.assertTrue(set(DAY4_PASTURES) <= set(WHEAT_TILES))
        self.assertNotIn(GOOSE_TILE, set(MELON_TILES) | set(WHEAT_TILES))
        self.assertEqual(choose_day4_animals({}), ["COW", "COW", "SHEEP"])
        self.assertEqual(day1_required_hands(), DAY1_HANDS)
        self.assertLessEqual(route_lengths()[0][0], 24)
        self.assertTrue(all(length <= 23 for length in route_lengths()[0][1:]))
        self.assertEqual(len(route_lengths()[3]), 9)

    def test_literal_layout(self):
        symbols = {}
        for position in MELON_TILES:
            symbols[position] = "M"
        for position in CARROT_TILES:
            symbols[position] = "C"
        for position in WHEAT_TILES:
            symbols[position] = "W"
        symbols[GOOSE_TILE] = "G"
        rows = tuple(
            "".join(symbols.get((x, y), ".") for x in range(10))
            for y in range(5)
        )
        self.assertEqual(rows, (
            "..CCWWCC..",
            ".CCWWWWCC.",
            "CCWGMMCCCC",
            "CWWMMMMWWC",
            "WWMMMMMMWW",
        ))


if __name__ == "__main__":
    unittest.main()
