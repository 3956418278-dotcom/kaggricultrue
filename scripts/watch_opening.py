#!/usr/bin/env python3
"""Run, validate, and view the fixed four-day opening."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_environment() -> None:
    try:
        import kaggle_environments  # noqa: F401
    except ModuleNotFoundError:
        python = ROOT / ".venv" / "bin" / "python"
        if not python.exists():
            raise SystemExit("kaggle-environments and .venv/bin/python are unavailable")
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


_ensure_environment()

from kaggle_environments import make  # noqa: E402

from src.kaggriculture_agent import rules  # noqa: E402
from src.kaggriculture_agent.scripted_opening import (  # noqa: E402
    ABANDONED_TILES,
    CARROT_TILES,
    DAY1_HANDS,
    DAY3_HANDS,
    DAY4_HANDS,
    DAY4_PASTURES,
    GOOSE_TILE,
    INNER_MELONS,
    MELON_TILES,
    OUTER_MELONS,
    SEED,
    WHEAT_DAY3_TILE,
    WHEAT_TILES,
    agent,
    day1_required_hands,
    route_lengths,
)
from src.kaggriculture_eval.player_days import observation, unit_effects  # noqa: E402
from src.kaggriculture_eval.replay import render_replay_html, write_replay  # noqa: E402


def _pass_agent(obs):
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in obs.farms[obs.player].hands],
        "market": [],
    }


def _animals(obs) -> Counter:
    result = Counter()
    for row in obs["farms"][0]["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and "animal" in tile:
                result[tile["animal"]] += 1
    return result


def _crops(obs) -> Counter:
    result = Counter()
    for row in obs["farms"][0]["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                result[tile["crop"]] += 1
    return result


def _action(replay, frame):
    return replay["steps"][frame + 1][0].get("action") or {}


def _audit(replay) -> tuple[bool, str]:
    actions = {}
    effects = defaultdict(list)
    by_frame = {}
    noops = []
    for frame in range(96):
        obs = observation(replay, frame, 0)
        actions[frame] = _action(replay, frame)
        turn_effects, turn_noops = unit_effects(
            obs, actions[frame], replay["configuration"]
        )
        by_frame[frame] = turn_effects
        effects[frame // 24].extend(turn_effects)
        noops.extend((frame, item) for item in turn_noops)

    day1 = observation(replay, 24, 0)
    day4_start = observation(replay, 72, 0)
    end = observation(replay, 96, 0)
    water = {
        day: {tuple(e["position"]) for e in records if e["action"][0] == "WATER"}
        for day, records in effects.items()
    }
    goose = {
        day: {e["action"][0] for e in effects[day]
              if tuple(e["position"]) == GOOSE_TILE}
        for day in (1, 2, 3)
    }
    carrot_yields = {
        day4_start["farms"][0]["tiles"][y][x].get("yield_units")
        for x, y in CARROT_TILES
    }
    wheat_day3 = sum(max(0, int(e["physical_delta"].get("WHEAT", 0)))
                     for e in effects[2])
    wheat_day4 = sum(max(0, int(e["physical_delta"].get("WHEAT", 0)))
                     for e in effects[3])
    carrot_day4 = sum(max(0, int(e["physical_delta"].get("CARROT", 0)))
                      for e in effects[3])
    animal_targets = {
        (2, 3): "SHEEP",
        (4, 1): "COW",
        (5, 1): "COW",
    }
    animal_feed = {tuple(e["position"]) for e in effects[3]
                   if e["action"][0] == "FEED"
                   and tuple(e["position"]) in animal_targets}
    animal_care = {tuple(e["position"]) for e in effects[3]
                   if e["action"][0] == "CARE"
                   and tuple(e["position"]) in animal_targets}

    harvest_frame = {}
    wheat_harvest_frame = {}
    build_frame = {}
    animal_action_frames = defaultdict(dict)
    for frame in range(72, 96):
        for event in by_frame[frame]:
            position = tuple(event["position"])
            if (event["action"][0] == "HARVEST"
                    and event["physical_delta"].get("CARROT", 0) > 0):
                harvest_frame[position] = frame
            if (event["action"][0] == "HARVEST"
                    and event["physical_delta"].get("WHEAT", 0) > 0):
                wheat_harvest_frame[position] = frame
            if event["action"][0] == "BUILD_PASTURE":
                build_frame[position] = frame
            if (position in animal_targets
                    and event["action"][0] in {"PLACE", "FEED", "CARE"}):
                animal_action_frames[position][event["action"][0]] = frame

    purchase_frames = [frame for frame in range(72, 96)
                       if ["BUY_ANIMAL", "COW", 2]
                       in actions[frame].get("market", [])
                       and ["BUY_ANIMAL", "SHEEP", 1]
                       in actions[frame].get("market", [])]
    carrot_sell_frames = [frame for frame in range(72, 96)
                          if any(order[:2] == ["SELL", "CARROT"]
                                 for order in actions[frame].get("market", []))]
    wheat_sell_frames = [frame for frame in range(72, 96)
                         if any(order[:2] == ["SELL", "WHEAT"]
                                for order in actions[frame].get("market", []))]
    day4_hires = sum(order == ["HIRE"]
                     for frame in range(72, 96)
                     for order in actions[frame].get("market", []))
    final_tiles = end["farms"][0]["tiles"]
    placed_animals = {
        position: final_tiles[position[1]][position[0]].get("animal")
        for position in DAY4_PASTURES
        if isinstance(final_tiles[position[1]][position[0]], dict)
    }
    post_opening_pass = all(
        not _action(replay, frame).get("market")
        and all(unit == ["PASS"] for unit in [
            _action(replay, frame).get("farmer", ["PASS"]),
            *_action(replay, frame).get("hands", []),
        ])
        for frame in range(96, len(replay["steps"]) - 1)
    )

    checks = {
        "official episode completed": replay.get("statuses") == ["DONE", "DONE"],
        "Day1 exactly five hands": day1_required_hands() == DAY1_HANDS == 5,
        "Day1 land purchased": "NE" in day1["farms"][0]["unlocked_quadrants"],
        "Day1 crop counts": _crops(day1) == Counter(
            {"MELON": 12, "CARROT": 16, "WHEAT": 15}
        ),
        "Day1 Goose placed": _animals(day1) == Counter({"GOOSE": 1}),
        "Day1 every crop watered": water[0]
            == set(MELON_TILES) | set(CARROT_TILES) | set(WHEAT_TILES),
        "abandoned tiles never serviced": not any(
            tuple(e["position"]) in ABANDONED_TILES
            for records in effects.values() for e in records
        ),
        "Day2 outer Melons watered": water[1] == set(OUTER_MELONS),
        "Day2 Goose maintained": goose[1]
            >= {"FEED", "CARE", "COLLECT_FERTILIZER"},
        "Day2 no harvest": not any(e["action"][0] == "HARVEST" for e in effects[1]),
        "Day3 inner Melons watered": water[2] & set(MELON_TILES) == set(INNER_MELONS),
        "Day3 all Carrot and Wheat watered":
            set(CARROT_TILES) <= water[2] and set(WHEAT_TILES) <= water[2],
        "Day3 Carrot yield two and zero harvest": carrot_yields == {2}
            and not any(e["action"][0] == "HARVEST"
                        and e["physical_delta"].get("CARROT", 0) > 0
                        for e in effects[2]),
        "Day3 Wheat harvest is two": wheat_day3 == 2
            and day4_start["farms"][0]["tiles"]
                [WHEAT_DAY3_TILE[1]][WHEAT_DAY3_TILE[0]] is None,
        "Day3 Goose maintained": goose[2]
            >= {"FEED", "CARE", "COLLECT_FERTILIZER"},
        "Day4 starts with expected shed stock":
            day4_start["private"]["shed"].get("WHEAT", 0) == 1
            and day4_start["private"]["shed"].get("FERTILIZER", 0) == 2,
        "Day4 hires exactly eight hands": day4_hires == DAY4_HANDS == 8,
        "Day4 hire cost is 54":
            sum(rules.fibonacci_hire_cost(i) for i in range(DAY4_HANDS)) == 54,
        "Day4 Carrot harvest is 48": carrot_day4 == 48,
        "Day4 all 16 Carrot tiles harvested":
            set(harvest_frame) == set(CARROT_TILES),
        "Day4 first Carrot and Wheat sale is t15":
            carrot_sell_frames == [87] and wheat_sell_frames == [87],
        "Day4 animals bought before final harvest": purchase_frames == [87]
            and max(harvest_frame.values()) > purchase_frames[0],
        "Day4 selected Wheat harvested before Pastures":
            set(wheat_harvest_frame) == set(DAY4_PASTURES)
            and all(wheat_harvest_frame[p] < build_frame[p]
                    for p in DAY4_PASTURES),
        "Day4 fixed Pastures built": set(build_frame) == set(DAY4_PASTURES),
        "Day4 Wheat harvest is nine": wheat_day4 == 9,
        "Day4 outer Melons watered": water[3] & set(MELON_TILES) == set(OUTER_MELONS),
        "Day4 Goose maintained": goose[3]
            >= {"FEED", "CARE", "COLLECT_FERTILIZER"},
        "Day4 Cow/Cow/Sheep placed at fixed Pastures":
            placed_animals == animal_targets,
        "Day4 all new animals fed": animal_feed == set(DAY4_PASTURES),
        "Day4 all new animals cared": animal_care == set(DAY4_PASTURES),
        "Day4 final animals":
            _animals(end) == Counter({"COW": 2, "SHEEP": 1, "GOOSE": 1}),
        "Day4 remaining crops": _crops(end) == Counter({"MELON": 12, "WHEAT": 11}),
        "no skipped/illegal/impossible unit action": not noops,
        "Day5 onward all PASS": post_opening_pass,
    }

    lines = ["Fixed opening audit", f"seed = {replay['info']['seed']}",
             f"route lengths = {route_lengths()}", ""]
    hires = {0: DAY1_HANDS, 1: 0, 2: DAY3_HANDS, 3: DAY4_HANDS}
    for day in range(4):
        beginning = observation(replay, day * 24, 0)
        ending = observation(replay, (day + 1) * 24, 0)
        max_hands = max(len(observation(replay, day * 24 + hour, 0)
                            ["farms"][0]["hands"]) for hour in range(24))
        lines.extend((
            f"Day {day + 1}",
            f"money start/end = {beginning['farms'][0]['money']:.0f} / "
            f"{ending['farms'][0]['money']:.0f}",
            f"hands count = {max_hands}",
            "hire cost = " + str(sum(rules.fibonacci_hire_cost(i)
                                      for i in range(hires[day]))),
        ))
        worker_turns = defaultdict(list)
        for hour in range(24):
            action = actions[day * 24 + hour]
            units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
            for worker, unit in enumerate(units):
                worker_turns[worker].append((hour, unit))
            if action.get("market"):
                lines.append(f"market t{hour:02d}: {action['market']}")
        for worker in sorted(worker_turns):
            lines.append(f"worker {worker}:")
            lines.extend(f"  t{hour:02d} {unit}"
                         for hour, unit in worker_turns[worker])
        if day == 2:
            lines.extend(("carrot yield = 2 on all 16 tiles",
                          "carrot harvested = 0",
                          f"wheat harvested = {wheat_day3}"))
        if day == 3:
            timing = []
            for position in DAY4_PASTURES:
                animal = animal_targets[position]
                turns = animal_action_frames[position]
                timing.append(
                    f"{animal} {position} PLACE/FEED/CARE = "
                    f"t{turns.get('PLACE', -72) - 72:02d}/"
                    f"t{turns.get('FEED', -72) - 72:02d}/"
                    f"t{turns.get('CARE', -72) - 72:02d}"
                )
            lines.extend(("carrot_expected = 48",
                          f"carrot_actual = {carrot_day4}",
                          "first Carrot SELL turn = "
                          + (f"t{carrot_sell_frames[0] % 24:02d}"
                             if carrot_sell_frames else "none"),
                          "first Wheat SELL turn = "
                          + (f"t{wheat_sell_frames[0] % 24:02d}"
                             if wheat_sell_frames else "none"),
                          "2 Cow + 1 Sheep purchase turn = "
                          + (f"t{purchase_frames[0] % 24:02d}"
                             if purchase_frames else "none"),
                          *timing,
                          f"Day4 end shed = {dict(end['private']['shed'])}",
                          "Day4 end inventories = "
                          + repr(end["private"]["inventories"]),
                          "16 / 16 Carrot harvested = "
                          + str(checks["Day4 all 16 Carrot tiles harvested"])))
        lines.append("")

    lines.append("Checks")
    lines.extend(f"[{'PASS' if passed else 'FAIL'}] {name}"
                 for name, passed in checks.items())
    if noops:
        lines.append(f"no-op details = {noops}")
    passed = all(checks.values())
    lines.extend(("", "OPENING PASS" if passed else "OPENING FAIL"))
    return passed, "\n".join(lines) + "\n"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="default: runs/fixed-opening-seed-<seed>")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_dir or ROOT / "runs" / f"fixed-opening-seed-{SEED}"
    output.mkdir(parents=True, exist_ok=True)
    environment = make("kaggriculture", configuration={"seed": SEED}, debug=True)
    environment.run([agent, _pass_agent])
    replay = environment.toJSON()
    replay_path = write_replay(replay, output / "replay.json")
    html_path = output / "replay.html"
    html_path.write_text(render_replay_html(replay), encoding="utf-8")
    passed, log = _audit(replay)
    log_path = output / "opening.log"
    log_path.write_text(log, encoding="utf-8")
    print(log)
    print(f"Replay: {replay_path.resolve()}")
    print(f"Offline HTML copy: {html_path.resolve()}")
    print(f"Action log: {log_path.resolve()}")
    if not passed:
        print("\033[31mFixed opening validation failed.\033[0m", file=sys.stderr)
        return 1
    if not args.no_open:
        print("Starting the existing official replay viewer...")
        sys.stdout.flush()
        viewer = ROOT / "scripts" / "view_replay.py"
        os.execv(sys.executable,
                 [sys.executable, str(viewer), str(replay_path.resolve())])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
