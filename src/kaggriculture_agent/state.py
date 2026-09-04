"""Observation normalization and owned-state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import rules


Position = tuple[int, int]


def get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): max(0, int(amount or 0)) for key, amount in value.items()}


@dataclass(frozen=True)
class WorkerState:
    index: int
    position: Position
    inventory: Mapping[str, int]

    @property
    def carried(self) -> int:
        return sum(self.inventory.values())


@dataclass(frozen=True)
class TileState:
    position: Position
    raw: Any

    @property
    def is_empty(self) -> bool:
        return self.raw is None

    @property
    def is_locked(self) -> bool:
        return self.raw == "LOCKED"

    @property
    def kind(self) -> str | None:
        return str(self.raw.get("kind")) if isinstance(self.raw, Mapping) else None

    @property
    def animal(self) -> str | None:
        if isinstance(self.raw, Mapping) and "animal" in self.raw:
            return str(self.raw["animal"])
        return None


@dataclass(frozen=True)
class OwnedState:
    step: int
    day: int
    hour: int
    player: int
    money: int
    board_size: int
    tiles: tuple[TileState, ...]
    workers: tuple[WorkerState, ...]
    unlocked_quadrants: tuple[str, ...]
    hires_today: int
    shed: Mapping[str, int]
    seeds: Mapping[str, int]
    market_inventory: Mapping[str, int]
    market_prices: Mapping[str, int]
    unlocked_shops: tuple[str, ...]

    @property
    def turns_left(self) -> int:
        return max(0, rules.TERMINAL_ACTION_STEP - self.step + 1)

    @property
    def days_left(self) -> float:
        return self.turns_left / rules.TURNS_PER_DAY

    @property
    def turns_left_today(self) -> int:
        return rules.TURNS_PER_DAY - self.hour

    @property
    def shed_used(self) -> int:
        return sum(self.shed.values())

    def tile_at(self, position: Position) -> TileState:
        index = position[1] * self.board_size + position[0]
        return self.tiles[index]

    def tiles_of_kind(self, kind: str) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.tiles if tile.kind == kind)

    def empty_tiles(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.tiles if tile.is_empty)

    def crop_tiles(self) -> tuple[TileState, ...]:
        return self.tiles_of_kind("PLANT")

    def animal_tiles(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.tiles if tile.animal is not None)

    def empty_structures(self, kind: str | None = None) -> tuple[TileState, ...]:
        return tuple(
            tile
            for tile in self.tiles
            if tile.kind in ("COOP", "PASTURE")
            and tile.animal is None
            and (kind is None or tile.kind == kind)
        )

    def carried_total(self, item: str) -> int:
        return sum(worker.inventory.get(item, 0) for worker in self.workers)

    def owned_total(self, item: str) -> int:
        return self.shed.get(item, 0) + self.carried_total(item)

    def owned_animals(self, animal: str) -> int:
        placed = sum(tile.animal == animal for tile in self.tiles)
        return placed + self.owned_total(animal)

    def projected_feed_need_today(self) -> int:
        return sum(
            1
            for tile in self.animal_tiles()
            if isinstance(tile.raw, Mapping) and not bool(tile.raw.get("fed_today", False))
        )


def reconstruct(observation: Any) -> OwnedState:
    """Build an immutable owned-state view using only the supplied observation."""
    player = int(get(observation, "player", 0) or 0)
    farms = get(observation, "farms", ()) or ()
    if player < 0 or player >= len(farms):
        raise ValueError("observation has no owned farm")
    farm = farms[player]
    private = get(observation, "private", {}) or {}
    raw_tiles = get(farm, "tiles", ()) or ()
    board_size = len(raw_tiles) or rules.BOARD_SIZE
    tiles = tuple(
        TileState((x, y), raw_tiles[y][x])
        for y in range(board_size)
        for x in range(board_size)
    )
    inventories = list(get(private, "inventories", ()) or ())
    positions = [get(farm, "farmer", (0, 0)), *(get(farm, "hands", ()) or ())]
    workers = tuple(
        WorkerState(
            index=index,
            position=(int(position[0]), int(position[1])),
            inventory=_int_map(inventories[index] if index < len(inventories) else {}),
        )
        for index, position in enumerate(positions)
    )
    market = get(observation, "market", {}) or {}
    town = get(observation, "town", {}) or {}
    step = int(get(observation, "step", 0) or 0)
    return OwnedState(
        step=step,
        day=int(get(observation, "day", step // rules.TURNS_PER_DAY) or 0),
        hour=int(get(observation, "hour", step % rules.TURNS_PER_DAY) or 0),
        player=player,
        money=int(get(farm, "money", 0) or 0),
        board_size=board_size,
        tiles=tiles,
        workers=workers,
        unlocked_quadrants=tuple(get(farm, "unlocked_quadrants", ("NW",)) or ("NW",)),
        hires_today=int(get(farm, "hires_today", 0) or 0),
        shed=_int_map(get(private, "shed", {})),
        seeds=_int_map(get(private, "seeds", {})),
        market_inventory=_int_map(get(market, "inventory", {})),
        market_prices=_int_map(get(market, "prices", {})),
        unlocked_shops=tuple(str(shop) for shop in (get(town, "unlocked_shops", ()) or ())),
    )
