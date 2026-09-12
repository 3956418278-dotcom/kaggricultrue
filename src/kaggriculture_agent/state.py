"""Canonical observation state for the D11+ controller.

The adapter deliberately keeps every official tile field. Planning may derive
counts, but never replaces transition-bearing animal/crop records by counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import rules

Position = tuple[int, int]


def get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _int_map(value: Any, *, clamp: bool = True) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(k): max(0, int(v or 0)) if clamp else int(v or 0) for k, v in value.items()}


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
        return str(self.raw["animal"]) if isinstance(self.raw, Mapping) and "animal" in self.raw else None

    @property
    def crop(self) -> str | None:
        return str(self.raw["crop"]) if isinstance(self.raw, Mapping) and self.raw.get("kind") == "PLANT" else None


@dataclass(frozen=True)
class AssetState:
    asset_type: str
    position: Position
    official: Mapping[str, Any]


@dataclass(frozen=True)
class MarketState:
    inventory: Mapping[str, int]
    price: Mapping[str, int]


@dataclass(frozen=True)
class OwnState:
    money: int
    shed_inventory: Mapping[str, int]
    seeds: Mapping[str, int]
    workers: tuple[WorkerState, ...]
    worker_positions: tuple[Position, ...]
    worker_inventory: tuple[Mapping[str, int], ...]
    animals: tuple[AssetState, ...]
    crops: tuple[AssetState, ...]
    owned_land: tuple[str, ...]
    usable_tiles: tuple[Position, ...]
    tiles: tuple[TileState, ...]
    hires_today: int


@dataclass(frozen=True)
class OppState:
    visible_workers: tuple[Position, ...]
    visible_animals: tuple[AssetState, ...]
    visible_crops: tuple[AssetState, ...]
    owned_land: tuple[str, ...]
    usable_tiles: tuple[Position, ...]
    tiles: tuple[TileState, ...]


@dataclass(frozen=True)
class State:
    step: int
    day: int
    turn: int
    player: int
    board_size: int
    shops: tuple[str, ...]
    market: MarketState
    own: OwnState
    opp: OppState

    @property
    def hour(self): return self.turn
    @property
    def money(self): return self.own.money
    @property
    def tiles(self): return self.own.tiles
    @property
    def workers(self): return self.own.workers
    @property
    def unlocked_quadrants(self): return self.own.owned_land
    @property
    def hires_today(self): return self.own.hires_today
    @property
    def shed(self): return self.own.shed_inventory
    @property
    def seeds(self): return self.own.seeds
    @property
    def market_inventory(self): return self.market.inventory
    @property
    def market_prices(self): return self.market.price
    @property
    def unlocked_shops(self): return self.shops
    @property
    def turns_left(self): return max(0, rules.TERMINAL_ACTION_STEP - self.step + 1)
    @property
    def days_left(self): return self.turns_left / rules.TURNS_PER_DAY
    @property
    def turns_left_today(self): return rules.TURNS_PER_DAY - self.turn
    @property
    def shed_used(self): return sum(self.shed.values())

    def tile_at(self, position: Position) -> TileState:
        return self.tiles[position[1] * self.board_size + position[0]]

    def tiles_of_kind(self, kind: str) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.tiles if tile.kind == kind)

    def empty_tiles(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.tiles if tile.is_empty)

    def crop_tiles(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.tiles if tile.crop is not None)

    def animal_tiles(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.tiles if tile.animal is not None)

    def empty_structures(self, kind: str | None = None) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.tiles if tile.kind in ("COOP", "PASTURE") and tile.animal is None and (kind is None or tile.kind == kind))

    def carried_total(self, item: str) -> int:
        return sum(worker.inventory.get(item, 0) for worker in self.workers)

    def owned_total(self, item: str) -> int:
        return self.shed.get(item, 0) + self.carried_total(item)

    def owned_animals(self, animal: str) -> int:
        return sum(a.asset_type == animal for a in self.own.animals) + self.owned_total(animal)


OwnedState = State


def _tiles(raw_tiles: Any, board_size: int) -> tuple[TileState, ...]:
    return tuple(TileState((x, y), dict(raw_tiles[y][x]) if isinstance(raw_tiles[y][x], Mapping) else raw_tiles[y][x])
                 for y in range(board_size) for x in range(board_size))


def _assets(tiles: tuple[TileState, ...]):
    animals, crops = [], []
    for tile in tiles:
        if not isinstance(tile.raw, Mapping):
            continue
        official = dict(tile.raw)
        if "animal" in official:
            animals.append(AssetState(str(official["animal"]), tile.position, official))
        elif official.get("kind") == "PLANT":
            crops.append(AssetState(str(official["crop"]), tile.position, official))
    return tuple(animals), tuple(crops)


def reconstruct(observation: Any) -> State:
    player = int(get(observation, "player", 0) or 0)
    farms = get(observation, "farms", ()) or ()
    if player < 0 or player >= len(farms):
        raise ValueError("observation has no owned farm")
    if len(farms) != 2:
        raise ValueError("the pinned Kaggriculture contract is two-player")
    own_farm, opp_farm = farms[player], farms[1 - player]
    raw_own_tiles = get(own_farm, "tiles", ()) or ()
    board_size = len(raw_own_tiles) or rules.BOARD_SIZE
    own_tiles = _tiles(raw_own_tiles, board_size)
    opp_tiles = _tiles(get(opp_farm, "tiles", ()) or (), board_size)
    own_animals, own_crops = _assets(own_tiles)
    opp_animals, opp_crops = _assets(opp_tiles)

    private = get(observation, "private", {}) or {}
    inventories = list(get(private, "inventories", ()) or ())
    positions = [get(own_farm, "farmer", (0, 0)), *(get(own_farm, "hands", ()) or ())]
    workers = tuple(WorkerState(i, (int(p[0]), int(p[1])), _int_map(inventories[i] if i < len(inventories) else {})) for i, p in enumerate(positions))
    opp_positions = tuple((int(p[0]), int(p[1])) for p in [get(opp_farm, "farmer", (0, 0)), *(get(opp_farm, "hands", ()) or ())])
    market = get(observation, "market", {}) or {}
    step = int(get(observation, "step", 0) or 0)
    own_land = tuple(str(q) for q in (get(own_farm, "unlocked_quadrants", ("NW",)) or ("NW",)))
    opp_land = tuple(str(q) for q in (get(opp_farm, "unlocked_quadrants", ("NW",)) or ("NW",)))
    usable = lambda tiles: tuple(t.position for t in tiles if not t.is_locked)
    return State(
        step=step,
        day=int(get(observation, "day", step // rules.TURNS_PER_DAY) or 0),
        turn=int(get(observation, "hour", step % rules.TURNS_PER_DAY) or 0),
        player=player,
        board_size=board_size,
        shops=tuple(str(s) for s in (get(get(observation, "town", {}) or {}, "unlocked_shops", ()) or ())),
        market=MarketState(_int_map(get(market, "inventory", {}), clamp=False), _int_map(get(market, "prices", {}))),
        own=OwnState(int(get(own_farm, "money", 0) or 0), _int_map(get(private, "shed", {})), _int_map(get(private, "seeds", {})), workers, tuple(w.position for w in workers), tuple(w.inventory for w in workers), own_animals, own_crops, own_land, usable(own_tiles), own_tiles, int(get(own_farm, "hires_today", 0) or 0)),
        opp=OppState(opp_positions, opp_animals, opp_crops, opp_land, usable(opp_tiles), opp_tiles),
    )
