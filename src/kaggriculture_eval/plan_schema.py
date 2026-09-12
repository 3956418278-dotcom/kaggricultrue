"""Frozen player-day record schema; not part of the submitted strategy."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping

Position=tuple[int,int]
@dataclass(frozen=True)
class TimedAmount: step:int; item:str; quantity:int
@dataclass(frozen=True)
class TimedCash: step:int; purpose:str; amount:int
@dataclass(frozen=True)
class CashDimension: upfront:int=0; scheduled:tuple[TimedCash,...]=(); reserve_required:int=0; sunk_cost:int=0
@dataclass(frozen=True)
class TimeDimension: start_step:int; completion_step:int; last_value_step:int; deadlines:tuple[int,...]=()
@dataclass(frozen=True)
class OccupancyInterval: position:Position|None; start_step:int; end_step:int
@dataclass(frozen=True)
class LandDimension: intervals:tuple[OccupancyInterval,...]=(); capacity_created:int=0
@dataclass(frozen=True)
class WorkAmount: day:int; kind:str; actions:int; travel_actions:int=0; position:Position|None=None; deadline_step:int|None=None
@dataclass(frozen=True)
class ActionDimension: work:tuple[WorkAmount,...]=(); capacity_supplied:Mapping[int,int]=field(default_factory=dict)
@dataclass(frozen=True)
class PhysicalDimension: inputs:tuple[TimedAmount,...]=(); outputs:tuple[TimedAmount,...]=(); peak_shed_units:int=0
@dataclass(frozen=True)
class RevenueDimension:
    realized_now:int=0; projected_sales:tuple[TimedAmount,...]=(); projected_gross:int=0; terminal_cash_effect:int=0; terminal_unsold_units:int=0
@dataclass(frozen=True)
class EconomicCommitment:
    identifier:str; kind:str; target:Position|None; existing:bool; cash:CashDimension
    time:TimeDimension; land:LandDimension; actions:ActionDimension
    physical:PhysicalDimension; revenue:RevenueDimension
    metadata:Mapping[str,object]=field(default_factory=dict)
    required_state:Mapping[str,object]=field(default_factory=dict)
    required_outputs:Mapping[str,int]=field(default_factory=dict)
@dataclass(frozen=True)
class EconomicWindow:
    start_turn:int; end_turn:int; market_orders:tuple[tuple[object,...],...]=()
    required_shed:Mapping[str,int]=field(default_factory=dict); minimum_cash:int=0; extra_hands_allowed:int=0
@dataclass(frozen=True)
class Plan:
    obligations:tuple[EconomicCommitment,...]; selected:tuple[EconomicCommitment,...]
    support:tuple[EconomicCommitment,...]; rejected:Mapping[str,str]
    fertilize_targets:frozenset[Position]; animal_purchases:tuple[str,...]
    buy_land:bool; feed_reserve:int; fertilizer_reserve:int
    diagnostics:Mapping[str,object]=field(default_factory=dict)
    starting_animals:Mapping[str,int]=field(default_factory=dict)
    day:int=-1; formed_step:int=-1; revision:int=0; replan_reason:str|None=None
    max_hands:int|None=5; economic_windows:tuple[EconomicWindow,...]=()
