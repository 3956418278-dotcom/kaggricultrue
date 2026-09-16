"""Conditional empirical scenario forecast for animal products (WOOL, MILK, EGG).

Replaces legacy single-point forecast_inventory heuristics for animal value
calculations with empirical scenario distributions reanchored to observed state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import rules
from .market import REVEAL_DAYS, sell_revenue
from .state import State

SHOP_NAMES = (
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)
SHOP_CODE = {s: i for i, s in enumerate(SHOP_NAMES)}


@dataclass(frozen=True)
class ScenarioDistribution:
    """Full-horizon scenario distribution for an animal product."""
    product: str
    step: int
    horizon: int
    weights: np.ndarray  # (S,)
    inventory_paths: np.ndarray  # (S, H)
    price_paths: np.ndarray  # (S, H)
    inventory_q: np.ndarray  # (5, H) ordered: q10, q25, q50, q75, q90
    price_q: np.ndarray  # (5, H) ordered: q10, q25, q50, q75, q90
    state: str
    conditioning_level: str
    scenario_count: int
    ood: dict[str, bool]
    confidence: float

    @property
    def horizon_steps(self) -> int:
        return self.horizon

    def expected_inventory(self, h: int | None = None) -> float | np.ndarray:
        if self.horizon == 0:
            return 0.0 if h is not None else np.array([])
        if h is not None:
            return float(self.weights @ self.inventory_paths[:, h])
        return self.weights @ self.inventory_paths

    def expected_price(self, h: int | None = None) -> float | np.ndarray:
        if self.horizon == 0:
            return 0.0 if h is not None else np.array([])
        if h is not None:
            return float(self.weights @ self.price_paths[:, h])
        return self.weights @ self.price_paths

    def quantile_inventory(self, q_idx: int, h: int | None = None) -> float | np.ndarray:
        if self.horizon == 0:
            return 0.0 if h is not None else np.array([])
        if h is not None:
            return float(self.inventory_q[q_idx, h])
        return self.inventory_q[q_idx]

    def quantile_price(self, q_idx: int, h: int | None = None) -> float | np.ndarray:
        if self.horizon == 0:
            return 0.0 if h is not None else np.array([])
        if h is not None:
            return float(self.price_q[q_idx, h])
        return self.price_q[q_idx]


def _inventory_to_price_paths(product: str, inv_paths: np.ndarray) -> np.ndarray:
    """Exact mapping of inventory paths to price paths via rules.market_price."""
    S, H = inv_paths.shape
    if S == 0 or H == 0:
        return np.zeros((S, H), dtype=float)
    rounded = np.rint(inv_paths).astype(int)
    unq, inv_idx = np.unique(rounded, return_inverse=True)
    unq_prices = np.array([rules.market_price(product, int(v)) for v in unq], dtype=float)
    return unq_prices[inv_idx].reshape(S, H)


def animal_incremental_market_path(
    events: Sequence[tuple[int, int]],
    horizon: int,
    *,
    inclusive: bool = False,
) -> np.ndarray:
    """Cumulative market inventory impact path of shape (horizon,).

    For a sale at step h_k with quantity q_k:
    - If inclusive is False (default for a candidate's own future sales):
      At h <= h_k, the transaction has not yet completed and entered the market,
      so it contributes 0 before/during the transaction.
      At h > h_k, the transaction has completed, so it contributes +q_k.
    - If inclusive is True (for prior accepted sales or counterfactual exits at the same step):
      The sale impact is present at h >= h_k.
    """
    impact = np.zeros(horizon, dtype=float)
    for step_h, qty in events:
        start = step_h if inclusive else step_h + 1
        if 0 <= start <= horizon:
            impact[start:] += qty
        elif start < 0:
            impact[:] += qty
    return impact


def candidate_inventory(
    reference_paths: np.ndarray,
    candidate_impact: np.ndarray | float | int = 0,
    baseline_impact: np.ndarray | float | int = 0,
) -> np.ndarray:
    """Counterfactual inventory I_candidate = I_ref + Delta_cand - Delta_base."""
    return reference_paths + np.asarray(candidate_impact) - np.asarray(baseline_impact)


def is_valid_shops(shops: Sequence[str]) -> bool:
    """1-8 unique legal shops allow empirical forecast; duplicate or invalid shops do not."""
    return len(set(shops)) == len(shops) and all(shop in rules.SHOPS for shop in shops)


def animal_batch_sale_events(
    state: State,
    animal_type: str,
    *,
    raw: Mapping[str, object] | None = None,
    is_new: bool = False,
    end_day: int | None = None,
) -> list[tuple[int, int]]:
    """Generate macro batch sale events [(relative_sale_step, quantity), ...] to terminal.

    An animal accumulates held product; when reaching max_held, it is harvested and
    sold as a batch. Any remaining partial batch at terminal is sold at the last
    production step (or current step if no further productions occur).

    For existing animals (raw provided, is_new=False):
    - Starts accumulation from current held quantity (raw['yield_units']).
    - Existing legal pending CARE bonus is realized on the next production refresh.
    - Future uncommitted care is NOT assumed.

    For new animals (is_new=True or raw=None):
    - Placed at state.day with initial held = 0 and no bonus.
    """
    rule = rules.ANIMALS[animal_type]
    max_held = rule.max_held
    last_refresh_day = end_day if end_day is not None else (rules.TERMINAL_ACTION_STEP // rules.TURNS_PER_DAY - 1)

    if is_new or raw is None:
        placed_day = state.day
        accum = 0
        pending_bonus = 0
        raw_dict: dict[str, object] = {}
    else:
        raw_dict = dict(raw)
        placed_day = int(raw_dict.get("placed_day", state.day))
        accum = int(raw_dict.get("yield_units", 0))
        pending_bonus = int(raw_dict.get("pending_care_bonus", 0))

    events: list[tuple[int, int]] = []
    last_h = 0

    # If current held already exceeds or meets max_held, harvest immediate batch at h = 0
    while accum >= max_held:
        events.append((0, max_held))
        accum -= max_held

    for d in range(state.day, last_refresh_day + 1):
        since = d + 1 - placed_day - rule.first_yield_day
        if since < 0 or since % rule.interval != 0:
            continue

        if is_new or raw is None:
            qty = 1
        else:
            prod_raw = dict(raw_dict)
            prod_raw["animal"] = animal_type
            prod_raw["placed_day"] = placed_day
            if d == state.day:
                if pending_bonus > 0:
                    prod_raw["fed_today"] = True
                    prod_raw["pending_care_bonus"] = pending_bonus
                    qty = rules.animal_production_on_refresh(prod_raw, d)
                    pending_bonus = 0
                else:
                    qty = rules.animal_production_on_refresh(prod_raw, d)
                    if qty <= 0:
                        qty = 1
            else:
                if pending_bonus > 0:
                    prod_raw["fed_today"] = True
                    prod_raw["pending_care_bonus"] = pending_bonus
                    qty = rules.animal_production_on_refresh(prod_raw, d)
                    pending_bonus = 0
                else:
                    prod_raw["fed_today"] = True
                    prod_raw["pending_care_bonus"] = 0
                    qty = rules.animal_production_on_refresh(prod_raw, d)
                    if qty <= 0:
                        qty = 1

        h = (d + 1) * rules.TURNS_PER_DAY - state.step
        last_h = h
        accum += qty
        while accum >= max_held:
            events.append((h, max_held))
            accum -= max_held

    if accum > 0:
        events.append((last_h, accum))

    return events


def scenario_batch_sale_value(
    product: str,
    reference_paths: np.ndarray,
    batch_sale_events: Sequence[tuple[int, int]],
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Sequential transaction price-impact revenue across scenarios for multi-batch sales.

    Evaluates batch sales strictly in chronological order:
    1. At sale step h_k, batch k sells into the current counterfactual inventory,
       which includes impacts of previous batches (h_j < h_k or earlier in sequence)
       but NOT this batch k or future batches.
    2. After sale step h_k (i.e. at h > h_k), market inventory increases by q_k.
    """
    S, H = reference_paths.shape
    if S == 0 or H == 0 or not batch_sale_events:
        return {"expected": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0}

    w = np.ones(S) / S if weights is None else np.asarray(weights) / np.sum(weights)
    current_inv = reference_paths.copy()
    scenario_revenue = np.zeros(S, dtype=float)

    sorted_events = sorted(batch_sale_events, key=lambda x: x[0])
    for step_h, qty in sorted_events:
        if qty <= 0:
            continue
        if 0 <= step_h < H:
            for s in range(S):
                scenario_revenue[s] += sell_revenue(
                    product, qty, int(round(current_inv[s, step_h]))
                )
            if step_h + 1 < H:
                current_inv[:, step_h + 1:] += qty
        elif step_h < 0:
            current_inv += qty

    return {
        "expected": float(w @ scenario_revenue),
        "q10": float(np.quantile(scenario_revenue, 0.10)),
        "q50": float(np.quantile(scenario_revenue, 0.50)),
        "q90": float(np.quantile(scenario_revenue, 0.90)),
    }


def scenario_revenue_value(
    product: str,
    candidate_inv: np.ndarray,
    quantities: Sequence[int],
    step_indices: Sequence[int],
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Sequential transaction price-impact revenue aggregated across scenarios."""
    S, H = candidate_inv.shape
    if S == 0 or H == 0:
        return {"expected": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0}
    w = np.ones(S) / S if weights is None else np.asarray(weights) / np.sum(weights)
    v = np.zeros(S, dtype=float)
    for s in range(S):
        tot = 0
        for q, h in zip(quantities, step_indices):
            if 0 <= h < H:
                tot += sell_revenue(product, q, int(round(candidate_inv[s, h])))
        v[s] = tot
    return {
        "expected": float(w @ v),
        "q10": float(np.quantile(v, 0.10)),
        "q50": float(np.quantile(v, 0.50)),
        "q90": float(np.quantile(v, 0.90)),
    }


def _model_dir(product: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "animal_calculation_value_model" / product.lower()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Model directory for {product} not found at {candidate}")


class _MilkEggForecaster:
    def __init__(self, product: str):
        self.product = product.lower()
        self.d = _model_dir(product)
        self.cfg = json.load(open(self.d / f"{self.product}_final_runtime_bundle_v3.json"))
        z = np.load(self.d / f"{self.product}_final_scenario_library_v3.npz", allow_pickle=True)
        self.z = {k: z[k] for k in z.files}
        self.cal = pd.read_csv(self.d / f"{self.product}_final_calibration_all_oof_0906.csv")
        self.rs = pd.read_csv(self.d / f"{self.product}_final_residual_support_0906.csv")

    def _gate(self, s: Mapping[str, Any]) -> int:
        if self.product == "milk":
            if s["cow_age35"] <= 2:
                return 0 if s["cow_age_mean"] <= 6.20714282989502 else 1
            return 2 if s["cow_total"] <= 17 else 3
        shops = s["shops"]
        fp = lambda x: min([i + 1 for i, y in enumerate(shops) if y == x], default=0)
        br, ba = fp("BRUNCH_SPOT"), fp("BAKERY")
        return (0 if ba == 0 else 1) if br == 0 else (2 if ba <= 1 else 3)

    def _branch(self, s: Mapping[str, Any]) -> int:
        if self.product == "milk":
            if s["cow_bonus"] <= 31.5:
                return 0 if s["cow_age_mean"] <= 9.20714282989502 else 1
            return 2 if s["cow_bonus"] <= 46.5 else 3
        shops = s["shops"]
        fp = lambda x: min([i + 1 for i, y in enumerate(shops) if y == x], default=0)
        br, ba = fp("BRUNCH_SPOT"), fp("BAKERY")
        if br == 0:
            return 0 if ba <= 1 else 1
        opp_money = s.get("opponent_money")
        if opp_money is None:
            return -1
        return 2 if opp_money <= 10134 else 3

    def _scenario_shops(self, idx: np.ndarray, k: int) -> list[list[str]]:
        return [[SHOP_NAMES[c] for c in row[:k] if c >= 0] for row in self.z["shop_codes"][idx]]

    def _sig(self, shops: Sequence[str]) -> tuple[str, ...]:
        if self.product == "milk":
            mp = {"PIZZA_SHOP": "P", "ICE_CREAM_SHOP": "I", "SMOOTHIE_SHOP": "S"}
        else:
            mp = {"BAKERY": "B", "BRUNCH_SPOT": "R"}
        return tuple(mp.get(x, "O") for x in shops)

    def _filter(self, pool: np.ndarray, shops: Sequence[str], stage: str) -> tuple[np.ndarray, str]:
        cfg = self.cfg["reveal_filter_0906_selected"][stage]
        p = cfg["policy"]
        th = int(cfg["threshold"])
        k = len(shops)
        if p == "none" or k == 0:
            return pool, "state"
        seq = self._scenario_shops(pool, k)
        exact = np.array([tuple(x) == tuple(shops) for x in seq])
        rel = np.array([self._sig(x) == self._sig(shops) for x in seq])
        a = pool[exact]
        b = pool[rel]
        if p == "relevant":
            return (b, "relevant") if len(b) >= th else (pool, "state_fallback")
        if p == "exact":
            return (a, "exact") if len(a) >= th else (pool, "state_fallback")
        if len(a) >= th:
            return a, "exact"
        if len(b) >= th:
            return b, "relevant"
        return pool, "state_fallback"

    def _cal(self, step: int) -> pd.Series:
        if step < 311:
            d = max(9, min(11, (step + 1) // 24 - 1))
            q = self.cal[self.cal.stage == f"D{d}"]
            return q.iloc[0] if len(q) else self.cal.iloc[0]
        od = max(0, (step - 311) / 24)
        q = self.cal[self.cal.stage.str.startswith("D12+")].copy()
        i = (q.obs_days - od).abs().argmin()
        return q.iloc[i]

    def forecast(self, s: Mapping[str, Any]) -> ScenarioDistribution:
        step = int(s["step"])
        cur = float(s["inventory"])
        shops = list(s.get("shops", []))
        N = len(self.z["inventory_full"])
        residual_ood = False

        if step > rules.TERMINAL_ACTION_STEP:
            return ScenarioDistribution(
                product=self.product.upper(), step=step, horizon=0,
                weights=np.array([]), inventory_paths=np.zeros((0, 0)),
                price_paths=np.zeros((0, 0)), inventory_q=np.zeros((5, 0)),
                price_q=np.zeros((5, 0)), state="terminal", conditioning_level="none",
                scenario_count=0, ood={"reveal_ood": False, "residual_ood": False, "price_ood": False},
                confidence=1.0,
            )

        if step < 239:
            pool = np.arange(N)
            cand, level = self._filter(pool, shops, "pre")
            paths = self.z["inventory_full"][cand, step:].astype(float) - self.z["inventory_full"][cand, step, None] + cur
            state_desc = "broad"
        elif step < 311:
            day = 9 if step < 264 else (10 if step < 288 else 11)
            g = self._gate(s)
            pool = np.where(self.z[f"gate_d{day}"] == g)[0]
            cand, level = self._filter(pool, shops, "pre")
            paths = self.z["inventory_full"][cand, step:].astype(float) - self.z["inventory_full"][cand, step, None] + cur
            state_desc = f"G{g}"
        elif step == 311:
            b = self._branch(s)
            if b < 0:
                # Insufficient information fallback for EGG when opponent money is unavailable
                cand, level = self._filter(np.arange(N), shops, "pre")
                paths = self.z["inventory_full"][cand, step:].astype(float) - self.z["inventory_full"][cand, step, None] + cur
                state_desc = "insufficient_info"
            else:
                d12 = cur
                pool = np.where(self.z["d12_branch"] == b)[0]
                cand, level = self._filter(pool, shops, "post")
                delta = self.z["d12_delta"][cand].astype(float)
                paths = d12 + delta
                state_desc = f"B{b}"
        else:
            # step > 311 (post-D12)
            has_b = "d12_branch" in s and s["d12_branch"] is not None and int(s["d12_branch"]) >= 0
            has_inv = "d12_inventory" in s and s["d12_inventory"] is not None and np.isfinite(float(s["d12_inventory"]))
            if has_b and has_inv:
                b = int(s["d12_branch"])
                d12 = float(s["d12_inventory"])
                pool = np.where(self.z["d12_branch"] == b)[0]
                cand, level = self._filter(pool, shops, "post")
                e = min(step - 311, self.z["d12_delta"].shape[1] - 1)
                cen = self.z[f"center_b{b}"].astype(float)
                base = self.z["d12_delta"][cand].astype(float)
                rho = self.z[f"rho_b{b}"][e, e:].astype(float)
                robs = (cur - d12) - cen[e]
                R = base - cen
                delta = cen[e:][None, :] + rho[None, :] * robs + (R[:, e:] - rho[None, :] * R[:, [e]])
                near = self.rs.iloc[(self.rs.obs_days - (e / 24)).abs().argsort()[:1]]
                r = near[near.branch == b]
                if len(r):
                    residual_ood = bool(robs < r.q01.iloc[0] or robs > r.q99.iloc[0])
                paths = d12 + delta
                state_desc = f"B{b}"
            else:
                # Late-attach fallback: re-anchor empirical scenarios without rho correction
                pool = np.arange(N)
                cand, level = self._filter(pool, shops, "pre")
                paths = self.z["inventory_full"][cand, step:].astype(float) - self.z["inventory_full"][cand, step, None] + cur
                state_desc = "late_fallback"
                level = "late_fallback"

        # Re-anchor strictly to current inventory at h = 0
        paths[:, 0] = cur

        qs = np.quantile(paths, [.1, .25, .5, .75, .9], axis=0)
        if state_desc != "late_fallback" and state_desc != "insufficient_info":
            c = self._cal(step)
            med = qs[2].copy()
            qs[0] = med - c["cal_mult80"] * (med - qs[0])
            qs[4] = med + c["cal_mult80"] * (qs[4] - med)
            qs[1] = med - c["cal_mult50"] * (med - qs[1])
            qs[3] = med + c["cal_mult50"] * (qs[3] - med)

        # Price paths mapped from integer rounded inventory using rules.market_price
        pp = _inventory_to_price_paths(self.product.upper(), paths)
        pq = np.stack([
            np.array([rules.market_price(self.product.upper(), int(round(x))) for x in qs[4 - i]])
            for i in range(5)
        ])
        pq = np.sort(pq, axis=0)

        reveal_ood = "fallback" in level
        flags = {
            "reveal_ood": bool(reveal_ood),
            "residual_ood": bool(residual_ood),
            "price_ood": False,
        }
        if state_desc == "late_fallback":
            flags["late_fallback"] = True
            conf = 0.3
        elif state_desc == "insufficient_info":
            flags["insufficient_info"] = True
            conf = 0.3
        else:
            conf = max(0.2, 1.0 - 0.2 * sum(flags.values()))

        return ScenarioDistribution(
            product=self.product.upper(),
            step=step,
            horizon=paths.shape[1],
            weights=np.ones(len(cand)) / len(cand),
            inventory_paths=paths,
            price_paths=pp,
            inventory_q=qs,
            price_q=pq,
            state=state_desc,
            conditioning_level=level,
            scenario_count=len(cand),
            ood=flags,
            confidence=conf,
        )


class _WoolForecaster:
    def __init__(self):
        self.product = "wool"
        self.d = _model_dir("WOOL")
        self.cfg = json.load(open(self.d / "wool_function_final_v2.json"))
        z = np.load(self.d / "wool_final_scenario_library_v2.npz", allow_pickle=True)
        self.z = {k: z[k] for k in z.files}
        self.trans = pd.read_csv(self.d / "wool_final_d9_to_d12_transition.csv")
        self.rs = pd.read_csv(self.d / "wool_final_residual_support_0906.csv")

    def _gate(self, s: Mapping[str, Any]) -> int:
        sheep_age9p = s.get("sheep_age9p", 0)
        shops = s.get("shops", [])
        yarn_indices = [i for i, shop in enumerate(shops) if shop == "YARN_STORE"]
        if yarn_indices and min(yarn_indices) < 2:
            return 2
        if len(shops) >= 3 and shops[2] == "YARN_STORE":
            return 3
        return 0 if sheep_age9p <= 3 else 1

    def _branch(self, s: Mapping[str, Any]) -> int:
        sheep_age9p = s.get("sheep_age9p", 0)
        shops = s.get("shops", [])
        yarn_indices = [i for i, shop in enumerate(shops) if shop == "YARN_STORE"]
        if not yarn_indices:
            return 0 if sheep_age9p <= 3 else 1
        pull = sum((13 - REVEAL_DAYS[i]) * 12 for i in yarn_indices if i < len(REVEAL_DAYS) and REVEAL_DAYS[i] <= 12)
        return 3 if pull > 126 else 2

    def forecast(self, s: Mapping[str, Any]) -> ScenarioDistribution:
        step = int(s["step"])
        cur = float(s["inventory"])
        shops = list(s.get("shops", []))
        residual_ood = False

        if step > rules.TERMINAL_ACTION_STEP:
            return ScenarioDistribution(
                product="WOOL", step=step, horizon=0,
                weights=np.array([]), inventory_paths=np.zeros((0, 0)),
                price_paths=np.zeros((0, 0)), inventory_q=np.zeros((5, 0)),
                price_q=np.zeros((5, 0)), state="terminal", conditioning_level="none",
                scenario_count=0, ood={"reveal_ood": False, "residual_ood": False, "price_ood": False},
                confidence=1.0,
            )

        if step < 239:
            raise ValueError(
                f"WOOL empirical scenario library only available from step 239 (Day 10); step {step} must use legacy forecast_inventory."
            )

        d12_full_delta = np.hstack([
            np.zeros((len(self.z["d12_branch"]), 1)),
            self.z["d12_delta_inventory"].astype(float),
        ])

        if step < 311:
            g = self._gate(s)
            cand = np.where(self.z["d9_state"] == g)[0]
            if len(cand) == 0:
                cand = np.arange(len(self.z["d9_state"]))
            e = max(0, step - 240)
            delta = self.z["d9_delta_inventory"][cand, e:].astype(float) - self.z["d9_delta_inventory"][cand, e, None].astype(float)
            paths = cur + delta
            state_desc = f"G{g}"
            level = "state"
        elif step == 311:
            b = int(s.get("d12_branch", self._branch(s)))
            cand = np.where(self.z["d12_branch"] == b)[0]
            if len(cand) == 0:
                cand = np.arange(len(self.z["d12_branch"]))
            delta = d12_full_delta[cand, 0:]
            paths = cur + delta
            state_desc = f"B{b}"
            level = "branch"
        else:
            # step > 311 (post-D12)
            has_b = "d12_branch" in s and s["d12_branch"] is not None and int(s["d12_branch"]) >= 0
            has_inv = "d12_inventory" in s and s["d12_inventory"] is not None and np.isfinite(float(s["d12_inventory"]))
            e = min(step - 311, d12_full_delta.shape[1] - 1)
            if has_b and has_inv:
                b = int(s["d12_branch"])
                d12 = float(s["d12_inventory"])
                cand = np.where(self.z["d12_branch"] == b)[0]
                if len(cand) == 0:
                    cand = np.arange(len(self.z["d12_branch"]))
                base = d12_full_delta[cand]
                cen = np.mean(base, axis=0)
                robs = (cur - d12) - cen[e]
                obs_days = e / 24
                obs_idx = int(np.argmin(np.abs(np.array([int(x) for x in self.z["residual_obs_days"]]) - obs_days)))
                rho_408 = self.z["residual_rho"][obs_idx, b, :].astype(float)
                rho_slice = rho_408[e - 1:]
                if len(rho_slice) > 0 and rho_slice[0] != 0:
                    rho_slice = rho_slice / rho_slice[0]
                R = base - cen
                delta = cen[e:][None, :] + rho_slice[None, :] * robs + (R[:, e:] - rho_slice[None, :] * R[:, [e]])
                near = self.rs.iloc[(self.rs.obs_days_after_D12 - obs_days).abs().argsort()[:1]]
                r = near[near.branch == b]
                if len(r):
                    residual_ood = bool(robs < r.residual_q01.iloc[0] or robs > r.residual_q99.iloc[0])
                paths = d12 + delta
                state_desc = f"B{b}"
                level = "branch"
            else:
                # Late-attach fallback without fake anchor or rho correction
                cand = np.arange(len(self.z["d12_branch"]))
                delta = d12_full_delta[cand, e:] - d12_full_delta[cand, e, None]
                paths = cur + delta
                state_desc = "late_fallback"
                level = "late_fallback"

        # Re-anchor strictly to current inventory at h = 0
        paths[:, 0] = cur

        qs = np.quantile(paths, [.1, .25, .5, .75, .9], axis=0)
        pp = _inventory_to_price_paths("WOOL", paths)
        pq = np.stack([
            np.array([rules.market_price("WOOL", int(round(x))) for x in qs[4 - i]])
            for i in range(5)
        ])
        pq = np.sort(pq, axis=0)

        flags = {
            "reveal_ood": False,
            "residual_ood": bool(residual_ood),
            "price_ood": False,
        }
        if state_desc == "late_fallback":
            flags["late_fallback"] = True
            conf = 0.3
        else:
            conf = max(0.2, 1.0 - 0.2 * sum(flags.values()))

        return ScenarioDistribution(
            product="WOOL",
            step=step,
            horizon=paths.shape[1],
            weights=np.ones(len(cand)) / len(cand),
            inventory_paths=paths,
            price_paths=pp,
            inventory_q=qs,
            price_q=pq,
            state=state_desc,
            conditioning_level=level,
            scenario_count=len(cand),
            ood=flags,
            confidence=conf,
        )


@lru_cache(maxsize=3)
def _get_forecaster(product: str) -> _WoolForecaster | _MilkEggForecaster:
    prod_upper = product.upper()
    if prod_upper == "WOOL":
        return _WoolForecaster()
    if prod_upper in ("MILK", "EGG"):
        return _MilkEggForecaster(prod_upper)
    raise ValueError(f"Scenario forecaster is only available for WOOL, MILK, EGG; got {product}")


@dataclass
class AnimalMarketContext:
    """Per-player session context maintaining anchor observations and turn forecast caches."""
    d12_anchors: dict[str, tuple[int, float]] = field(default_factory=dict)
    reference_cache: dict[tuple[int, str, float], ScenarioDistribution] = field(default_factory=dict)

    def reset(self) -> None:
        self.d12_anchors.clear()
        self.reference_cache.clear()

    def observe_turn(self, state: State) -> None:
        self.reference_cache = {k: v for k, v in self.reference_cache.items() if k[0] == state.step}
        if state.step != 311:
            return
        for prod in ("WOOL", "MILK", "EGG"):
            if prod in self.d12_anchors:
                continue
            forecaster = _get_forecaster(prod)
            feats = _extract_state_features(state, prod)
            branch = forecaster._branch(feats)
            cur_inv = float(state.market.inventory.get(prod, rules.MARKET_I0))
            if branch >= 0:
                self.d12_anchors[prod] = (branch, cur_inv)


def _extract_state_features(state: State, product: str, **kwargs) -> dict[str, Any]:
    prod_upper = product.upper()
    cur_inv = float(kwargs.get("inventory", state.market.inventory.get(prod_upper, rules.MARKET_I0)))
    features: dict[str, Any] = {
        "step": int(kwargs.get("step", state.step)),
        "inventory": cur_inv,
        "shops": list(kwargs.get("shops", state.shops)),
    }
    for k in ("d12_branch", "d12_inventory"):
        if k in kwargs:
            features[k] = kwargs[k]

    if prod_upper == "WOOL":
        sheep = [a for a in (*state.own.animals, *state.opp.visible_animals) if a.asset_type == "SHEEP"]
        features["sheep_total"] = len(sheep)
        if state.day >= 9:
            features["sheep_age9p"] = sum(1 for a in sheep if (state.day - int(a.official.get("placed_day", 0))) >= 9)
        else:
            features["sheep_age9p"] = sum(1 for a in sheep if int(a.official.get("placed_day", 0)) <= 0)
    elif prod_upper == "MILK":
        cows = [a for a in (*state.own.animals, *state.opp.visible_animals) if a.asset_type == "COW"]
        cow_ages = [state.day - int(a.official.get("placed_day", 0)) for a in cows]
        features["cow_total"] = len(cows)
        features["cow_ages"] = cow_ages
        features["cow_age35"] = sum(1 for age in cow_ages if 3 <= age <= 5)
        features["cow_age_mean"] = float(np.mean(cow_ages)) if cow_ages else 0.0
        features["cow_bonus"] = float(sum(int(a.official.get("pending_care_bonus", 0)) for a in cows))
    elif prod_upper == "EGG":
        geese = [a for a in (*state.own.animals, *state.opp.visible_animals) if a.asset_type == "GOOSE"]
        features["goose_total"] = len(geese)
        features["opponent_money"] = kwargs.get("opponent_money", getattr(state.opp, "money", None))

    return features


def forecast_product_distribution(
    state: State,
    product: str,
    context: AnimalMarketContext | None = None,
    **kwargs,
) -> ScenarioDistribution:
    """Public entry point: forecast full scenario distribution for WOOL, MILK, or EGG."""
    prod_upper = product.upper()
    cur_inv = float(kwargs.get("inventory", state.market.inventory.get(prod_upper, rules.MARKET_I0)))
    step = int(kwargs.get("step", state.step))
    key = (step, prod_upper, cur_inv)
    if context is not None and not kwargs and key in context.reference_cache:
        return context.reference_cache[key]

    forecaster = _get_forecaster(product)
    s = _extract_state_features(state, product, **kwargs)
    if context is not None and prod_upper in context.d12_anchors:
        if "d12_branch" not in s:
            s["d12_branch"] = context.d12_anchors[prod_upper][0]
        if "d12_inventory" not in s:
            s["d12_inventory"] = context.d12_anchors[prod_upper][1]

    dist = forecaster.forecast(s)
    if context is not None and not kwargs:
        context.reference_cache[key] = dist
    return dist
