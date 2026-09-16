"""Conditional empirical scenario forecast for animal products (WOOL, MILK, EGG).

Replaces legacy single-point forecast_inventory heuristics for animal value
calculations with empirical scenario distributions reanchored to observed state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
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


def candidate_inventory(
    reference_paths: np.ndarray,
    candidate_impact: np.ndarray | float | int = 0,
    baseline_impact: np.ndarray | float | int = 0,
) -> np.ndarray:
    """Counterfactual inventory I_candidate = I_ref + Delta_cand - Delta_base."""
    return reference_paths + np.asarray(candidate_impact) - np.asarray(baseline_impact)


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
    base = repo_root / "animal_calculation_value_model"
    prod_upper = product.upper()
    if prod_upper == "WOOL":
        candidate = base / "Kaggriculture_WOOL_final_v2"
        if candidate.exists():
            return candidate
    elif prod_upper in ("MILK", "EGG"):
        sub = "milk" if prod_upper == "MILK" else "egg"
        candidates = [
            base / "Kaggriculture_MILK_EGG_final_v3" / "Kaggriculture_MILK_EGG_final_v3" / sub,
            base / "Kaggriculture_MILK_EGG_final_v3" / sub,
        ]
        for c in candidates:
            if c.exists():
                return c
    raise FileNotFoundError(f"Model directory for {product} not found under {base}")


class _MilkEggForecaster:
    def __init__(self, product: str):
        self.product = product.lower()
        self.d = _model_dir(product)
        self.cfg = json.load(open(self.d / f"{self.product}_final_runtime_bundle_v3.json"))
        z = np.load(self.d / f"{self.product}_final_scenario_library_v3.npz", allow_pickle=True)
        self.z = {k: z[k] for k in z.files}
        self.price = pd.read_csv(self.d / f"{self.product}_price_map_0906_v1.csv").sort_values("inventory")
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
        return (0 if ba <= 1 else 1) if br == 0 else (2 if s["opponent_money"] <= 10134 else 3)

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

    def _price(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xx = self.price.inventory.to_numpy(float)
        yy = self.price.price.to_numpy(float)
        a = np.asarray(x, float)
        v = np.interp(a, xx, yy, left=yy[0], right=yy[-1])
        return v, (a < xx[0]) | (a > xx[-1])

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
        else:
            b = int(s.get("d12_branch", self._branch(s)))
            pool = np.where(self.z["d12_branch"] == b)[0]
            cand, level = self._filter(pool, shops, "post")
            e = min(step - 311, self.z["d12_delta"].shape[1] - 1)
            cen = self.z[f"center_b{b}"].astype(float)
            base = self.z["d12_delta"][cand].astype(float)
            if e == 0:
                delta = base[:, e:]
                d12 = cur
            else:
                cen_d12 = float(np.mean(self.z["inventory_full"][cand, 311]))
                d12 = float(s.get("d12_inventory", cen_d12))
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

        # Re-anchor strictly to current inventory at h = 0
        paths[:, 0] = cur

        qs = np.quantile(paths, [.1, .25, .5, .75, .9], axis=0)
        c = self._cal(step)
        med = qs[2].copy()
        qs[0] = med - c["cal_mult80"] * (med - qs[0])
        qs[4] = med + c["cal_mult80"] * (qs[4] - med)
        qs[1] = med - c["cal_mult50"] * (med - qs[1])
        qs[3] = med + c["cal_mult50"] * (qs[3] - med)
        qs = np.sort(qs, axis=0)

        pp, pood = self._price(paths)
        pq = np.stack([self._price(qs[4])[0], self._price(qs[3])[0], self._price(qs[2])[0], self._price(qs[1])[0], self._price(qs[0])[0]])
        pq = np.sort(pq, axis=0)

        reveal_ood = "fallback" in level
        flags = {
            "reveal_ood": bool(reveal_ood),
            "residual_ood": bool(residual_ood),
            "price_ood": bool(np.any(pood)),
        }
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

    def _price(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xx = self.z["price_inventory_keys"].astype(float)
        yy = self.z["price_values"].astype(float)
        a = np.asarray(x, float)
        v = np.interp(a, xx, yy, left=yy[0], right=yy[-1])
        return v, (a < xx[0]) | (a > xx[-1])

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
            g = self._gate(s)
            cand = np.where(self.z["d9_state"] == g)[0]
            if len(cand) == 0:
                cand = np.arange(len(self.z["d9_state"]))

            # Deterministic simulation from step to 239
            inv_hist = [cur]
            yarn_revealed = any(shop == "YARN_STORE" for shop in shops)
            for t in range(step, 239):
                delta_t = 0
                if yarn_revealed and (t + 1) % 4 == 0:
                    delta_t -= 2
                if (t + 1) % 24 == 0:
                    delta_t -= 1
                inv_hist.append(inv_hist[-1] + delta_t)

            anchor_239 = inv_hist[-1]
            pre_paths = np.tile(np.array(inv_hist), (len(cand), 1))
            delta_240 = self.z["d9_delta_inventory"][cand].astype(float)
            post_paths = anchor_239 + delta_240
            paths = np.hstack([pre_paths, post_paths])
            state_desc = f"G{g}"
            level = "state"

        elif step < 311:
            g = self._gate(s)
            cand = np.where(self.z["d9_state"] == g)[0]
            if len(cand) == 0:
                cand = np.arange(len(self.z["d9_state"]))
            e = step - 240
            delta = self.z["d9_delta_inventory"][cand, e:].astype(float) - self.z["d9_delta_inventory"][cand, e, None].astype(float)
            paths = cur + delta
            state_desc = f"G{g}"
            level = "state"

        else:
            b = int(s.get("d12_branch", self._branch(s)))
            cand = np.where(self.z["d12_branch"] == b)[0]
            if len(cand) == 0:
                cand = np.arange(len(self.z["d12_branch"]))
            d12_full_delta = np.hstack([np.zeros((len(self.z["d12_branch"]), 1)), self.z["d12_delta_inventory"].astype(float)])
            base = d12_full_delta[cand]
            cen = np.mean(base, axis=0)
            e = min(step - 311, base.shape[1] - 1)
            if e == 0:
                delta = base[:, e:]
                d12 = cur
            else:
                d12 = float(s.get("d12_inventory", cur - cen[e]))
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

        # Re-anchor strictly to current inventory at h = 0
        paths[:, 0] = cur

        qs = np.quantile(paths, [.1, .25, .5, .75, .9], axis=0)
        qs = np.sort(qs, axis=0)
        pp, pood = self._price(paths)
        pq = np.stack([self._price(qs[4])[0], self._price(qs[3])[0], self._price(qs[2])[0], self._price(qs[1])[0], self._price(qs[0])[0]])
        pq = np.sort(pq, axis=0)

        flags = {
            "reveal_ood": False,
            "residual_ood": bool(residual_ood),
            "price_ood": bool(np.any(pood)),
        }
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
        features["opponent_money"] = getattr(state.opp, "money", 0)

    return features


def forecast_product_distribution(state: State, product: str, **kwargs) -> ScenarioDistribution:
    """Public entry point: forecast full scenario distribution for WOOL, MILK, or EGG."""
    forecaster = _get_forecaster(product)
    s = _extract_state_features(state, product, **kwargs)
    return forecaster.forecast(s)
