# Codex: integrate FINAL MILK/EGG v3

Use v3 only; v1/v2 are intermediate. First inspect FINAL WOOL and preserve it. Integrate each product's `*_runtime_v3.py`, runtime JSON, scenario NPZ, OOF calibration, residual support, structural support and price map through the same planner forecast abstraction.

The runtime objective is current PUBLIC state -> full-turn conditional inventory/price scenario distribution to D29. D9-D11 uses current structural gate + the persisted reveal-conditioning policy + reanchoring to current inventory. D12 collapses to the public D12 branch. After D12, every new shop reveal immediately reconditions the scenario set and every current inventory observation updates the whole future path with the stored full-turn rho formula. Apply only 9/06 OOF calibration. 9/14 is frozen validation only.

Persist `d12_inventory` and `d12_branch` in planner state. Never use future shops/state. Return scenario paths/weights, inventory and price Q10/Q25/Q50/Q75/Q90, conditioning level, OOD and confidence. Counterfactual inventory is `I_reference + impact(candidate)-impact(baseline)`. Revenue MUST call official sequential transaction/price-impact semantics per scenario; never q*spot.

Keep patch narrow; do not change opening/routing/trading unrelated logic. Add tests for no future leakage, reveal reconditioning, D12 collapse, exact rho update, ordered quantiles, weights sum 1, candidate==baseline, OOD, sequential revenue, and unchanged WOOL. Implement and run tests, not just a design note.
