#!/usr/bin/env python3
"""Inspect a retained player-day as real official game scenes and intent tables."""
import argparse
from copy import deepcopy
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kaggriculture_eval.player_days import read_samples
from src.kaggriculture_eval.replay import render_replay_html


def render_sample(sample):
    observations = [t["state_before"] for t in sample["demonstrated_realization"]] + [sample["day_end_state"]]
    frames = []
    for obs in observations:
        pair = []
        for side in (0, 1):
            view = deepcopy(obs)
            view["player"] = side
            if side != sample["side"]:
                view["private"] = {"shed": {}, "seeds": {}, "inventories": [{}]}
            pair.append({"observation": view, "action": {}, "reward": 0, "status": "ACTIVE"})
        frames.append(pair)
    replay = {"name": "kaggriculture", "version": "0.1.0", "module_version": "1.32.7",
              "steps": frames, "configuration": sample["environment"]["configuration"], "info": {}}
    def scene(step):
        return '<iframe title="Official game scene" srcdoc="' + html.escape(render_replay_html(replay, initial_step=step), quote=True) + '"></iframe>'
    def safe(value):
        return html.escape(str(value))
    rows = []
    plan = sample["plan"]
    for goal in plan["obligations"] + plan["selected"] + plan["support"]:
        meta = goal["metadata"]
        effects = "; ".join(f"{k}: {v['before']} → {v['after']}" for k,v in meta.get("required_effect", {}).items())
        rows.append(f"<tr><td>{safe(meta.get('entity', goal['kind']))}</td><td>{safe(goal['target'] or 'open placement')}</td>"
                    f"<td>{safe(effects or meta.get('quadrant', ''))}</td><td>{safe(meta.get('completion_observed', True))}</td></tr>")
    trace = []
    for t in sample["demonstrated_realization"]:
        a = t["action"]
        trace.append(f"<tr><td>{t['step']}</td><td>{safe(a.get('farmer', ['PASS']))}</td>"
                     f"<td>{safe(a.get('hands', []))}</td><td>{safe(a.get('market', []))}</td><td>{len(t['noops'])}</td></tr>")
    return f'''<!doctype html><meta charset="utf-8"><title>Player-day {safe(sample['sample_id'])}</title>
<style>body{{font:15px system-ui;margin:24px;background:#f6f7f3;color:#243128}}iframe{{width:100%;height:680px;border:0;background:white}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:1100px){{.pair{{display:block}}}}</style>
<h1>Player-day {safe(sample['sample_id'])}</h1><p>Reference side {sample['side']}; {sample['actionable_turns']} actionable turns. Only the reference side's private inventory is retained; the opponent's private inventory is unknown, not zero.</p>
<div class="pair"><section><h2>Day-start state</h2>{scene(0)}</section><section><h2>Day-end state</h2>{scene(len(frames)-1)}</section></div>
<h2>Reconstructed Plan</h2><p>Economic effects only. Staffing, routes and selling are not Plan goals. False completion marks an attempted, unfulfilled effect.</p>
<table><tr><th>Entity / goal</th><th>Location constraint</th><th>Required economic effect</th><th>Demonstrated completion</th></tr>{''.join(rows)}</table>
<h2>Demonstrated realization</h2>{scene(0)}<table><tr><th>Step</th><th>Farmer</th><th>Hands</th><th>Separate market actions</th><th>No-ops</th></tr>{''.join(trace)}</table>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--sample", required=True, help="episode:side:day")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = next((s for s in read_samples(args.shard) if s["sample_id"] == args.sample), None)
    if sample is None:
        parser.error("sample not found in shard")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_sample(sample))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
