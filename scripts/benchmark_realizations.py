#!/usr/bin/env python3
"""Freeze an episode split and compare complete models, not ablated variants."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import importlib.metadata
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kaggriculture_agent.temporal_model import TemporalConfig
from src.kaggriculture_eval.player_days import read_samples
from src.kaggriculture_eval.realization_benchmark import compare_sample, episode_partition, source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--episode-limit", type=int, default=2)
    parser.add_argument("--days", type=int, nargs="+", default=[0, 8, 16, 24, 29])
    parser.add_argument("--deterministic-time", type=float, default=2)
    args = parser.parse_args()
    files = sorted(args.samples.glob("episode-*.jsonl.gz"))
    if not files:
        parser.error("no audited episode shards found")
    config = TemporalConfig(deterministic_time=args.deterministic_time)
    sources = source_identity()
    members = {p.stem.split(".")[0].removeprefix("episode-"): {
        "partition": episode_partition(p.stem.split(".")[0].removeprefix("episode-")),
        "path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in files}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"split_rule": "sha256(temporal-reference-split-20260906:<episode>) mod 5; zero held-out",
        "members": members, "sources": sources, "configuration": asdict(config),
        "days": args.days, "episode_limit": args.episode_limit, "evidence": "development-only",
        "comparison_background": "official interpreter, opponent PASS, no random weeds; recorded reference scored separately"}
    from kaggle_environments.envs.kaggriculture import kaggriculture as official
    manifest["runtime"] = {"python": sys.version, "distributions": {name: importlib.metadata.version(name)
        for name in ("ortools", "protobuf", "kaggle-environments")},
        "official_source_sha256": sha256(Path(official.__file__).read_bytes()).hexdigest()}
    for source, digest in sources.items():
        data = Path(source).read_bytes()
        if sha256(data).hexdigest() != digest:
            raise RuntimeError("source changed during benchmark preparation")
        target = args.output / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if args.freeze_only:
        print(json.dumps({"episodes": len(members), "held_out": sum(m["partition"] == "held-out" for m in members.values())}))
        return
    chosen = sorted((e for e, m in members.items() if m["partition"] == "development"),
                    key=lambda e: sha256(f"temporal-development-order:{e}".encode()).hexdigest())[:args.episode_limit]
    for episode in chosen:
        for sample in read_samples(members[episode]["path"]):
            if sample["day"] not in args.days:
                continue
            result = compare_sample(sample, config)
            target = args.output / (sample["sample_id"].replace(":", "-")+".json")
            target.write_text(json.dumps(result, separators=(",", ":"))+"\n")
            print(json.dumps({"sample": sample["sample_id"], **result["comparison"],
                "reference_completion": result["reference"]["completed"],
                "candidate_completion": result["candidate"]["completed"],
                "diagnostics": {k: result["candidate"]["diagnostics"].get(k) for k in
                    ("status", "seconds", "deterministic_time", "initialization", "search_improved_start", "planning_failure")}}), flush=True)


if __name__ == "__main__":
    main()
