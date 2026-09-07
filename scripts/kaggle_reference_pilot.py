#!/usr/bin/env python3
"""Prepare a private CPU collection + reconstruction notebook (pilot or scale).

Run with the authenticated tooling environment (conda run -n kaggle python).
Publishing the resulting output is a separate explicit Kaggle CLI operation.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kaggriculture_eval.reference_pipeline import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--daily-manifest", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--resume-source", help="attach a prior private checkpoint Dataset")
    parser.add_argument("--selection-source", type=Path, help="reuse a frozen leaderboard/sampling snapshot")
    parser.add_argument("--metadata-cache", type=Path, help="optional joined metadata bootstrap; cloud performs further collection")
    parser.add_argument("--kernel-slug", default="kaggriculture-player-day-pilot")
    parser.add_argument("--index-version", type=int)
    parser.add_argument("--daily-version", type=int, required=True, help="verified official daily Dataset version")
    parser.add_argument("--extraction-seconds", type=int, default=10800)
    parser.add_argument("--daily-source", default="kaggle/kaggriculture-episodes-2026-09-04")
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--metadata-queries", type=int, default=24)
    args = parser.parse_args()
    if args.candidate_limit < 1 or args.metadata_queries < 0 or args.extraction_seconds < 1:
        raise ValueError("positive bounded candidate/runtime limits and nonnegative query limit required")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    owner = api.get_config_value("username")
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "selection.json"
    if args.selection_source:
        frozen = json.loads(args.selection_source.read_text())
    else:
        leaderboard = api.competition_leaderboard_view("kaggriculture", page_size=10)
        teams = [{"teamId": r.team_id, "teamName": r.team_name, "score": r.score,
                  "submissionDate": r.submission_date.isoformat()} for r in leaderboard]
        frozen = {"leaderboard": {"observed_at": datetime.now(timezone.utc).isoformat(), "teams": teams},
                  "sampling_salt": "kaggriculture-player-days-pilot-v1"}
    index_version = args.index_version or frozen.get("index_version")
    if not index_version:
        raise ValueError("provide verified index version or an identified frozen selection")
    config = {"leaderboard": frozen["leaderboard"], "sampling_salt": frozen["sampling_salt"],
        "candidate_limit": args.candidate_limit, "metadata_queries": args.metadata_queries,
        "extraction_seconds": args.extraction_seconds,
        "daily_source": args.daily_source, "index_source": "kaggle/kaggriculture-episodes-index",
        "index_version": index_version, "daily_version": args.daily_version,
        "daily_manifest_sha256": hashlib.sha256(args.daily_manifest.read_bytes()).hexdigest()}
    metadata = json.loads(args.metadata_cache.read_text()) if args.metadata_cache else {"episodes": {}}
    identity = json.loads((ROOT / "references/official/kaggriculture-environment-1.32.7.json").read_text())
    config["official_source_sha256"] = identity["key_file_sha256"]["kaggriculture.py"]
    import csv
    with args.index_manifest.open() as handle:
        index_rows = list(csv.DictReader(handle))
    if not any(r["daily_dataset_slug"] == args.daily_source.split("/")[-1] for r in index_rows):
        raise ValueError("daily source is not present in the official index snapshot")
    config["index_manifest_sha256"] = hashlib.sha256(args.index_manifest.read_bytes()).hexdigest()
    with args.daily_manifest.open() as handle:
        candidate_rows = list(csv.DictReader(handle))
    candidate_rows.sort(key=lambda r: hashlib.sha256((config["sampling_salt"] + r["episode_id"]).encode()).hexdigest())
    candidate_ids = {r["episode_id"] for r in candidate_rows[:config["candidate_limit"]]}
    if args.candidate_limit > len(candidate_rows):
        raise ValueError("candidate cap exceeds the attached source; choose an explicit source-sized cap")
    metadata = {**metadata, "episodes": {k: v for k, v in metadata["episodes"].items() if k in candidate_ids}}
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        # Explicit allowlist; no environments, credentials, user notes or caches.
        paths = [ROOT / "src/__init__.py", *sorted((ROOT / "src/kaggriculture_agent").glob("*.py")),
                 ROOT / "src/kaggriculture_eval/__init__.py", ROOT / "src/kaggriculture_eval/player_days.py",
                 ROOT / "src/kaggriculture_eval/reference_pipeline.py",
                 ROOT / "src/kaggriculture_eval/reference_audit.py",
                 ROOT / "src/kaggriculture_eval/reference_storage.py",
                 ROOT / "src/kaggriculture_eval/replay.py",
                 ROOT / "scripts/inspect_player_day.py", ROOT / "scripts/audit_player_days.py",
                 ROOT / "scripts/restore_reference_dataset.py"]
        config["source_hashes"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for path in paths:
            archive.writestr(str(path.relative_to(ROOT)), path.read_bytes())
        archive.writestr("selection.json", json.dumps(config))
        archive.writestr("episode-metadata.json", json.dumps(metadata))
        archive.writestr("index-manifest.csv", args.index_manifest.read_bytes())
        archive.writestr("daily-manifest.csv", args.daily_manifest.read_bytes())
    payload = base64.b64encode(bundle.getvalue()).decode()
    code = f'''import base64, hashlib, importlib.metadata, io, json, pathlib, shutil, subprocess, sys, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "kaggle-environments==1.32.7"], check=True)
root = pathlib.Path("/kaggle/working/extractor")
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
    archive.extractall(root)
sys.path.insert(0, str(root))
from src.kaggriculture_eval.reference_pipeline import run_collection
config = json.loads((root / "selection.json").read_text())
metadata = json.loads((root / "episode-metadata.json").read_text())
for name, expected in config["source_hashes"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("extractor source hash mismatch")
if hashlib.sha256((root / "index-manifest.csv").read_bytes()).hexdigest() != config["index_manifest_sha256"]:
    raise RuntimeError("index snapshot hash mismatch")
candidates = [p.parent for p in pathlib.Path("/kaggle/input").rglob("manifest.csv") if p.parent.name == config["daily_source"].split("/")[-1]]
if len(candidates) != 1:
    raise RuntimeError("daily input mount missing or ambiguous")
resume = {args.resume_source!r}
prior = None
if resume:
    prior = [p for p in pathlib.Path("/kaggle/input").rglob("extraction-manifest.json") if p.parent.name == resume.split("/")[-1]]
    if len(prior) != 1:
        raise RuntimeError("resume checkpoint missing or ambiguous")
    from src.kaggriculture_eval.reference_storage import restore_dataset
    restore_dataset(prior[0].parent, "/kaggle/working/restored-checkpoint")
    prior = pathlib.Path("/kaggle/working/restored-checkpoint")
run_collection(candidates[0], "/kaggle/working/player-days", config, metadata,
               cache="/kaggle/working/metadata-cache", resume=prior)
out = pathlib.Path("/kaggle/working/player-days")
for name in ("index-manifest.csv", "daily-manifest.csv"):
    shutil.copy2(root / name, out / name)
with zipfile.ZipFile(out / "extractor-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in config["source_hashes"]:
        archive.write(root / name, name)
    archive.write(root / "episode-metadata.json", "bootstrap-metadata.json")
    archive.write(root / "selection.json", "selection.json")
packages = sorted(d.metadata["Name"] + "==" + d.version for d in importlib.metadata.distributions() if d.metadata["Name"])
(out / "cloud-requirements.txt").write_text("\\n".join(packages) + "\\n")
print("Collection, reconstruction and audit finished in this run. Dataset-ready output:", out)
'''
    write_json(args.output / "pilot.ipynb", {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(True)}]})
    if (args.output / "pilot.ipynb").stat().st_size >= 1_000_000:
        raise ValueError("Kaggle notebook source must be under 1 MB; reduce metadata payload")
    write_json(args.output / "kernel-metadata.json", {"id": f"{owner}/{args.kernel_slug}",
        "title": "Kaggriculture Player Day Collection", "code_file": "pilot.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": [args.daily_source] + ([args.resume_source] if args.resume_source else []),
        "competition_sources": [], "kernel_sources": []})
    write_json(config_path, config)
    print(f"Prepared private CPU notebook for {owner}; {len(metadata['episodes'])} metadata records. No upload performed.")


if __name__ == "__main__":
    main()
