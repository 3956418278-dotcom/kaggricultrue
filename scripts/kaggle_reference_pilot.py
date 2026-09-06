#!/usr/bin/env python3
"""Prepare private CPU notebook; extraction runs in Kaggle, never this command.

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
from src.kaggriculture_eval.reference_pipeline import discover_metadata, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--daily-manifest", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--resume-source", help="attach a prior private checkpoint Dataset")
    parser.add_argument("--daily-source", default="kaggle/kaggriculture-episodes-2026-09-04")
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--metadata-queries", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.candidate_limit <= 100:
        raise ValueError("approved pilot cap is 100 candidate episodes")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    owner = api.get_config_value("username")
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "selection.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        leaderboard = api.competition_leaderboard_view("kaggriculture", page_size=10)
        teams = [{"teamId": r.team_id, "teamName": r.team_name, "score": r.score,
                  "submissionDate": r.submission_date.isoformat()} for r in leaderboard]
        config = {"leaderboard": {"observed_at": datetime.now(timezone.utc).isoformat(), "teams": teams},
            "candidate_limit": args.candidate_limit, "sampling_salt": "kaggriculture-player-days-pilot-v1",
            "daily_source": args.daily_source, "index_source": "kaggle/kaggriculture-episodes-index",
            "index_version": 37, "daily_version": 1,
            "daily_manifest_sha256": hashlib.sha256(args.daily_manifest.read_bytes()).hexdigest()}
        write_json(config_path, config)
    metadata = discover_metadata(args.output / "metadata", config["leaderboard"], limit=args.metadata_queries)
    identity = json.loads((ROOT / "references/official/kaggriculture-environment-1.32.7.json").read_text())
    config["official_source_sha256"] = identity["key_file_sha256"]["kaggriculture.py"]
    import csv
    index_rows = list(csv.DictReader(args.index_manifest.open()))
    if not any(r["daily_dataset_slug"] == args.daily_source.split("/")[-1] for r in index_rows):
        raise ValueError("daily source is not present in the official index snapshot")
    config["index_manifest_sha256"] = hashlib.sha256(args.index_manifest.read_bytes()).hexdigest()
    candidate_rows = list(csv.DictReader(args.daily_manifest.open()))
    candidate_rows.sort(key=lambda r: hashlib.sha256((config["sampling_salt"] + r["episode_id"]).encode()).hexdigest())
    candidate_ids = {r["episode_id"] for r in candidate_rows[:config["candidate_limit"]]}
    metadata = {**metadata, "episodes": {k: v for k, v in metadata["episodes"].items() if k in candidate_ids}}
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        # Explicit allowlist; no environments, credentials, user notes or caches.
        paths = [ROOT / "src/__init__.py", *sorted((ROOT / "src/kaggriculture_agent").glob("*.py")),
                 ROOT / "src/kaggriculture_eval/__init__.py", ROOT / "src/kaggriculture_eval/player_days.py",
                 ROOT / "src/kaggriculture_eval/reference_pipeline.py"]
        config["source_hashes"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for path in paths:
            archive.writestr(str(path.relative_to(ROOT)), path.read_bytes())
        archive.writestr("selection.json", json.dumps(config))
        archive.writestr("episode-metadata.json", json.dumps(metadata))
    payload = base64.b64encode(bundle.getvalue()).decode()
    code = f'''import base64, io, json, pathlib, shutil, subprocess, sys, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "kaggle-environments==1.32.7"], check=True)
root = pathlib.Path("/kaggle/working/extractor")
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
    archive.extractall(root)
sys.path.insert(0, str(root))
from src.kaggriculture_eval.reference_pipeline import run_pilot
config = json.loads((root / "selection.json").read_text())
metadata = json.loads((root / "episode-metadata.json").read_text())
candidates = [p.parent for p in pathlib.Path("/kaggle/input").rglob("manifest.csv") if p.parent.name == config["daily_source"].split("/")[-1]]
if len(candidates) != 1:
    raise RuntimeError("daily input mount missing or ambiguous")
resume = {args.resume_source!r}
if resume:
    prior = [p for p in pathlib.Path("/kaggle/input").rglob("extraction-manifest.json") if p.parent.name == resume.split("/")[-1]]
    if len(prior) != 1:
        raise RuntimeError("resume checkpoint missing or ambiguous")
    shutil.copytree(prior[0].parent, "/kaggle/working/player-days", dirs_exist_ok=True)
run_pilot(candidates[0], "/kaggle/working/player-days", config, metadata)
'''
    write_json(args.output / "pilot.ipynb", {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(True)}]})
    if (args.output / "pilot.ipynb").stat().st_size >= 1_000_000:
        raise ValueError("Kaggle notebook source must be under 1 MB; reduce metadata payload")
    write_json(args.output / "kernel-metadata.json", {"id": f"{owner}/kaggriculture-player-day-pilot",
        "title": "Kaggriculture Player Day Pilot", "code_file": "pilot.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": [args.daily_source] + ([args.resume_source] if args.resume_source else []),
        "competition_sources": [], "kernel_sources": []})
    write_json(config_path, config)
    print(f"Prepared private CPU notebook for {owner}; {len(metadata['episodes'])} metadata records. No upload performed.")


if __name__ == "__main__":
    main()
