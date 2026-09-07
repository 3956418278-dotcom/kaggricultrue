#!/usr/bin/env python3
"""Stage audited, allowlisted files for a private Kaggle Dataset; no upload.

Run in the pinned project environment. Upload the resulting directory with the
authenticated Kaggle tooling environment, with CSV conversion disabled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kaggriculture_eval.reference_audit import audit_collection
from src.kaggriculture_eval.reference_pipeline import write_json


def prepare(source, output, dataset_id, *, extractor=None):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new empty staging directory; never upload stale extra files")
    audit = audit_collection(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "extraction-manifest.json").read_text())
    selection = json.loads((source / "selection.json").read_text())
    names = [r["shard"] for r in manifest["episodes"].values() if r["status"] == "extracted"]
    names += ["extraction-manifest.json", "selection.json", "episode-metadata.json"]
    names += [name for name in ("collection.json", "index-manifest.csv", "daily-manifest.csv", "cloud-requirements.txt")
              if (source / name).exists()]
    for name in names:
        shutil.copy2(source / name, output / name)
    archive_path = source / "extractor-source.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            for name, expected in selection["source_hashes"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError("archived extractor source hash mismatch")
        shutil.copy2(archive_path, output / archive_path.name)
    else:
        if extractor is None:
            raise ValueError("pilot output requires its exact downloaded extractor source")
        with zipfile.ZipFile(output / "extractor-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name, expected in selection["source_hashes"].items():
                data = (Path(extractor) / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("pilot extractor source hash mismatch")
                archive.writestr(name, data)
    write_json(output / "audit.json", audit)
    # Current reader/auditor, identified independently of the original extractor.
    tools = ["src/__init__.py", *selection["source_hashes"], "src/kaggriculture_eval/reference_audit.py",
             "src/kaggriculture_eval/replay.py", "scripts/audit_player_days.py", "scripts/inspect_player_day.py",
             "src/kaggriculture_eval/reference_storage.py", "scripts/restore_reference_dataset.py"]
    with zipfile.ZipFile(output / "inspection-tools.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        hashes = {}
        for name in sorted(set(tools)):
            data = (ROOT / name).read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
        archive.writestr("tool-hashes.json", json.dumps(hashes, indent=2))
    description = f"""Private Kaggriculture Plan-to-realization research references.

{audit['totals'].get('player_days', 0)} player-days; extraction complete: {manifest['complete']}.
Each sample contains day-start state, a semantically reconstructed fixed daily Plan,
demonstrated realization and day-end state. Selling and staffing are not Plan goals.
Selection: frozen top-10 team snapshot AND side-specific pre-game rating cutoff.
This is a snapshot-qualified high-rating cohort, not historical rank or optimal execution.
Official kaggle-environments 1.32.7 replay reexecution precedes every extracted shard.
See selection.json, extraction-manifest.json and audit.json for identities and exclusions.
Schema: {manifest['schema_version']}. Episode shards are deterministic gzip JSONL.
Kaggle automatically expands gzip shards to .jsonl and zip bundles to directories.
Before auditing or resuming, use the maintained restore_reference_dataset.py command
to rebuild gzip checkpoints and verify their original hashes. This only decodes
storage; it never reconstructs Plans or changes sample content.
The pilot is exploratory, not an accepted planner benchmark or model-training dataset.
Attempts and unresolved no-ops remain distinct from accomplished goals. Standalone
stock accumulation is not reconstructed as work; placement prerequisites and goal
equivalence need review before defining planner-comparison acceptance metrics.

Original official replay material: CC0-1.0, from {selection['daily_source']} version {selection['daily_version']}.
Repository source is included for private reproducibility; no additional code license
is granted by this Dataset's Other designation. No credentials or opponent agent code included.
extractor-source.zip preserves the original extractor; inspection-tools.zip contains
separately hashed reader/audit tools. Use Python 3.12 and kaggle-environments==1.32.7.

Inspection (after extracting inspection-tools.zip):
On Kaggle the tools may already be expanded under inspection-tools/.
python scripts/restore_reference_dataset.py EXPANDED_DATASET --output RESTORED_DIRECTORY
python scripts/inspect_player_day.py episode-EPISODE.jsonl.gz --sample EPISODE:SIDE:DAY --output day.html
python scripts/audit_player_days.py DATASET_DIRECTORY

Future collection and reconstruction execute together in one private CPU Kaggle run,
with per-episode checkpoints. Versioned Dataset uploads preserve the completed output;
they do not perform a second reconstruction stage.
"""
    (output / "README.md").write_text(description)
    write_json(output / "dataset-metadata.json", {"id": dataset_id,
        "title": "Kaggriculture Player Day References", "licenses": [{"name": "other"}],
        "description": description})
    write_json(output / "artifact-hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.iterdir()) if p.is_file()})
    return {"directory": str(output), "dataset": dataset_id, "files": len(list(output.iterdir())),
            "bytes": sum(p.stat().st_size for p in output.iterdir()), "audit_passed": audit["passed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--extractor", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.dataset, extractor=args.extractor)))
