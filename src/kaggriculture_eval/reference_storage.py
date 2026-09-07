"""Restore Kaggle's automatically expanded Dataset files to exact checkpoints.

This is lossless storage decoding, not player-day or Plan reconstruction. The
original compressed hash is checked before a restored checkpoint is accepted.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def restore_dataset(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("restore into a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "extraction-manifest.json").read_text())
    restored = 0
    for record in manifest["episodes"].values():
        if record.get("status") != "extracted":
            continue
        name, expected = record["shard"], record["sha256"]
        target = output / name
        if not target.resolve().is_relative_to(output):
            raise ValueError("unsafe checkpoint name")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        packed = source / name
        if packed.exists():
            shutil.copy2(packed, temporary)
        else:
            unpacked = packed.with_suffix("")  # episode-ID.jsonl.gz -> .jsonl
            with unpacked.open("rb") as raw, temporary.open("wb") as destination:
                with gzip.GzipFile(fileobj=destination, mode="wb", filename="", mtime=0) as compressed:
                    shutil.copyfileobj(raw, compressed, length=1024 * 1024)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError("restored checkpoint hash mismatch; incompatible content or compression runtime")
        temporary.replace(target)
        restored += 1
    # Keep frozen provenance verbatim, including selection/collection identities.
    for name in ("extraction-manifest.json", "selection.json", "episode-metadata.json", "collection.json",
                 "audit.json", "index-manifest.csv", "daily-manifest.csv", "cloud-requirements.txt",
                 "artifact-hashes.json", "README.md"):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    for name in ("extractor-source", "inspection-tools"):
        if (source / name).is_dir():
            shutil.copytree(source / name, output / name, dirs_exist_ok=True)
        if (source / (name + ".zip")).is_file():
            shutil.copy2(source / (name + ".zip"), output / (name + ".zip"))
    # Expanded source retains its own original hashes even though zip bytes are
    # no longer available. Never pretend a newly zipped archive is byte-identical.
    if (output / "extractor-source").is_dir():
        selection = json.loads((output / "selection.json").read_text())
        for name, expected in selection["source_hashes"].items():
            path = output / "extractor-source" / name
            if not path.resolve().is_relative_to(output / "extractor-source"):
                raise ValueError("unsafe source path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("expanded extractor source hash mismatch")
    return {"restored_shards": restored, "directory": str(output), "operation": "storage restoration; no Plan reconstruction"}
