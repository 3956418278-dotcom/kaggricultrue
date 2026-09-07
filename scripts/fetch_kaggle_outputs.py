#!/usr/bin/env python3
"""Fetch selected private notebook outputs without logging signed download URLs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default=r"^(player-days/|extractor/.*\.py$)")
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    owner, slug = args.kernel.split("/")
    files, token = [], None
    while True:
        with api.build_kaggle_client() as client:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug, request.page_size = owner, slug, 200
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        files.extend(f for f in response.files if re.search(args.pattern, f.file_name))
        token = response.next_page_token
        if not token:
            break
    hashes = {}

    def fetch(item):
        path = (args.output / item.file_name).resolve()
        if not path.is_relative_to(args.output.resolve()):
            raise ValueError("unsafe output path")
        expected = hashes.get(item.file_name)
        if expected and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return "verified existing"
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                with requests.get(item.url, stream=True, timeout=(15, 90)) as download:
                    if download.status_code in (401, 403, 429):
                        raise RuntimeError(f"access stopped: HTTP {download.status_code}")
                    download.raise_for_status()
                    temporary = path.with_suffix(path.suffix + ".partial")
                    sha = hashlib.sha256()
                    with temporary.open("wb") as target:
                        for chunk in download.iter_content(1024 * 1024):
                            target.write(chunk)
                            sha.update(chunk)
                    if expected and sha.hexdigest() != expected:
                        raise ValueError("download hash mismatch")
                    temporary.replace(path)
                    return "downloaded"
            except (requests.exceptions.SSLError, requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == 2:
                    raise RuntimeError("transient transfer failed after three attempts") from None
                time.sleep(2 ** attempt)
        raise RuntimeError("transfer incomplete")

    # Obtain the CURRENT output manifest before trusting any cached shard.
    # Otherwise a newer notebook version could silently retain old valid files.
    manifest_item = next((f for f in files if f.file_name == "player-days/extraction-manifest.json"), None)
    if manifest_item:
        try:
            fetch(manifest_item)
        except Exception as error:
            print("manifest transfer failed: " + type(error).__name__, flush=True)
            raise SystemExit(1) from None
        manifest = args.output / manifest_item.file_name
        hashes = {"player-days/" + r["shard"]: r["sha256"]
                  for r in json.loads(manifest.read_text())["episodes"].values() if r.get("shard")}
        files.remove(manifest_item)
        print("Current extraction manifest downloaded; verifying this output version.", flush=True)
    errors = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, f): f.file_name for f in files}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as error:
                errors += 1
                result = "FAILED " + type(error).__name__  # Never expose signed URLs.
            print(f"{n}/{len(files)} {futures[future]}: {result}", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
