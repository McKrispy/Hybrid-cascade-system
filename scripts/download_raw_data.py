"""Script to download raw datasets for HybridCascade Batch B01.

Timestamp: 2026-11-09 13:10 (Asia/Shanghai)

Datasets:
1. Banking77: test.csv and categories.json from PolyAI-LDN
2. CLINC150: data_full.json from clinc/oos-eval
3. ANLI: anli_v0.1.zip from dl.fbaipublicfiles.com (R2/test.jsonl and R3/test.jsonl)
4. StrategyQA: train.json from eladsegal/strategyqa
"""

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
import httpx

DEV_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = DEV_ROOT / "data" / "raw"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HybridCascade-Downloader/1.0"
}


def download_file(urls: list[str], dest_path: Path, min_size: int = 100, max_retries: int = 4) -> None:
    """Download a file from urls (with fallback mirrors and retries) to dest_path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size >= min_size:
        print(f"[CACHE] {dest_path.name} already exists ({dest_path.stat().st_size} bytes). Skipping.")
        return

    client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60.0)
    for url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[DOWNLOAD] Attempt {attempt}/{max_retries} fetching {url} -> {dest_path.name}...")
                resp = client.get(url)
                if resp.status_code != 200:
                    print(f"[WARN] HTTP {resp.status_code} from {url}")
                    break
                content = resp.content
                if len(content) < min_size:
                    print(f"[WARN] Incomplete content ({len(content)} bytes) from {url}")
                    time.sleep(1)
                    continue

                temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
                with open(temp_path, "wb") as f:
                    f.write(content)
                temp_path.replace(dest_path)
                print(f"[SUCCESS] Saved {dest_path.name} ({len(content)} bytes)")
                return
            except Exception as e:
                print(f"[WARN] Error on attempt {attempt}: {e}")
                time.sleep(1.5 * attempt)
    
    raise RuntimeError(f"Failed to download {dest_path.name} after trying all mirrors.")


def fetch_banking77():
    print("\n--- 1. Banking77 ---")
    dest_dir = DATA_RAW / "banking77"
    
    # test.csv
    csv_urls = [
        "https://cdn.jsdelivr.net/gh/PolyAI-LDN/task-specific-datasets@master/banking_data/test.csv",
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        "https://ghproxy.net/https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    ]
    download_file(csv_urls, dest_dir / "test.csv", min_size=5000)

    # categories.json
    cat_urls = [
        "https://cdn.jsdelivr.net/gh/PolyAI-LDN/task-specific-datasets@master/banking_data/categories.json",
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json",
        "https://ghproxy.net/https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json",
    ]
    download_file(cat_urls, dest_dir / "categories.json", min_size=500)


def fetch_clinc150():
    print("\n--- 2. CLINC150 ---")
    dest_dir = DATA_RAW / "clinc150"
    urls = [
        "https://cdn.jsdelivr.net/gh/clinc/oos-eval@master/data/data_full.json",
        "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
        "https://ghproxy.net/https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
    ]
    download_file(urls, dest_dir / "data_full.json", min_size=500000)


def fetch_anli():
    print("\n--- 3. ANLI (R2 and R3) ---")
    dest_dir = DATA_RAW / "anli"
    dest_dir.mkdir(parents=True, exist_ok=True)
    r2_file = dest_dir / "r2_test.jsonl"
    r3_file = dest_dir / "r3_test.jsonl"

    if r2_file.exists() and r3_file.exists():
        print(f"[CACHE] ANLI R2 and R3 test files already exist in {dest_dir}. Skipping.")
        return

    # Download official zip
    zip_urls = [
        "https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip",
        "https://ghproxy.net/https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip",
    ]
    zip_dest = dest_dir / "anli_v0.1.zip"
    download_file(zip_urls, zip_dest, min_size=10000000)

    print(f"[EXTRACT] Extracting R2/test.jsonl and R3/test.jsonl from {zip_dest}...")
    with zipfile.ZipFile(zip_dest, "r") as zf:
        names = zf.namelist()
        r2_name = [n for n in names if n.endswith("R2/test.jsonl") or n.endswith("R2\\test.jsonl")][0]
        r3_name = [n for n in names if n.endswith("R3/test.jsonl") or n.endswith("R3\\test.jsonl")][0]

        with zf.open(r2_name) as src, open(r2_file, "wb") as dst:
            dst.write(src.read())
        with zf.open(r3_name) as src, open(r3_file, "wb") as dst:
            dst.write(src.read())

    print(f"[SUCCESS] Extracted ANLI R2 ({r2_file.stat().st_size} bytes) and R3 ({r3_file.stat().st_size} bytes)")


def fetch_strategyqa():
    print("\n--- 4. StrategyQA ---")
    dest_dir = DATA_RAW / "strategyqa"
    urls = [
        "https://cdn.jsdelivr.net/gh/eladsegal/strategyqa@master/data/strategyqa/train.json",
        "https://raw.githubusercontent.com/eladsegal/strategyqa/master/data/strategyqa/train.json",
        "https://ghproxy.net/https://raw.githubusercontent.com/eladsegal/strategyqa/master/data/strategyqa/train.json",
    ]
    download_file(urls, dest_dir / "train.json", min_size=500000)


def main():
    print(f"Starting raw dataset downloads for HybridCascade B01...")
    print(f"Destination root: {DATA_RAW}")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    fetch_banking77()
    fetch_clinc150()
    fetch_anli()
    fetch_strategyqa()

    print("\n==========================================")
    print("All four raw datasets downloaded successfully!")
    print("==========================================")


if __name__ == "__main__":
    main()
