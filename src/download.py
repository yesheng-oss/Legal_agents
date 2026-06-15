import os
import requests
from pathlib import Path
from config import RAW_DATA_DIR


DEFAULT_DEMO_LIMIT = 1000


def _resolve_download_limit():
    value = os.environ.get("DOWNLOAD_LIMIT", str(DEFAULT_DEMO_LIMIT)).strip().lower()
    if value in {"", "all", "none", "0"}:
        return None
    return int(value)


def download():
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    base = "SunSpace0923/Refined-Chinese-Legal-Dataset"
    files = ["train.json"]
    limit = _resolve_download_limit()

    for name in files:
        path = Path(RAW_DATA_DIR) / name
        if path.exists():
            print(f"{name} already exists, skipping")
            continue

        url = f"https://huggingface.co/datasets/{base}/resolve/main/{name}"
        print(f"Downloading {name}...")
        r = requests.get(url, timeout=600, stream=True)
        r.encoding = "utf-8"
        r.raise_for_status()

        saved = 0
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                f.write(line + "\n")
                saved += 1
                if limit and saved >= limit:
                    break
        print(f"Saved {name}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            count = sum(1 for _ in f)
        print(f"  {count} records")

if __name__ == "__main__":
    download()
