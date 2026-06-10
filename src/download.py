import json
import requests
from pathlib import Path
from config import RAW_DATA_DIR

def download():
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    base = "SunSpace0923/Refined-Chinese-Legal-Dataset"
    files = ["train.json", "validation.json"]

    for name in files:
        path = Path(RAW_DATA_DIR) / name
        if path.exists():
            print(f"{name} already exists, skipping")
            continue

        url = f"https://huggingface.co/datasets/{base}/resolve/main/{name}"
        print(f"Downloading {name}...")
        r = requests.get(url, timeout=600, stream=True)
        r.encoding = "utf-8"
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Saved {name}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            count = sum(1 for _ in f)
        print(f"  {count} records")

if __name__ == "__main__":
    download()
