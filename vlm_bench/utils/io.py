from pathlib import Path
import json

def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_parquet(path: Path, rows):
    import pandas as pd
    df = pd.json_normalize(rows)
    df.to_parquet(path)
