import json
from pathlib import Path
from typing import List

def read_jsonl(path: str) -> List[dict]:
    p = Path(path)
    out = []
    if not p.exists():
        return out
    with p.open() as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out

def write_jsonl(path: str, rows: List[dict]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
