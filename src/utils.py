import json
from pathlib import Path

def read_jsonl(path):
    path = Path(path)
    out = []
    with path.open() as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
