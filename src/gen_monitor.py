import json
import pandas as pd
from pathlib import Path

def main(logfile="logs/gen_calls.log", out="metrics/gen_monitoring_report.csv"):
    Path("metrics").mkdir(parents=True, exist_ok=True)
    rows = []
    if Path(logfile).exists():
        with open(logfile) as f:
            for l in f:
                if not l.strip(): continue
                try:
                    rec = json.loads(l)
                except:
                    continue
                rec["answer_len"] = len(rec.get("resp",""))
                rec["low_confidence"] = any(x in rec.get("resp","").lower() for x in ["i don't know","cannot","unable"])
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("Wrote gen monitoring report:", out)

if __name__ == "__main__":
    main()
