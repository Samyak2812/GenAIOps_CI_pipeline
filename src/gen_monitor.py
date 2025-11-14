# Aggregate logs to metrics CSV (simple)
import pandas as pd
from pathlib import Path
import json

def main(logfile="logs/gen_calls.log", out="metrics/gen_monitoring_report.csv"):
    Path("metrics").mkdir(parents=True, exist_ok=True)
    rows = []
    if Path(logfile).exists():
        with open(logfile) as f:
            for l in f:
                if l.strip():
                    rows.append(json.loads(l))
    if rows:
        df = pd.DataFrame(rows)
        df["answer_len"] = df["resp"].apply(lambda x: len(x) if isinstance(x,str) else 0)
        df.to_csv(out, index=False)
        print("Wrote monitoring CSV:", out)
    else:
        print("No logs yet.")

if __name__ == "__main__":
    main()
