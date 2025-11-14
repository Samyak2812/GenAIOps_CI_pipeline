# Minimal evaluator: replace with real eval comparing outputs to references
import argparse
import json
from pathlib import Path
import random

def compute_factuality():
    # placeholder: compute with real metrics
    return round(random.uniform(0.7, 0.95), 3)

def compute_hallucination_rate():
    return round(random.uniform(0.01, 0.12), 3)

def main(output="metrics/gen_eval_report.json"):
    Path("metrics").mkdir(parents=True, exist_ok=True)
    report = {
        "factuality": compute_factuality(),
        "hallucination_rate": compute_hallucination_rate(),
        "notes": "dummy evaluation - replace with a real harness"
    }
    with open(output,"w") as f:
        json.dump(report, f, indent=2)
    print("Wrote evaluation report to", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="metrics/gen_eval_report.json")
    args = parser.parse_args()
    main(args.output)
