# Placeholder for finetune. For production replace with HF+PEFT or cloud training trigger.
import argparse
import json
from pathlib import Path
import os
import sys

def finetune_local(train_data="data/gen/finetune_dataset.jsonl", output="models/gen_model", epochs=1):
    Path(output).mkdir(parents=True, exist_ok=True)
    info = {"finetuned": True, "train_data": train_data, "epochs": epochs}
    Path(output, "FINETUNE_INFO.json").write_text(json.dumps(info))
    print("Wrote finetune marker to", output)

def trigger_remote(api_url: str, payload_path: str = None):
    # Replace with actual cloud trigger: curl or SDK call
    print("Trigger remote finetune at", api_url)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local","trigger"], default="local")
    parser.add_argument("--train_data", default="data/gen/finetune_dataset.jsonl")
    parser.add_argument("--output", default="models/gen_model")
    parser.add_argument("--api_url", default="")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "local":
        # guard: skip finetune if in CI unless explicitly allowed
        if os.getenv("CI","") == "true" and os.getenv("ALLOW_FINETUNE_IN_CI","0") != "1":
            print("Skipping local finetune inside CI runner (safety).")
            sys.exit(0)
        finetune_local(args.train_data, args.output, args.epochs)
    else:
        rc = trigger_remote(args.api_url)
        sys.exit(rc)
