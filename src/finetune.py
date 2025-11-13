import argparse
import os
from pathlib import Path
import json
import sys
import subprocess

def finetune_local(train_data="data/gen/finetune_dataset.jsonl", output_dir="models/gen_model", epochs=1):
    # Placeholder: implement your local HF + PEFT training pipeline here.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # For now write a marker file to indicate finetune was "run"
    info = {"finetune": True, "train_data": train_data, "epochs": epochs}
    Path(output_dir, "FINETUNE_INFO.json").write_text(json.dumps(info, indent=2))
    print("Local finetune placeholder done:", output_dir)

def trigger_external(api_url, payload_path=None, api_key_env="CLOUD_API_KEY"):
    # Placeholder to trigger remote training. Replace with real cloud API trigger.
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        print("No API key found in env var", api_key_env)
        return 2
    payload = {}
    if payload_path and os.path.exists(payload_path):
        payload = json.load(open(payload_path))
    # Example curl call (disabled by default); uncomment/replace for real use.
    print(f"Would trigger external training at {api_url} with payload keys {list(payload.keys())}")
    # Example:
    # subprocess.run(["curl","-X","POST",api_url,"-H",f"Authorization: Bearer {api_key}","-d",json.dumps(payload)])
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local","trigger"], default="local")
    parser.add_argument("--output", default="models/gen_model")
    parser.add_argument("--train_data", default="data/gen/finetune_dataset.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--api_url", default="")
    parser.add_argument("--payload", default="")
    args = parser.parse_args()

    if args.mode == "local":
        if os.getenv("SKIP_FINETUNE", "1") == "1" and os.getenv("CI", "") == "true":
            print("Finetune skipped in CI environment (safety). Set SKIP_FINETUNE=0 to override.")
            sys.exit(0)
        finetune_local(args.train_data, args.output, args.epochs)
    else:
        rc = trigger_external(args.api_url, args.payload)
        sys.exit(rc)
