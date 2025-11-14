# Create a knowledge base jsonl and a prompt template
import argparse
from pathlib import Path
from src.utils import write_jsonl
import json

def main(seed_dir="data/seed", out_kb="data/gen/knowledge_base.jsonl", out_prompt="data/gen/prompts_template.txt"):
    Path("data/gen").mkdir(parents=True, exist_ok=True)
    rows = []
    s = Path(seed_dir)
    if s.exists():
        for p in s.glob("**/*"):
            if p.suffix.lower() == ".txt":
                rows.append({"id": p.stem, "text": p.read_text(), "meta": {"source": str(p)}})
            elif p.suffix.lower() == ".jsonl":
                for line in p.read_text().splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
    if not rows:
        # minimal default KB
        for i in range(3):
            rows.append({"id": f"demo_{i}", "text": f"Demo knowledge entry {i}. Use this as example context.", "meta": {"demo": True}})
    write_jsonl(out_kb, rows)
    prompt = """You are a helpful assistant. Use the provided context (if any) to answer the user's question concisely.

Context:
{context}

Question:
{question}

Answer:"""
    Path(out_prompt).write_text(prompt)
    print("Wrote KB and prompt")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", default="data/seed")
    parser.add_argument("--out_kb", default="data/gen/knowledge_base.jsonl")
    parser.add_argument("--out_prompt", default="data/gen/prompts_template.txt")
    args = parser.parse_args()
    main(args.seed_dir, args.out_kb, args.out_prompt)
