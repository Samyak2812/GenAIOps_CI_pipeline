import json
from pathlib import Path
import argparse
from src.utils import write_jsonl

def main(seed_dir="data/seed", out_kb="data/gen/knowledge_base.jsonl", out_prompt="data/gen/prompts_template.txt"):
    Path("data/gen").mkdir(parents=True, exist_ok=True)
    # Load any seed files in seed_dir (txt/ jsonl). For demo: create small KB if none exist.
    kb = []
    seed = Path(seed_dir)
    if seed.exists():
        for p in seed.glob("**/*"):
            if p.suffix.lower() in [".txt"]:
                kb.append({"id": p.stem, "text": p.read_text(), "meta": {"source": str(p)}})
            if p.suffix.lower() in [".jsonl"]:
                for line in p.read_text().splitlines():
                    if line.strip():
                        try:
                            kb.append(json.loads(line))
                        except:
                            pass
    if not kb:
        # default demo KB
        for i in range(5):
            kb.append({"id": f"demo_{i}", "text": f"Example incident description number {i}. Severity low. Risk: medium.", "meta": {"demo": True}})
    write_jsonl(out_kb, kb)
    prompt = """You are an assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""
    Path(out_prompt).write_text(prompt)
    print("Wrote KB:", out_kb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", default="data/seed")
    parser.add_argument("--out_kb", default="data/gen/knowledge_base.jsonl")
    parser.add_argument("--out_prompt", default="data/gen/prompts_template.txt")
    args = parser.parse_args()
    main(args.seed_dir, args.out_kb, args.out_prompt)
