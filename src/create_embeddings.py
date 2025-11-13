import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from src.utils import read_jsonl
from pathlib import Path

def create_embeddings(kb_path="data/gen/knowledge_base.jsonl", out_emb="data/gen/embeddings.npy", out_ids="data/gen/ids.npy", model_name="all-MiniLM-L6-v2"):
    Path("data/gen").mkdir(parents=True, exist_ok=True)
    kb = read_jsonl(kb_path)
    texts = [item.get("text","") for item in kb]
    ids = [item.get("id", str(i)) for i,item in enumerate(kb)]
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False)
    np.save(out_emb, embs)
    np.save(out_ids, np.array(ids))
    print(f"Saved embeddings ({embs.shape}) to {out_emb}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="data/gen/knowledge_base.jsonl")
    parser.add_argument("--out_emb", default="data/gen/embeddings.npy")
    parser.add_argument("--out_ids", default="data/gen/ids.npy")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    create_embeddings(args.kb, args.out_emb, args.out_ids, args.model)
