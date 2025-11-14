# create embeddings (supports HF sentence-transformers or OpenAI)
import argparse
from pathlib import Path
import numpy as np
from src.utils import read_jsonl
import os

def create_embeddings(kb_path="data/gen/knowledge_base.jsonl", out_emb="data/gen/embeddings.npy", out_ids="data/gen/ids.npy", model_name=None, backend="hf"):
    Path("data/gen").mkdir(parents=True, exist_ok=True)
    kb = read_jsonl(kb_path)
    texts = [item.get("text","") for item in kb]
    ids = [item.get("id", str(i)) for i,item in enumerate(kb)]
    if backend == "openai":
        # Use OpenAI embeddings
        import openai
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set for openai embeddings")
        openai.api_key = key
        embs = []
        for t in texts:
            resp = openai.Embedding.create(model=model_name, input=t)
            embs.append(resp["data"][0]["embedding"])
        embs = np.array(embs, dtype=np.float32)
    else:
        # HF sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name or "sentence-transformers/all-MiniLM-L6-v2")
        embs = model.encode(texts, show_progress_bar=False)
        embs = np.array(embs, dtype=np.float32)
    np.save(out_emb, embs)
    np.save(out_ids, np.array(ids, dtype=object))
    print("Saved embeddings:", out_emb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="data/gen/knowledge_base.jsonl")
    parser.add_argument("--out_emb", default="data/gen/embeddings.npy")
    parser.add_argument("--out_ids", default="data/gen/ids.npy")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default="hf", choices=["hf","openai"])
    args = parser.parse_args()
    create_embeddings(args.kb, args.out_emb, args.out_ids, args.model, args.backend)
