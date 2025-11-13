import numpy as np
import faiss
from pathlib import Path
import argparse

def build_index(emb_path="data/gen/embeddings.npy", ids_path="data/gen/ids.npy", out_index="data/gen/faiss_index.faiss"):
    Path("data/gen").mkdir(parents=True, exist_ok=True)
    embs = np.load(emb_path)
    ids = np.load(ids_path, allow_pickle=True)
    d = embs.shape[1]
    # normalize then index
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, out_index)
    print("Saved faiss index:", out_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="data/gen/embeddings.npy")
    parser.add_argument("--ids", default="data/gen/ids.npy")
    parser.add_argument("--out", default="data/gen/faiss_index.faiss")
    args = parser.parse_args()
    build_index(args.emb, args.ids, args.out)
