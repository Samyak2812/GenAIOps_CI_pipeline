import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List
import json

class Retriever:
    def __init__(self, emb_model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="data/gen/faiss_index.faiss", ids_path="data/gen/ids.npy", kb_path="data/gen/knowledge_base.jsonl"):
        self.emb_model = SentenceTransformer(emb_model_name)
        self.index = None
        self.ids = None
        if Path(index_path).exists():
            self.index = faiss.read_index(index_path)
        if Path(ids_path).exists():
            self.ids = np.load(ids_path, allow_pickle=True)
        self.kb = {}
        if Path(kb_path).exists():
            with open(kb_path) as f:
                for l in f:
                    if l.strip():
                        obj = json.loads(l)
                        self.kb[obj["id"]] = obj.get("text","")

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        if self.index is None or self.ids is None:
            return []
        emb = self.emb_model.encode([query])
        faiss.normalize_L2(emb)
        D,I = self.index.search(emb, top_k)
        hits = []
        for i in I[0]:
            try:
                key = self.ids[int(i)]
                hits.append(self.kb.get(key,""))
            except:
                pass
        return hits
