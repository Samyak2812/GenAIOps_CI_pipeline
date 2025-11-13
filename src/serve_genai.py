from fastapi import FastAPI, Query
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
from typing import List

app = FastAPI()
EMB_PATH = "data/gen/embeddings.npy"
IDS_PATH = "data/gen/ids.npy"
KB_PATH = "data/gen/knowledge_base.jsonl"
PROMPT_PATH = "data/gen/prompts_template.txt"

# load on first request lazily
_state = {"model": None, "index": None, "ids": None, "kb": None, "prompt": None}

def _ensure():
    if _state["model"] is None:
        if Path(EMB_PATH).exists():
            _state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            _state["model"] = SentenceTransformer("all-MiniLM-L6-v2")  # still create
    if _state["index"] is None and Path("data/gen/faiss_index.faiss").exists():
        _state["index"] = faiss.read_index("data/gen/faiss_index.faiss")
    if _state["ids"] is None and Path(IDS_PATH).exists():
        _state["ids"] = np.load(IDS_PATH, allow_pickle=True)
    if _state["kb"] is None and Path(KB_PATH).exists():
        with open(KB_PATH) as f:
            _state["kb"] = {json.loads(l)["id"]: json.loads(l)["text"] for l in f}
    if _state["prompt"] is None and Path(PROMPT_PATH).exists():
        _state["prompt"] = Path(PROMPT_PATH).read_text()

@app.get("/")
def home():
    return {"message": "GenAI server running"}

def retrieve(question, k=3):
    _ensure()
    model = _state["model"]
    index = _state["index"]
    ids = _state["ids"]
    if index is None or ids is None:
        return []
    emb = model.encode([question])
    faiss.normalize_L2(emb)
    D,I = index.search(emb, k)
    hits = []
    for i in I[0]:
        hits.append(ids[int(i)])
    # map to texts
    return [_state["kb"].get(h,"") for h in hits]

def generate_simulated(prompt):
    return "SIMULATED ANSWER: " + prompt[:200]

@app.get("/generate")
def generate(q: str = Query(...), top_k: int = 3):
    ctx = retrieve(q, k=top_k)
    prompt_template = _state["prompt"] or "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    prompt = prompt_template.format(context="\n".join(ctx), question=q)
    resp = generate_simulated(prompt)
    # log
    Path("logs").mkdir(parents=True, exist_ok=True)
    with open("logs/gen_calls.log","a") as f:
        f.write(json.dumps({"q": q, "prompt": prompt, "resp": resp}) + "\n")
    return {"answer": resp, "ctx": ctx}
