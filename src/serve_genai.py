from fastapi import FastAPI, Query
from typing import List
from src.retrieval import Retriever
from src.llm import LLM
from pathlib import Path
import json

app = FastAPI()
# init components
retriever = Retriever()
# choose backend from environment or params; default local
import os
backend = os.getenv("GENAI_BACKEND", "local")
llm = LLM(backend=backend, hf_model=os.getenv("HF_MODEL","gpt2"), openai_model=os.getenv("OPENAI_MODEL","gpt-4o-mini"))

PROMPT_PATH = "data/gen/prompts_template.txt"
if Path(PROMPT_PATH).exists():
    PROMPT_TEMPLATE = Path(PROMPT_PATH).read_text()
else:
    PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

@app.get("/")
def home():
    return {"message": "GenAI server running"}

@app.get("/generate")
def generate(q: str = Query(...), top_k: int = Query(4), temperature: float = Query(0.0)):
    ctxs = retriever.retrieve(q, top_k)
    context = "\n\n".join(ctxs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=q)
    answer = llm.generate(prompt, max_new_tokens=128, temperature=temperature)
    # log
    Path("logs").mkdir(parents=True, exist_ok=True)
    with open("logs/gen_calls.log","a") as f:
        f.write(json.dumps({"q": q, "ctx_ids": ctxs, "resp": answer}) + "\n")
    return {"answer": answer, "context_used": ctxs}
