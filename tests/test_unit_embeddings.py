import os
from pathlib import Path
from src.create_embeddings import create_embeddings
from src.gen_prep import main as prep_main

def test_create_embeddings_runs(tmp_path, monkeypatch):
    # prepare a tiny KB in tmp dir
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "a.txt").write_text("hello world test")
    monkeypatch.chdir(tmp_path)
    # run gen_prep
    prep_main(seed_dir=str(seed), out_kb="data/gen/knowledge_base.jsonl", out_prompt="data/gen/prompts_template.txt")
    create_embeddings(kb_path="data/gen/knowledge_base.jsonl", out_emb="data/gen/embeddings.npy", out_ids="data/gen/ids.npy")
    assert Path("data/gen/embeddings.npy").exists()
    assert Path("data/gen/ids.npy").exists()
