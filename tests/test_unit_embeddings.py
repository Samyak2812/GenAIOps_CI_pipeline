from pathlib import Path
from src.gen_prep import main as prep_main
from src.create_embeddings import create_embeddings

def test_create_embeddings_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed = tmp_path / "data" / "seed"
    seed.mkdir(parents=True)
    (seed / "a.txt").write_text("This is a test knowledge entry.")
    prep_main(seed_dir=str(seed), out_kb="data/gen/knowledge_base.jsonl", out_prompt="data/gen/prompts_template.txt")
    create_embeddings(kb_path="data/gen/knowledge_base.jsonl", out_emb="data/gen/embeddings.npy", out_ids="data/gen/ids.npy")
    assert Path("data/gen/embeddings.npy").exists()
    assert Path("data/gen/ids.npy").exists()
