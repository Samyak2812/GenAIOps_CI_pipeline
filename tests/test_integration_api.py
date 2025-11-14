from fastapi.testclient import TestClient
from src.serve_genai import app
from pathlib import Path

def test_home_and_generate(tmp_path, monkeypatch):
    # create minimal KB to avoid empty retrieval
    kb_dir = tmp_path / "data" / "gen"
    kb_dir.mkdir(parents=True)
    (kb_dir / "knowledge_base.jsonl").write_text('{"id":"1","text":"Test context entry","meta":{}}'+"\n")
    (kb_dir / "prompts_template.txt").write_text("Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
    # monkeypatch working dir
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    r2 = client.get("/generate", params={"q":"hello","top_k":1})
    assert r2.status_code == 200
    assert "answer" in r2.json()
