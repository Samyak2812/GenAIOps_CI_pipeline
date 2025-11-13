from fastapi.testclient import TestClient
from src.serve_genai import app

def test_home():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()

def test_generate_no_kb():
    client = TestClient(app)
    r = client.get("/generate", params={"q": "hello", "top_k": 1})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
