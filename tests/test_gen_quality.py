from src.gen_evaluate import main
import json
from pathlib import Path

def test_eval_generates_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data/gen").mkdir(parents=True, exist_ok=True)
    main(output="metrics/gen_eval_report.json")
    assert Path("metrics/gen_eval_report.json").exists()
    r = json.load(open("metrics/gen_eval_report.json"))
    assert "factuality" in r
    assert "hallucination_rate" in r
