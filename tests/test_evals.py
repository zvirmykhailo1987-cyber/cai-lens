import json

from cailens import evals, synth


def _wrap(monkeypatch, model):
    orig = synth.generate

    def gen(seed, split="dev"):
        t, img = orig(seed, split)
        model.add(t)
        return t, img

    monkeypatch.setattr(evals, "generate", gen)


def test_eval_run_writes_a_report(monkeypatch, model, tmp_path):
    _wrap(monkeypatch, model)
    monkeypatch.setattr(evals, "RESULTS", tmp_path)
    rep = evals.run("dev", 3)
    s = rep["summary"]
    assert s["documents"] == 3 and s["registered_pct"] == 100.0
    assert s["deadline"]["never_later_pct"] == 100.0
    assert s["cost"]["usd_per_doc_live"] > 0
    saved = json.loads((tmp_path / "dev-llm-3.json").read_text(encoding="utf-8"))
    assert saved["summary"]["documents"] == 3


def test_failures_list_shows_wrong_text(monkeypatch, model, tmp_path):
    _wrap(monkeypatch, model)
    monkeypatch.setattr(evals, "RESULTS", tmp_path)
    model.mistakes["B.targa"] = "ZZ999ZZ"
    rep = evals.run("dev", 2, save=False)
    assert rep["failures"] and all("B.targa" in f["text_errors"] for f in rep["failures"] if f["text_errors"])


def test_baseline_comparison(monkeypatch, model, tmp_path):
    _wrap(monkeypatch, model)
    monkeypatch.setattr(evals, "RESULTS", tmp_path)
    rep = evals.run_baseline("dev", 3)
    c = rep["comparison"]
    assert c["ibrido"]["documents"] == 3 and c["solo_modello"]["documents"] == 3
    assert c["ibrido"]["box_wrong"] <= c["solo_modello"]["box_wrong"]  # the fake baseline drops a cross per case
    assert (tmp_path / "baseline-dev-3.json").exists()


def test_local_env_loader(monkeypatch, tmp_path):
    from cailens import llm
    monkeypatch.setattr(llm, "HOME", tmp_path)
    (tmp_path / ".env.local").write_text("OTHER=1\nANTHROPIC_API_KEY=test-value\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    evals.load_local_env()
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "test-value"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
