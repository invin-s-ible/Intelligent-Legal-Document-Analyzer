from analyzer import LegalAnalyzer

TEXT = open("data/sample_contract.txt", encoding="utf-8").read()


def test_sample_contract():
    r = LegalAnalyzer().analyze(TEXT)
    assert "date" in r["ents"] and "money" in r["ents"]
    assert any("Veda Technologies" in p for p in r["ents"]["party"])
    labels = {c["k"] for c in r["clauses"]}
    assert {"Payment", "Termination"} <= labels
    assert any(x["label"] == "Unlimited liability" for x in r["risks"])
    assert r["score"] >= 60 and r["summary"] and r["obl"]
