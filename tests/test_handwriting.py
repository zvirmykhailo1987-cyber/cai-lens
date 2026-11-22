from cailens.agents import handwriting


def test_confidence_comes_from_readability_and_shape():
    assert handwriting.confidence("A.targa", "AB123CD", False) == ("AB123CD", 0.92)
    assert handwriting.confidence("A.targa", "AB12CD", False) == ("AB12CD", 0.4)  # wrong shape: a person looks
    assert handwriting.confidence("luogo", "MILANO, VIA ROMA 1", False)[1] == 0.85  # free text
    assert handwriting.confidence("A.targa", None, False) == (None, 0.8)  # read as empty
    assert handwriting.confidence("A.targa", "AB123CD", True) == (None, 0.2)  # illegible


def test_comparison_canon():
    from cailens import formats
    assert formats.compact("A.compagnia", "LAGUNA ASS.NI") == formats.compact("A.compagnia", "LAGUNA ASSNI")
    assert formats.compact("testimoni", "nessuno") is None
    assert formats.compact("A.targa", "ab 123 cd") == "AB123CD"
