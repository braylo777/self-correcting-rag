import pytest

from scrag.pipeline import (Corpus, Result, SelfCorrectingConfig, gold_label, naive, score, self_correcting,
                            tokenize)
from scripts.run_eval import rethreshold

DOCS = [
    {"doc_id": 1, "title": "Aspirin lowers stroke risk", "abstract": ["Aspirin reduced ischemic stroke in trials."]},
    {"doc_id": 2, "title": "Vitamin D and fractures", "abstract": ["Vitamin D did not reduce fractures."]},
    {"doc_id": 3, "title": "Coffee and sleep", "abstract": ["Caffeine delays sleep onset."]},
    {"doc_id": 4, "title": "Acetylsalicylic acid cardiology", "abstract": ["ASA prevents myocardial infarction."]},
]


class FakeLLM:
    """Grades by keyword overlap; verdict probs and rewrites are scripted."""

    def __init__(self, relevant=("aspirin", "acetylsalicylic", "vitamin"), verdict=(0.8, 0.1, 0.1),
                 rewrite="acetylsalicylic acid"):
        self.relevant, self.verdict, self.rewrite = relevant, verdict, rewrite
        self.grades = self.rewrites = 0

    def yes_prob(self, prompt):
        self.grades += 1
        doc = prompt.split("Claim:")[0].lower()
        claim = prompt.split("Claim:")[1].lower()
        return 0.9 if any(k in doc and k in claim for k in self.relevant) else 0.1

    def choice_probs(self, prompt, options):
        return list(self.verdict[: len(options)])

    def generate(self, prompt, max_new_tokens=32):
        self.rewrites += 1
        return self.rewrite


@pytest.fixture
def corpus():
    return Corpus(DOCS)


def test_tokenize_drops_stopwords():
    assert tokenize("The effect of Aspirin on stroke") == ["effect", "aspirin", "stroke"]


def test_bm25_search_ranks(corpus):
    assert corpus.search("aspirin stroke", 2)[0]["doc_id"] == 1


def test_naive_always_answers(corpus):
    r = naive(FakeLLM(verdict=(0.4, 0.6)), corpus, 1, "unrelated claim about mars")
    assert r.answer == "CONTRADICT" and len(r.kept_docs) == 3


def test_naive_nei_can_abstain(corpus):
    r = naive(FakeLLM(verdict=(0.1, 0.2, 0.7)), corpus, 1, "x", allow_nei=True)
    assert r.answer == "ABSTAIN"


def test_sc_answers_when_relevant(corpus):
    llm = FakeLLM()
    r = self_correcting(llm, corpus, 1, "aspirin reduces stroke", SelfCorrectingConfig(k=2))
    assert r.answer == "SUPPORT" and r.rewrites == 0 and r.kept_docs[0] == 1


def test_sc_rewrites_then_finds(corpus):
    llm = FakeLLM()
    r = self_correcting(llm, corpus, 1, "ASA prevents heart attacks acetylsalicylic", SelfCorrectingConfig(k=1))
    assert r.answer == "SUPPORT" and 4 in r.kept_docs


def test_sc_rewrite_triggered_when_nothing_relevant(corpus):
    llm = FakeLLM(relevant=())
    r = self_correcting(llm, corpus, 1, "aspirin stroke", SelfCorrectingConfig(k=2, max_rewrites=2))
    assert r.answer == "ABSTAIN" and r.rewrites == 2 and llm.rewrites == 2 and len(r.queries) == 3
    assert "no relevant evidence" in r.reason


def test_sc_grades_each_doc_once(corpus):
    llm = FakeLLM(relevant=())
    self_correcting(llm, corpus, 1, "aspirin stroke", SelfCorrectingConfig(k=4, max_rewrites=2))
    assert llm.grades == 4  # rewrites return already-seen docs; no regrading


def test_sc_abstains_on_nei_verdict(corpus):
    r = self_correcting(FakeLLM(verdict=(0.2, 0.1, 0.7)), corpus, 1, "aspirin stroke", SelfCorrectingConfig())
    assert r.answer == "ABSTAIN" and "not enough" in r.reason


def test_sc_abstains_on_low_confidence(corpus):
    r = self_correcting(FakeLLM(verdict=(0.45, 0.4, 0.15)), corpus, 1, "aspirin stroke",
                        SelfCorrectingConfig(answer_threshold=0.6))
    assert r.answer == "ABSTAIN" and "low confidence" in r.reason


def test_sc_refutes(corpus):
    r = self_correcting(FakeLLM(verdict=(0.1, 0.8, 0.1)), corpus, 1, "vitamin d reduces fractures",
                        SelfCorrectingConfig())
    assert r.answer == "CONTRADICT"


def test_gold_label():
    assert gold_label({"evidence": {}}) == "NEI"
    assert gold_label({"evidence": {"1": [{"label": "SUPPORT"}]}}) == "SUPPORT"
    assert gold_label({"evidence": {"1": [{"label": "CONTRADICT"}]}}) == "CONTRADICT"
    assert gold_label({"evidence": {"1": [{"label": "SUPPORT"}], "2": [{"label": "CONTRADICT"}]}}) == "MIXED"


def test_score_metrics():
    rs = [Result(1, "SUPPORT"), Result(2, "ABSTAIN"), Result(3, "SUPPORT"), Result(4, "ABSTAIN")]
    s = score(rs, {1: "SUPPORT", 2: "NEI", 3: "CONTRADICT", 4: "SUPPORT"})
    assert s["accuracy"] == 0.5 and s["abstention_rate"] == 0.5 and s["false_answer_rate"] == 0.25
    assert s["answered_precision"] == 0.5 and s["abstained_on_nei"] == 1


def test_rethreshold():
    r = Result(1, "SUPPORT", probs=[0.5, 0.4, 0.1], kept_docs=[1])
    assert rethreshold(r, 0.5) == "SUPPORT" and rethreshold(r, 0.6) == "ABSTAIN"
    assert rethreshold(Result(1, "ABSTAIN"), 0.5) == "ABSTAIN"


def test_score_all_abstain_precision_none():
    s = score([Result(1, "ABSTAIN")], {1: "SUPPORT"})
    assert s["answered_precision"] is None and s["accuracy"] == 0.0
