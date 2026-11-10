"""Naive RAG vs self-correcting RAG on SciFact claim verification.

Task: given a scientific claim, answer SUPPORTED or REFUTED from retrieved
abstracts - or abstain when the corpus holds no evidence (SciFact NEI claims).

Naive:          BM25 top-3 -> verdict (always answers)
Naive+NEI:      BM25 top-3 -> verdict with a third "not enough info" option
Self-correcting:
  1. retrieve BM25 top-k
  2. grade every chunk: P(relevant) from the LLM's yes/no logits
  3. no chunk >= grade threshold -> LLM rewrites the query -> retrieve again (<= max_rewrites)
  4. still nothing relevant -> ABSTAIN
  5. verdict over graded chunks (A supported / B refuted / C not enough info)
  6. abstain if C wins or the normalized A/B confidence < answer threshold
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[a-z0-9]+")
STOP = set(
    "the of and a an in to is for on with by as are was were be that this from at or not".split()
)


def tokenize(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in STOP]


class Corpus:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.ids = [d["doc_id"] for d in docs]
        self.bm25 = BM25Okapi(
            [tokenize(d["title"] + " " + " ".join(d["abstract"])) for d in docs]
        )

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        return cls(
            [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            {"doc_id": self.ids[i], "text": self.text(i), "bm25": float(scores[i])}
            for i in top
        ]

    def text(self, i: int, max_chars: int = 900) -> str:
        d = self.docs[i]
        return (d["title"] + ". " + " ".join(d["abstract"]))[:max_chars]


class LLM(Protocol):
    def yes_prob(self, prompt: str) -> float: ...
    def choice_probs(self, prompt: str, options: list[str]) -> list[float]: ...
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str: ...


GRADE = (
    "Document:\n{doc}\n\nClaim: {claim}\n\nDoes the document contain evidence that directly supports "
    "or refutes the claim? Answer yes or no."
)
VERDICT = (
    "Evidence:\n{ctx}\n\nClaim: {claim}\n\nBased only on the evidence, is the claim:\n"
    "A) supported\nB) refuted\n{extra}Answer with the letter only."
)
REWRITE = (
    "Rewrite this scientific claim as a short keyword search query for finding abstracts that test it. "
    "Expand abbreviations, add synonyms.\nClaim: {claim}\nQuery:"
)


@dataclass
class Result:
    claim_id: int
    answer: str  # SUPPORT | CONTRADICT | ABSTAIN
    confidence: float = 0.0
    rewrites: int = 0
    queries: list[str] = field(default_factory=list)
    kept_docs: list[int] = field(default_factory=list)
    grades: list[float] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    reason: str = ""


def naive(
    llm: LLM, corpus: Corpus, cid: int, claim: str, k: int = 3, allow_nei: bool = False
) -> Result:
    docs = corpus.search(claim, k)
    ctx = "\n\n".join(f"[{i + 1}] {d['text']}" for i, d in enumerate(docs))
    opts = ["A", "B", "C"] if allow_nei else ["A", "B"]
    p = llm.choice_probs(
        VERDICT.format(
            ctx=ctx,
            claim=claim,
            extra="C) not enough information\n" if allow_nei else "",
        ),
        opts,
    )
    best = max(range(len(opts)), key=lambda i: p[i])
    ans = ["SUPPORT", "CONTRADICT", "ABSTAIN"][best]
    return Result(cid, ans, p[best], kept_docs=[d["doc_id"] for d in docs], probs=p)


@dataclass
class SelfCorrectingConfig:
    k: int = 5
    grade_threshold: float = 0.5
    answer_threshold: float = 0.6
    max_rewrites: int = 2
    max_ctx_docs: int = 3


def self_correcting(
    llm: LLM, corpus: Corpus, cid: int, claim: str, cfg: SelfCorrectingConfig
) -> Result:
    res = Result(cid, "ABSTAIN", queries=[claim])
    query, seen = claim, {}
    for attempt in range(cfg.max_rewrites + 1):
        for d in corpus.search(query, cfg.k):
            if d["doc_id"] not in seen:
                seen[d["doc_id"]] = (
                    llm.yes_prob(GRADE.format(doc=d["text"], claim=claim)),
                    d,
                )
        kept = sorted(
            (v for v in seen.values() if v[0] >= cfg.grade_threshold),
            key=lambda v: -v[0],
        )
        if kept:
            break
        if attempt < cfg.max_rewrites:
            query = (
                llm.generate(REWRITE.format(claim=claim)).strip().split("\n")[0]
                or claim
            )
            query = f"{query} {claim}" if attempt == 1 else query
            res.queries.append(query)
            res.rewrites += 1
    res.grades = sorted((round(v[0], 4) for v in seen.values()), reverse=True)
    if not kept:
        res.reason = "no relevant evidence after rewrites"
        return res
    kept = kept[: cfg.max_ctx_docs]
    res.kept_docs = [d["doc_id"] for _, d in kept]
    ctx = "\n\n".join(f"[{i + 1}] {d['text']}" for i, (_, d) in enumerate(kept))
    p = llm.choice_probs(
        VERDICT.format(ctx=ctx, claim=claim, extra="C) not enough information\n"),
        ["A", "B", "C"],
    )
    res.probs = p
    if p[2] > max(p[0], p[1]):
        res.reason = "verdict: not enough information"
        return res
    conf = max(p[0], p[1]) / (p[0] + p[1])
    res.confidence = conf
    if conf < cfg.answer_threshold:
        res.reason = f"low confidence {conf:.2f}"
        return res
    res.answer = "SUPPORT" if p[0] >= p[1] else "CONTRADICT"
    return res


def gold_label(claim: dict) -> str:
    labels = {e["label"] for ev in claim["evidence"].values() for e in ev}
    if not labels:
        return "NEI"
    return (
        "SUPPORT"
        if labels == {"SUPPORT"}
        else "CONTRADICT" if labels == {"CONTRADICT"} else "MIXED"
    )


def score(results: list[Result], gold: dict[int, str]) -> dict:
    n = len(results)
    correct = wrong = abst = abst_ok = answered_ok = answered = 0
    for r in results:
        g = gold[r.claim_id]
        if r.answer == "ABSTAIN":
            abst += 1
            if g == "NEI":
                correct += 1
                abst_ok += 1
        else:
            answered += 1
            if r.answer == g:
                correct += 1
                answered_ok += 1
            else:
                wrong += 1
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "abstention_rate": round(abst / n, 4),
        "false_answer_rate": round(wrong / n, 4),
        "answered_precision": round(answered_ok / answered, 4) if answered else None,
        "correct": correct,
        "wrong_answers": wrong,
        "abstained": abst,
        "abstained_on_nei": abst_ok,
    }
