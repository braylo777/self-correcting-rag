"""Run naive vs self-correcting RAG on SciFact (download: see README).

1. Tune the answer-confidence threshold on --tune N claims from claims_train (grid, max accuracy).
2. Evaluate all systems on claims_dev (--limit to subsample). MIXED-label claims are skipped.
Writes results/<model>/{rows.jsonl,summary.json}.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from scrag.hf_llm import HFLLM
from scrag.pipeline import Corpus, Result, SelfCorrectingConfig, gold_label, naive, score, self_correcting

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data"


def load(name, limit):
    cl = [json.loads(l) for l in (DATA / name).read_text().splitlines()]
    cl = [c for c in cl if gold_label(c) != "MIXED"]
    return cl[:limit] if limit else cl


def rethreshold(r: Result, t: float) -> str:
    if not r.probs or len(r.probs) < 3 or r.answer == "ABSTAIN" and not r.kept_docs:
        return "ABSTAIN"
    p = r.probs
    if p[2] > max(p[0], p[1]) or max(p[0], p[1]) / (p[0] + p[1]) < t:
        return "ABSTAIN"
    return "SUPPORT" if p[0] >= p[1] else "CONTRADICT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tune", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    corpus = Corpus.load(DATA / "corpus.jsonl")
    llm = HFLLM(a.model)
    cfg = SelfCorrectingConfig(answer_threshold=0.0)
    t0 = time.time()
    # --- tune answer threshold on train ---
    train = load("claims_train.jsonl", a.tune)
    tr = [self_correcting(llm, corpus, c["id"], c["claim"], cfg) for c in train]
    gtr = {c["id"]: gold_label(c) for c in train}
    grid = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    tune = {}
    for t in grid:
        rs = [Result(r.claim_id, rethreshold(r, t)) for r in tr]
        tune[t] = score(rs, gtr)["accuracy"]
    best_t = max(grid, key=lambda t: (tune[t], -t))
    # --- dev ---
    dev = load("claims_dev.jsonl", a.limit)
    gold = {c["id"]: gold_label(c) for c in dev}
    rows, sys_res = [], {"naive": [], "naive_nei": [], "self_correcting": []}
    for i, c in enumerate(dev):
        n1 = naive(llm, corpus, c["id"], c["claim"])
        n2 = naive(llm, corpus, c["id"], c["claim"], allow_nei=True)
        sc = self_correcting(llm, corpus, c["id"], c["claim"], cfg)
        sc.answer = rethreshold(sc, best_t) if sc.kept_docs else "ABSTAIN"
        for k, r in (("naive", n1), ("naive_nei", n2), ("self_correcting", sc)):
            sys_res[k].append(r)
        gold_docs = [int(d) for d in c["evidence"]]
        rows.append({"id": c["id"], "gold": gold[c["id"]], "gold_docs": gold_docs, "naive": n1.answer,
                     "naive_nei": n2.answer, "sc": asdict(sc),
                     "naive_hit": bool(set(gold_docs) & set(n1.kept_docs)),
                     "sc_hit": bool(set(gold_docs) & set(sc.kept_docs))})
        print(f"[{i + 1}/{len(dev)}] gold={gold[c['id']]} naive={n1.answer} nei={n2.answer} sc={sc.answer} "
              f"rw={sc.rewrites}", flush=True)
    summ = {"model": a.model, "device": llm.device, "dev_claims": len(dev), "tune_claims": len(train),
            "gold_distribution": {g: sum(v == g for v in gold.values()) for g in ("SUPPORT", "CONTRADICT", "NEI")},
            "tuned_answer_threshold": best_t, "tune_grid_accuracy": tune,
            "grade_threshold": cfg.grade_threshold, "max_rewrites": cfg.max_rewrites,
            "systems": {k: score(v, gold) for k, v in sys_res.items()},
            "sc_rewrite_rate": round(sum(r.rewrites > 0 for r in sys_res["self_correcting"]) / len(dev), 4),
            "sc_mean_rewrites": round(sum(r.rewrites for r in sys_res["self_correcting"]) / len(dev), 4),
            "llm_calls": llm.calls, "wall_seconds": round(time.time() - t0, 1)}
    out = ROOT / "results" / a.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
