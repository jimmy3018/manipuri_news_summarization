# -*- coding: utf-8 -*-
"""
Improved Evaluation code for graph-guided summarization
- Soft matching for entities/events
- Fact-aware evaluation
- Reduced hallucination bias
@author: jimmy
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from collections import Counter


# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "cleaned_dataset_with_manipuri_summary.json"
REALIZER_OUTPUT_DIR = "realizer_outputs"
OUTPUT_EVAL_JSON = "evaluation_results.json"
OUTPUT_EVAL_CSV = "evaluation_results.csv"

REFERENCE_FIELD = "manipuri_human_summary"
SYSTEM_SUMMARY_FIELD = "summary_mni"


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class DocumentEvalResult:
    doc_id: str
    reference_exists: bool
    rouge1_f1: float
    rouge2_f1: float
    rougel_f1: float
    bleu: float
    entity_coverage: float
    event_coverage: float
    fact_precision: float
    unsupported_entity_ratio: float
    unsupported_trigger_ratio: float
    consistency_score: float
    summary_token_count: int
    reference_token_count: int
    compression_ratio: float
    evidence_marker_count: int


# =========================================================
# UTILS
# =========================================================

def safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def normalize_ws(text: str) -> str:
    return " ".join(safe_text(text).split())


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tokenize(text: str) -> List[str]:
    text = normalize_ws(text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t.strip()]


def ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


# =========================================================
# SOFT MATCH
# =========================================================

def soft_match(candidate: str, text: str) -> bool:
    c_tokens = set(tokenize(candidate))
    t_tokens = set(tokenize(text))
    if not c_tokens:
        return False
    overlap = len(c_tokens & t_tokens)
    return overlap / len(c_tokens) >= 0.5


def extract_summary_mentions(summary: str, candidates: List[str]) -> List[str]:
    summary = safe_text(summary)
    found = []

    for c in candidates:
        if c and (c in summary or soft_match(c, summary)):
            found.append(c)

    return list(dict.fromkeys(found))


# =========================================================
# ROUGE
# =========================================================

def rouge_n_f1(ref, pred, n=1):
    r = Counter(ngrams(tokenize(ref), n))
    p = Counter(ngrams(tokenize(pred), n))
    overlap = sum((r & p).values())

    if overlap == 0:
        return 0.0

    precision = overlap / max(1, sum(p.values()))
    recall = overlap / max(1, sum(r.values()))
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(ref, pred):
    r = tokenize(ref)
    p = tokenize(pred)

    dp = [[0]*(len(p)+1) for _ in range(len(r)+1)]
    for i in range(1, len(r)+1):
        for j in range(1, len(p)+1):
            if r[i-1] == p[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0

    precision = lcs / len(p)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


# =========================================================
# BLEU
# =========================================================

def bleu_score(ref, pred, max_n=4):
    r = tokenize(ref)
    p = tokenize(pred)

    if not r or not p:
        return 0.0

    precisions = []
    for n in range(1, max_n+1):
        ref_ng = Counter(ngrams(r, n))
        pred_ng = Counter(ngrams(p, n))
        overlap = sum((ref_ng & pred_ng).values())
        total = sum(pred_ng.values())
        precisions.append(max(overlap / total if total else 0, 1e-9))

    bp = 1 if len(p) > len(r) else math.exp(1 - len(r)/len(p)) if len(p) > 0 else 0
    return bp * math.exp(sum(math.log(p) for p in precisions)/max_n)


# =========================================================
# GRAPH CONSISTENCY
# =========================================================

def extract_gold_entities(doc):
    return list(set(
        [safe_text(e.get("text")) for e in doc.get("ner", [])] +
        [safe_text(e.get("text_en")) for e in doc.get("ner", [])]
    ))


def extract_gold_triggers(doc):
    return list(set(
        [safe_text(e.get("trigger")) for e in doc.get("events", [])] +
        [safe_text(e.get("trigger_en")) for e in doc.get("events", [])]
    ))


def estimate_unsupported_ratio(summary, gold_items, facts):
    summary_tokens = tokenize(summary)

    support = set()
    for g in gold_items:
        support.update(tokenize(g))

    for f in facts:
        support.update(tokenize(f.get("text", "")))

    content = [t for t in summary_tokens if len(t) >= 3]
    if not content:
        return 0.0

    unsupported = [t for t in content if t not in support]
    return len(unsupported) / len(content)


def graph_consistency_metrics(doc, realizer_obj):
    summary = safe_text(realizer_obj.get("summary_mni"))
    facts = realizer_obj.get("realized_facts", [])

    gold_entities = extract_gold_entities(doc)
    gold_triggers = extract_gold_triggers(doc)

    entity_hits = extract_summary_mentions(summary, gold_entities)

    fact_text = " ".join(f.get("text", "") for f in facts)
    trigger_hits = extract_summary_mentions(fact_text, gold_triggers)

    entity_cov = len(entity_hits) / max(1, len(gold_entities))
    event_cov = len(trigger_hits) / max(1, len(gold_triggers))

    grounded = sum(
        1 for f in facts
        if any(e in f.get("text", "") for e in gold_entities)
    )
    fact_precision = grounded / max(1, len(facts))

    ue = estimate_unsupported_ratio(summary, gold_entities, facts)
    ut = estimate_unsupported_ratio(summary, gold_triggers, facts)

    consistency = (
        0.3*entity_cov +
        0.3*event_cov +
        0.2*fact_precision +
        0.1*(1-ue) +
        0.1*(1-ut)
    )

    return {
        "entity_coverage": round(entity_cov,4),
        "event_coverage": round(event_cov,4),
        "fact_precision": round(fact_precision,4),
        "unsupported_entity_ratio": round(ue,4),
        "unsupported_trigger_ratio": round(ut,4),
        "consistency_score": round(consistency,4)
    }


# =========================================================
# EVALUATION
# =========================================================

def evaluate_document(doc, realizer_obj):
    ref = safe_text(doc.get(REFERENCE_FIELD))
    pred = safe_text(realizer_obj.get(SYSTEM_SUMMARY_FIELD)) if realizer_obj else ""

    rouge1 = rouge_n_f1(ref, pred)
    rouge2 = rouge_n_f1(ref, pred, 2)
    rougel = rouge_l_f1(ref, pred)
    bleu = bleu_score(ref, pred)

    graph = graph_consistency_metrics(doc, realizer_obj or {})

    return DocumentEvalResult(
        doc_id=safe_text(doc.get("doc_id")),
        reference_exists=bool(ref),
        rouge1_f1=round(rouge1,4),
        rouge2_f1=round(rouge2,4),
        rougel_f1=round(rougel,4),
        bleu=round(bleu,4),
        entity_coverage=graph["entity_coverage"],
        event_coverage=graph["event_coverage"],
        fact_precision=graph["fact_precision"],
        unsupported_entity_ratio=graph["unsupported_entity_ratio"],
        unsupported_trigger_ratio=graph["unsupported_trigger_ratio"],
        consistency_score=graph["consistency_score"],
        summary_token_count=len(tokenize(pred)),
        reference_token_count=len(tokenize(ref)),
        compression_ratio=round(len(tokenize(pred))/max(1,len(tokenize(" ".join(s.get("text","") for s in doc.get("sentences",[]))))),4),
        evidence_marker_count=len(re.findall(r"\[s\d+", pred))
    )


# =========================================================
# RUN
# =========================================================

def run_evaluation():
    dataset = load_json(DATASET_PATH)
    if isinstance(dataset, dict):
        dataset = [dataset]

    realizer_index = {
        safe_text(load_json(f).get("doc_id")): load_json(f)
        for f in Path(REALIZER_OUTPUT_DIR).glob("*.json")
    }

    results = []
    for doc in dataset:
        r = evaluate_document(doc, realizer_index.get(safe_text(doc.get("doc_id"))))
        results.append(r)

    macro = {
        k: round(sum(getattr(r, k) for r in results)/len(results),4)
        for k in asdict(results[0]).keys()
        if isinstance(getattr(results[0], k), (int,float))
    }

    print("\n===== MACRO RESULTS =====")
    for k,v in macro.items():
        print(f"{k:25s}: {v}")

    save_json({
        "macro": macro,
        "documents": [asdict(r) for r in results]
    }, OUTPUT_EVAL_JSON)


if __name__ == "__main__":
    run_evaluation()