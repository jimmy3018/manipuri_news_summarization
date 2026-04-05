# -*- coding: utf-8 -*-
"""
Graph-based evaluation for all baselines

Evaluates:
- entity_coverage
- event_coverage
- fact_precision
- consistency_score

@author: jimmy
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "cleaned_dataset_with_manipuri_summary.json"
BASELINE_DIR = "baseline_outputs/mt5_outputs"  # change for each baseline

# =========================================================
# UTILS
# =========================================================

def safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", safe_text(text))
    return [t for t in text.split() if t]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# EXTRACTION
# =========================================================

def extract_gold_entities(doc):
    ents = []
    for e in doc.get("ner", []):
        ents.append(safe_text(e.get("text")))
        ents.append(safe_text(e.get("text_en")))
    return list(set([e for e in ents if e]))


def extract_gold_events(doc):
    evs = []
    for ev in doc.get("events", []):
        evs.append(safe_text(ev.get("trigger")))
        evs.append(safe_text(ev.get("trigger_en")))
    return list(set([e for e in evs if e]))


def extract_summary_mentions(summary, candidates):
    return [c for c in candidates if c in summary]


# =========================================================
# METRICS
# =========================================================

def compute_metrics(doc, summary):

    gold_entities = extract_gold_entities(doc)
    gold_events = extract_gold_events(doc)

    pred_entities = extract_summary_mentions(summary, gold_entities)
    pred_events = extract_summary_mentions(summary, gold_events)

    # coverage
    entity_cov = len(pred_entities) / max(1, len(gold_entities))
    event_cov = len(pred_events) / max(1, len(gold_events))

    # precision
    summary_tokens = tokenize(summary)
    gold_tokens = set(tokenize(" ".join(gold_entities + gold_events)))

    supported = [t for t in summary_tokens if t in gold_tokens]

    fact_precision = len(supported) / max(1, len(summary_tokens))

    # consistency
    consistency = 0.5 * entity_cov + 0.3 * event_cov + 0.2 * fact_precision

    return {
        "entity_coverage": round(entity_cov, 4),
        "event_coverage": round(event_cov, 4),
        "fact_precision": round(fact_precision, 4),
        "consistency_score": round(consistency, 4)
    }


# =========================================================
# MAIN
# =========================================================

def run_evaluation():

    dataset = load_json(DATASET_PATH)
    results = []

    for doc in dataset:
        doc_id = safe_text(doc.get("doc_id"))

        baseline_file = Path(BASELINE_DIR) / f"{doc_id}.json"
        if not baseline_file.exists():
            continue

        baseline = load_json(baseline_file)
        summary = safe_text(baseline.get("summary_mni"))

        metrics = compute_metrics(doc, summary)
        metrics["doc_id"] = doc_id

        results.append(metrics)

    # macro avg
    avg = {}
    keys = ["entity_coverage", "event_coverage", "fact_precision", "consistency_score"]

    for k in keys:
        avg[k] = round(sum(r[k] for r in results) / len(results), 4)

    print("\n===== RESULTS =====")
    for k, v in avg.items():
        print(f"{k:20s}: {v}")

    print(f"\nDocuments evaluated: {len(results)}")


if __name__ == "__main__":
    run_evaluation()