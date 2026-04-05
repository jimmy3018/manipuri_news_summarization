#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Paired bootstrap significance testing for structure-aware summarization metrics.

Metrics:
1. entity_coverage
2. event_coverage
3. fact_precision
4. consistency_score

Compares:
- proposed system vs one or more baselines

Expected:
- dataset JSON with doc_id, ner, events
- system output folders with one JSON per document

@author: jimmy
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "cleaned_dataset_with_manipuri_summary.json"

OURS_DIR = "realizer_outputs"

BASELINE_DIRS = {
    "TextRank": "baseline_outputs/textrank_outputs",
    "LexRank": "baseline_outputs/lexrank_outputs",
    "mT5": "baseline_outputs/mt5_outputs",
}

OUTPUT_JSON = "bootstrap_significance_results.json"

OURS_SUMMARY_FIELD = "summary_mni"
BASELINE_SUMMARY_FIELD = "summary_mni"

N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# weights for consistency
ALPHA = 0.33
BETA = 0.33
GAMMA = 0.34


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class DocMetrics:
    doc_id: str
    entity_coverage: float
    event_coverage: float
    fact_precision: float
    consistency_score: float


# =========================================================
# UTILS
# =========================================================

def safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_ws(text: str) -> str:
    return " ".join(safe_text(text).split())


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tokenize(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [tok for tok in text.split() if tok.strip()]


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    data = load_json(dataset_path)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Dataset must be a JSON object or a list of JSON objects.")


def build_output_index(folder: str) -> Dict[str, Dict[str, Any]]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Output folder not found: {folder}")

    index: Dict[str, Dict[str, Any]] = {}
    for fp in folder_path.glob("*.json"):
        try:
            obj = load_json(fp)
            doc_id = safe_text(obj.get("doc_id"))
            if doc_id:
                index[doc_id] = obj
        except Exception as exc:
            print(f"Warning: could not load {fp}: {exc}")
    return index


# =========================================================
# SOURCE GRAPH EXTRACTION
# =========================================================

def extract_gold_entities(doc: Dict[str, Any]) -> List[str]:
    entities = []

    for ent in doc.get("ner", []):
        txt = safe_text(ent.get("text"))
        txt_en = safe_text(ent.get("text_en"))
        if txt:
            entities.append(txt)
        if txt_en:
            entities.append(txt_en)

    for ev in doc.get("events", []):
        for arg in ev.get("arguments", []):
            arg_text = safe_text(arg.get("text"))
            arg_text_en = safe_text(arg.get("text_en"))
            if arg_text:
                entities.append(arg_text)
            if arg_text_en:
                entities.append(arg_text_en)

    # preserve order
    seen = set()
    result = []
    for x in entities:
        if x and x not in seen:
            seen.add(x)
            result.append(x)
    return result


def extract_gold_events(doc: Dict[str, Any]) -> List[str]:
    events = []
    for ev in doc.get("events", []):
        trig = safe_text(ev.get("trigger"))
        trig_en = safe_text(ev.get("trigger_en"))
        if trig:
            events.append(trig)
        if trig_en:
            events.append(trig_en)

    seen = set()
    result = []
    for x in events:
        if x and x not in seen:
            seen.add(x)
            result.append(x)
    return result


# =========================================================
# METRICS
# =========================================================

def extract_summary_mentions(summary: str, candidates: List[str]) -> List[str]:
    found = []
    summary = safe_text(summary)
    for c in candidates:
        if c and c in summary:
            found.append(c)

    seen = set()
    result = []
    for x in found:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def compute_entity_coverage(gold_entities: List[str], summary: str) -> float:
    if not gold_entities:
        return 0.0
    pred_entities = extract_summary_mentions(summary, gold_entities)
    return len(pred_entities) / len(gold_entities)


def compute_event_coverage(gold_events: List[str], summary: str) -> float:
    if not gold_events:
        return 0.0
    pred_events = extract_summary_mentions(summary, gold_events)
    return len(pred_events) / len(gold_events)


def compute_fact_precision(gold_entities: List[str], gold_events: List[str], summary: str) -> float:
    summary_tokens = tokenize(summary)
    if not summary_tokens:
        return 0.0

    gold_token_set = set()
    for item in gold_entities + gold_events:
        for tok in tokenize(item):
            gold_token_set.add(tok)

    supported_tokens = [tok for tok in summary_tokens if tok in gold_token_set]
    return len(supported_tokens) / len(summary_tokens)


def compute_consistency_score(entity_cov: float, event_cov: float, fact_precision: float) -> float:
    return ALPHA * entity_cov + BETA * event_cov + GAMMA * fact_precision


def evaluate_one(doc: Dict[str, Any], summary: str) -> DocMetrics:
    gold_entities = extract_gold_entities(doc)
    gold_events = extract_gold_events(doc)

    ec = compute_entity_coverage(gold_entities, summary)
    evc = compute_event_coverage(gold_events, summary)
    fp = compute_fact_precision(gold_entities, gold_events, summary)
    cs = compute_consistency_score(ec, evc, fp)

    return DocMetrics(
        doc_id=safe_text(doc.get("doc_id")),
        entity_coverage=ec,
        event_coverage=evc,
        fact_precision=fp,
        consistency_score=cs,
    )


# =========================================================
# PER-DOCUMENT METRICS
# =========================================================

def collect_metrics_for_system(
    dataset: List[Dict[str, Any]],
    output_index: Dict[str, Dict[str, Any]],
    summary_field: str
) -> List[DocMetrics]:
    results: List[DocMetrics] = []

    for doc in dataset:
        doc_id = safe_text(doc.get("doc_id"))
        out_obj = output_index.get(doc_id)
        if out_obj is None:
            continue

        summary = safe_text(out_obj.get(summary_field))
        metrics = evaluate_one(doc, summary)
        results.append(metrics)

    return results


def align_results(
    ours: List[DocMetrics],
    baseline: List[DocMetrics]
) -> Tuple[List[DocMetrics], List[DocMetrics]]:
    ours_map = {x.doc_id: x for x in ours}
    base_map = {x.doc_id: x for x in baseline}

    common_ids = sorted(set(ours_map.keys()) & set(base_map.keys()))

    return [ours_map[i] for i in common_ids], [base_map[i] for i in common_ids]


# =========================================================
# BOOTSTRAP
# =========================================================

def mean_metric(results: List[DocMetrics], metric_name: str) -> float:
    if not results:
        return 0.0
    return sum(getattr(r, metric_name) for r in results) / len(results)


def paired_bootstrap_p_value(
    ours: List[DocMetrics],
    baseline: List[DocMetrics],
    metric_name: str,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    if len(ours) != len(baseline):
        raise ValueError("Aligned system result lists must have same length.")

    random.seed(seed)
    n = len(ours)

    observed_diff = mean_metric(ours, metric_name) - mean_metric(baseline, metric_name)

    diffs = []
    for _ in range(n_bootstrap):
        sample_indices = [random.randrange(n) for _ in range(n)]
        ours_sample = [ours[i] for i in sample_indices]
        base_sample = [baseline[i] for i in sample_indices]
        diff = mean_metric(ours_sample, metric_name) - mean_metric(base_sample, metric_name)
        diffs.append(diff)

    # one-sided empirical p-value:
    # probability that baseline >= ours, i.e. diff <= 0
    count_nonpositive = sum(1 for d in diffs if d <= 0.0)
    p_value = (count_nonpositive + 1) / (n_bootstrap + 1)

    return {
        "metric": metric_name,
        "observed_diff": round(observed_diff, 6),
        "p_value": round(p_value, 6),
        "n_documents": n,
        "n_bootstrap": n_bootstrap,
    }


# =========================================================
# MAIN
# =========================================================

def run_significance_testing() -> None:
    dataset = load_dataset(DATASET_PATH)

    ours_index = build_output_index(OURS_DIR)
    ours_metrics = collect_metrics_for_system(dataset, ours_index, OURS_SUMMARY_FIELD)

    all_results = {
        "config": {
            "dataset_path": DATASET_PATH,
            "ours_dir": OURS_DIR,
            "baseline_dirs": BASELINE_DIRS,
            "ours_summary_field": OURS_SUMMARY_FIELD,
            "baseline_summary_field": BASELINE_SUMMARY_FIELD,
            "n_bootstrap": N_BOOTSTRAP,
            "random_seed": RANDOM_SEED,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
        },
        "comparisons": {}
    }

    metric_names = [
        "entity_coverage",
        "event_coverage",
        "fact_precision",
        "consistency_score",
    ]

    for baseline_name, baseline_dir in BASELINE_DIRS.items():
        print(f"\nComparing Ours vs {baseline_name}")

        baseline_index = build_output_index(baseline_dir)
        baseline_metrics = collect_metrics_for_system(dataset, baseline_index, BASELINE_SUMMARY_FIELD)

        ours_aligned, base_aligned = align_results(ours_metrics, baseline_metrics)

        comparison_result = {
            "n_documents": len(ours_aligned),
            "metrics": {}
        }

        for metric in metric_names:
            res = paired_bootstrap_p_value(
                ours_aligned,
                base_aligned,
                metric_name=metric,
                n_bootstrap=N_BOOTSTRAP,
                seed=RANDOM_SEED
            )
            comparison_result["metrics"][metric] = res

            print(
                f"{metric:20s} | diff = {res['observed_diff']:.6f} | p = {res['p_value']:.6f}"
            )

        all_results["comparisons"][baseline_name] = comparison_result

    save_json(all_results, OUTPUT_JSON)
    print(f"\nSaved significance results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_significance_testing()