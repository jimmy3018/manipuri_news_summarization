# -*- coding: utf-8 -*-
"""
Graph-based evaluation for Manipuri summarization

Computes:
1. Entity Coverage (EC)
2. Event Coverage (EvC)
3. Fact Precision (FP)
4. Consistency Score (CS)

Expected:
- dataset JSON with source graph information
- realizer output JSON files with generated summaries

@author: jimmy
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "cleaned_dataset_with_manipuri_summary.json"
REALIZER_OUTPUT_DIR = "realizer_outputs"

OUTPUT_JSON = "graph_based_evaluation_results.json"
OUTPUT_CSV = "graph_based_evaluation_results.csv"

SYSTEM_SUMMARY_FIELD = "summary_mni"

# Weights for consistency score
ALPHA = 0.33   # Entity Coverage
BETA = 0.33    # Event Coverage
GAMMA = 0.34   # Fact Precision


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class DocumentEvalResult:
    doc_id: str
    entity_coverage: float
    event_coverage: float
    fact_precision: float
    consistency_score: float
    num_source_entities: int
    num_source_events: int
    num_summary_entities: int
    num_summary_events: int
    included_entities: List[str]
    covered_event_ids: List[str]
    summary_length_tokens: int


# =========================================================
# UTILITIES
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
    # Keep unicode words, remove punctuation loosely
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [tok for tok in text.split() if tok.strip()]


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = safe_text(item)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    data = load_json(dataset_path)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Dataset must be either a JSON object or a list of JSON objects.")


def build_realizer_index(realizer_dir: str) -> Dict[str, Dict[str, Any]]:
    realizer_path = Path(realizer_dir)
    if not realizer_path.exists():
        raise FileNotFoundError(f"Realizer output directory not found: {realizer_dir}")

    index: Dict[str, Dict[str, Any]] = {}

    for fp in realizer_path.glob("*.json"):
        try:
            obj = load_json(fp)
            doc_id = safe_text(obj.get("doc_id"))
            if doc_id:
                index[doc_id] = obj
        except Exception as exc:
            print(f"Warning: Could not load {fp}: {exc}")

    return index


# =========================================================
# GRAPH ITEM EXTRACTION
# =========================================================

def extract_source_entities(doc: Dict[str, Any]) -> List[str]:
    """
    Extract all source entities from NER and event arguments.
    """
    entities: List[str] = []

    for ent in doc.get("ner", []):
        text = safe_text(ent.get("text"))
        if text:
            entities.append(text)

    for ev in doc.get("events", []):
        for arg in ev.get("arguments", []):
            text = safe_text(arg.get("text"))
            if text:
                entities.append(text)

    return unique_preserve_order(entities)


def extract_source_event_info(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Keep source event ids, triggers, and arguments.
    """
    events_info: List[Dict[str, Any]] = []

    for ev in doc.get("events", []):
        event_id = safe_text(ev.get("event_id"))
        trigger = safe_text(ev.get("trigger"))
        arguments = []

        for arg in ev.get("arguments", []):
            arg_text = safe_text(arg.get("text"))
            if arg_text:
                arguments.append(arg_text)

        events_info.append({
            "event_id": event_id,
            "trigger": trigger,
            "arguments": unique_preserve_order(arguments),
        })

    return events_info


# =========================================================
# SUMMARY MATCHING
# =========================================================

def extract_summary_entities(summary: str, source_entities: List[str]) -> List[str]:
    """
    Entity mention matching by exact substring lookup.
    """
    summary = normalize_ws(summary)
    found: List[str] = []

    for ent in source_entities:
        if ent and ent in summary:
            found.append(ent)

    return unique_preserve_order(found)


def extract_summary_event_matches(
    summary: str,
    source_events: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """
    Match source events inside summary.

    Strategy:
    - event considered covered if trigger appears in summary
    - OR at least one argument appears together with trigger-like support
    """
    summary = normalize_ws(summary)

    covered_event_ids: List[str] = []
    matched_trigger_texts: List[str] = []

    for ev in source_events:
        event_id = ev["event_id"]
        trigger = ev["trigger"]
        arguments = ev["arguments"]

        trigger_found = bool(trigger and trigger in summary)
        arg_hits = sum(1 for arg in arguments if arg and arg in summary)

        # Event is covered if trigger appears,
        # or if enough supporting arguments appear
        if trigger_found or arg_hits >= 2:
            covered_event_ids.append(event_id)
            if trigger:
                matched_trigger_texts.append(trigger)

    return unique_preserve_order(covered_event_ids), unique_preserve_order(matched_trigger_texts)


# =========================================================
# METRICS
# =========================================================

def compute_entity_coverage(
    source_entities: List[str],
    summary_entities: List[str]
) -> float:
    if not source_entities:
        return 0.0
    return len(set(summary_entities) & set(source_entities)) / len(set(source_entities))


def compute_event_coverage(
    source_event_ids: List[str],
    covered_event_ids: List[str]
) -> float:
    if not source_event_ids:
        return 0.0
    return len(set(covered_event_ids) & set(source_event_ids)) / len(set(source_event_ids))


def compute_fact_precision(
    summary_entities: List[str],
    covered_event_ids: List[str],
    source_entities: List[str],
    source_event_ids: List[str]
) -> float:
    """
    Fact precision estimates how much of the summary-supported structure
    is actually grounded in the source graph.

    Here:
    - summary facts = detected summary entities + detected summary events
    - grounded facts = overlap with source entities + source events
    """
    summary_fact_count = len(set(summary_entities)) + len(set(covered_event_ids))
    if summary_fact_count == 0:
        return 0.0

    grounded_entity_count = len(set(summary_entities) & set(source_entities))
    grounded_event_count = len(set(covered_event_ids) & set(source_event_ids))

    grounded_fact_count = grounded_entity_count + grounded_event_count
    return grounded_fact_count / summary_fact_count


def compute_consistency_score(
    ec: float,
    evc: float,
    fp: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA
) -> float:
    total = alpha + beta + gamma
    if total == 0:
        raise ValueError("Alpha, beta, gamma must not all be zero.")
    alpha /= total
    beta /= total
    gamma /= total
    return (alpha * ec) + (beta * evc) + (gamma * fp)


# =========================================================
# DOCUMENT EVALUATION
# =========================================================

def evaluate_document(
    doc: Dict[str, Any],
    realizer_obj: Optional[Dict[str, Any]]
) -> DocumentEvalResult:
    doc_id = safe_text(doc.get("doc_id"))
    system_summary = ""
    if realizer_obj is not None:
        system_summary = safe_text(realizer_obj.get(SYSTEM_SUMMARY_FIELD))

    source_entities = extract_source_entities(doc)
    source_events = extract_source_event_info(doc)
    source_event_ids = [ev["event_id"] for ev in source_events if ev["event_id"]]

    summary_entities = extract_summary_entities(system_summary, source_entities)
    covered_event_ids, _ = extract_summary_event_matches(system_summary, source_events)

    ec = compute_entity_coverage(source_entities, summary_entities)
    evc = compute_event_coverage(source_event_ids, covered_event_ids)
    fp = compute_fact_precision(
        summary_entities=summary_entities,
        covered_event_ids=covered_event_ids,
        source_entities=source_entities,
        source_event_ids=source_event_ids,
    )
    cs = compute_consistency_score(ec, evc, fp)

    return DocumentEvalResult(
        doc_id=doc_id,
        entity_coverage=round(ec, 4),
        event_coverage=round(evc, 4),
        fact_precision=round(fp, 4),
        consistency_score=round(cs, 4),
        num_source_entities=len(source_entities),
        num_source_events=len(source_event_ids),
        num_summary_entities=len(summary_entities),
        num_summary_events=len(covered_event_ids),
        included_entities=summary_entities,
        covered_event_ids=covered_event_ids,
        summary_length_tokens=len(tokenize(system_summary)),
    )


# =========================================================
# AGGREGATION
# =========================================================

def macro_average(results: List[DocumentEvalResult]) -> Dict[str, float]:
    if not results:
        return {}

    fields = [
        "entity_coverage",
        "event_coverage",
        "fact_precision",
        "consistency_score",
        "num_source_entities",
        "num_source_events",
        "num_summary_entities",
        "num_summary_events",
        "summary_length_tokens",
    ]

    agg: Dict[str, float] = {}
    for field in fields:
        vals = [getattr(r, field) for r in results]
        agg[field] = round(sum(vals) / len(vals), 4)

    agg["num_documents"] = len(results)
    return agg


# =========================================================
# CSV EXPORT
# =========================================================

def save_results_csv(results: List[DocumentEvalResult], path: str | Path) -> None:
    path = Path(path)
    if not results:
        return

    fields = list(asdict(results[0]).keys())

    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for result in results:
            row_data = asdict(result)
            row = []
            for field in fields:
                value = row_data[field]
                if isinstance(value, list):
                    row.append('"' + "; ".join(map(str, value)) + '"')
                else:
                    row.append(str(value))
            f.write(",".join(row) + "\n")


# =========================================================
# MAIN
# =========================================================

def run_evaluation() -> None:
    dataset = load_dataset(DATASET_PATH)
    realizer_index = build_realizer_index(REALIZER_OUTPUT_DIR)

    results: List[DocumentEvalResult] = []

    for doc in dataset:
        doc_id = safe_text(doc.get("doc_id"))
        realizer_obj = realizer_index.get(doc_id)
        result = evaluate_document(doc, realizer_obj)
        results.append(result)

    macro = macro_average(results)

    output = {
        "config": {
            "dataset_path": DATASET_PATH,
            "realizer_output_dir": REALIZER_OUTPUT_DIR,
            "system_summary_field": SYSTEM_SUMMARY_FIELD,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
        },
        "macro_average": macro,
        "documents": [asdict(r) for r in results],
    }

    save_json(output, OUTPUT_JSON)
    save_results_csv(results, OUTPUT_CSV)

    print("\n===== GRAPH-BASED EVALUATION RESULTS =====")
    for k, v in macro.items():
        print(f"{k:24s}: {v}")

    print(f"\nSaved JSON: {OUTPUT_JSON}")
    print(f"Saved CSV : {OUTPUT_CSV}")


if __name__ == "__main__":
    run_evaluation()