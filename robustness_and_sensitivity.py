#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Robustness and sensitivity analysis for graph-based Manipuri summarization.

Computes:
1. Robustness across document complexity groups
2. Sensitivity to salience selection thresholds

@author: jimmy
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import networkx as nx


# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "cleaned_dataset_with_manipuri_summary.json"
OURS_DIR = "realizer_outputs"

OUTPUT_JSON = "robustness_sensitivity_results.json"

SUMMARY_FIELD = "summary_mni"

# Thresholds for sensitivity analysis
SALience_PERCENTAGES = [0.3, 0.5, 0.7]

# Consistency weights
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
    num_entities: int
    num_events: int


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
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [tok for tok in text.split() if tok.strip()]


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = safe_text(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# =========================================================
# LOADING
# =========================================================

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    data = load_json(dataset_path)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Dataset must be a JSON object or list of JSON objects.")


def build_output_index(folder: str, summary_field: str) -> Dict[str, str]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Output folder not found: {folder}")

    index: Dict[str, str] = {}
    for fp in folder_path.glob("*.json"):
        try:
            obj = load_json(fp)
            doc_id = safe_text(obj.get("doc_id"))
            summary = safe_text(obj.get(summary_field))
            if doc_id:
                index[doc_id] = summary
        except Exception as exc:
            print(f"Warning: could not load {fp}: {exc}")
    return index


# =========================================================
# GOLD EXTRACTION
# =========================================================

def extract_gold_entities(doc: Dict[str, Any]) -> List[str]:
    entities: List[str] = []

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

    return unique_preserve_order(entities)


def extract_gold_events(doc: Dict[str, Any]) -> List[str]:
    triggers: List[str] = []
    for ev in doc.get("events", []):
        trig = safe_text(ev.get("trigger"))
        trig_en = safe_text(ev.get("trigger_en"))
        if trig:
            triggers.append(trig)
        if trig_en:
            triggers.append(trig_en)
    return unique_preserve_order(triggers)


# =========================================================
# METRICS
# =========================================================

def extract_summary_mentions(summary: str, candidates: List[str]) -> List[str]:
    summary = safe_text(summary)
    found = []
    for c in candidates:
        if c and c in summary:
            found.append(c)
    return unique_preserve_order(found)


def compute_entity_coverage(gold_entities: List[str], summary: str) -> float:
    if not gold_entities:
        return 0.0
    pred = extract_summary_mentions(summary, gold_entities)
    return len(pred) / len(gold_entities)


def compute_event_coverage(gold_events: List[str], summary: str) -> float:
    if not gold_events:
        return 0.0
    pred = extract_summary_mentions(summary, gold_events)
    return len(pred) / len(gold_events)


def compute_fact_precision(gold_entities: List[str], gold_events: List[str], summary: str) -> float:
    summary_tokens = tokenize(summary)
    if not summary_tokens:
        return 0.0

    gold_token_set = set()
    for item in gold_entities + gold_events:
        for tok in tokenize(item):
            gold_token_set.add(tok)

    supported = [tok for tok in summary_tokens if tok in gold_token_set]
    return len(supported) / len(summary_tokens)


def compute_consistency_score(entity_cov: float, event_cov: float, fact_precision: float) -> float:
    return ALPHA * entity_cov + BETA * event_cov + GAMMA * fact_precision


def evaluate_summary(doc: Dict[str, Any], summary: str) -> DocMetrics:
    gold_entities = extract_gold_entities(doc)
    gold_events = extract_gold_events(doc)

    ec = compute_entity_coverage(gold_entities, summary)
    evc = compute_event_coverage(gold_events, summary)
    fp = compute_fact_precision(gold_entities, gold_events, summary)
    cs = compute_consistency_score(ec, evc, fp)

    return DocMetrics(
        doc_id=safe_text(doc.get("doc_id")),
        entity_coverage=round(ec, 6),
        event_coverage=round(evc, 6),
        fact_precision=round(fp, 6),
        consistency_score=round(cs, 6),
        num_entities=len(gold_entities),
        num_events=len(doc.get("events", [])),
    )


def average_metrics(metrics_list: List[DocMetrics]) -> Dict[str, float]:
    if not metrics_list:
        return {
            "entity_coverage": 0.0,
            "event_coverage": 0.0,
            "fact_precision": 0.0,
            "consistency_score": 0.0,
            "count": 0
        }

    n = len(metrics_list)
    return {
        "entity_coverage": round(sum(m.entity_coverage for m in metrics_list) / n, 4),
        "event_coverage": round(sum(m.event_coverage for m in metrics_list) / n, 4),
        "fact_precision": round(sum(m.fact_precision for m in metrics_list) / n, 4),
        "consistency_score": round(sum(m.consistency_score for m in metrics_list) / n, 4),
        "count": n
    }


# =========================================================
# ROBUSTNESS BY EVENT COMPLEXITY
# =========================================================

def event_complexity_group(num_events: int) -> str:
    if num_events <= 3:
        return "1-3"
    if num_events <= 6:
        return "4-6"
    return "7+"


def compute_robustness_by_complexity(
    dataset: List[Dict[str, Any]],
    summary_index: Dict[str, str]
) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[DocMetrics]] = {
        "1-3": [],
        "4-6": [],
        "7+": []
    }

    for doc in dataset:
        doc_id = safe_text(doc.get("doc_id"))
        if doc_id not in summary_index:
            continue

        summary = summary_index[doc_id]
        metrics = evaluate_summary(doc, summary)
        group = event_complexity_group(len(doc.get("events", [])))
        groups[group].append(metrics)

    return {group: average_metrics(items) for group, items in groups.items()}


# =========================================================
# SIMPLE GRAPH + SALIENCE FOR SENSITIVITY
# =========================================================

def build_event_entity_graph(doc: Dict[str, Any]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    entity_nodes: Dict[str, str] = {}
    for i, ent in enumerate(doc.get("ner", []), start=1):
        text = safe_text(ent.get("text"))
        if not text:
            continue
        node_id = f"ent::{i}"
        graph.add_node(
            node_id,
            node_type="entity",
            surface=text,
            text_en=safe_text(ent.get("text_en")),
            label=safe_text(ent.get("label"))
        )
        entity_nodes[text] = node_id

    event_nodes: Dict[str, str] = {}
    for i, ev in enumerate(doc.get("events", []), start=1):
        ev_id = safe_text(ev.get("event_id", f"ev_{i}"))
        node_id = f"evt::{ev_id}"
        graph.add_node(
            node_id,
            node_type="event",
            event_id=ev_id,
            trigger=safe_text(ev.get("trigger")),
            trigger_en=safe_text(ev.get("trigger_en")),
            event_type=safe_text(ev.get("type")),
            event_order=i
        )
        event_nodes[ev_id] = node_id

    for ev in doc.get("events", []):
        ev_id = safe_text(ev.get("event_id"))
        ev_node = event_nodes.get(ev_id)
        if ev_node is None:
            continue

        for arg in ev.get("arguments", []):
            arg_text = safe_text(arg.get("text"))
            role = safe_text(arg.get("role"))
            if not arg_text:
                continue

            ent_node = entity_nodes.get(arg_text)
            if ent_node is None:
                ent_node = f"ent::auto::{arg_text}"
                if ent_node not in graph:
                    graph.add_node(
                        ent_node,
                        node_type="entity",
                        surface=arg_text,
                        text_en=safe_text(arg.get("text_en")),
                        label="UNLINKED_ARG"
                    )

            graph.add_edge(
                ev_node,
                ent_node,
                edge_type="event_entity_role",
                role=role,
                weight=1.0
            )
            graph.add_edge(
                ent_node,
                ev_node,
                edge_type="entity_event_role_reverse",
                role=role,
                weight=1.0
            )

    return graph


def compute_salience_scores(graph: nx.MultiDiGraph) -> Dict[str, float]:
    events = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "event"]
    entities = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "entity"]

    s_event = {u: 1.0 for u in events}
    s_entity = {e: 1.0 for e in entities}

    for _ in range(20):
        new_s_event = {}
        new_s_entity = {}

        for u in events:
            neighbors = [
                (v, d.get("weight", 1.0))
                for _, v, d in graph.out_edges(u, data=True)
                if graph.nodes[v].get("node_type") == "entity"
                and d.get("edge_type") == "event_entity_role"
            ]
            if neighbors:
                total = sum(w for _, w in neighbors)
                score = sum((w / total) * s_entity[v] for v, w in neighbors)
            else:
                score = 0.0
            order = int(graph.nodes[u].get("event_order", 1))
            prior = 1.0 / order
            new_s_event[u] = 0.15 * prior + 0.85 * score

        for e in entities:
            neighbors = [
                (u, d.get("weight", 1.0))
                for u, _, d in graph.in_edges(e, data=True)
                if graph.nodes[u].get("node_type") == "event"
                and d.get("edge_type") == "event_entity_role"
            ]
            if neighbors:
                total = sum(w for _, w in neighbors)
                score = sum((w / total) * s_event[u] for u, w in neighbors)
            else:
                score = 0.0
            surface = safe_text(graph.nodes[e].get("surface"))
            tf = 1.0 + len(tokenize(surface)) * 0.0
            new_s_entity[e] = 0.15 * tf + 0.85 * score

        s_event, s_entity = new_s_event, new_s_entity

    scores = {}
    scores.update(s_event)
    scores.update(s_entity)
    return scores


def build_threshold_summary(doc: Dict[str, Any], salience_ratio: float) -> str:
    graph = build_event_entity_graph(doc)
    if graph.number_of_nodes() == 0:
        return ""

    scores = compute_salience_scores(graph)

    event_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "event"]
    entity_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "entity"]

    ranked_events = sorted(event_nodes, key=lambda n: scores.get(n, 0.0), reverse=True)
    ranked_entities = sorted(entity_nodes, key=lambda n: scores.get(n, 0.0), reverse=True)

    top_events_n = max(1, int(len(ranked_events) * salience_ratio)) if ranked_events else 0
    top_entities_n = max(1, int(len(ranked_entities) * salience_ratio)) if ranked_entities else 0

    selected_events = ranked_events[:top_events_n]
    selected_entities = set(ranked_entities[:top_entities_n])

    realized_sentences: List[str] = []

    for ev_node in selected_events:
        ev_attrs = graph.nodes[ev_node]
        trigger = safe_text(ev_attrs.get("trigger"))
        if not trigger:
            continue

        parts: List[str] = []
        args: List[str] = []

        for _, ent_node, edge_attrs in graph.out_edges(ev_node, data=True):
            if edge_attrs.get("edge_type") != "event_entity_role":
                continue
            if ent_node not in selected_entities:
                continue

            ent_surface = safe_text(graph.nodes[ent_node].get("surface"))
            if ent_surface:
                args.append(ent_surface)

        args = unique_preserve_order(args)

        if args:
            parts.append(" ".join(args[:4]))
        parts.append(trigger)

        sent = normalize_ws(" ".join(parts))
        if sent:
            realized_sentences.append(sent)

    return normalize_ws(". ".join(unique_preserve_order(realized_sentences)))


def compute_sensitivity_to_salience(dataset: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, List[DocMetrics]] = defaultdict(list)

    for ratio in SALience_PERCENTAGES:
        for doc in dataset:
            summary = build_threshold_summary(doc, ratio)
            metrics = evaluate_summary(doc, summary)
            key = f"top_{int(ratio * 100)}"
            results[key].append(metrics)

    return {k: average_metrics(v) for k, v in results.items()}


# =========================================================
# MAIN
# =========================================================

def run_analysis() -> None:
    dataset = load_dataset(DATASET_PATH)
    summary_index = build_output_index(OURS_DIR, SUMMARY_FIELD)

    robustness = compute_robustness_by_complexity(dataset, summary_index)
    sensitivity = compute_sensitivity_to_salience(dataset)

    output = {
        "config": {
            "dataset_path": DATASET_PATH,
            "ours_dir": OURS_DIR,
            "summary_field": SUMMARY_FIELD,
            "salience_percentages": SALience_PERCENTAGES,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA
        },
        "robustness_by_event_complexity": robustness,
        "sensitivity_to_salience": sensitivity
    }

    save_json(output, OUTPUT_JSON)

    print("\n===== ROBUSTNESS BY EVENT COMPLEXITY =====")
    for group, vals in robustness.items():
        print(
            f"{group:>4} | "
            f"EC={vals['entity_coverage']:.4f} | "
            f"EvC={vals['event_coverage']:.4f} | "
            f"FP={vals['fact_precision']:.4f} | "
            f"CS={vals['consistency_score']:.4f} | "
            f"N={vals['count']}"
        )

    print("\n===== SENSITIVITY TO SALIENCE =====")
    for threshold, vals in sensitivity.items():
        print(
            f"{threshold:>8} | "
            f"EC={vals['entity_coverage']:.4f} | "
            f"EvC={vals['event_coverage']:.4f} | "
            f"FP={vals['fact_precision']:.4f} | "
            f"CS={vals['consistency_score']:.4f} | "
            f"N={vals['count']}"
        )

    print(f"\nSaved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_analysis()