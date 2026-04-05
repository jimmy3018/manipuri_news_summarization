#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Event-Entity Graph + Planner + Realizer
Manipuri-to-Manipuri grounded summarization

Key improvements:
1. Event quality scoring
2. Weak event filtering
3. Stronger entity coverage in planner
4. Strict fact-based realizer
5. Event-type-aware realization
6. Removed generic unsupported fallback entity sentences

@author: jimmy
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["font.family"] = [
    "Noto Sans Bengali",
    "Noto Sans",
    "Arial Unicode MS",
    "DejaVu Sans",
]


# =========================================================
# CONFIG
# =========================================================

@dataclass
class GraphBuildConfig:
    add_sentence_nodes: bool = False
    add_entity_entity_cooccurrence: bool = True
    add_event_event_temporal: bool = True
    cooccurrence_window: int = 1


@dataclass
class PlannerConfig:
    top_events: int = 8
    top_entities: int = 12
    max_content_units: int = 5
    min_event_salience: float = 0.0
    min_event_arguments: int = 2
    force_entity_coverage: bool = True


@dataclass
class RealizerConfig:
    max_sentences: int = 4
    cite_evidence: bool = True
    include_location_in_lead: bool = False
    max_facts_per_unit: int = 2
    keep_one_sentence_per_fact: bool = True
    allow_unit_level_merge: bool = False


# =========================================================
# UTILS
# =========================================================

def safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_ws(text: str) -> str:
    return " ".join(safe_text(text).split())


def ensure_period(text: str) -> str:
    text = normalize_ws(text)
    if not text:
        return ""
    if text[-1] not in ".!?।":
        text += "."
    return text


def clean_event_type(event_type: str) -> str:
    return safe_text(event_type).upper()


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = safe_text(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def save_json(data: Dict[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(output_path)


# =========================================================
# GRAPH BUILDER
# =========================================================

class EventEntityGraphBuilder:
    def __init__(self, config: Optional[GraphBuildConfig] = None) -> None:
        self.config = config or GraphBuildConfig()

    def build_graph(self, doc: Dict[str, Any]) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()

        graph.graph["doc_id"] = safe_text(doc.get("doc_id", "UNKNOWN_DOC"))
        graph.graph["headline"] = safe_text(doc.get("headline"))
        graph.graph["headline_en"] = safe_text(doc.get("headline_en"))
        graph.graph["location"] = safe_text(doc.get("location"))
        graph.graph["source_language"] = safe_text(doc.get("source_language"))

        sentence_map = self._index_sentences(doc)
        graph.graph["sentence_map"] = sentence_map

        if self.config.add_sentence_nodes:
            self._add_sentence_nodes(graph, sentence_map)

        entity_index = self._add_entity_nodes(graph, doc)
        event_index = self._add_event_nodes(graph, doc)

        self._add_event_entity_edges(graph, doc, entity_index, event_index)

        if self.config.add_event_event_temporal:
            self._add_event_event_edges(graph, doc, event_index)

        if self.config.add_entity_entity_cooccurrence:
            self._add_entity_cooccurrence_edges(graph, doc, entity_index, sentence_map)

        return graph

    def _index_sentences(self, doc: Dict[str, Any]) -> Dict[int, str]:
        sentence_map: Dict[int, str] = {}
        for sent in doc.get("sentences", []):
            sid = int(sent["sid"])
            sentence_map[sid] = safe_text(sent["text"])
        return sentence_map

    def _add_sentence_nodes(self, graph: nx.MultiDiGraph, sentence_map: Dict[int, str]) -> None:
        for sid, text in sentence_map.items():
            graph.add_node(
                f"sent::{sid}",
                node_type="sentence",
                sid=sid,
                text=text,
            )

    def _add_entity_nodes(self, graph: nx.MultiDiGraph, doc: Dict[str, Any]) -> Dict[str, str]:
        entity_index: Dict[str, str] = {}

        for i, ent in enumerate(doc.get("ner", []), start=1):
            surface = safe_text(ent.get("text"))
            node_id = f"ent::{i}"

            graph.add_node(
                node_id,
                node_type="entity",
                surface=surface,
                text_en=safe_text(ent.get("text_en")),
                label=safe_text(ent.get("label")),
                confidence=float(ent.get("confidence", 1.0) or 1.0),
            )
            if surface:
                entity_index[surface] = node_id

        return entity_index

    def _add_event_nodes(self, graph: nx.MultiDiGraph, doc: Dict[str, Any]) -> Dict[str, str]:
        event_index: Dict[str, str] = {}

        for idx, ev in enumerate(doc.get("events", []), start=1):
            event_id = safe_text(ev.get("event_id", f"ev_{idx}"))
            node_id = f"evt::{event_id}"

            graph.add_node(
                node_id,
                node_type="event",
                event_id=event_id,
                trigger=safe_text(ev.get("trigger")),
                trigger_en=safe_text(ev.get("trigger_en")),
                event_type=safe_text(ev.get("type")),
                confidence=float(ev.get("confidence", 1.0) or 1.0),
                event_order=idx,
            )
            event_index[event_id] = node_id

        return event_index

    def _add_event_entity_edges(
        self,
        graph: nx.MultiDiGraph,
        doc: Dict[str, Any],
        entity_index: Dict[str, str],
        event_index: Dict[str, str],
    ) -> None:
        for ev in doc.get("events", []):
            ev_node = event_index.get(safe_text(ev.get("event_id")))
            if ev_node is None:
                continue

            ev_conf = float(graph.nodes[ev_node].get("confidence", 1.0))

            for arg in ev.get("arguments", []):
                arg_text = safe_text(arg.get("text"))
                role = safe_text(arg.get("role")).upper()

                ent_node = entity_index.get(arg_text)
                if ent_node is None:
                    ent_node = f"ent::auto::{arg_text}"
                    if ent_node not in graph:
                        graph.add_node(
                            ent_node,
                            node_type="entity",
                            surface=arg_text,
                            text_en=safe_text(arg.get("text_en")),
                            label="UNLINKED_ARG",
                            confidence=0.7,
                            auto_created=True,
                        )

                ent_conf = float(graph.nodes[ent_node].get("confidence", 1.0))
                weight = (0.6 * ev_conf) + (0.4 * ent_conf)

                graph.add_edge(
                    ev_node,
                    ent_node,
                    edge_type="event_entity_role",
                    role=role,
                    role_text=arg_text,
                    role_text_en=safe_text(arg.get("text_en")),
                    weight=weight,
                )
                graph.add_edge(
                    ent_node,
                    ev_node,
                    edge_type="entity_event_role_reverse",
                    role=role,
                    weight=weight,
                )

    def _add_event_event_edges(
        self,
        graph: nx.MultiDiGraph,
        doc: Dict[str, Any],
        event_index: Dict[str, str],
    ) -> None:
        events = doc.get("events", [])
        for i in range(len(events) - 1):
            src = event_index.get(safe_text(events[i].get("event_id")))
            dst = event_index.get(safe_text(events[i + 1].get("event_id")))
            if src and dst:
                graph.add_edge(
                    src,
                    dst,
                    edge_type="event_event_next",
                    relation="next",
                    weight=0.5,
                )

    def _add_entity_cooccurrence_edges(
        self,
        graph: nx.MultiDiGraph,
        doc: Dict[str, Any],
        entity_index: Dict[str, str],
        sentence_map: Dict[int, str],
    ) -> None:
        ner_items = doc.get("ner", [])
        if not ner_items:
            return

        for sid, sent_text in sentence_map.items():
            present_entities: List[str] = []
            for ent in ner_items:
                surface = safe_text(ent.get("text"))
                if surface and surface in sent_text and surface in entity_index:
                    present_entities.append(surface)

            present_entities = unique_preserve_order(present_entities)

            for i in range(len(present_entities)):
                for j in range(i + 1, len(present_entities)):
                    e1 = entity_index[present_entities[i]]
                    e2 = entity_index[present_entities[j]]

                    graph.add_edge(
                        e1,
                        e2,
                        edge_type="entity_entity_cooccurs",
                        relation="cooccurs",
                        sid=sid,
                        weight=0.3,
                    )
                    graph.add_edge(
                        e2,
                        e1,
                        edge_type="entity_entity_cooccurs",
                        relation="cooccurs",
                        sid=sid,
                        weight=0.3,
                    )


# =========================================================
# LOADING
# =========================================================

def load_dataset_json(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError("Dataset must contain either a single JSON object or a list of JSON objects.")


# =========================================================
# VISUALIZATION
# =========================================================

def print_graph_summary(graph: nx.MultiDiGraph) -> None:
    print("=" * 80)
    print(f"Document ID     : {graph.graph.get('doc_id')}")
    print(f"Headline        : {graph.graph.get('headline')}")
    print(f"Location        : {graph.graph.get('location')}")
    print(f"Source Language : {graph.graph.get('source_language')}")
    print(f"Nodes           : {graph.number_of_nodes()}")
    print(f"Edges           : {graph.number_of_edges()}")
    print("=" * 80)

    node_type_counts: Dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    print("Node type counts:")
    for k, v in sorted(node_type_counts.items()):
        print(f"  - {k}: {v}")

    edge_type_counts: Dict[str, int] = {}
    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs.get("edge_type", "unknown")
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

    print("\nEdge type counts:")
    for k, v in sorted(edge_type_counts.items()):
        print(f"  - {k}: {v}")


def to_simple_digraph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    simple_graph = nx.DiGraph()

    for node_id, attrs in graph.nodes(data=True):
        simple_graph.add_node(node_id, **attrs)

    for u, v, attrs in graph.edges(data=True):
        if simple_graph.has_edge(u, v):
            simple_graph[u][v]["weight"] += float(attrs.get("weight", 1.0))
        else:
            simple_graph.add_edge(
                u,
                v,
                weight=float(attrs.get("weight", 1.0)),
                edge_type=attrs.get("edge_type", "unknown"),
            )
    return simple_graph


def build_node_label(node_id: str, attrs: Dict[str, Any]) -> str:
    node_type = attrs.get("node_type", "")

    if node_type == "entity":
        return safe_text(attrs.get("surface")) or safe_text(attrs.get("text_en")) or "ENT"
    if node_type == "event":
        return safe_text(attrs.get("trigger")) or safe_text(attrs.get("event_type")) or "EVT"
    if node_type == "sentence":
        sid = attrs.get("sid", "")
        return f"s{sid}"
    return node_id


def visualize_graph(graph: nx.MultiDiGraph, figsize: Tuple[int, int] = (14, 10)) -> None:
    simple_graph = to_simple_digraph(graph)
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(simple_graph, k=1.0, seed=42)

    entity_nodes = [n for n, a in simple_graph.nodes(data=True) if a.get("node_type") == "entity"]
    event_nodes = [n for n, a in simple_graph.nodes(data=True) if a.get("node_type") == "event"]
    sentence_nodes = [n for n, a in simple_graph.nodes(data=True) if a.get("node_type") == "sentence"]

    nx.draw_networkx_nodes(simple_graph, pos, nodelist=entity_nodes, node_size=1200, node_shape="o", alpha=0.9)
    nx.draw_networkx_nodes(simple_graph, pos, nodelist=event_nodes, node_size=1600, node_shape="s", alpha=0.9)

    if sentence_nodes:
        nx.draw_networkx_nodes(simple_graph, pos, nodelist=sentence_nodes, node_size=900, node_shape="d", alpha=0.8)

    nx.draw_networkx_edges(simple_graph, pos, arrows=True, arrowstyle="->", arrowsize=15, width=1.5, alpha=0.6)

    labels = {node_id: build_node_label(node_id, attrs) for node_id, attrs in simple_graph.nodes(data=True)}
    nx.draw_networkx_labels(simple_graph, pos, labels=labels, font_size=8)

    title = graph.graph.get("headline") or graph.graph.get("doc_id", "Graph")
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =========================================================
# SALIENCE
# =========================================================

def compute_salience_scores(
    graph: nx.MultiDiGraph,
    alpha: float = 0.85,
    max_iter: int = 30,
    tol: float = 1e-4,
    pagerank_beta: float = 0.20,
) -> Dict[str, float]:
    events = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "event"]
    entities = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "entity"]

    s_event = {u: 1.0 for u in events}
    s_entity = {e: 1.0 for e in entities}

    entity_tf: Dict[str, float] = {}
    for e in entities:
        surface = safe_text(graph.nodes[e].get("surface"))
        freq = sum(
            1 for _, _, d in graph.edges(data=True)
            if safe_text(d.get("role_text")) == surface
        )
        entity_tf[e] = float(max(freq, 1))

    max_tf = max(entity_tf.values()) if entity_tf else 1.0
    for e in entity_tf:
        entity_tf[e] /= max_tf

    for _ in range(max_iter):
        new_s_event: Dict[str, float] = {}
        new_s_entity: Dict[str, float] = {}

        for u in events:
            neighbors = [
                (v, float(d.get("weight", 1.0)))
                for _, v, d in graph.out_edges(u, data=True)
                if graph.nodes[v].get("node_type") == "entity" and d.get("edge_type") == "event_entity_role"
            ]

            if neighbors:
                total_weight = sum(w for _, w in neighbors) or 1.0
                score = sum((w / total_weight) * s_entity[v] for v, w in neighbors)
            else:
                score = 0.0

            event_order = int(graph.nodes[u].get("event_order", 1))
            prior = 1.0 / (1 + max(event_order - 1, 0))
            new_s_event[u] = (1 - alpha) * prior + alpha * score

        for e in entities:
            neighbors = [
                (u, float(d.get("weight", 1.0)))
                for u, _, d in graph.in_edges(e, data=True)
                if graph.nodes[u].get("node_type") == "event" and d.get("edge_type") == "event_entity_role"
            ]

            if neighbors:
                total_weight = sum(w for _, w in neighbors) or 1.0
                score = sum((w / total_weight) * s_event[u] for u, w in neighbors)
            else:
                score = 0.0

            new_s_entity[e] = (1 - alpha) * entity_tf[e] + alpha * score

        diff = sum(abs(new_s_event[u] - s_event[u]) for u in events) + \
               sum(abs(new_s_entity[e] - s_entity[e]) for e in entities)

        s_event, s_entity = new_s_event, new_s_entity
        if diff < tol:
            break

    base_scores: Dict[str, float] = {}
    base_scores.update(s_event)
    base_scores.update(s_entity)

    simple_graph = to_simple_digraph(graph)
    if simple_graph.number_of_nodes() > 0 and simple_graph.number_of_edges() > 0:
        try:
            pr = nx.pagerank(simple_graph, weight="weight")
            final_scores = {
                n: (1 - pagerank_beta) * base_scores.get(n, 0.0) + pagerank_beta * pr.get(n, 0.0)
                for n in simple_graph.nodes()
            }
            return final_scores
        except Exception:
            return base_scores

    return base_scores


def print_salience(graph: nx.MultiDiGraph) -> None:
    scores = compute_salience_scores(graph)
    print("\n===== SALIENCE SCORES =====")
    for node, score in sorted(scores.items(), key=lambda x: -x[1]):
        attrs = graph.nodes[node]
        label = (
            safe_text(attrs.get("surface"))
            or safe_text(attrs.get("trigger"))
            or safe_text(attrs.get("text_en"))
            or node
        )
        print(f"{label:45s} -> {score:.4f}")


# =========================================================
# EVENT QUALITY
# =========================================================

def compute_event_quality(event: Dict[str, Any]) -> float:
    salience = float(event.get("salience", 0.0))
    confidence = float(event.get("confidence", 1.0) or 1.0)
    arg_count = len(event.get("arguments", []))

    completeness = min(arg_count / 3.0, 1.0)
    return (0.5 * salience) + (0.3 * confidence) + (0.2 * completeness)


# =========================================================
# EVIDENCE
# =========================================================

def find_supporting_sentences_for_text(sentence_map: Dict[int, str], text_value: Optional[str]) -> List[int]:
    text_value = safe_text(text_value)
    if not text_value:
        return []

    matched: List[int] = []
    for sid, sent_text in sentence_map.items():
        if text_value in sent_text:
            matched.append(sid)
    return matched


def collect_event_evidence(graph: nx.MultiDiGraph, event_node: str) -> List[int]:
    sentence_map: Dict[int, str] = graph.graph.get("sentence_map", {})
    attrs = graph.nodes[event_node]

    evidence_ids = set(find_supporting_sentences_for_text(sentence_map, attrs.get("trigger")))

    for _, entity_node, edge_attrs in graph.out_edges(event_node, data=True):
        if edge_attrs.get("edge_type") != "event_entity_role":
            continue

        role_text = edge_attrs.get("role_text")
        ent_surface = graph.nodes[entity_node].get("surface")

        evidence_ids.update(find_supporting_sentences_for_text(sentence_map, role_text))
        evidence_ids.update(find_supporting_sentences_for_text(sentence_map, ent_surface))

    return sorted(evidence_ids)


def collect_entity_evidence(graph: nx.MultiDiGraph, entity_node: str) -> List[int]:
    sentence_map: Dict[int, str] = graph.graph.get("sentence_map", {})
    surface = graph.nodes[entity_node].get("surface")
    return sorted(find_supporting_sentences_for_text(sentence_map, surface))


# =========================================================
# PLANNER-READY INPUT
# =========================================================

def get_top_salient_nodes(
    graph: nx.MultiDiGraph,
    scores: Dict[str, float],
    top_events: int = 8,
    top_entities: int = 12,
) -> Tuple[List[str], List[str]]:
    event_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "event"]
    entity_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "entity"]

    ranked_events = sorted(event_nodes, key=lambda n: scores.get(n, 0.0), reverse=True)[:top_events]
    ranked_entities = sorted(entity_nodes, key=lambda n: scores.get(n, 0.0), reverse=True)[:top_entities]
    return ranked_events, ranked_entities


def build_planner_ready_input(
    doc: Dict[str, Any],
    graph: nx.MultiDiGraph,
    planner_config: Optional[PlannerConfig] = None,
) -> Dict[str, Any]:
    planner_config = planner_config or PlannerConfig()

    scores = compute_salience_scores(graph)
    ranked_events, ranked_entities = get_top_salient_nodes(
        graph,
        scores,
        top_events=planner_config.top_events,
        top_entities=planner_config.top_entities,
    )

    sentence_map: Dict[int, str] = graph.graph.get("sentence_map", {})

    planner_events: List[Dict[str, Any]] = []
    for evt_node in ranked_events:
        evt_attrs = graph.nodes[evt_node]

        arguments: List[Dict[str, Any]] = []
        for _, ent_node, edge_attrs in graph.out_edges(evt_node, data=True):
            if edge_attrs.get("edge_type") != "event_entity_role":
                continue
            ent_attrs = graph.nodes[ent_node]
            arguments.append({
                "role": safe_text(edge_attrs.get("role")),
                "text": safe_text(ent_attrs.get("surface")),
                "text_en": safe_text(ent_attrs.get("text_en")),
                "label": safe_text(ent_attrs.get("label")),
            })

        planner_events.append({
            "event_id": safe_text(evt_attrs.get("event_id")),
            "trigger": safe_text(evt_attrs.get("trigger")),
            "trigger_en": safe_text(evt_attrs.get("trigger_en")),
            "event_type": safe_text(evt_attrs.get("event_type")),
            "confidence": float(evt_attrs.get("confidence", 1.0)),
            "salience": round(float(scores.get(evt_node, 0.0)), 4),
            "arguments": arguments,
            "evidence_ids": collect_event_evidence(graph, evt_node),
        })

    planner_entities: List[Dict[str, Any]] = []
    for ent_node in ranked_entities:
        ent_attrs = graph.nodes[ent_node]
        planner_entities.append({
            "text": safe_text(ent_attrs.get("surface")),
            "text_en": safe_text(ent_attrs.get("text_en")),
            "label": safe_text(ent_attrs.get("label")),
            "salience": round(float(scores.get(ent_node, 0.0)), 4),
            "evidence_ids": collect_entity_evidence(graph, ent_node),
        })

    return {
        "doc_id": safe_text(doc.get("doc_id")),
        "headline": safe_text(doc.get("headline")),
        "headline_en": safe_text(doc.get("headline_en")),
        "location": safe_text(doc.get("location")),
        "source_language": safe_text(doc.get("source_language")),
        "summary_target_language": "Manipuri",
        "planner_goal": "থৌদোক অমসুং মীওইশিংগী মরু ওইবা ৱাফমশিং খনদুনা evidence sentence-শিংগী মথক্তা grounded summary plan শেম্বা.",
        "salient_events": planner_events,
        "salient_entities": planner_entities,
        "evidence_sentences": [{"sid": sid, "text": text} for sid, text in sorted(sentence_map.items())],
    }


def print_planner_ready_input(planner_input: Dict[str, Any]) -> None:
    print("\n===== PLANNER-READY INPUT =====")
    print(json.dumps(planner_input, ensure_ascii=False, indent=2))


# =========================================================
# PLANNER
# =========================================================

def extract_event_entity_names(event: Dict[str, Any]) -> Set[str]:
    names = set()
    for arg in event.get("arguments", []):
        val = safe_text(arg.get("text")) or safe_text(arg.get("text_en"))
        if val:
            names.add(val)
    return names


def infer_discourse_role(event: Dict[str, Any]) -> str:
    et = clean_event_type(event.get("event_type", ""))

    if any(k in et for k in ["STATEMENT", "WARNING", "APPEAL", "REACTION", "OPINION", "DEMAND", "CLARIFICATION"]):
        return "reaction"
    if any(k in et for k in ["CASUALTY", "INJURY", "KILLING", "EXPLOSION", "ATTACK", "ARREST", "PROTEST", "STRIKE", "CORRUPTION"]):
        return "main"
    if any(k in et for k in ["SUPPORT", "RELIEF", "REWARD", "TREATMENT", "TRAINING", "AWARENESS"]):
        return "consequence"
    if any(k in et for k in ["ANNOUNCEMENT", "PLANNING", "MERGER", "ALIGNMENT", "POLICY_CHANGE", "TRANSFER"]):
        return "background"

    return "detail"


def build_atomic_fact(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": safe_text(event.get("event_id")),
        "event_type": safe_text(event.get("event_type")),
        "trigger": safe_text(event.get("trigger")),
        "trigger_en": safe_text(event.get("trigger_en")),
        "arguments": event.get("arguments", []),
        "evidence_ids": event.get("evidence_ids", []),
        "salience": float(event.get("salience", 0.0)),
        "confidence": float(event.get("confidence", 1.0)),
        "quality": compute_event_quality(event),
        "discourse_role": infer_discourse_role(event),
    }


def cluster_events_into_units(events: List[Dict[str, Any]], max_units: int = 5) -> List[List[Dict[str, Any]]]:
    remaining = events[:]
    units: List[List[Dict[str, Any]]] = []

    while remaining and len(units) < max_units:
        seed = remaining.pop(0)
        seed_entities = extract_event_entity_names(seed)
        seed_evidence = set(seed.get("evidence_ids", []))
        cluster = [seed]

        still_left = []
        for ev in remaining:
            ev_entities = extract_event_entity_names(ev)
            ev_evidence = set(ev.get("evidence_ids", []))

            entity_sim = jaccard(seed_entities, ev_entities)
            evidence_overlap = len(seed_evidence & ev_evidence)

            if entity_sim >= 0.20 or evidence_overlap > 0:
                cluster.append(ev)
                seed_entities |= ev_entities
                seed_evidence |= ev_evidence
            else:
                still_left.append(ev)

        remaining = still_left
        units.append(cluster)

    if remaining and units:
        for ev in remaining:
            units[-1].append(ev)

    return units


def infer_summary_focus(planner_input: Dict[str, Any]) -> str:
    headline = safe_text(planner_input.get("headline"))
    if headline:
        return headline

    events = planner_input.get("salient_events", [])
    if events:
        top = events[0]
        return safe_text(top.get("trigger")) or safe_text(top.get("event_type"))

    return "মরু ওইবা থৌদোকশিং"


def build_content_unit(unit_id: int, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    events_sorted = sorted(events, key=lambda e: compute_event_quality(e), reverse=True)
    lead = events_sorted[0]

    role_priority = {"main": 0, "detail": 1, "reaction": 2, "consequence": 3, "background": 4}
    purposes = [infer_discourse_role(e) for e in events_sorted]
    purpose = sorted(purposes, key=lambda p: role_priority.get(p, 99))[0]

    evidence_ids = sorted(unique_preserve_order([
        str(sid) for ev in events_sorted for sid in ev.get("evidence_ids", [])
    ]))
    evidence_ids_int = [int(x) for x in evidence_ids if str(x).isdigit()]

    entities = unique_preserve_order([
        safe_text(arg.get("text"))
        for ev in events_sorted
        for arg in ev.get("arguments", [])
    ])

    atomic_facts = [build_atomic_fact(ev) for ev in events_sorted]

    return {
        "unit_id": f"u{unit_id}",
        "purpose": purpose,
        "lead_event_id": safe_text(lead.get("event_id")),
        "event_ids": [safe_text(ev.get("event_id")) for ev in events_sorted],
        "event_types": unique_preserve_order([safe_text(ev.get("event_type")) for ev in events_sorted]),
        "salience": round(max(float(ev.get("salience", 0.0)) for ev in events_sorted), 4),
        "quality": round(max(compute_event_quality(ev) for ev in events_sorted), 4),
        "entities": entities,
        "evidence_ids": evidence_ids_int,
        "atomic_facts": atomic_facts,
    }


def select_must_include_entities(planner_input: Dict[str, Any], max_entities: int = 6) -> List[str]:
    entities = []
    for ent in planner_input.get("salient_entities", [])[:max_entities]:
        name = safe_text(ent.get("text"))
        if name and name not in entities:
            entities.append(name)
    return entities


def select_must_include_events(planner_input: Dict[str, Any], max_events: int = 4) -> List[str]:
    out = []
    for ev in planner_input.get("salient_events", [])[:max_events]:
        eid = safe_text(ev.get("event_id"))
        if eid and eid not in out:
            out.append(eid)
    return out


def build_planner_output(
    planner_input: Dict[str, Any],
    planner_config: Optional[PlannerConfig] = None,
) -> Dict[str, Any]:
    planner_config = planner_config or PlannerConfig()

    events = planner_input.get("salient_events", [])

    filtered_events = [
        e for e in events
        if len(e.get("arguments", [])) >= planner_config.min_event_arguments
    ]

    if planner_config.min_event_salience > 0:
        filtered_events = [
            e for e in filtered_events
            if float(e.get("salience", 0.0)) >= planner_config.min_event_salience
        ]

    sorted_events = sorted(
        filtered_events,
        key=lambda e: compute_event_quality(e),
        reverse=True
    )

    if not sorted_events:
        sorted_events = sorted(events, key=lambda e: compute_event_quality(e), reverse=True)

    clusters = cluster_events_into_units(sorted_events, max_units=planner_config.max_content_units)
    content_units = [build_content_unit(i + 1, cluster) for i, cluster in enumerate(clusters)]

    role_priority = {"main": 0, "detail": 1, "reaction": 2, "consequence": 3, "background": 4}
    content_units = sorted(
        content_units,
        key=lambda u: (role_priority.get(u.get("purpose", "detail"), 99), -float(u.get("quality", 0.0)))
    )

    for i, unit in enumerate(content_units, start=1):
        unit["priority"] = i

    must_entities = select_must_include_entities(planner_input)
    must_events = select_must_include_events(planner_input)

    if planner_config.force_entity_coverage and content_units:
        covered_entities = set()
        for unit in content_units:
            covered_entities.update(unit["entities"])

        missing_entities = [e for e in must_entities if e not in covered_entities]
        if missing_entities:
            content_units[-1]["entities"].extend(missing_entities)
            content_units[-1]["entities"] = unique_preserve_order(content_units[-1]["entities"])

    lead_event_id = content_units[0]["lead_event_id"] if content_units else ""

    planner_output = {
        "doc_id": safe_text(planner_input.get("doc_id")),
        "headline": safe_text(planner_input.get("headline")),
        "headline_en": safe_text(planner_input.get("headline_en")),
        "location": safe_text(planner_input.get("location")),
        "summary_focus": infer_summary_focus(planner_input),
        "lead_event_id": lead_event_id,
        "content_units": content_units,
        "must_include_entities": must_entities,
        "must_include_events": must_events,
        "ordering_strategy": "main_then_detail_then_reaction",
        "constraints": {
            "avoid_new_facts": True,
            "use_only_supported_evidence": True,
            "max_summary_sentences": 4,
            "summary_language": "Manipuri",
            "force_entity_coverage": planner_config.force_entity_coverage,
        },
    }
    return planner_output


def print_planner_output(planner_output: Dict[str, Any]) -> None:
    print("\n===== PLANNER OUTPUT =====")
    print(json.dumps(planner_output, ensure_ascii=False, indent=2))


# =========================================================
# REALIZER (REDESIGNED, STRICT, EVENT-FAITHFUL)
# =========================================================

def split_role_map(arguments: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    role_map: Dict[str, List[str]] = defaultdict(list)
    for arg in arguments:
        role = safe_text(arg.get("role", "ARG")).upper()
        val = safe_text(arg.get("text"))
        if val:
            role_map[role].append(val)
    return {k: unique_preserve_order(v) for k, v in role_map.items()}


def role_vals(role_map: Dict[str, List[str]], *keys: str) -> List[str]:
    out: List[str] = []
    for key in keys:
        out.extend(role_map.get(key.upper(), []))
    return unique_preserve_order(out)


def join_names(values: List[str], max_items: int = 4) -> str:
    vals = unique_preserve_order([safe_text(v) for v in values if safe_text(v)])
    if not vals:
        return ""
    return " অমসুং ".join(vals[:max_items])


def attach_evidence(sentence: str, evidence_ids: List[int], include_evidence: bool = True) -> str:
    sentence = ensure_period(sentence)
    if not include_evidence or not evidence_ids:
        return sentence
    marker = ", ".join(f"s{sid}" for sid in evidence_ids)
    return f"{sentence[:-1]} [{marker}]."


def build_generic_fact_sentence(event: Dict[str, Any]) -> str:
    trigger = safe_text(event.get("trigger"))
    if not trigger:
        return ""

    role_map = split_role_map(event.get("arguments", []))

    subject = join_names(role_vals(
        role_map,
        "AGENT", "PARTICIPANT", "GROUP", "SPEAKER",
        "ORGANIZER", "REQUESTER", "COMPLAINANT", "BENEFICIARY"
    ))

    obj = join_names(role_vals(
        role_map,
        "TARGET", "VICTIM", "SUSPECT", "AFFECTED", "OBJECT",
        "THEME", "TOPIC", "ISSUE", "CAUSE"
    ))

    loc = join_names(role_vals(role_map, "LOC", "PLACE", "DESTINATION"))
    time = join_names(role_vals(role_map, "TIME"))
    amount = join_names(role_vals(role_map, "AMOUNT", "COUNT", "QUANTITY"))
    org = join_names(role_vals(role_map, "ORG", "ORGANIZATION"))

    parts: List[str] = []

    if subject:
        parts.append(subject)
    if obj:
        parts.append(obj)
    if loc:
        parts.append(loc)
    if time:
        parts.append(time)
    if amount:
        parts.append(amount)
    if org:
        parts.append(org)

    parts.append(trigger)

    sentence = " ".join([p for p in parts if p]).strip()
    return ensure_period(sentence)


def realize_fact_by_type(event: Dict[str, Any]) -> str:
    event_type = clean_event_type(event.get("event_type", ""))
    trigger = safe_text(event.get("trigger"))
    if not trigger:
        return ""

    role_map = split_role_map(event.get("arguments", []))

    agent = join_names(role_vals(role_map, "AGENT", "SPEAKER", "REQUESTER", "ORGANIZER"))
    participant = join_names(role_vals(role_map, "PARTICIPANT", "GROUP", "BENEFICIARY"))
    target = join_names(role_vals(role_map, "TARGET", "VICTIM", "OBJECT", "AFFECTED", "SUSPECT"))
    theme = join_names(role_vals(role_map, "THEME", "TOPIC", "ISSUE", "CAUSE"))
    loc = join_names(role_vals(role_map, "LOC", "PLACE", "DESTINATION"))
    time = join_names(role_vals(role_map, "TIME"))
    amount = join_names(role_vals(role_map, "AMOUNT", "COUNT", "QUANTITY"))
    org = join_names(role_vals(role_map, "ORG", "ORGANIZATION"))

    if "SIGNATURE_COLLECTION" in event_type:
        group = participant or agent
        if group and loc:
            return ensure_period(f"{group}না {loc}দা {trigger}")
        if group:
            return ensure_period(f"{group}না {trigger}")
        return build_generic_fact_sentence(event)

    if any(k in event_type for k in ["PUBLIC_STATEMENT", "CLARIFICATION", "REACTION", "APPEAL", "DEMAND", "WARNING", "ANNOUNCEMENT"]):
        speaker = agent or participant
        if speaker and theme:
            return ensure_period(f"{speaker}না {theme}গী মরমদা {trigger}")
        if speaker and target:
            return ensure_period(f"{speaker}না {target}গী মরমদা {trigger}")
        if speaker:
            return ensure_period(f"{speaker}না {trigger}")
        return build_generic_fact_sentence(event)

    if any(k in event_type for k in ["PROTEST", "RALLY", "STRIKE", "MEETING", "DEMONSTRATION"]):
        group = participant or agent
        if group and loc:
            return ensure_period(f"{group}না {loc}দা {trigger}")
        if group:
            return ensure_period(f"{group}না {trigger}")
        return build_generic_fact_sentence(event)

    if any(k in event_type for k in ["ARREST", "DETENTION", "CAPTURE"]):
        if agent and target and loc:
            return ensure_period(f"{agent}না {loc}দা {target} {trigger}")
        if target and loc:
            return ensure_period(f"{loc}দা {target} {trigger}")
        if target:
            return ensure_period(f"{target} {trigger}")
        return build_generic_fact_sentence(event)

    if any(k in event_type for k in ["ATTACK", "EXPLOSION", "KILLING", "INJURY", "CASUALTY", "SHOOTING"]):
        if target and loc:
            return ensure_period(f"{loc}দা {target}গী মরমদা {trigger}")
        if loc:
            return ensure_period(f"{loc}দা {trigger}")
        if target:
            return ensure_period(f"{target}গী মরমদা {trigger}")
        return build_generic_fact_sentence(event)

    if any(k in event_type for k in ["SECURITY_REQUEST", "SECURITY_DEPLOYMENT"]):
        requester = agent or participant
        if requester and target:
            return ensure_period(f"{requester}না {target}গী মরমদা {trigger}")
        if requester and loc:
            return ensure_period(f"{requester}না {loc}দা {trigger}")
        return build_generic_fact_sentence(event)

    if any(k in event_type for k in ["TRANSFER", "POLICY_CHANGE", "PLANNING", "SUPPORT", "RELIEF", "REWARD", "TREATMENT", "TRAINING"]):
        if agent and target:
            return ensure_period(f"{agent}না {target}গী মরমদা {trigger}")
        if target:
            return ensure_period(f"{target}গী মরমদা {trigger}")
        return build_generic_fact_sentence(event)

    if any(k in event_type for k in ["CROSS_BORDER_MOVEMENT", "MOVEMENT", "TRAVEL"]):
        mover = agent or participant or target
        if mover and loc:
            return ensure_period(f"{mover} {loc}দা {trigger}")
        if mover:
            return ensure_period(f"{mover} {trigger}")
        return build_generic_fact_sentence(event)

    if amount and target and trigger:
        return ensure_period(f"{target} {amount} {trigger}")

    return build_generic_fact_sentence(event)


def realize_fact_strict(event: Dict[str, Any], include_evidence: bool = True) -> Dict[str, Any]:
    text = realize_fact_by_type(event)
    text = attach_evidence(text, event.get("evidence_ids", []), include_evidence=include_evidence)

    return {
        "event_id": safe_text(event.get("event_id")),
        "event_type": safe_text(event.get("event_type")),
        "text": text,
        "evidence_ids": event.get("evidence_ids", []),
    }


def deduplicate_realized_facts(realized_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []

    for fact in realized_facts:
        text = normalize_ws(fact.get("text", ""))
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(fact)

    return output


def select_summary_facts(realized_facts: List[Dict[str, Any]], max_sentences: int = 4) -> List[Dict[str, Any]]:
    return realized_facts[:max_sentences]


def realize_unit(
    unit: Dict[str, Any],
    cite_evidence: bool = True,
    max_facts_per_unit: int = 2,
) -> List[Dict[str, Any]]:
    facts = unit.get("atomic_facts", [])
    if not facts:
        return []

    facts = sorted(
        facts,
        key=lambda f: float(f.get("quality", f.get("salience", 0.0))),
        reverse=True
    )

    realized_facts: List[Dict[str, Any]] = []
    for fact in facts[:max_facts_per_unit]:
        realized = realize_fact_strict(fact, include_evidence=cite_evidence)
        if safe_text(realized.get("text")):
            realized_facts.append(realized)

    return realized_facts


def realize_summary_from_plan(
    planner_output: Dict[str, Any],
    realizer_config: Optional[RealizerConfig] = None,
) -> Dict[str, Any]:
    realizer_config = realizer_config or RealizerConfig()

    units = sorted(planner_output.get("content_units", []), key=lambda u: int(u.get("priority", 999)))
    if not units:
        return {
            "doc_id": planner_output.get("doc_id"),
            "headline": planner_output.get("headline"),
            "summary_mni": "",
            "sentences": [],
            "realized_facts": [],
            "grounded": True,
        }

    all_realized_facts: List[Dict[str, Any]] = []
    for unit in units:
        unit_facts = realize_unit(
            unit,
            cite_evidence=realizer_config.cite_evidence,
            max_facts_per_unit=realizer_config.max_facts_per_unit,
        )
        all_realized_facts.extend(unit_facts)

    all_realized_facts = deduplicate_realized_facts(all_realized_facts)
    selected_facts = select_summary_facts(all_realized_facts, max_sentences=realizer_config.max_sentences)

    realized_sentences = [safe_text(f.get("text")) for f in selected_facts if safe_text(f.get("text"))]
    summary_mni = " ".join(realized_sentences).strip()

    return {
        "doc_id": safe_text(planner_output.get("doc_id")),
        "headline": safe_text(planner_output.get("headline")),
        "location": safe_text(planner_output.get("location")),
        "summary_focus": safe_text(planner_output.get("summary_focus")),
        "summary_language": "Manipuri",
        "summary_mni": summary_mni,
        "sentences": realized_sentences,
        "realized_facts": selected_facts,
        "grounded": True,
        "evidence_style": "inline_sentence_ids" if realizer_config.cite_evidence else "none",
    }


def print_realized_output(realized_output: Dict[str, Any]) -> None:
    print("\n===== REALIZER OUTPUT =====")
    print(json.dumps(realized_output, ensure_ascii=False, indent=2))
    print("\n===== FINAL SUMMARY (MANIPURI) =====")
    print(realized_output.get("summary_mni", ""))


# =========================================================
# EVENT TYPE INSPECTION
# =========================================================

def get_all_event_types(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for doc in docs:
        for ev in doc.get("events", []):
            et = clean_event_type(ev.get("type", "UNKNOWN"))
            counts[et] += 1
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def print_event_type_inventory(docs: List[Dict[str, Any]]) -> None:
    counts = get_all_event_types(docs)
    print("\n===== EVENT TYPE INVENTORY =====")
    for et, cnt in counts.items():
        print(f"{et:35s} -> {cnt}")


# =========================================================
# VERIFY
# =========================================================

def verify_summary_against_plan(realized_output: Dict[str, Any], planner_output: Dict[str, Any]) -> Dict[str, Any]:
    must_entities = [e for e in planner_output.get("must_include_entities", []) if e]
    must_events = set(planner_output.get("must_include_events", []))
    text = safe_text(realized_output.get("summary_mni"))

    included_entities = [e for e in must_entities if e in text]
    entity_coverage = len(included_entities) / max(1, len(must_entities))

    covered_event_ids = set()
    for fact in realized_output.get("realized_facts", []):
        trig_text = safe_text(fact.get("text"))
        event_id = safe_text(fact.get("event_id"))
        if trig_text and event_id:
            covered_event_ids.add(event_id)

    event_coverage = len(covered_event_ids & must_events) / max(1, len(must_events))
    consistency_score = round((entity_coverage + event_coverage) / 2, 4)

    return {
        "doc_id": safe_text(realized_output.get("doc_id")),
        "entity_coverage": round(entity_coverage, 4),
        "event_coverage": round(event_coverage, 4),
        "consistency_score": consistency_score,
        "included_entities": included_entities,
        "covered_event_ids": sorted(covered_event_ids),
    }


def print_verification(verification: Dict[str, Any]) -> None:
    print("\n===== VERIFICATION =====")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


# =========================================================
# PROCESSING
# =========================================================

def process_single_document(
    doc: Dict[str, Any],
    builder: EventEntityGraphBuilder,
    planner_config: PlannerConfig,
    realizer_config: RealizerConfig,
    visualize: bool,
    print_planner_ready: bool,
    save_planner_ready: bool,
    planner_ready_output_dir: str,
    print_plan: bool,
    save_plan: bool,
    plan_output_dir: str,
    print_realizer: bool,
    save_realizer: bool,
    realizer_output_dir: str,
    print_verify: bool,
    save_verify: bool,
    verify_output_dir: str,
) -> None:
    graph = builder.build_graph(doc)

    print_graph_summary(graph)
    print_salience(graph)

    planner_input = build_planner_ready_input(doc, graph, planner_config=planner_config)
    if print_planner_ready:
        print_planner_ready_input(planner_input)

    if save_planner_ready:
        out_path = Path(planner_ready_output_dir) / f"{safe_text(planner_input.get('doc_id'))}_planner_ready.json"
        print(f"\nPlanner-ready JSON saved to: {save_json(planner_input, out_path)}")

    planner_output = build_planner_output(planner_input, planner_config=planner_config)
    if print_plan:
        print_planner_output(planner_output)

    if save_plan:
        out_path = Path(plan_output_dir) / f"{safe_text(planner_output.get('doc_id'))}_content_plan.json"
        print(f"\nContent plan JSON saved to: {save_json(planner_output, out_path)}")

    realized_output = realize_summary_from_plan(planner_output, realizer_config=realizer_config)
    if print_realizer:
        print_realized_output(realized_output)

    if save_realizer:
        out_path = Path(realizer_output_dir) / f"{safe_text(realized_output.get('doc_id'))}_realized_summary.json"
        print(f"\nRealized summary JSON saved to: {save_json(realized_output, out_path)}")

    verification = verify_summary_against_plan(realized_output, planner_output)
    if print_verify:
        print_verification(verification)

    if save_verify:
        out_path = Path(verify_output_dir) / f"{safe_text(verification.get('doc_id'))}_verification.json"
        print(f"\nVerification JSON saved to: {save_json(verification, out_path)}")

    if visualize:
        visualize_graph(graph)


def process_dataset(
    file_path: str,
    graph_config: Optional[GraphBuildConfig] = None,
    planner_config: Optional[PlannerConfig] = None,
    realizer_config: Optional[RealizerConfig] = None,
    visualize: bool = True,
    max_docs: Optional[int] = None,
    print_planner_ready: bool = False,
    save_planner_ready: bool = True,
    planner_ready_output_dir: str = "planner_ready_outputs",
    print_plan: bool = True,
    save_plan: bool = True,
    plan_output_dir: str = "planner_outputs",
    print_realizer: bool = True,
    save_realizer: bool = True,
    realizer_output_dir: str = "realizer_outputs",
    print_verify: bool = True,
    save_verify: bool = True,
    verify_output_dir: str = "verification_outputs",
) -> None:
    docs = load_dataset_json(file_path)
    if max_docs is not None:
        docs = docs[:max_docs]

    builder = EventEntityGraphBuilder(config=graph_config or GraphBuildConfig())
    planner_config = planner_config or PlannerConfig()
    realizer_config = realizer_config or RealizerConfig()

    print(f"\nLoaded {len(docs)} document(s) from {file_path}\n")

    for idx, doc in enumerate(docs, start=1):
        try:
            print(f"\n########## DOCUMENT {idx}/{len(docs)} ##########")
            process_single_document(
                doc=doc,
                builder=builder,
                planner_config=planner_config,
                realizer_config=realizer_config,
                visualize=visualize,
                print_planner_ready=print_planner_ready,
                save_planner_ready=save_planner_ready,
                planner_ready_output_dir=planner_ready_output_dir,
                print_plan=print_plan,
                save_plan=save_plan,
                plan_output_dir=plan_output_dir,
                print_realizer=print_realizer,
                save_realizer=save_realizer,
                realizer_output_dir=realizer_output_dir,
                print_verify=print_verify,
                save_verify=save_verify,
                verify_output_dir=verify_output_dir,
            )
        except Exception as exc:
            doc_id = safe_text(doc.get("doc_id", f"UNKNOWN_{idx}")) if isinstance(doc, dict) else f"UNKNOWN_{idx}"
            print(f"\nError processing document {doc_id}: {exc}")


# =========================================================
# MENU
# =========================================================

def print_menu() -> None:
    print("\n" + "=" * 72)
    print("IMPROVED MANIPURI EVENT-ENTITY GRAPH + PLANNER + REALIZER MENU")
    print("=" * 72)
    print("1. Process all documents")
    print("2. Process first N documents")
    print("3. Process one document by index")
    print("4. Toggle visualization ON/OFF")
    print("5. Toggle save planner-ready JSON ON/OFF")
    print("6. Toggle save planner output ON/OFF")
    print("7. Toggle save realizer output ON/OFF")
    print("8. Toggle save verification output ON/OFF")
    print("9. Toggle evidence markers ON/OFF")
    print("10. Show current settings")
    print("11. Show all event types from dataset")
    print("12. Run Ablation Study")
    print("0. Exit")
    print("=" * 72)


def run_menu() -> None:
    dataset_path = "cleaned_dataset.json"

    graph_config = GraphBuildConfig(
        add_sentence_nodes=False,
        add_entity_entity_cooccurrence=True,
        add_event_event_temporal=True,
    )

    planner_config = PlannerConfig(
        top_events=8,
        top_entities=12,
        max_content_units=5,
        min_event_salience=0.0,
        min_event_arguments=2,
        force_entity_coverage=True,
    )

    realizer_config = RealizerConfig(
        max_sentences=4,
        cite_evidence=True,
        include_location_in_lead=False,
        max_facts_per_unit=2,
        keep_one_sentence_per_fact=True,
        allow_unit_level_merge=False,
    )

    visualize = True
    save_planner_ready = True
    save_plan = True
    save_realizer = True
    save_verify = True

    print_planner_ready = False
    print_plan = True
    print_realizer = True
    print_verify = True

    planner_ready_output_dir = "planner_ready_outputs"
    plan_output_dir = "planner_outputs"
    realizer_output_dir = "realizer_outputs"
    verify_output_dir = "verification_outputs"

    docs = load_dataset_json(dataset_path)
    builder = EventEntityGraphBuilder(config=graph_config)

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            process_dataset(
                file_path=dataset_path,
                graph_config=graph_config,
                planner_config=planner_config,
                realizer_config=realizer_config,
                visualize=visualize,
                max_docs=None,
                print_planner_ready=print_planner_ready,
                save_planner_ready=save_planner_ready,
                planner_ready_output_dir=planner_ready_output_dir,
                print_plan=print_plan,
                save_plan=save_plan,
                plan_output_dir=plan_output_dir,
                print_realizer=print_realizer,
                save_realizer=save_realizer,
                realizer_output_dir=realizer_output_dir,
                print_verify=print_verify,
                save_verify=save_verify,
                verify_output_dir=verify_output_dir,
            )

        elif choice == "2":
            n_text = input("Enter N: ").strip()
            try:
                n = int(n_text)
                process_dataset(
                    file_path=dataset_path,
                    graph_config=graph_config,
                    planner_config=planner_config,
                    realizer_config=realizer_config,
                    visualize=visualize,
                    max_docs=n,
                    print_planner_ready=print_planner_ready,
                    save_planner_ready=save_planner_ready,
                    planner_ready_output_dir=planner_ready_output_dir,
                    print_plan=print_plan,
                    save_plan=save_plan,
                    plan_output_dir=plan_output_dir,
                    print_realizer=print_realizer,
                    save_realizer=save_realizer,
                    realizer_output_dir=realizer_output_dir,
                    print_verify=print_verify,
                    save_verify=save_verify,
                    verify_output_dir=verify_output_dir,
                )
            except ValueError:
                print("Invalid number.")

        elif choice == "3":
            idx_text = input(f"Enter document index (1 to {len(docs)}): ").strip()
            try:
                idx = int(idx_text)
                if idx < 1 or idx > len(docs):
                    print("Index out of range.")
                    continue

                print(f"\n########## DOCUMENT {idx}/{len(docs)} ##########")
                process_single_document(
                    doc=docs[idx - 1],
                    builder=builder,
                    planner_config=planner_config,
                    realizer_config=realizer_config,
                    visualize=visualize,
                    print_planner_ready=print_planner_ready,
                    save_planner_ready=save_planner_ready,
                    planner_ready_output_dir=planner_ready_output_dir,
                    print_plan=print_plan,
                    save_plan=save_plan,
                    plan_output_dir=plan_output_dir,
                    print_realizer=print_realizer,
                    save_realizer=save_realizer,
                    realizer_output_dir=realizer_output_dir,
                    print_verify=print_verify,
                    save_verify=save_verify,
                    verify_output_dir=verify_output_dir,
                )
            except ValueError:
                print("Invalid index.")

        elif choice == "4":
            visualize = not visualize
            print(f"Visualization set to: {visualize}")

        elif choice == "5":
            save_planner_ready = not save_planner_ready
            print(f"Save planner-ready JSON set to: {save_planner_ready}")

        elif choice == "6":
            save_plan = not save_plan
            print(f"Save planner output JSON set to: {save_plan}")

        elif choice == "7":
            save_realizer = not save_realizer
            print(f"Save realizer output JSON set to: {save_realizer}")

        elif choice == "8":
            save_verify = not save_verify
            print(f"Save verification output JSON set to: {save_verify}")

        elif choice == "9":
            realizer_config.cite_evidence = not realizer_config.cite_evidence
            print(f"Evidence markers set to: {realizer_config.cite_evidence}")

        elif choice == "10":
            print("\nCurrent settings:")
            print(f"  Dataset path              : {dataset_path}")
            print(f"  Visualization             : {visualize}")
            print(f"  Save planner-ready JSON   : {save_planner_ready}")
            print(f"  Save planner JSON         : {save_plan}")
            print(f"  Save realizer JSON        : {save_realizer}")
            print(f"  Save verification JSON    : {save_verify}")
            print(f"  Evidence markers          : {realizer_config.cite_evidence}")
            print(f"  Planner-ready dir         : {planner_ready_output_dir}")
            print(f"  Planner output dir        : {plan_output_dir}")
            print(f"  Realizer output dir       : {realizer_output_dir}")
            print(f"  Verification output dir   : {verify_output_dir}")
            print(f"  Total documents           : {len(docs)}")
            print(f"  top_events                : {planner_config.top_events}")
            print(f"  top_entities              : {planner_config.top_entities}")
            print(f"  max_content_units         : {planner_config.max_content_units}")
            print(f"  min_event_arguments       : {planner_config.min_event_arguments}")
            print(f"  force_entity_coverage     : {planner_config.force_entity_coverage}")
            print(f"  max_summary_sentences     : {realizer_config.max_sentences}")
            print(f"  max_facts_per_unit        : {realizer_config.max_facts_per_unit}")
            print(f"  one_sentence_per_fact     : {realizer_config.keep_one_sentence_per_fact}")
            print(f"  allow_unit_merge          : {realizer_config.allow_unit_level_merge}")

        elif choice == "11":
            print_event_type_inventory(docs)
        elif choice == "12":
            run_ablation_study(
                docs=docs,
                builder=builder,
                planner_config=planner_config,
                realizer_config=realizer_config,
                max_docs=20,
            )

        elif choice == "0":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please try again.")
# =========================================================
# ABLATION STUDY
# =========================================================

def run_ablation_study(
    docs: List[Dict[str, Any]],
    builder: EventEntityGraphBuilder,
    planner_config: PlannerConfig,
    realizer_config: RealizerConfig,
    max_docs: int = 20,
):
    print("\n===== RUNNING ABLATION STUDY =====")

    results = {}

    variants = {
        "full": {"planner": True, "verify": True, "features": True},
        "no_planner": {"planner": False, "verify": True, "features": True},
        "no_verification": {"planner": True, "verify": False, "features": True},
        "no_features": {"planner": True, "verify": True, "features": False},
    }

    for variant_name, config in variants.items():
        print(f"\n--- Variant: {variant_name} ---")

        entity_covs = []
        event_covs = []
        consistency_scores = []

        for doc in docs[:max_docs]:
            graph = builder.build_graph(doc)

            # Remove structured features (simulate weak system)
            if not config["features"]:
                for n in list(graph.nodes()):
                    if graph.nodes[n].get("node_type") == "event":
                        graph.nodes[n]["confidence"] = 0.1

            planner_input = build_planner_ready_input(doc, graph, planner_config)

            # Skip planner
            if config["planner"]:
                planner_output = build_planner_output(planner_input, planner_config)
            else:
                planner_output = {
                    "doc_id": planner_input["doc_id"],
                    "content_units": [{
                        "priority": 1,
                        "atomic_facts": planner_input["salient_events"]
                    }],
                    "must_include_entities": [],
                    "must_include_events": [],
                }

            realized_output = realize_summary_from_plan(planner_output, realizer_config)

            if config["verify"]:
                verification = verify_summary_against_plan(realized_output, planner_output)
                entity_covs.append(verification["entity_coverage"])
                event_covs.append(verification["event_coverage"])
                consistency_scores.append(verification["consistency_score"])

        results[variant_name] = {
            "entity_cov": round(sum(entity_covs)/len(entity_covs), 4),
            "event_cov": round(sum(event_covs)/len(event_covs), 4),
            "consistency": round(sum(consistency_scores)/len(consistency_scores), 4),
        }

    print("\n===== ABLATION RESULTS =====")
    for k, v in results.items():
        print(f"{k:20s} | EC={v['entity_cov']} | EvC={v['event_cov']} | CS={v['consistency']}")

    return results

if __name__ == "__main__":
    run_menu()