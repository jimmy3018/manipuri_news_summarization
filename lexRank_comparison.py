# -*- coding: utf-8 -*-
"""
Standalone LexRank baseline for Manipuri news summarization.

Reads:
    cleaned_dataset_with_manipuri_summary.json

Writes:
    baseline_outputs/lexrank_outputs/<doc_id>.json

@author: jimmy
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "cleaned_dataset_with_manipuri_summary.json"
OUTPUT_DIR = "baseline_outputs/lexrank_outputs"


@dataclass
class LexRankConfig:
    top_k_sentences: int = 3
    similarity_threshold: float = 0.1


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    data = load_json(dataset_path)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Dataset must be a JSON object or a list of JSON objects.")


def tokenize(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [tok for tok in text.split() if tok.strip()]


def sentence_texts_from_doc(doc: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    for sent in doc.get("sentences", []):
        txt = normalize_ws(sent.get("text", ""))
        if txt:
            sentences.append(txt)
    return sentences


# =========================================================
# SIMILARITY
# =========================================================

def cosine_sim_from_token_overlap(a: str, b: str) -> float:
    """
    Lightweight cosine similarity using token-frequency overlap.
    """
    ta = tokenize(a)
    tb = tokenize(b)

    if not ta or not tb:
        return 0.0

    vocab = sorted(set(ta) | set(tb))
    idx = {word: i for i, word in enumerate(vocab)}

    va = np.zeros(len(vocab), dtype=float)
    vb = np.zeros(len(vocab), dtype=float)

    for word in ta:
        va[idx[word]] += 1.0
    for word in tb:
        vb[idx[word]] += 1.0

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))


def build_similarity_matrix(sentences: List[str]) -> np.ndarray:
    n = len(sentences)
    sim = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim[i, j] = cosine_sim_from_token_overlap(sentences[i], sentences[j])

    return sim


# =========================================================
# LEXRANK
# =========================================================

def build_lexrank_graph(sentences: List[str], threshold: float) -> nx.Graph:
    """
    Build LexRank graph using thresholded sentence similarity.
    """
    sim = build_similarity_matrix(sentences)
    n = len(sentences)

    binary_adj = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j and sim[i, j] >= threshold:
                binary_adj[i, j] = 1.0

    graph = nx.from_numpy_array(binary_adj)
    return graph


def rank_sentences_lexrank(sentences: List[str], threshold: float = 0.1) -> List[int]:
    """
    Rank sentences using LexRank centrality.
    """
    if not sentences:
        return []

    graph = build_lexrank_graph(sentences, threshold)
    scores = nx.pagerank(graph, weight="weight")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked]


def select_top_sentences(sentences: List[str], ranked_idxs: List[int], top_k: int) -> List[str]:
    """
    Select top-ranked sentences and restore original order.
    """
    chosen_idxs = sorted(ranked_idxs[:top_k])
    return [sentences[i] for i in chosen_idxs]


def summarize_lexrank(doc: Dict[str, Any], config: LexRankConfig) -> Dict[str, Any]:
    doc_id = safe_text(doc.get("doc_id"))
    sentences = sentence_texts_from_doc(doc)

    if not sentences:
        return {
            "doc_id": doc_id,
            "model": "LexRank",
            "summary_mni": "",
            "selected_sentence_count": 0
        }

    ranked_idxs = rank_sentences_lexrank(
        sentences,
        threshold=config.similarity_threshold
    )
    selected_sentences = select_top_sentences(
        sentences,
        ranked_idxs,
        config.top_k_sentences
    )
    summary = " ".join(selected_sentences)

    return {
        "doc_id": doc_id,
        "model": "LexRank",
        "summary_mni": normalize_ws(summary),
        "selected_sentence_count": len(selected_sentences),
        "selected_sentences": selected_sentences
    }


# =========================================================
# SAVE
# =========================================================

def save_lexrank_output(output_dir: str, result: Dict[str, Any]) -> None:
    doc_id = safe_text(result.get("doc_id", "UNKNOWN_DOC"))
    out_path = Path(output_dir) / f"{doc_id}.json"
    save_json(result, out_path)


# =========================================================
# MAIN
# =========================================================

def run_lexrank_baseline(
    dataset_path: str,
    output_dir: str,
    config: Optional[LexRankConfig] = None,
    max_docs: Optional[int] = None
) -> None:
    config = config or LexRankConfig()
    docs = load_dataset(dataset_path)

    if max_docs is not None:
        docs = docs[:max_docs]

    print(f"Loaded {len(docs)} documents")

    for idx, doc in enumerate(docs, start=1):
        doc_id = safe_text(doc.get("doc_id", f"DOC_{idx}"))
        print(f"Processing {idx}/{len(docs)} -> {doc_id}")

        result = summarize_lexrank(doc, config)
        save_lexrank_output(output_dir, result)

    print("\nDone.")
    print(f"LexRank outputs saved to: {output_dir}")


if __name__ == "__main__":
    cfg = LexRankConfig(
        top_k_sentences=3,
        similarity_threshold=0.1
    )

    run_lexrank_baseline(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        config=cfg,
        max_docs=None
    )