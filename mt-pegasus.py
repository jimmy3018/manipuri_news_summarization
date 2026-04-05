# -*- coding: utf-8 -*-
"""
Standalone mT5 baseline for Manipuri summarization.

Reads:
    cleaned_dataset_with_manipuri_summary.json

Writes:
    baseline_outputs/mt5_outputs/<doc_id>.json

Requires:
    pip install torch transformers sentencepiece

@author: jimmy
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "cleaned_dataset_with_manipuri_summary.json"
OUTPUT_DIR = "baseline_outputs/mt5_outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MT5_MODEL_NAME = "google/mt5-small"


@dataclass
class MT5Config:
    min_new_tokens: int = 20
    max_new_tokens: int = 80
    num_beams: int = 4
    max_input_length: int = 1024
    prompt_prefix: str = "summarize: "


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


def sentence_texts_from_doc(doc: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    for sent in doc.get("sentences", []):
        txt = normalize_ws(sent.get("text", ""))
        if txt:
            sentences.append(txt)
    return sentences


def get_document_text(doc: Dict[str, Any]) -> str:
    return " ".join(sentence_texts_from_doc(doc))


# =========================================================
# MODEL
# =========================================================

class MT5Summarizer:
    def __init__(self, model_name: str, config: MT5Config):
        self.model_name = model_name
        self.config = config

        print(f"Loading mT5 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    def summarize(self, text: str) -> str:
        text = normalize_ws(text)
        if not text:
            return ""

        prompt = f"{self.config.prompt_prefix}{text}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=self.config.min_new_tokens,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                early_stopping=True
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return normalize_ws(summary)


# =========================================================
# PIPELINE
# =========================================================

def summarize_with_mt5(
    doc: Dict[str, Any],
    summarizer: MT5Summarizer
) -> Dict[str, Any]:
    doc_id = safe_text(doc.get("doc_id"))
    source_text = get_document_text(doc)

    summary = summarizer.summarize(source_text)

    return {
        "doc_id": doc_id,
        "model": "mT5",
        "source_text": source_text,
        "summary_mni": summary
    }


# =========================================================
# RUNNER
# =========================================================

def run_mt5_baseline(
    dataset_path: str,
    output_dir: str,
    config: Optional[MT5Config] = None,
    max_docs: Optional[int] = None
) -> None:
    config = config or MT5Config()
    docs = load_dataset(dataset_path)

    if max_docs is not None:
        docs = docs[:max_docs]

    print(f"Loaded {len(docs)} documents")

    summarizer = MT5Summarizer(MT5_MODEL_NAME, config)

    for idx, doc in enumerate(docs, start=1):
        doc_id = safe_text(doc.get("doc_id", f"DOC_{idx}"))
        print(f"Processing {idx}/{len(docs)} -> {doc_id}")

        try:
            result = summarize_with_mt5(doc, summarizer)
            save_json(result, Path(output_dir) / f"{doc_id}.json")
        except Exception as exc:
            print(f"Failed on {doc_id}: {exc}")

    print("\nDone.")
    print(f"mT5 outputs saved to: {output_dir}")


if __name__ == "__main__":
    cfg = MT5Config(
        min_new_tokens=20,
        max_new_tokens=80,
        num_beams=4,
        max_input_length=1024,
        prompt_prefix="summarize: "
    )

    run_mt5_baseline(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        config=cfg,
        max_docs=None
    )