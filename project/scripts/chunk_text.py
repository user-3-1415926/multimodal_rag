from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List

from .common import CHUNK_DIR, RAW_TEXT_DIR, ensure_directories


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paragraphs:
        return paragraphs
    return [text.strip()] if text.strip() else []


def _windowed_chunks(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    words = re.split(r"\s+", text.strip())
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]
        if window:
            chunks.append(" ".join(window))
    return chunks


def _infer_chunk_type(idx: int) -> str:
    if idx == 0:
        return "category"
    if idx == 1:
        return "detail"
    if idx == 2:
        return "location"
    return "description"


def chunk_text(item_id: str, chunk_size: int = 200, overlap: int = 40) -> List[Dict]:
    """
    切分文本并将结果写入 data/chunks/<item_id>.jsonl
    """
    ensure_directories()
    text_path = RAW_TEXT_DIR / f"{item_id}.txt"
    if not text_path.exists():
        raise FileNotFoundError(f"Raw text not found: {text_path}")

    raw_text = text_path.read_text(encoding="utf-8")
    paragraphs = _split_paragraphs(raw_text)

    chunks: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
        else:
            chunks.extend(_windowed_chunks(paragraph, chunk_size=chunk_size, overlap=overlap))

    if not chunks:
        chunks = [raw_text.strip()]

    chunk_records: List[Dict] = []
    jsonl_path = CHUNK_DIR / f"{item_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, chunk_text_content in enumerate(chunks, start=1):
            chunk_id = f"{item_id}-{idx}"
            payload = {
                "chunk_id": chunk_id,
                "parent": item_id,
                "text": chunk_text_content,
                "type": _infer_chunk_type(idx - 1),
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
            chunk_records.append(payload)

    return chunk_records

